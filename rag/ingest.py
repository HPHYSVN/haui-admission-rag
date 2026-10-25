"""Safe, review-preserving source ingestion for the HaUI corpus.

The corpus is curated: nothing in this module promotes a document to the index or
marks it as fact. It only writes a source record and records a workflow state, so
the existing human review and chunk-review pipeline still decides what reaches
retrieval.

Two boundaries are enforced here:

* **Local files** are resolved first (so a symlink cannot escape an allowed root),
  must be regular files, and are size-limited before parsing.
* **URLs** are limited to ``http(s)``, reject credentials, loopback/private/
  link-local/reserved addresses, re-validate every redirect, and are read with a
  hard byte cap. This blocks the usual SSRF pivots (``localhost``, ``169.254``
  metadata services, decimal IPs, and redirects into private networks).

DNS rebinding between validation and connection is a residual risk of any
``requests``-based fetcher; run ingestion in a network namespace with no access
to internal services when the source is untrusted.
"""

from __future__ import annotations

import hashlib
import io
import ipaddress
import json
import re
import socket
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests

from rag.config import DEFAULT_DOCUMENTS_DIR
from rag.document_workflow import DEFAULT_STATE_FILE, review_document, set_state

#: Hard ceiling for one imported source, before and after decoding.
MAX_SOURCE_BYTES = 5 * 1024 * 1024
#: Hard ceiling for extracted text (keeps prompts and review files manageable).
MAX_CONTENT_CHARS = 500_000
#: Redirect chain limit. Every hop is validated again.
MAX_REDIRECTS = 3
#: Accepted suffix for local imports. PDF is parsed only when pdfplumber is present.
ALLOWED_FILE_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".html", ".htm", ".json", ".pdf"})
ALLOWED_CONTENT_TYPES = frozenset(
    {
        "text/html",
        "application/xhtml+xml",
        "text/plain",
        "text/markdown",
        "application/json",
        "application/pdf",
    }
)
VALID_STATUSES = frozenset({"active", "archived", "draft"})
VALID_REVIEW_ACTIONS = frozenset({"pending", "approve", "reject"})
CATEGORY_RE = re.compile(r"^[0-9]{2}_[a-z0-9_]+$")
DOCUMENT_ID_RE = re.compile(r"^(?P<category>[0-9]{2}_[a-z0-9_]+)_(?P<number>[0-9]{3,})$")


class IngestError(ValueError):
    """The source, category, or duplicate state cannot be imported safely."""


@dataclass(frozen=True)
class FetchedSource:
    data: bytes
    final_url: str
    content_type: str
    title_hint: str = ""


@dataclass(frozen=True)
class ExtractedDocument:
    content: str
    title_hint: str = ""


@dataclass(frozen=True)
class IngestResult:
    document: dict[str, Any]
    path: Path
    created: bool
    duplicate_of: str | None = None
    wrote: bool = False
    context: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Path and category safety
# ---------------------------------------------------------------------------


def validate_category(category: str, documents_dir: Path = DEFAULT_DOCUMENTS_DIR) -> str:
    """Return a safe category slug that can never traverse out of the corpus."""
    if not isinstance(category, str) or not CATEGORY_RE.fullmatch(category):
        raise IngestError(
            "Category must look like '01_admission' (two digits, underscore, lowercase slug)"
        )
    directory = Path(documents_dir)
    target = (directory / f"{category}.jsonl").resolve()
    try:
        target.relative_to(directory.resolve())
    except ValueError as exc:  # defensive: the regex already removes separators
        raise IngestError("Category resolves outside the documents directory") from exc
    return category


def document_path(category: str, documents_dir: Path = DEFAULT_DOCUMENTS_DIR) -> Path:
    category = validate_category(category, documents_dir)
    return Path(documents_dir) / f"{category}.jsonl"


def resolve_local_file(source: str, *, allowed_root: Path | None = None) -> Path:
    """Resolve a local import path, rejecting non-files and root escapes."""
    candidate = Path(source).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise IngestError(f"Source file not found: {source}") from exc
    if not resolved.is_file():
        raise IngestError(f"Source is not a regular file: {source}")
    if allowed_root is not None:
        root = Path(allowed_root).expanduser().resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise IngestError("Source file is outside the allowed import root") from exc
    suffix = resolved.suffix.casefold()
    if suffix not in ALLOWED_FILE_SUFFIXES:
        raise IngestError(
            f"Unsupported file type {suffix or '(none)'}; "
            f"allowed: {', '.join(sorted(ALLOWED_FILE_SUFFIXES))}"
        )
    if resolved.stat().st_size > MAX_SOURCE_BYTES:
        raise IngestError(f"Source file exceeds {MAX_SOURCE_BYTES} bytes")
    return resolved


# ---------------------------------------------------------------------------
# URL safety and fetching
# ---------------------------------------------------------------------------


def validate_url(url: str) -> str:
    """Validate an http(s) URL and reject every non-public destination.

    Resolution happens during validation, so a hostname that points at
    ``127.0.0.1`` or a private range is rejected before any request is sent.
    """
    if not isinstance(url, str) or not url.strip():
        raise IngestError("URL must not be empty")
    parsed = urlsplit(url.strip())
    if parsed.scheme.casefold() not in {"http", "https"}:
        raise IngestError("Only http:// and https:// URLs are allowed")
    if parsed.username or parsed.password:
        raise IngestError("URLs with embedded credentials are not allowed")
    hostname = parsed.hostname
    if not hostname:
        raise IngestError("URL has no hostname")
    host = hostname.rstrip(".").casefold()
    if _is_blocked_hostname(host):
        raise IngestError("URL points at a local or reserved hostname")
    if _is_blocked_ip(host):
        raise IngestError("URL points at a private, loopback, or reserved address")

    # Resolve once up front so DNS names cannot hide an internal target. The
    # actual connection resolves again; see the module docstring for the caveat.
    if not _is_ip_literal(host):
        port = parsed.port or (443 if parsed.scheme.casefold() == "https" else 80)
        try:
            resolved = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise IngestError(f"Could not resolve URL hostname: {host}") from exc
        if not resolved:
            raise IngestError(f"Could not resolve URL hostname: {host}")
        for info in resolved:
            address = str(info[4][0]).split("%", 1)[0]
            if _is_blocked_ip(address):
                raise IngestError("URL hostname resolves to a local or reserved address")
    return urlunsplit(parsed)


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _is_blocked_ip(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        # Not an IP literal (a hostname); hostname checks happen separately.
        return False
    return not address.is_global


def _is_blocked_hostname(host: str) -> bool:
    if host in {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "metadata",
        "metadata.google.internal",
    }:
        return True
    return host.endswith((".localhost", ".local", ".internal", ".home.arpa"))


def fetch_url(
    url: str,
    *,
    session: requests.Session | None = None,
    timeout: tuple[float, float] = (5.0, 20.0),
    max_bytes: int = MAX_SOURCE_BYTES,
) -> FetchedSource:
    """Download one public source with redirect and size validation."""
    active = session or requests.Session()
    current = validate_url(url)
    headers = {
        "User-Agent": "HaUI-Admissions-Ingest/1.0 (+https://haui.edu.vn)",
        "Accept": "text/html, text/plain, application/json, application/pdf;q=0.9",
    }
    for _ in range(MAX_REDIRECTS + 1):
        try:
            response = active.get(
                current,
                headers=headers,
                timeout=timeout,
                allow_redirects=False,
                stream=True,
            )
        except requests.RequestException as exc:
            raise IngestError(f"Could not fetch URL ({type(exc).__name__})") from exc
        try:
            status = int(getattr(response, "status_code", 0) or 0)
            if 300 <= status < 400:
                location = response.headers.get("location", "")
                if not location:
                    raise IngestError(f"Redirect {status} without a Location header")
                current = validate_url(urljoin(current, location))
                continue
            if status >= 400:
                raise IngestError(f"Source answered HTTP {status}")
            if not 200 <= status < 300:
                raise IngestError(f"Unexpected source status {status}")
            declared = response.headers.get("content-length")
            if declared:
                try:
                    if int(declared) > max_bytes:
                        raise IngestError(f"Source exceeds {max_bytes} bytes")
                except ValueError:
                    pass
            content_type = (
                response.headers.get("content-type", "").split(";", 1)[0].strip().casefold()
            )
            if content_type and content_type not in ALLOWED_CONTENT_TYPES:
                raise IngestError(f"Unsupported content type: {content_type}")
            data = _read_limited(response, max_bytes)
            return FetchedSource(
                data=data,
                final_url=current,
                content_type=content_type or "text/plain",
            )
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
    raise IngestError(f"Too many redirects (limit {MAX_REDIRECTS})")


def _read_limited(response: Any, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            raise IngestError(f"Source exceeds {max_bytes} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------


def extract_content(raw: bytes, *, content_type: str = "text/plain", suffix: str = "") -> ExtractedDocument:
    """Turn a fetched/local payload into clean text plus an optional title."""
    if len(raw) > MAX_SOURCE_BYTES:
        raise IngestError(f"Source exceeds {MAX_SOURCE_BYTES} bytes")
    kind = content_type.split(";", 1)[0].strip().casefold()
    suffix = suffix.casefold()

    if kind == "application/pdf" or suffix == ".pdf":
        return ExtractedDocument(content=_normalize_text(_extract_pdf(raw)))
    if kind in {"text/html", "application/xhtml+xml"} or suffix in {".html", ".htm"}:
        return _extract_html(raw)
    if kind == "application/json" or suffix == ".json":
        return _extract_json(raw)
    return ExtractedDocument(content=_normalize_text(_decode(raw)))


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "windows-1258", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_html(raw: bytes) -> ExtractedDocument:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_decode(raw), "lxml")
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    for tag in soup(
        ["script", "style", "noscript", "svg", "iframe", "form", "nav", "footer", "header"]
    ):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    text = root.get_text("\n", strip=True)
    return ExtractedDocument(content=_normalize_text(text), title_hint=title)


def _extract_json(raw: bytes) -> ExtractedDocument:
    try:
        payload = json.loads(_decode(raw))
    except json.JSONDecodeError as exc:
        raise IngestError("JSON source is not valid JSON") from exc
    if isinstance(payload, dict):
        for key in ("content", "text", "body"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                title = payload.get("title")
                return ExtractedDocument(
                    content=_normalize_text(value),
                    title_hint=title.strip() if isinstance(title, str) else "",
                )
        raise IngestError("JSON source has no 'content', 'text', or 'body' string")
    if isinstance(payload, list) and all(isinstance(item, str) for item in payload):
        return ExtractedDocument(content=_normalize_text("\n\n".join(payload)))
    raise IngestError("JSON source must be an object with content or a list of strings")


def _extract_pdf(raw: bytes) -> str:
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise IngestError("PDF import requires pdfplumber") from exc
    pages: list[str] = []
    try:
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
    except Exception as exc:
        raise IngestError("Could not read PDF source") from exc
    return "\n\n".join(pages)


# ---------------------------------------------------------------------------
# Corpus records
# ---------------------------------------------------------------------------


def load_documents(documents_dir: Path = DEFAULT_DOCUMENTS_DIR) -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    directory = Path(documents_dir)
    if not directory.exists():
        return records
    for path in sorted(directory.glob("*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IngestError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise IngestError(f"{path}:{line_no}: document must be a JSON object")
            records.append((path, record))
    return records


def next_document_id(category: str, documents_dir: Path = DEFAULT_DOCUMENTS_DIR) -> str:
    category = validate_category(category, documents_dir)
    pattern = re.compile(rf"^{re.escape(category)}_([0-9]+)$")
    highest = 0
    existing: set[str] = set()
    for _, record in load_documents(documents_dir):
        document_id = str(record.get("id") or "")
        existing.add(document_id)
        match = pattern.fullmatch(document_id)
        if match:
            highest = max(highest, int(match.group(1)))
    candidate = f"{category}_{highest + 1:03d}"
    while candidate in existing:
        highest += 1
        candidate = f"{category}_{highest + 1:03d}"
    return candidate


def _normalized_content(value: str) -> str:
    return " ".join(value.casefold().split())


def _hash_content(value: str) -> str:
    return hashlib.sha256(_normalized_content(value).encode("utf-8")).hexdigest()


def _normalized_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if not parsed.scheme:
        return value.strip().rstrip("/")
    path = parsed.path.rstrip("/")
    return urlunsplit(
        (parsed.scheme.casefold(), parsed.netloc.casefold(), path, parsed.query, "")
    )


def find_duplicate(
    content: str,
    source_url: str = "",
    *,
    documents_dir: Path = DEFAULT_DOCUMENTS_DIR,
    exclude_id: str | None = None,
) -> tuple[Path, dict[str, Any]] | None:
    """Return the first record with the same URL or normalized content hash."""
    wanted_url = _normalized_url(source_url) if source_url else ""
    wanted_hash = _hash_content(content)
    for path, record in load_documents(documents_dir):
        if exclude_id and record.get("id") == exclude_id:
            continue
        existing_url = str(record.get("source_url") or "")
        if wanted_url and existing_url and _normalized_url(existing_url) == wanted_url:
            return path, record
        existing_content = record.get("content")
        if isinstance(existing_content, str) and _hash_content(existing_content) == wanted_hash:
            return path, record
    return None


def _atomic_write(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    text = "\n".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records
    )
    temp.write_text(text + "\n", encoding="utf-8")
    temp.replace(path)


def _validate_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    clean = dict(metadata)
    title = clean.get("title")
    if not isinstance(title, str) or not title.strip():
        raise IngestError("A non-empty title is required")
    if len(title) > 300:
        raise IngestError("Title must be 300 characters or fewer")
    clean["title"] = title.strip()

    year = clean.get("year")
    if year is not None:
        if not isinstance(year, int) or isinstance(year, bool) or not 1990 <= year <= 2100:
            raise IngestError("year must be an integer between 1990 and 2100")
    for field_name in ("published_date", "effective_date", "expiry_date"):
        value = clean.get(field_name)
        if value in (None, ""):
            clean.pop(field_name, None)
            continue
        try:
            date.fromisoformat(str(value))
        except ValueError as exc:
            raise IngestError(f"{field_name} must be YYYY-MM-DD") from exc
    status = clean.get("status", "active")
    if status not in VALID_STATUSES:
        raise IngestError(f"status must be one of {sorted(VALID_STATUSES)}")
    clean["status"] = status
    return clean


def _resolve_source(
    source: str,
    *,
    allowed_root: Path | None,
    session: requests.Session | None,
) -> tuple[str, str, str]:
    """Return ``(content, source_url, title_hint)`` for a file path or URL."""
    parsed = urlsplit(source)
    if parsed.scheme.casefold() in {"http", "https"}:
        fetched = fetch_url(source, session=session)
        extracted = extract_content(
            fetched.data,
            content_type=fetched.content_type,
            suffix=Path(urlsplit(fetched.final_url).path).suffix,
        )
        return extracted.content, fetched.final_url, extracted.title_hint
    if parsed.scheme:
        raise IngestError("Only local paths and http(s) URLs are supported")
    path = resolve_local_file(source, allowed_root=allowed_root)
    extracted = extract_content(path.read_bytes(), suffix=path.suffix)
    return extracted.content, path.as_uri(), extracted.title_hint


# ---------------------------------------------------------------------------
# Import/update
# ---------------------------------------------------------------------------


def import_source(
    source: str,
    *,
    category: str,
    documents_dir: Path = DEFAULT_DOCUMENTS_DIR,
    state_file: Path = DEFAULT_STATE_FILE,
    metadata: Mapping[str, Any] | None = None,
    update_id: str | None = None,
    duplicate: str = "error",
    review: str = "pending",
    allowed_root: Path | None = None,
    session: requests.Session | None = None,
    dry_run: bool = False,
) -> IngestResult:
    """Import one file/URL as a reviewed source record.

    ``duplicate='error'`` refuses a second import of the same URL/content and
    names the existing document. ``duplicate='update'`` refreshes that record in
    place. Both paths reset the workflow state to ``review_pending`` (unless an
    explicit ``review`` action is requested), so imported text never bypasses
    review.
    """
    if duplicate not in {"error", "update"}:
        raise IngestError("duplicate must be 'error' or 'update'")
    if review not in VALID_REVIEW_ACTIONS:
        raise IngestError(f"review must be one of {sorted(VALID_REVIEW_ACTIONS)}")
    category = validate_category(category, documents_dir)
    content, source_url, title_hint = _resolve_source(
        source, allowed_root=allowed_root, session=session
    )
    content = _normalize_text(content)
    if not content:
        raise IngestError("Source produced no extractable text")
    if len(content) > MAX_CONTENT_CHARS:
        raise IngestError(
            f"Extracted text exceeds {MAX_CONTENT_CHARS} characters; split or trim the source"
        )

    supplied = dict(metadata or {})
    reserved = {"id", "content", "source_url", "category"} & set(supplied)
    if reserved:
        raise IngestError(f"Metadata must not override: {', '.join(sorted(reserved))}")
    if not supplied.get("title"):
        supplied["title"] = title_hint or Path(source).stem or source_url
    document_metadata = _validate_metadata(supplied)

    target_record: dict[str, Any] | None = None
    duplicate_id: str | None = None
    if update_id:
        for path, record in load_documents(documents_dir):
            if record.get("id") == update_id:
                if record.get("category") != category:
                    raise IngestError(
                        f"Document {update_id!r} belongs to category {record.get('category')!r}"
                    )
                target_record = record
                target_path = path
                break
        if target_record is None:
            raise IngestError(f"Document ID {update_id!r} was not found")
    else:
        match = find_duplicate(
            content,
            source_url,
            documents_dir=documents_dir,
            exclude_id=update_id,
        )
        if match is not None:
            path, record = match
            duplicate_id = str(record.get("id"))
            if duplicate != "update":
                raise IngestError(
                    f"Duplicate of existing document {duplicate_id!r} "
                    f"(use duplicate='update' or --update {duplicate_id})"
                )
            if record.get("category") != category:
                raise IngestError(
                    f"Duplicate document {duplicate_id!r} belongs to "
                    f"category {record.get('category')!r}"
                )
            target_record = record
            target_path = path
        else:
            target_path = document_path(category, documents_dir)

    now = datetime.now(timezone.utc)
    if target_record is None:
        document = {
            "id": next_document_id(category, documents_dir),
            "category": category,
            "university": "HaUI",
            "language": "vi",
            "source_type": "official_website" if source_url.startswith(("http://", "https://")) else "file_import",
            "source_name": "HaUI Admissions Ingestion",
            "crawled_date": now.date().isoformat(),
        }
        created = True
    else:
        document = dict(target_record)
        created = False

    document.update(document_metadata)
    document["content"] = content
    document["source_url"] = source_url
    document["category"] = category
    if update_id or target_record is not None:
        document["updated_at"] = now.isoformat()
    document["id"] = str(document["id"])

    records = [
        record
        for path, record in load_documents(documents_dir)
        if path == target_path and record.get("id") != document["id"]
    ]
    insert_at = 0
    for index, (path, record) in enumerate(load_documents(documents_dir)):
        if path == target_path:
            insert_at += 1
    records.insert(min(insert_at, len(records)), document)

    if not dry_run:
        _atomic_write(target_path, records)
        if review == "pending":
            set_state(document["id"], "review_pending", state_file=state_file)
        else:
            review_document(document["id"], review, note="imported via ingest", state_file=state_file)

    return IngestResult(
        document=document,
        path=target_path,
        created=created,
        duplicate_of=duplicate_id,
        wrote=not dry_run,
        context={
            "source": source,
            "source_url": source_url,
            "review": review,
            "dry_run": dry_run,
        },
    )


def set_document_decision(
    document_id: str,
    decision: str,
    *,
    state_file: Path = DEFAULT_STATE_FILE,
    note: str = "",
) -> str:
    """Approve or exclude (reject) an imported document without deleting it."""
    if decision not in {"approve", "reject"}:
        raise IngestError("decision must be 'approve' or 'reject'")
    return review_document(document_id, decision, note=note, state_file=state_file)


__all__ = [
    "ALLOWED_CONTENT_TYPES",
    "ALLOWED_FILE_SUFFIXES",
    "IngestError",
    "IngestResult",
    "MAX_SOURCE_BYTES",
    "document_path",
    "extract_content",
    "fetch_url",
    "find_duplicate",
    "import_source",
    "load_documents",
    "next_document_id",
    "resolve_local_file",
    "set_document_decision",
    "validate_category",
    "validate_url",
]
