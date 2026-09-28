#!/usr/bin/env python3
"""Structure-aware chunking for the admission corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from rag.document_workflow import get_state, set_state

ROOT_DIR = Path(__file__).resolve().parents[1]
HEADING_RE = re.compile(
    r"(?m)^(?P<heading>(?:#{1,6}\s+.+|(?:\d+(?:\.\d+)*|[IVX]+)\.\s+.+))$"
)
SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")


def split_sections(text: str) -> list[tuple[str, str]]:
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [("", text.strip())]
    sections: list[tuple[str, str]] = []
    if text[: matches[0].start()].strip():
        sections.append(("", text[: matches[0].start()].strip()))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        heading = match.group("heading").strip()
        body = text[match.end() : end].strip()
        sections.append((heading, body))
    return sections


def split_body(body: str, max_chars: int = 3200, overlap_chars: int = 400) -> list[str]:
    if len(body) <= max_chars:
        return [body]
    del overlap_chars
    paragraphs: list[str] = []
    for paragraph in (part.strip() for part in re.split(r"\n{2,}", body) if part.strip()):
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        is_list_block = len(lines) > 1 and all(
            line.startswith(("-", "*", "+")) or re.match(r"^\d+[.)]\s", line)
            for line in lines
        )
        if (
            len(paragraph) > max_chars
            and not is_list_block
            and not paragraph.lstrip().startswith(("[Bảng", "|"))
        ):
            sentences = [sentence.strip() for sentence in SENTENCE_RE.split(paragraph) if sentence.strip()]
            if len(sentences) > 1:
                sentence_group = ""
                for sentence in sentences:
                    candidate = f"{sentence_group} {sentence}".strip()
                    if sentence_group and len(candidate) > max_chars:
                        paragraphs.append(sentence_group)
                        sentence_group = sentence
                    else:
                        sentence_group = candidate
                if sentence_group:
                    paragraphs.append(sentence_group)
                continue
        paragraphs.append(paragraph)
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def classify_content(text: str) -> tuple[str, dict | None]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "paragraph", None
    is_table = any(
        marker in text.casefold()
        for marker in ("[bảng", "|---", "| ---", "│", "tt:")
    )
    is_list = sum(
        line.startswith(("-", "*", "+")) or bool(re.match(r"^\d+[.)]\s", line))
        for line in lines
    ) >= max(1, len(lines) // 2)
    is_image = any(marker in text.casefold() for marker in ("![", "[ảnh", "[image"))
    types = sum((is_table, is_list, is_image))
    if types > 1:
        return "mixed", None
    if is_table:
        return "table", {"format": "text", "content": text}
    if is_list:
        return "list", None
    if is_image:
        return "image", {"format": "text", "content": text}
    return "paragraph", None


def chunk_document(document: dict) -> list[dict]:
    chunks: list[dict] = []
    document_revision = hashlib.sha256(
        "\0".join(
            str(document.get(key, ""))
            for key in ("id", "title", "content", "source_url")
        ).encode("utf-8")
    ).hexdigest()
    heading_stack: list[str] = []
    for section_index, (heading, body) in enumerate(split_sections(document["content"]), 1):
        if heading:
            depth = 1
            numbered = re.match(r"^(\d+(?:\.\d+)*)\.", heading)
            markdown = re.match(r"^(#{1,6})\s", heading)
            if numbered:
                numeric_depth = numbered.group(1).count(".") + 1
                base_depth = 0
                for parent_heading in heading_stack:
                    if re.match(r"^\d+(?:\.\d+)*\.", parent_heading):
                        break
                    base_depth += 1
                depth = base_depth + numeric_depth
            elif markdown:
                depth = len(markdown.group(1))
            heading_stack = heading_stack[: depth - 1] + [heading]
        heading_path = list(heading_stack)
        if not body:
            continue
        for local_index, text in enumerate(split_body(body, max_chars=2800), 1):
            content_type, structured_content = classify_content(text)
            prefix = f"{' > '.join(heading_path)}\n\n" if heading_path else ""
            chunks.append(
                {
                    "chunk_id": f"{document['id']}__section-{section_index:03d}__chunk-{local_index:03d}",
                    "document_id": document["id"],
                    "title": document["title"],
                    "text": prefix + text,
                    "heading_path": heading_path,
                    "section_path": heading_path,
                    "content_type": content_type,
                    "structured_content": structured_content,
                    "category": document["category"],
                    "year": document.get("year"),
                    "source_url": document["source_url"],
                    "source_title": document["title"],
                    "source_position": {
                        "section_index": section_index,
                        "chunk_index": local_index,
                    },
                    "proposal_revision": hashlib.sha256(
                        f"{document_revision}\0{section_index}\0{local_index}\0{text}".encode("utf-8")
                    ).hexdigest(),
                    "status": "proposal",
                    "text_override": None,
                    "review_note": "",
                }
            )
    return chunks


def write_proposals(documents: list[dict], output_file: Path) -> int:
    if not documents:
        return 0
    document_ids = {document["id"] for document in documents}
    existing = []
    if output_file.exists():
        existing = [
            json.loads(line)
            for line in output_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    retained = [chunk for chunk in existing if chunk.get("document_id") not in document_ids]
    generated = [chunk for document in documents for chunk in chunk_document(document)]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as stream:
        for chunk in [*retained, *generated]:
            stream.write(json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n")
    return len(generated)


def remove_reviewed_document(reviewed_file: Path, document_id: str) -> int:
    if not reviewed_file.exists():
        return 0
    records = [
        json.loads(line)
        for line in reviewed_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    retained = [record for record in records if record.get("document_id") != document_id]
    removed = len(records) - len(retained)
    if removed:
        output_file = reviewed_file.with_suffix(reviewed_file.suffix + ".tmp")
        with output_file.open("w", encoding="utf-8") as output:
            for record in retained:
                output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        output_file.replace(reviewed_file)
    return removed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, nargs="*", help="Document JSONL files; defaults to all corpus files")
    parser.add_argument("--document", help="Generate proposals only for this approved document ID")
    parser.add_argument(
        "--replace-review",
        action="store_true",
        help="Discard prior chunk decisions for regenerated approved documents",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "data" / "chunks" / "proposals",
        help="Proposal output directory",
    )
    args = parser.parse_args()
    files = args.input or sorted((ROOT_DIR / "data" / "documents").glob("*.jsonl"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    selected_document = False
    for path in files:
        documents = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        selected = [
            document for document in documents
            if args.document is None or document.get("id") == args.document
        ]
        if args.document and not selected:
            continue
        approved_documents = []
        for document in selected:
            if get_state(document["id"]) != "approved":
                if args.document:
                    raise ValueError(
                        f"Document {document['id']!r} must be explicitly approved before chunking."
                    )
                continue
            selected_document = True
            approved_documents.append(document)
        if approved_documents:
            output_file = args.output_dir / path.name
            if args.replace_review:
                reviewed_file = ROOT_DIR / "data" / "chunks" / "reviewed" / path.name
                for document in approved_documents:
                    remove_reviewed_document(reviewed_file, document["id"])
            total += write_proposals(approved_documents, output_file)
            for document in approved_documents:
                set_state(document["id"], "chunk_proposed")
    if args.document and not selected_document:
        raise ValueError(f"Document {args.document!r} was not found in the supplied corpus files")
    if not total:
        print("No proposals generated; documents must first be approved.")
    else:
        print(f"Wrote {total} chunk proposals to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
