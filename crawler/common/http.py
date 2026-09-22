from __future__ import annotations

import logging
import time
from urllib.parse import urlparse

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 (HaUIAdmissionRAG/1.0)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8",
    "Accept-Language": "vi,en;q=0.9",
}


class HttpClient:
    """Safe HTTP client with retries, timeout, and polite rate limiting."""

    def __init__(self, delay_seconds: float = 0.5, timeout: int = 30):
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self.delay_seconds = delay_seconds
        self.timeout = timeout
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)
        self._last_request_time = time.time()

    def get(self, url: str, retries: int = 3) -> tuple[bytes, str]:
        """Fetch URL content safely.

        Returns: (content_bytes, content_type)
        """
        self._rate_limit()
        last_exc: Exception | None = None

        for attempt in range(1, retries + 1):
            try:
                response = self.session.get(url, timeout=self.timeout, verify=False)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                return response.content, content_type
            except Exception as exc:
                last_exc = exc
                logger.warning(f"Fetch attempt {attempt}/{retries} failed for {url}: {exc}")
                time.sleep(1.0 * attempt)

        raise RuntimeError(f"Failed to fetch {url} after {retries} attempts: {last_exc}")
