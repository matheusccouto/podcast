"""Web Story content extraction protocol and Jina Reader adapter."""

from contextlib import nullcontext
from typing import Protocol, runtime_checkable

import httpx

_MIN_CONTENT_LENGTH = 200
_REQUEST_TIMEOUT_SECONDS = 20.0
_UNREADABLE_STATUS_CODES = frozenset({401, 403, 404, 406, 410, 422, 451})


@runtime_checkable
class ContentFetcher(Protocol):
    """Structural interface for extracting readable text from a Story URL."""

    async def fetch_content(self, url: str) -> str | None:
        """Return readable text for `url`, or None when the URL is unreadable."""
        ...


class JinaFetcher:
    """Markdown content extractor backed by Jina Reader (`r.jina.ai`)."""

    def __init__(
        self,
        *,
        base_url: str = "https://r.jina.ai",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure the Jina Reader base URL and optional shared HTTP client."""
        self.base_url = base_url.rstrip("/")
        self.client = client

    async def fetch_content(self, url: str) -> str | None:
        """Extract readable Markdown via Jina Reader, returning None if unreadable.

        Expected client/paywall HTTP errors (401/403/404/406/410/422/451) and
        short extractions (< 200 chars) return None so the Story can be skipped;
        upstream 5xx and transport failures raise loudly.
        """
        target = f"{self.base_url}/{url}"
        ctx = (
            nullcontext(self.client) if self.client is not None else httpx.AsyncClient()
        )
        async with ctx as active_client:
            response = await active_client.get(
                target,
                timeout=_REQUEST_TIMEOUT_SECONDS,
                follow_redirects=True,
            )

        if response.status_code in _UNREADABLE_STATUS_CODES:
            return None
        response.raise_for_status()

        cleaned = response.text.strip()
        if len(cleaned) < _MIN_CONTENT_LENGTH:
            return None
        return cleaned
