"""Web Story content extraction protocol and Exa adapter."""

from contextlib import nullcontext
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

import httpx
from pydantic import SecretStr

from podcast.models import MAX_CONTENT_CHARS

_MIN_CONTENT_LENGTH = 200
_REQUEST_TIMEOUT_SECONDS = 20.0
_SEARCH_NUM_RESULTS = 5


@runtime_checkable
class ContentFetcher(Protocol):
    """Structural interface for extracting readable text from a Story URL."""

    async def fetch_content(
        self,
        url: str,
        *,
        title: str | None = None,
    ) -> str | None:
        """Return readable text for `url`, or None when unreadable."""
        ...


class ExaFetcher:
    """Web content extractor backed by Exa `/contents` and `/search` APIs."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        base_url: str = "https://api.exa.ai",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure Exa credentials, base URL, and optional shared HTTP client."""
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.client = client

    def _headers(self) -> dict[str, str]:
        """Return Exa API authentication and JSON content headers."""
        return {
            "x-api-key": self.api_key.get_secret_value(),
            "Content-Type": "application/json",
        }

    @staticmethod
    def _first_readable_text(payload: dict[str, Any]) -> str | None:
        """Return the first extracted text body meeting minimum readable length."""
        results: list[dict[str, Any]] = payload.get("results") or []
        for result in results:
            raw_text = result.get("text")
            if isinstance(raw_text, str):
                cleaned = raw_text.strip()
                if len(cleaned) >= _MIN_CONTENT_LENGTH:
                    return cleaned
        return None

    async def fetch_content(
        self,
        url: str,
        *,
        title: str | None = None,
    ) -> str | None:
        """Extract readable text via Exa `/contents`, falling back to `/search`.

        When the original URL is gated or unreadable and `title` is provided,
        queries Exa `/search` for open public coverage excluding the gated host.
        Upstream HTTP errors raise loudly.
        """
        ctx = (
            nullcontext(self.client) if self.client is not None else httpx.AsyncClient()
        )
        async with ctx as active_client:
            response = await active_client.post(
                f"{self.base_url}/contents",
                headers=self._headers(),
                json={
                    "urls": [url],
                    "text": {"maxCharacters": MAX_CONTENT_CHARS},
                },
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            extracted = self._first_readable_text(response.json())
            if extracted is not None or not title:
                return extracted

            host = (urlparse(url).hostname or "").removeprefix("www.")
            search_body: dict[str, Any] = {
                "query": title,
                "type": "auto",
                "numResults": _SEARCH_NUM_RESULTS,
                "contents": {"text": {"maxCharacters": MAX_CONTENT_CHARS}},
            }
            if host:
                search_body["excludeDomains"] = [host]

            search_response = await active_client.post(
                f"{self.base_url}/search",
                headers=self._headers(),
                json=search_body,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
            search_response.raise_for_status()
            return self._first_readable_text(search_response.json())
