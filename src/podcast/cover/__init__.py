"""Per-Episode cover artwork generation through OpenRouter image models."""

import base64
from importlib import resources
from pathlib import Path

import httpx
from ffmpeg.asyncio import FFmpeg
from pydantic import SecretStr

from podcast.models import Script

_COVER_SIZE = 1400
_HEADLINE_WORDS = 4
_REQUEST_TIMEOUT_SECONDS = 180.0
_PROMPT_TEMPLATE = (
    resources.files(__package__).joinpath("prompt.md").read_text(encoding="utf-8")
)


def headline(script: Script) -> str:
    """Render the lead Segment title as a short uppercase cover headline."""
    return " ".join(script.segments[0].title.split()[:_HEADLINE_WORDS]).upper()


def render_prompt(script: Script) -> str:
    """Render the cover generation prompt from the Episode lead Segment."""
    lead = script.segments[0]
    facts = "\n".join(f"- {fact}" for fact in lead.key_facts)
    return _PROMPT_TEMPLATE.format(
        title=lead.title,
        facts=facts,
        headline=headline(script),
    )


async def to_square_jpeg(image_bytes: bytes, *, size: int) -> bytes:
    """Scale-crop any image payload into a square JPEG of exactly `size` pixels."""
    ffmpeg = (
        FFmpeg()
        .option("y")
        .input("pipe:0")
        .output(
            "pipe:1",
            vf=(
                f"scale={size}:{size}:force_original_aspect_ratio=increase,"
                f"crop={size}:{size}"
            ),
            vcodec="mjpeg",
            qscale=2,
            f="image2pipe",
        )
    )
    return await ffmpeg.execute(image_bytes)


class OpenRouterCover:
    """Generates a 1400x1400 Episode cover from the staged master artwork."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        master_cover_path: Path,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Store OpenRouter credentials, the master cover, and an optional client."""
        self.api_key = api_key
        self.master_cover_path = master_cover_path
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.client = client

    def _reference(self) -> dict[str, object]:
        """Return the master cover as an OpenRouter image_url reference."""
        encoded = base64.b64encode(self.master_cover_path.read_bytes()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
        }

    async def generate(self, script: Script) -> bytes:
        """Request the Episode artwork once and return a 1400x1400 JPEG."""
        payload: dict[str, object] = {
            "model": self.model,
            "prompt": render_prompt(script),
            "input_references": [self._reference()],
            "aspect_ratio": "1:1",
            "resolution": "2K",
            "response_format": "b64_json",
        }
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        if self.client is None:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/images",
                    headers=headers,
                    json=payload,
                    timeout=_REQUEST_TIMEOUT_SECONDS,
                )
        else:
            response = await self.client.post(
                f"{self.base_url}/images",
                headers=headers,
                json=payload,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        response.raise_for_status()

        data = response.json().get("data")
        if not data:
            msg = "OpenRouter image response holds no data payload."
            raise ValueError(msg)
        return await to_square_jpeg(
            base64.b64decode(data[0]["b64_json"]),
            size=_COVER_SIZE,
        )
