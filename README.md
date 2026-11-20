# podcast

Stateless daily CLI that produces a **5–8 minute Brazilian Portuguese (`pt-BR`) curated tech news podcast** tailored to a listener's [Taste Profile](src/podcast/agents/curator/instructions.md).

- **Ingestion:** Extensible upstream `Source` adapters and web content extraction behind `Protocol` interfaces.
- **Editorial pipeline:** Multi-step agent workflow (`curate` → `write_script`) producing a two-`Persona` `Script`.
- **Audio & Hosting:** Multi-speaker TTS synthesis normalized to MP3 via `ffmpeg`, published as an RSS 2.0 `Feed` on GitHub Pages (`feed.xml` + metadata JSON) and GitHub Releases (`.mp3` enclosures).
- **Failure policy:** No fallbacks — external errors fail loudly, unreadable `Story` URLs are skipped, and quiet days still publish whatever valid stories passed.

See [GLOSSARY.md](GLOSSARY.md) for domain vocabulary and [docs/architecture/context.mmd](docs/architecture/context.mmd) for C4 diagrams.

## Quickstart

1. Copy `.env.example` to `.env` and configure your API key and `Persona` voice IDs.
2. Ensure `ffmpeg` is installed on your `PATH`.
3. Run:

```bash
uv run podcast --help
uv run podcast
```

## Development

```bash
uv run pytest                                  # Offline unit tests
uv run pytest -m live                          # Live tests against external services
uv run --env-file .env deepeval test run evals/ # Scriptwriter evaluations
uvx ruff check . && uvx ruff format --check .  # Lint & format
uvx ty check                                   # Typecheck
```
