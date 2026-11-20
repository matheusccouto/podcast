# podcast

Stateless daily CLI that produces a **5–8 minute Brazilian Portuguese (`pt-BR`) curated tech news podcast** tailored to a listener's [Taste Profile](src/podcast/agents/curator/instructions.md).

- **Sources:** Official Hacker News Firebase API (extensible via the `Source` protocol) + keyless Jina Reader (`r.jina.ai`) for article extraction.
- **Editorial pipeline:** LangGraph `StateGraph` (`curate` → `write_script`) running LangChain `create_agent` subagents backed by OpenRouter.
- **Audio & Hosting:** Multi-speaker Fish Audio TTS via OpenRouter, `ffmpeg` loudness normalization to MP3, published as an RSS 2.0 `Feed` on GitHub Pages (`feed.xml` + metadata JSON) and GitHub Releases (`.mp3` enclosures).
- **Failure policy:** No fallbacks — external errors fail loudly, unreadable `Story` URLs are skipped, and quiet days still publish whatever valid stories passed.

See [GLOSSARY.md](GLOSSARY.md) for domain vocabulary and [docs/architecture/context.mmd](docs/architecture/context.mmd) for C4 diagrams.

## Quickstart

1. Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`, `PERSONA_A_VOICE_ID`, and `PERSONA_B_VOICE_ID`.
2. Ensure `ffmpeg` is installed on your `PATH`.
3. Run:

```bash
uv run podcast --help
uv run podcast
```

## Development

```bash
uv run pytest                                  # Fast offline unit tests
uv run pytest -m live                          # Live network / LLM / TTS tests
uv run --env-file .env deepeval test run evals/ # Scriptwriter quality evals
uvx ruff check . && uvx ruff format --check .  # Lint & format
uvx ty check                                   # Typecheck
```
