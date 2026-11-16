# Podcast — Technical Specification

## 1. Overview & Goals

Build a stateless daily CLI application that produces a **5–8 minute Brazilian Portuguese (`pt-BR`) curated tech news podcast** tailored to a single listener's [Taste Profile](../GLOSSARY.md).

- **Primary Source (v1):** Official **Hacker News Firebase API** (`https://hacker-news.firebaseio.com/v0/`, free, no auth key).
- **Editorial Flow:** LangGraph `StateGraph` (`curate` -> `write_script`) connecting two module-level `create_agent` subagents: `curator` (`Taste Profile` + Hacker News / Jina tools + `CuratorState` + `CuratorContext`) and `writer` (Jinja2-parameterized scriptwriter emitting `Dialogue` with Fish Audio `SpeechTurn[]` JSON). Both agents and `podcast_graph` use standard `.ainvoke(..., context=..., config=...)` / `.invoke(...)`.
- **Format & Tone:** Two-`Persona` conversational dialogue (~800–1,150 spoken words) with light humor, structured around **1 lead `Story` + 1–4 highlights**.
- **Providers:** **OpenRouter** for LLM (`langchain` `create_agent(..., response_format=ToolStrategy(...))` + `langchain-openrouter`, defaulting to `baidu/ernie-5.5-flash-preview:free`), TTS (`fish-audio/s2.1-pro-free:free` via `/api/v1/audio/speech`), and Jev (`jev-1.13` via `/api/v1/systemone`), plus keyless **Jina Reader** (`https://r.jina.ai/<url>`) for fetching linked `Story` content.
- **Hosting:** **GitHub-only** — GitHub Pages (`gh-pages` branch) hosts `feed.xml` and `episodes/YYYY-MM-DD.json`; GitHub Releases hosts `episodes/YYYY-MM-DD.mp3`. Listeners subscribe via any standard Podcast App polling `feed.xml`.
- **Simplicity & Failure Policy:** Hobbyist project — **no fallbacks**. External errors fail loudly. Unreadable `Story` URLs are strictly skipped. Quiet days still publish whatever valid `Story` items passed.

See [GLOSSARY.md](../GLOSSARY.md) for canonical terminology and [docs/architecture/](./architecture/) for nested Mermaid C4 `.mmd` diagrams.

---

## 2. Repository & Package Layout

```text
podcast/
├── .env.example                          # Reference env vars for agents & CI
├── .github/
│   └── workflows/
│       └── daily.yaml                    # Disabled cron (commented) + workflow_dispatch → Releases + Pages
├── AGENTS.md                             # Project commands & standards
├── GLOSSARY.md                             # Canonical domain vocabulary
├── pyproject.toml                        # Entrypoint: podcast = "podcast.__main__:main"
├── docs/
│   ├── architecture/
│   │   ├── context.mmd                   # C4Context: System Context
│   │   └── podcast-system/
│   │       ├── container.mmd             # C4Container: CLI, GH Pages, GH Releases
│   │       ├── dynamic.mmd               # C4Dynamic: Daily Episode Run
│   │       └── cli/
│   │           ├── component.mmd         # C4Component: Podcast CLI (src/podcast/)
│   │           └── agents/
│   │               └── component.mmd     # C4Component: Editorial Agents (src/podcast/agents/)
│   ├── research/                         # Verified primary-source research
│   └── spec.md                           # This specification
├── src/podcast/
│   ├── __init__.py                       # Package init
│   ├── __main__.py                       # Click CLI entrypoint (`uv run podcast` / `python -m podcast`)
│   ├── models.py                         # Zero-dependency Pydantic models + validation
│   ├── settings.py                       # `Settings(BaseSettings)` + `Persona` definitions
│   ├── fetcher.py                        # `JinaFetcher` (`r.jina.ai`)
│   ├── tts.py                            # `OpenRouterTTS` (`fish-audio/s2.1-pro-free:free` PCM -> `python-ffmpeg` MP3)
│   ├── feed.py                           # `Feed` (`publish` -> `episodes/` + `feed.xml`)
│   ├── sources/
│   │   ├── __init__.py
│   │   └── hackernews.py                 # `HackerNewsSource` (Firebase JSON API)
│   └── agents/                           # Module-level `create_agent` subagents + LangGraph workflow
│       ├── __init__.py
│       ├── model.py                      # Configurable `init_chat_model` (`model`, `api_key`)
│       ├── workflow.py                   # `PodcastContext`, `PodcastState`, and compiled `podcast_graph`
│       ├── curator/
│       │   ├── __init__.py
│       │   ├── agent.py                  # `CuratorContext`, `CuratorState`, and compiled `curator`
│       │   └── instructions.md           # Jinja2 Listener Taste Profile + curation instructions
│       └── writer/
│           ├── __init__.py
│           ├── agent.py                  # `WriterContext`, `format_writer_prompt`, and compiled `writer`
│           ├── instructions.md           # Jinja2 system instructions (`{{ language }}`, `{{ personas }}`)
│           └── prompt.md                 # Jinja2 curated `Story` batch template
├── tests/
│   ├── conftest.py                       # Shared `live_settings` and `sample_turns` fixtures
│   ├── test_cli.py                       # Click CLI `--help` and flag override tests
│   ├── test_sources.py                   # `HackerNewsSource` live + HTTP error test
│   ├── test_fetcher.py                   # `JinaFetcher` live + parametrized 404/500 tests
│   ├── test_tts.py                       # `SpeechTurn` + offline ffmpeg + live Fish Audio
│   ├── test_feed.py                      # `Feed.publish` + RSS ordering on `tmp_path`
│   └── test_podcast.py                   # `ResearchBrief` validation + live `podcast_graph`
└── evals/
    ├── conftest.py                       # `OpenRouterJevModel`, `EvaluationDataset`, `JevEval` fixtures
    ├── download_golden.py                # Downloads multi-source NPR Goldens in Fish Audio shape
    ├── golden/                           # Curated `{date, stories, expected_output}` fixtures
    └── test_script_evals.py              # DeepEval `assert_test` + `Golden` evals (`deepeval test run`)
```

---

## 3. Configuration (`settings.py` & `.env.example`)

`Settings` extends `pydantic_settings.BaseSettings` and is instantiated inside `__main__.py` (or injected in tests), never at module import time. `Settings` also owns the two `Persona(name, role, tone, voice_id)` definitions and `llm_config` shared by `writer` and `OpenRouterTTS`.

### Environment Variables (`.env.example`)
| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENROUTER_API_KEY` | Yes | — | Auth for OpenRouter LLM, TTS, and Jev endpoints |
| `LLM_MODEL` | No | `baidu/ernie-5.5-flash-preview:free` | OpenRouter model ID for research & scripting |
| `TTS_MODEL` | No | `fish-audio/s2.1-pro-free:free` | OpenRouter TTS model ID |
| `PODCAST_LANGUAGE` | No | `pt-BR` | Target Episode language |
| `PODCAST_TITLE` | No | `HN Daily BR` | RSS `<title>` |
| `PODCAST_DESCRIPTION` | No | `Resumo diário de tecnologia e Hacker News.` | RSS `<description>` |
| `PODCAST_AUTHOR` | No | `Matheus Couto` | `<itunes:author>` |
| `PODCAST_BASE_URL` | Yes in CI | `http://localhost:8000` | Public GitHub Pages base URL (`feed.xml`) |
| `PODCAST_RELEASES_BASE_URL` | Yes in CI | `http://localhost:8000/episodes` | Base URL for MP3 `<enclosure>` assets |
| `PODCAST_DIST_DIR` | No | `dist` | Output directory for published episode artifacts |
| `PERSONA_A_VOICE_ID` | Yes for TTS | — | Fish Audio `reference_id` for Persona A |
| `PERSONA_B_VOICE_ID` | Yes for TTS | — | Fish Audio `reference_id` for Persona B |

---

## 4. Module Interfaces

### 4.1 `sources/hackernews.py` & `fetcher.py`
- **`HackerNewsSource`:** `fetch_stories(limit=30)` and `fetch_comments(story_id, limit=15)` call the official HN Firebase API via `httpx.AsyncClient`; upstream HTTP failures raise loudly.
- **`JinaFetcher`:** `fetch_content(url)` calls `GET https://r.jina.ai/{url}`. Returns `None` on expected client/unreadable HTTP responses (`401/403/404/406/410/422/451`) or bodies under `200` chars; raises loudly on `5xx` and transport errors.

### 4.2 `agents/` (`curator`, `writer`, `podcast_graph`)
- **`curator` (`agents/curator/`):** Module-level `create_agent` with `list_stories`, `fetch_story_content`, and `fetch_comments` `ToolRuntime` tools reading `CuratorContext`, `@dynamic_prompt` rendering Jinja2 `instructions.md`, `@after_agent` running `ResearchBrief.build_curated_stories(...)`, and `response_format=ToolStrategy(ResearchBrief)`.
- **`writer` (`agents/writer/`):** Module-level `create_agent(..., tools=[], response_format=ToolStrategy(Dialogue))` with `WriterContext`, `@dynamic_prompt` rendering Jinja2 `instructions.md`, co-located Jinja2 `prompt.md`, and `@after_agent` verifying `Dialogue.checked_turns(personas)`.
- **`podcast_graph` (`agents/workflow.py`):** Module-level compiled `StateGraph(PodcastState, context_schema=PodcastContext)` (`START -> curate -> write_script -> END`) invoked via `.ainvoke(PodcastState(...), context=PodcastContext(...), config=settings.llm_config)`.

### 4.3 `tts.py`
- **`OpenRouterTTS.synthesize(turns, *, title)`:** Verifies `ffmpeg` upfront, sends `{"model": model, "input": [t.model_dump() for t in turns], "response_format": "pcm"}` to `{base_url}/audio/speech`, requires explicit `rate=` and `channels=` in PCM `Content-Type`, and encodes normalized ID3 MP3 via `python-ffmpeg` (`ffmpeg.asyncio.FFmpeg`) with `-af loudnorm`, returning `SynthesizedAudio`.

### 4.4 `feed.py` & `__main__.py`
- **`Feed.from_settings(settings)` & `Feed.publish(script, audio)`:** Writes `dist/episodes/YYYY-MM-DD.mp3`, `dist/episodes/YYYY-MM-DD.json`, and regenerates `dist/feed.xml` newest-first.
- **`__main__.py`:** Click `@click.command()` CLI supporting `--date`, `--llm-model`, `--tts-model`, `--language`, and `--dist-dir`, executing `podcast_graph.ainvoke(...) -> tts.synthesize(...) -> Feed.from_settings(...).publish(...)`.

---

## 5. CI/CD & Hosting (`.github/workflows/daily.yaml`)

Currently disabled from automatic runs (`schedule` cron `09:00 UTC` / `06:00 BRT` commented out; `workflow_dispatch` retained):
1. Checkout `main`, install `uv` dependencies and `ffmpeg`.
2. Hydrate prior `feed.xml` and `episodes/*.json` from existing `gh-pages` branch into `dist/`.
3. Run `uv run podcast` using GitHub Actions Secrets (`OPENROUTER_API_KEY`, `PERSONA_A_VOICE_ID`, `PERSONA_B_VOICE_ID`).
4. **Upload MP3 to GitHub Releases first** (`episodes/YYYY-MM-DD.mp3` with `--clobber`) so `<enclosure>` URLs resolve before `feed.xml` updates.
5. Deploy updated `dist/feed.xml` and `dist/episodes/YYYY-MM-DD.json` (never `.mp3` binaries) to the `gh-pages` branch.

---

## 6. Testing & Evals Strategy

- **Tests (`uv run pytest`):** Live keyless endpoints (`HackerNewsSource`, `JinaFetcher`), `tmp_path` (`Feed`), offline `ffmpeg` + live `OpenRouterTTS`, and Pydantic validations + live `podcast_graph`.
- **Evals (`uv run --env-file .env deepeval test run evals/`):** Invokes `writer.ainvoke(...)` directly on cherry-picked multi-source NPR *The Indicator from Planet Money* goldens (`{date, stories, expected_output}` in Fish Audio JSON shape) and evaluates with `deepeval.assert_test` + `JevEval` (`OpenRouterJevModel`).
