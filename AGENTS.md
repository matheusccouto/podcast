# podcast

## Commands

- `uv run podcast`: Generate and publish today's Episode.
- `uv run pytest`: Run offline tests.
- `uv run pytest -m live`: Run live tests against external services.
- `uv run --env-file .env deepeval test run evals/`: Run evals.
- `uvx ruff check .`: Lint code.
- `uvx ruff format --check .`: Format code.
- `uvx ty check`: Typecheck code.

## Code Standards

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!

## Principles

- Prefer built-in capabilities of installed libraries over custom abstractions; reach for library primitives before writing wrapper code.
- Let external errors fail loudly rather than adding silent fallbacks.
- Keep lists and categorizations MECE (mutually exclusive, collectively exhaustive).
- Check `docs/architecture/` after codebase changes and update any affected C4 Mermaid diagrams.

## Structure

```text
.
├── .github/workflows/      # Scheduled and manual CI publishing
├── docs/
│   ├── agents/             # Issue tracker, triage labels, and domain doc conventions
│   └── architecture/       # C4 Mermaid diagrams
├── evals/
│   ├── conftest.py         # Judge adapter, metrics, and dataset loader
│   ├── golden/             # Golden reference fixtures
│   └── ...                 # Evaluation test suites
├── src/podcast/
│   ├── __main__.py         # CLI entrypoint and daily pipeline orchestration
│   ├── settings.py         # Configuration and Persona definitions
│   ├── models.py           # Immutable domain models and validation
│   ├── sources/            # Upstream Sources implementing Source (add new sources here)
│   ├── fetcher.py          # Content extraction protocol and implementation
│   ├── agents/             # Editorial workflow and nested subagent packages (add/remove here)
│   ├── tts.py              # Audio synthesis and MP3 encoding
│   └── feed/               # RSS 2.0 publisher and XML template
└── tests/
    ├── conftest.py         # Shared test fixtures
    └── ...                 # Offline unit tests (default) and @pytest.mark.live tests
```

## Resources

- https://docs.astral.sh/uv/llms.txt
- https://docs.astral.sh/ruff/llms.txt
- https://docs.astral.sh/ty/llms.txt
- https://openrouter.ai/docs/llms.txt
- https://openrouter.ai/<model_id>/llms.txt
- https://docs.langchain.com/llms.txt
- https://docs.exa.ai/llms.txt
- https://deepeval.com/llms.txt
- https://pydantic.dev/docs/llms.txt
- https://c4model.com/llms.txt

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues (`matheusccouto/podcast`) via `gh`. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses the five canonical triage labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout (`GLOSSARY.md` and `docs/adr/` at repo root). See `docs/agents/domain.md`.
