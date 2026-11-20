# podcast

## Commands

- `uv run podcast`: Generate and publish today's Episode.
- `uv run pytest`: Run tests.
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

## Resources

- https://docs.astral.sh/uv/llms.txt
- https://docs.astral.sh/ruff/llms.txt
- https://docs.astral.sh/ty/llms.txt
- https://openrouter.ai/docs/llms.txt
- https://openrouter.ai/fish-audio/s2.1-pro-free:free/llms.txt
- https://docs.langchain.com/llms.txt
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
