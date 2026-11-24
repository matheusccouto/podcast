"""Shared detection of unset live API keys for skipping live test suites."""

from pydantic import ValidationError

LIVE_KEY_FIELDS = frozenset({"openrouter_api_key", "exa_api_key"})


def live_key_skip_reason(error: ValidationError, *, suite: str) -> str | None:
    """Return a skip reason when only live API keys are missing, else None."""
    missing = {
        detail["loc"][0] for detail in error.errors() if detail["type"] == "missing"
    }
    if not missing or not missing <= LIVE_KEY_FIELDS:
        return None
    reason = ", ".join(sorted(map(str, missing)))
    return f"live API keys not configured for {suite}: {reason}"
