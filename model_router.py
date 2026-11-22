"""Task-based model routing.

Different jobs need different amounts of model capability. A conversation
title is cheap, high-volume work; a 4-option architecture analysis is the
hardest thing the app asks for. Rather than sending everything to one model,
each *task* maps to a *tier*, and each tier maps to a concrete model.

    task  ──►  tier  ──►  model / token budget

Everything is configurable through environment variables, and every tier
falls back to ``LLM_MODEL`` (then the built-in default), so an unconfigured
deployment behaves exactly like a single-model setup.

Env vars:
    LLM_MODEL                    default model for every tier
    LLM_MODEL_LIGHT / _STANDARD / _STRONG     per-tier model
    LLM_MAX_TOKENS_LIGHT / _STANDARD / _STRONG per-tier output budget
    LLM_TIER_TITLE / _CHAT / _ORACLE           per-task tier override
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from enum import Enum

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"


class Tier(str, Enum):
    LIGHT = "light"
    STANDARD = "standard"
    STRONG = "strong"


class Task(str, Enum):
    TITLE = "title"
    CHAT = "chat"
    ORACLE = "oracle"


# Default task → tier policy: cheapest tier that does the job well.
DEFAULT_TASK_TIERS: dict[Task, Tier] = {
    Task.TITLE: Tier.LIGHT,
    Task.CHAT: Tier.STANDARD,
    Task.ORACLE: Tier.STRONG,
}

# The budget is a ceiling, not a target: short replies cost nothing extra. It
# must cover a reasoning model's hidden thinking too. The default model
# (nemotron-3.5-lightning) spends ~250+ tokens reasoning before it writes a
# title, so a 256 cap returned empty titles; 4512 matches the pre-routing budget.
DEFAULT_MAX_TOKENS: dict[Tier, int] = {
    Tier.LIGHT: 4512,
    Tier.STANDARD: 4512,
    Tier.STRONG: 8192,
}


@dataclass(frozen=True)
class ModelRoute:
    task: str
    tier: str
    model: str
    max_tokens: int


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d", name, raw, default)
        return default


def _tier_for(task: Task) -> Tier:
    raw = os.getenv(f"LLM_TIER_{task.name}", "").strip().lower()
    if raw:
        try:
            return Tier(raw)
        except ValueError:
            logger.warning("Ignoring invalid LLM_TIER_%s=%r", task.name, raw)
    return DEFAULT_TASK_TIERS[task]


def _model_for(tier: Tier) -> str:
    return (
        os.getenv(f"LLM_MODEL_{tier.name}", "").strip()
        or os.getenv("LLM_MODEL", "").strip()
        or DEFAULT_MODEL
    )


def resolve(task: Task | str) -> ModelRoute:
    """Resolve the model and budget for a task. Reads env on every call so
    configuration changes apply without code changes (and tests can patch env)."""
    task = Task(task)
    tier = _tier_for(task)
    return ModelRoute(
        task=task.value,
        tier=tier.value,
        model=_model_for(tier),
        max_tokens=_env_int(f"LLM_MAX_TOKENS_{tier.name}", DEFAULT_MAX_TOKENS[tier]),
    )


def routing_table() -> list[dict]:
    """The effective routing for every task, for display in the admin UI."""
    return [asdict(resolve(task)) for task in Task]
