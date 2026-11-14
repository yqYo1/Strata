"""Model execution explicitly deferred in this fork; see AGENTS.md.

Apply this at catalog and launch boundaries so shared upstream control/API
updates do not enable a deferred engine. Inference dispatch is unchanged.
"""
from __future__ import annotations

import re


def deferred(value: object) -> bool:
    if not isinstance(value, str):
        return False
    name = value.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return (name.startswith(("deepseek", "qwen36", "qwen35moe")) or
            re.search(r"(?<![0-9])35[-_ ]?b(?:[^a-z0-9]|$)", name) is not None)


def require_config(cfg: dict) -> None:
    values = [cfg.get(k) for k in ("arch", "family", "model", "model_name")]
    args = cfg.get("args") or []
    for i, arg in enumerate(args):
        if arg in ("--native", "--model", "--pack", "-m") and i + 1 < len(args):
            values.append(args[i + 1])
        elif isinstance(arg, str) and arg.startswith(("--native=", "--model=", "--pack=")):
            values.append(arg.partition("=")[2])
    if any(deferred(v) for v in values):
        raise ValueError("DeepSeek and 35B model execution is deferred in this fork; "
                         "an explicit user request is required to add it")
