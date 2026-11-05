"""Run-config argument repair and JSON diagnostics.

Ported from Niko1221/Strata serve/server.py at ffa80d73133ea5c548aefe57889a16e8085a938f.
Copyright (c) 2026 Niko1221 and the Strata contributors. See repository LICENSE (MIT).
"""
import json
from pathlib import Path
import re


def split_joined_args(cfg: dict) -> None:
    """#2267: an engine option written as ONE element of the config's "args" - "--ple-io mmap" or "--ple-io=mmap" - reaches
    the engine as a single argument ("unknown argument: --ple-io mmap"; the usual form is two elements, "--ple-io",
    "mmap").  Recommend, never force: such an element is split into the flag and its value, and said at startup.  Only an
    element that starts with a flag name (--letters-digits-dashes) followed by a space or "=" is touched, and never one
    that sits in the value place of the flag before it."""
    args = cfg.get("args")
    if not isinstance(args, list):
        return
    out, changed = [], []
    for a in args:
        m = re.match(r"^(--[A-Za-z0-9][A-Za-z0-9-]*)(?:\s+|=)(\S.*)$", a) if isinstance(a, str) else None
        prev = out[-1] if out else None
        in_value_place = isinstance(prev, str) and re.match(r"^--[A-Za-z0-9][A-Za-z0-9-]*$", prev) is not None
        if m is None or in_value_place:
            out.append(a)
            continue
        out += [m.group(1), m.group(2).strip()]
        changed.append(a)
    if changed:
        cfg["args"] = out
        print("[strata] the config's args hold " + ", ".join(repr(c) for c in changed) + " as one element each: split into the "
              'flag and its value (write them as two elements, e.g. "--ple-io", "mmap")', flush=True)


def read_config(path) -> dict:
    """The run config's JSON.  A mistake in it (a missing comma between two args, a trailing comma) is said with its
    place in the file instead of a traceback (#2267: editing "args" by hand is where it happens)."""
    return parse_config_text(Path(path).read_text(encoding="utf-8-sig"), path)   # Notepad adds a BOM


def parse_config_text(text: str, path) -> dict:
    """read_config on text already read (the owner-launched server hashes the bytes it parses, launch attestation)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"[strata] {path} is not valid JSON: {e.msg} at line {e.lineno}, column {e.colno}. Check the commas "
                         'between the entries of "args" (each option and each value is its own quoted entry, e.g. '
                         '"--ple-io", "mmap",) and that no entry is followed by a comma at the end of a list.') from None


