# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Split a launcher's allowed physical cores between independent Linux instances."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def partition_cpu_sets(count: int, allowed: set[int], reserved: set[int] | None = None,
                       sys_root: Path = Path("/sys")) -> list[list[int]]:
    if type(count) is not int or count < 1:
        raise ValueError("the instance count must be positive")
    if any(type(cpu) is not int or cpu < 0 for cpu in allowed | (reserved or set())):
        raise ValueError("CPU sets must contain nonnegative integer CPU IDs")
    if not allowed:
        raise ValueError("no CPUs are allowed for this launcher")
    if count == 1 and not reserved:
        return [sorted(allowed)]
    cores: dict[tuple[int, int], list[int]] = {}
    occupied: set[tuple[int, int]] = set()
    for cpu in sorted(allowed | (reserved or set())):
        topology = sys_root / "devices/system/cpu" / f"cpu{cpu}" / "topology"
        try:
            key = (int((topology / "physical_package_id").read_text()),
                   int((topology / "core_id").read_text()))
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot read physical-core topology for CPU {cpu}: {error}") from error
        if any(value < 0 for value in key):
            raise ValueError(f"unknown physical-core topology for CPU {cpu}")
        if cpu in allowed:
            cores.setdefault(key, []).append(cpu)
        if reserved and cpu in reserved:
            occupied.add(key)
    available = sorted((cpus for key, cpus in cores.items() if key not in occupied), key=min)
    if len(available) < count:
        raise ValueError(
            f"{count} instances need at least {count} free physical cores, but only {len(available)} "
            "are available; restart the managed instances together or expand the launcher's CPU set"
        )
    size, extra = divmod(len(available), count)
    result, start = [], 0
    for index in range(count):
        end = start + size + (index < extra)
        result.append(sorted(cpu for siblings in available[start:end] for cpu in siblings))
        start = end
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instances", type=int, nargs="?")
    parser.add_argument("--cpus", help="child-only affinity before exec; use with --command")
    parser.add_argument("--command", nargs=argparse.REMAINDER)
    parser.add_argument("--exclude-pid", type=int, action="append", default=[])
    options = parser.parse_args()
    try:
        if options.cpus is not None:
            cpus = [int(x) for x in options.cpus.split(",")]
            if (not cpus or len(cpus) != len(set(cpus)) or any(x < 0 for x in cpus)
                    or not set(cpus) <= os.sched_getaffinity(0)):
                raise ValueError("child CPU set is not a nonempty subset of the inherited allowed mask")
            command = options.command
            if not command or not os.path.isabs(command[0]) or not os.access(command[0], os.X_OK):
                raise ValueError("--command needs an absolute executable path")
            os.sched_setaffinity(0, cpus)
            os.execv(command[0], command)   # same owned child pid, never the threaded server
        if options.command is not None or options.instances is None:
            raise ValueError("give an instance count, or --cpus and --command")
        reserved: set[int] = set()
        for pid in options.exclude_pid:
            try:
                reserved.update(os.sched_getaffinity(pid))
            except ProcessLookupError:
                continue
        groups = partition_cpu_sets(options.instances, os.sched_getaffinity(0), reserved)
    except (OSError, ValueError) as error:
        print(f"CPU affinity: {error}", file=sys.stderr)
        return 1
    for group in groups:
        print(",".join(str(cpu) for cpu in group))
    return 0


def launch_args(exe: str, args: list[str], cpus: list[int] | None) -> list[str]:
    """No preexec_fn: an isolated child pins itself and execs the exact configured executable."""
    if cpus is None:
        return [exe, *args]
    if (not cpus or len(cpus) != len(set(cpus)) or any(type(x) is not int or x < 0 for x in cpus)
            or not set(cpus) <= os.sched_getaffinity(0)):
        raise ValueError("engine CPU set must be within the server's allowed mask")
    if not os.path.isabs(exe) or not os.access(exe, os.X_OK):
        raise ValueError("affinity launch requires an absolute executable path")
    return [sys.executable, "-I", str(Path(__file__).resolve()), "--cpus",
            ",".join(map(str, cpus)), "--command", exe, *args]


if __name__ == "__main__":
    raise SystemExit(main())
