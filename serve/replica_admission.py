# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Fail-closed, opt-in replica resource admission; native allocation checks remain authoritative."""
from pathlib import Path
import re
import sys
from typing import Iterable

from serve.replicas import ReplicaConfigError

MIB = 1 << 20
SYSTEM_RESERVE_MIB = 6144


def positive(cfg, key):
    value = cfg.get(key)
    if type(value) is not int or value <= 0:
        raise ReplicaConfigError(f'"{key}" must be a positive integer MiB reservation for each replica')
    return value


def full_cgroup_root(proc_root, root, filesystem="cgroup2") -> bool:
    """Prove the visible filesystem root; do not assume a bind-mounted subtree exposes every ancestor."""
    with (proc_root / "self/mountinfo").open() as stream:
        text = stream.read(262145)
    if len(text) > 262144:
        raise ValueError("mountinfo exceeds the replica admission budget")
    for line in text.splitlines():
        fields = line.split()
        if "-" in fields:
            separator = fields.index("-")
            if (separator + 1 < len(fields) and len(fields) >= 6 and fields[3] == "/"
                    and fields[4] == str(root) and fields[separator + 1] == filesystem):
                return True
    return False


def memory_available_mib(proc_root=Path("/proc"), cg_root=Path("/sys/fs/cgroup")) -> int:
    """Host availability intersected with every v1/v2 ancestor limit; unsupported layouts fail closed."""
    try:
        info = dict(line.split(":", 1) for line in (proc_root / "meminfo").read_text().splitlines())
        free = int(info["MemAvailable"].split()[0]) * 1024
        entries = (proc_root / "self/cgroup").read_text().splitlines()
        memberships = []
        for line in entries:
            _, controllers, relative = line.split(":", 2)
            if ".." in Path(relative).parts or not relative.startswith("/"):
                raise ValueError("invalid cgroup membership")
            if not controllers:
                memberships.append((cg_root, relative, "memory.max", "memory.current", "max"))
            elif "memory" in controllers.split(","):
                memberships.append((cg_root / "memory", relative,
                                    "memory.limit_in_bytes", "memory.usage_in_bytes", ""))
        if not memberships:
            raise ValueError("no memory cgroup membership")
        for root, relative, limit_name, used_name, unlimited in memberships:
            if not full_cgroup_root(proc_root, root, "cgroup2" if unlimited == "max" else "cgroup"):
                raise ValueError("unsupported hidden/subtree memory cgroup mount")
            current = root / relative.lstrip("/")
            while True:
                limit_path = current / limit_name
                if (current == root and unlimited == "max" and not limit_path.exists()
                        and full_cgroup_root(proc_root, root)):
                    break
                limit = limit_path.read_text().strip()
                if limit != unlimited:
                    cap = int(limit)
                    used = int((current / used_name).read_text().strip())
                    if cap < 0 or used < 0:
                        raise ValueError("negative cgroup counter")
                    free = min(free, max(0, cap - used))
                if current == root:
                    break
                current = current.parent
        if free < 0:
            raise ValueError("negative MemAvailable")
        return free // MIB
    except (OSError, ValueError, KeyError) as error:
        raise ReplicaConfigError(f"cannot establish replica RAM admission: {error}") from error


def model_file_floor_mib(cfg) -> int:
    """Conservative file-backed model floor, not a claim about peak model/session RSS."""
    args = cfg.get("args") or []
    names = [args[i + 1] for i, flag in enumerate(args[:-1]) if flag in ("--native", "--pack", "--mtp")]
    if not names:
        raise ReplicaConfigError("replicas require explicit model paths in args")
    total, seen, files = 0, set(), 0
    try:
        for name in names:
            path = Path(name)
            if not path.is_absolute():
                path = Path(cfg.get("cwd") or ".") / path
            shard = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", path.name)
            candidates: Iterable[Path]
            if shard:
                count = int(shard[3])
                if not 1 <= int(shard[2]) <= count <= 4096:
                    raise ReplicaConfigError("invalid or oversized native GGUF shard set")
                candidates = [path.with_name(f"{shard[1]}-{index:05d}-of-{count:05d}.gguf")
                              for index in range(1, count + 1)]
                if any(not item.is_file() for item in candidates):
                    raise ReplicaConfigError("a native GGUF shard is missing from the replica RAM floor")
            else:
                candidates = path.rglob("*") if path.is_dir() else [path]
            for item in candidates:
                if not item.is_file():
                    continue
                resolved = item.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                files += 1
                if files > 4096:
                    raise ReplicaConfigError("replica model file scan exceeds 4096 files")
                total += item.stat().st_size
        if not total:
            raise ReplicaConfigError("no readable model files for replica RAM floor")
    except OSError as error:
        raise ReplicaConfigError(f"cannot inspect replica model files: {error}") from error
    return (total + MIB - 1) // MIB


def validate_resources(cfg, groups, pending=None, free_vram=None):
    """Before startup/reload/heal: check reservation for pending processes and every selected GPU."""
    if sys.platform != "linux":
        raise ReplicaConfigError("this Xe replica port supports Linux only")
    ram = positive(cfg, "replica_ram_mib")
    vram = positive(cfg, "replica_vram_mib")
    if ram < model_file_floor_mib(cfg):
        raise ReplicaConfigError("replica_ram_mib is smaller than the conservative model-file floor")
    count = len(groups) if pending is None else pending
    needed = count * ram + SYSTEM_RESERVE_MIB
    free = memory_available_mib()
    if free < needed:
        raise ReplicaConfigError(f"replicas need {needed} MiB free RAM including the system floor; "
                                 f"only {free} MiB available")
    if free_vram is None:
        from serve.telemetry import free_vram_mib
        free_vram = free_vram_mib
    for group in groups:
        for pci in group:
            if not (Path("/sys/bus/pci/devices") / pci).is_dir():
                raise ReplicaConfigError(f"replica GPU {pci} is missing")
            available = free_vram(pci)
            if type(available) is not int or available < vram:
                raise ReplicaConfigError(f"replica GPU {pci} needs {vram} MiB free VRAM; observed {available!r}")
