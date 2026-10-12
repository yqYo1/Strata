# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Opt-in Linux SYCL replicas: independent engines on disjoint explicit PCI device groups.

A numeric replicas count partitions gpu_pci plus split_pci evenly; explicit entries use
{"devices": ["05:00.0"]}. Each replica owns its cache, profile, log, CPU cores and live process.
Routing uses bounded hashed conversation prefixes plus current load; native prefix checks remain authoritative.
Required replica_ram_mib and replica_vram_mib reservations are checked before any child starts.
"""
from __future__ import annotations

from typing import Any, Callable

import collections
import copy
import os
import math
import re
import threading
import time

BLOCK = 64                          # tokens per hashed block of a prompt
LOAD_TOKENS = 3000                  # the cost of one request already running on a replica, in prompt tokens
INDEX_MAX = 262144                  # block hashes kept (about 17M tokens of prompts)
HEAL_COOLDOWN_S = 60.0              # a replica that died is started again at most this often


class ReplicaConfigError(ValueError):
    pass


def gpu_groups(cfg: dict) -> list[list[str]]:
    """The cards of each replica, from the config's "replicas" (see the module doc); [] when there are none or one."""
    spec = cfg.get("replicas")
    if spec is None or spec is False:
        return []
    split = cfg.get("split_pci") or []
    if not isinstance(split, list):
        raise ReplicaConfigError("split_pci must be a list of PCI addresses")
    cards = _cards([cfg.get("gpu_pci"), *split])
    if isinstance(spec, bool) or (not isinstance(spec, (int, list))):
        raise ReplicaConfigError(f'"replicas" must be a number or a list of {{"devices": [...]}}, not {spec!r}')
    if isinstance(spec, int):
        if spec > 16:
            raise ReplicaConfigError("this bounded replica port supports at most 16 groups")
        if spec < 1:
            raise ReplicaConfigError("replicas must be positive")
        if spec == 1:
            return []
        if not cards:
            raise ReplicaConfigError(f'"replicas": {spec} needs gpu_pci and split_pci to cut into groups')
        if len(cards) % spec:
            raise ReplicaConfigError(f'"replicas": {spec} does not divide the {len(cards)} cards {cards} evenly: '
                                     'name the groups with devices PCI lists or change the count')
        n = len(cards) // spec
        groups = [cards[i * n:(i + 1) * n] for i in range(spec)]
        if len(set(cards)) != len(cards):
            raise ReplicaConfigError("a PCI device appears more than once")
        return groups
    if len(spec) > 16:
        raise ReplicaConfigError("this bounded replica port supports at most 16 groups")
    groups = []
    for e in spec:
        g = e.get("devices") if isinstance(e, dict) else e
        groups.append(_cards(g))
    if any(not g for g in groups):
        raise ReplicaConfigError('every replica needs a nonempty devices PCI list')
    flat = [c for g in groups for c in g]
    if len(set(flat)) != len(flat):
        raise ReplicaConfigError(f"a card is in two replicas: {groups}")
    if len(groups) < 2:
        raise ReplicaConfigError("explicit replicas need at least two nonempty groups")
    return groups


def _cards(g) -> list[str]:
    if g is None or g == "":
        return []
    items = g if isinstance(g, (list, tuple)) else str(g).split(",")
    out = []
    for value in items:
        if value is None:
            continue
        text = str(value).strip().lower()
        if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", text):
            text = "0000:" + text
        if not re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", text):
            raise ReplicaConfigError("replica devices must be explicit PCI addresses, never GPU indices")
        if int(text[8:10], 16) > 31:
            raise ReplicaConfigError("PCI device number exceeds 31")
        out.append(text)
    return out


def _suffixed(path, i: int):
    if not isinstance(path, str) or not path:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}.r{i}{ext}"


def replica_cfg(cfg: dict, i: int, gpus: list[str]) -> dict:
    """The config of replica i: owned device group, log, persistence and profile; optional entry env."""
    c = copy.deepcopy(cfg)
    spec = cfg.get("replicas")
    entry = spec[i] if isinstance(spec, list) and isinstance(spec[i], dict) else {}
    c.pop("replicas", None)
    c.pop("gpu", None)
    c["gpu_pci"], c["split_pci"] = gpus[0], gpus[1:]
    if len(gpus) == 1:
        c.pop("layer_split", None)
    if not isinstance(c.get("args"), list) or not all(isinstance(a, str) for a in c["args"]):
        raise ReplicaConfigError("replica engine args must be a list of strings")
    extra = entry.get("args") or []
    if not isinstance(extra, list) or not all(isinstance(a, str) for a in extra):
        raise ReplicaConfigError("replica args must be a list of strings")
    locked = {"--native", "--pack", "--mtp", "--gpu-pci", "--split-device", "--layer-split"}
    if any(a.split("=", 1)[0] in locked for a in extra):
        raise ReplicaConfigError("per-replica args cannot replace the shared model or device ownership")
    args = list(c["args"]) + extra
    if any(a.split("=", 1)[0] in ("--layer-split", "--split-device", "--gpu-pci") for a in args):
        raise ReplicaConfigError("put replica GPU selection in devices, not raw engine arguments")
    c.pop("hip_ordinal", None)                 # a one-card config's Windows ordinal: not for a group
    c["log"] = _suffixed(cfg.get("log"), i)
    if cfg.get("expert_profile_save"):
        c["expert_profile_save"] = _suffixed(cfg["expert_profile_save"], i)
    if "--resident-budget-gib" in args:
        try:
            budget = float(args[args.index("--resident-budget-gib") + 1])
        except (IndexError, ValueError) as error:
            raise ReplicaConfigError("invalid replica resident RAM budget") from error
        reservation = c.get("replica_ram_mib")
        if (not math.isfinite(budget) or budget <= 0 or type(reservation) is not int
                or budget * 1024 > reservation):
            raise ReplicaConfigError("resident RAM budget exceeds the declared replica reservation")
    c["args"] = args
    for flag in ("--conversation-save", "--expert-profile-save"):
        if flag in args:
            j = args.index(flag) + 1
            if j >= len(args):
                raise ReplicaConfigError(f"missing value for {flag}")
            args[j] = _suffixed(args[j], i)
    if c.get("conversation_save"):
        c["conversation_save"] = _suffixed(c["conversation_save"], i)
    if entry.get("env"):
        if not isinstance(entry["env"], dict):
            raise ReplicaConfigError("replica env must be an object")
        forbidden = {"STRATA_GPU_PCI", "STRATA_LAYER_SPLIT", "ONEAPI_DEVICE_SELECTOR", "SYCL_DEVICE_FILTER"}
        if forbidden.intersection(entry["env"]):
            raise ReplicaConfigError("replica env cannot override its device group")
        c["env"] = {**(c.get("env") or {}), **{str(k): str(v) for k, v in entry["env"].items()}}
    if "STRATA_LAYER_SPLIT" in (c.get("env") or {}):
        raise ReplicaConfigError("replica env cannot override its device group")
    c["env"] = {**(c.get("env") or {}), "STRATA_GPU_PCI": gpus[0]}
    return c


def cpu_sets(n: int, enabled: bool = True):
    """Whole physical cores per replica. Explicitly off returns None; unavailable topology is an error."""
    if not enabled or n < 2 or not hasattr(os, "sched_getaffinity"):
        return None
    try:
        from serve.cpu_affinity import partition_cpu_sets
        return partition_cpu_sets(n, set(os.sched_getaffinity(0)))
    except (ValueError, OSError, ImportError) as e:
        raise ReplicaConfigError(f"cannot partition replica CPUs: {e}; "
                                 "explicitly use replica_cpus:false to disable") from e


class Router:
    """Which replica holds the start of a prompt: chained hashes of its 64-token blocks, each with the set of replicas
    that served it (a bit mask)."""

    def __init__(self, n: int, cap: int = INDEX_MAX):
        self.n, self.cap = n, cap
        self.blocks: collections.OrderedDict[int, int] = collections.OrderedDict()
        self.lock = threading.Lock()

    @staticmethod
    def chain(ids) -> list[int]:
        out, h = [], 0
        for k in range(len(ids) // BLOCK):
            h = hash((h, tuple(ids[k * BLOCK:(k + 1) * BLOCK])))
            out.append(h)
        return out

    def matched(self, ids) -> list[int]:
        """Per replica: how many leading tokens of `ids` it holds (a multiple of BLOCK)."""
        got = [0] * self.n
        alive = (1 << self.n) - 1
        with self.lock:
            for k, h in enumerate(self.chain(ids)):
                m = self.blocks.get(h, 0) & alive
                if not m:
                    break
                alive = m
                for r in range(self.n):
                    if m >> r & 1:
                        got[r] = (k + 1) * BLOCK
        return got

    def note(self, r: int, ids) -> None:
        with self.lock:
            for h in self.chain(ids):
                self.blocks[h] = self.blocks.pop(h, 0) | (1 << r)
            while len(self.blocks) > self.cap:
                self.blocks.popitem(last=False)

    def forget(self, r: int) -> None:
        """Replica r lost its cache (it restarted)."""
        with self.lock:
            bit = ~(1 << r)
            for h in [h for h, m in self.blocks.items() if m >> r & 1]:
                m = self.blocks[h] & bit
                if m:
                    self.blocks[h] = m
                else:
                    del self.blocks[h]


class ReplicaEngine:
    """Stands in for one engine (the interface Service uses), over several."""

    # One shared file / VRAM resize / control line is not offered by the front.
    NOT_HERE = {"session_file", "vram", "ctl", "slot_cv"}
    PROPAGATE = {"silence_s"}
    engines: list
    router: Router
    load_tokens: int
    inflight: list[int]
    run_locks: list
    lock: Any
    tl: threading.local
    healing: dict
    heal_at: list[float]
    routed: list[int]
    closing: bool
    heal_threads: list[threading.Thread]
    admit: Callable[[int], None] | None
    post_admit: Callable[[int], None] | None
    lifecycle: Any
    gen_seen: list[int]
    gpu_groups: list[list[str]]

    def __init__(self, engines: list, load_tokens: int = LOAD_TOKENS):
        if len(engines) < 2:
            raise ValueError("ReplicaEngine needs two or more engines")
        d = self.__dict__
        d["engines"] = list(engines)
        d["router"] = Router(len(engines))
        d["load_tokens"] = load_tokens
        d["inflight"] = [0] * len(engines)
        d["run_locks"] = [threading.Lock() for _ in engines]    # an engine without batch slots: one request at a time
        d["lock"] = threading.Lock()
        d["tl"] = threading.local()
        d["healing"] = {}
        d["heal_at"] = [0.0] * len(engines)
        d["routed"] = [0] * len(engines)
        d["closing"] = False
        d["heal_threads"] = []
        d["admit"] = None
        d["post_admit"] = None
        d["lifecycle"] = threading.RLock()
        d["gen_seen"] = [getattr(e, "gen", 0) for e in engines]

    # ---- attribute plumbing
    @property
    def current(self):
        return getattr(self.tl, "cur", None) or self.engines[0]

    def __getattr__(self, name):
        if name in self.NOT_HERE or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.current, name)

    def __setattr__(self, name, value):
        if name in self.PROPAGATE:
            for e in self.engines:
                setattr(e, name, value)
        elif name == "last":
            self.tl.last = value
        else:
            self.__dict__[name] = value

    @property
    def last(self):
        return getattr(self.tl, "last", self.current.last)

    @last.setter
    def last(self, value):
        self.tl.last = value

    @property
    def batch(self) -> int:
        """Parallel slots, in total: the Service lets this many requests run at once (it holds no FIFO).  An engine
        without slots counts one."""
        return sum(max(1, int(getattr(e, "batch", 0) or 0)) for e in self.engines)

    @property
    def waiting(self) -> int:
        return sum(int(getattr(e, "waiting", 0) or 0) for e in self.engines)

    @property
    def max_context(self) -> int:
        v = [e.max_context for e in self.engines if getattr(e, "max_context", 0) and e.alive()]
        return min(v) if v else 0

    @property
    def known_ctx(self) -> int:
        v = [getattr(e, "known_ctx", 0) for e in self.engines]
        return min(x for x in v if x) if any(v) else 0

    @property
    def unloaded(self) -> bool:
        return all(getattr(e, "unloaded", False) for e in self.engines)

    @property
    def starting(self) -> bool:
        return not self.alive() and any(getattr(e, "starting", False) for e in self.engines)

    @property
    def info(self) -> dict:
        i = dict(self.engines[0].info)
        i["replicas"] = len(self.engines)
        i["replicas_alive"] = sum(1 for e in self.engines if e.alive())
        for key in ("arena_mib", "conversation_cache_mib", "conversation_cache_slots"):
            values = [e.info.get(key) for e in self.engines]
            i[key] = sum(values) if all(type(v) is int and v >= 0 for v in values) else None
        return i

    @info.setter
    def info(self, value):
        pass

    @property
    def spawn(self):
        return self.engines[0].spawn

    @property
    def log_path(self):
        return self.engines[0].log_path

    @property
    def progress(self):
        c = getattr(self.tl, "cur", None)
        if c is not None:
            return c.progress
        return next((e.progress for e in self.engines if getattr(e, "progress", None)), None)

    @property
    def prefill_tok_s_mean(self):
        c = getattr(self.tl, "cur", None)
        if c is not None:
            return c.prefill_tok_s_mean
        return next((e.prefill_tok_s_mean for e in self.engines if getattr(e, "prefill_tok_s_mean", None)), None)

    def alive(self) -> bool:
        return any(e.alive() for e in self.engines)

    def exit_code(self):
        return next((c for c in (e.exit_code() for e in self.engines if not e.alive()) if c is not None), None)

    def death_note(self) -> str:
        return next((e.death_note() for e in self.engines if not e.alive()), "")

    def slots_view(self) -> list[dict]:
        with self.lock:
            inflight = list(self.inflight)
        out, base = [], 0
        for k, e in enumerate(self.engines):
            n = int(getattr(e, "batch", 0) or 0)
            view = e.slots_view() if n and hasattr(e, "slots_view") else \
                [{"slot": 0, "state": "busy" if inflight[k] else "idle", "held_tokens": 0}]
            for s in view:
                out.append({**s, "slot": base + int(s["slot"]), "replica": k})
            base += max(1, n)
        return out

    def replica_view(self) -> list[dict]:
        """For /metrics: each replica's state, requests running, and requests routed to it so far."""
        with self.lock:
            inflight, routed = list(self.inflight), list(self.routed)
        return [{"replica": k, "alive": e.alive(), "running": inflight[k], "routed": routed[k],
                 "batch": int(getattr(e, "batch", 0) or 0),
                 "devices": list(getattr(e, "gpu_pcis", None) or []), "info": dict(e.info)}
                for k, e in enumerate(self.engines)]

    # ---- life cycle
    def restart(self, tries: int = 3):
        with self.lifecycle:
            if self.closing:
                raise RuntimeError("replica service is closing")
            errors = []
            previously_live = self.alive()
            for k, e in enumerate(self.engines):
                if e.alive():
                    continue
                try:
                    if self.admit is not None:
                        e.close()                    # reap the old owner before capacity is sampled
                        self.admit(k)
                    e.restart(tries)
                    if self.post_admit is not None:
                        self.post_admit(k)
                    self.router.forget(k)
                except Exception as error:  # noqa: BLE001 - another replica can remain available
                    errors.append(error)
            if errors and not previously_live:
                for e in self.engines:
                    try:
                        e.close()
                    except Exception as error:  # noqa: BLE001 - preserve first load failure and close all children
                        print(f"[strata] replica load rollback failed: {error}", flush=True)
                raise errors[0]
            if errors and not self.alive():
                raise errors[0]
            for failure in errors:
                print(f"[strata] replica did not start ({failure}); the others serve", flush=True)

    def unload(self):
        with self.lifecycle:
            errors = []
            for k, e in enumerate(self.engines):
                try:
                    e.unload()
                    self.router.forget(k)
                except Exception as error:  # noqa: BLE001 - visit every owned child before surfacing failure
                    errors.append(error)
            if errors:
                raise errors[0]

    def close(self):
        with self.lock:
            self.closing = True
            threads = list(self.heal_threads)
        for thread in threads:
            thread.join()                         # no owned recovery may launch after close returns
        with self.lifecycle:
            error = None
            for e in self.engines:
                try:
                    e.close()
                except Exception as failure:  # noqa: BLE001 - still close every other owned child
                    error = error or failure
            if error:
                raise error

    # ---- routing
    def route(self, ids) -> int:
        """The replica for this prompt (and its running count raised): lowest cost among the living ones."""
        with self.lock:
            changed = [k for k, e in enumerate(self.engines) if getattr(e, "gen", 0) != self.gen_seen[k]]
            for k in changed:
                self.gen_seen[k] = getattr(self.engines[k], "gen", 0)
        for k in changed:
            self.router.forget(k)
        got = self.router.matched(ids)
        with self.lock:
            alive = [k for k, e in enumerate(self.engines) if e.alive()]
            if self.closing or not alive:
                return -1
            cost = {k: (len(ids) - got[k]) + self.load_tokens * self.inflight[k] for k in alive}
            # the least loaded wins a tie of cost (equal cards, nothing cached): spread, do not pile on replica 0
            k = min(alive, key=lambda k: (cost[k], self.inflight[k], self.routed[k]))
            self.inflight[k] += 1
            self.routed[k] += 1
        if len(alive) < len(self.engines):
            self.heal()
        return k

    def heal(self) -> None:
        """Start a replica that died again, in the background (the others keep serving)."""
        with self.lock:
            if self.closing:
                return
            now = time.monotonic()
            self.heal_threads = [t for t in self.heal_threads if t.is_alive()]
            for k, e in enumerate(self.engines):
                if e.alive() or getattr(e, "unloaded", False) or self.healing.get(k) or now < self.heal_at[k]:
                    continue
                self.heal_at[k], self.healing[k] = now + HEAL_COOLDOWN_S, True

                def run(k=k):
                    try:
                        with self.lifecycle:
                            if self.closing or getattr(self.engines[k], "unloaded", False):
                                return
                            if self.admit is not None:
                                self.engines[k].close()
                                self.admit(k)
                            self.engines[k].restart()
                            if self.post_admit is not None:
                                self.post_admit(k)
                            self.router.forget(k)
                    except Exception as error:  # noqa: BLE001 - recovery cannot fail another request
                        print(f"[strata] replica {k} did not restart: {error}", flush=True)
                    finally:
                        with self.lock:
                            self.healing[k] = False
                thread = threading.Thread(target=run, daemon=True)
                self.heal_threads.append(thread)
                thread.start()

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        # imported here: serve.server imports this module
        from serve.server import EngineDied, EngineNonFinite
        k = self.route(ids)
        if k < 0:
            raise EngineDied("no replica is running; this request was not sent")
        e = self.engines[k]
        self.tl.cur = e
        produced: list[int] = []
        held = False
        nonfinite = False
        complete = False
        gen = None
        before = None
        try:
            if not int(getattr(e, "batch", 0) or 0):          # no slots: this replica takes one request at a time
                lock = self.run_locks[k]
                beat = time.monotonic()
                while not lock.acquire(timeout=0.25):
                    if cancel is not None and cancel.is_set():
                        return
                    if time.monotonic() - beat >= 10:
                        beat = time.monotonic()
                        yield None
                held = True
            if cancel is not None and cancel.is_set():
                return
            before = e.last
            if embeddings is not None:
                gen = e.generate(ids, max_new, sampling, cancel, embeddings=embeddings)
            else:
                gen = e.generate(ids, max_new, sampling, cancel)
            for t in gen:
                if t is not None:
                    produced.append(t)
                yield t
            complete = (not (cancel is not None and cancel.is_set()) and e.last is not before
                        and isinstance(e.last, dict) and e.last.get("finish") in ("stop", "length")
                        and e.last.get("prompt_tokens") == len(ids))
        except EngineNonFinite:
            nonfinite = True
            raise
        finally:
            try:
                if gen is not None:
                    getattr(gen, "close", lambda: None)()   # STOP/drain while this replica's turn is held
            finally:
                if gen is not None and e.last is not before:
                    self.tl.last = dict(e.last) if isinstance(e.last, dict) else e.last
                if held:
                    self.run_locks[k].release()
                with self.lock:
                    self.inflight[k] -= 1
                if nonfinite or not e.alive():
                    self.router.forget(k)
                elif complete:
                    self.router.note(k, list(ids) + produced)
                elif (gen is not None and e.last is not before and isinstance(e.last, dict)
                      and e.last.get("finish") in ("stop", "length")
                      and e.last.get("prompt_tokens") == len(ids) and produced
                      and not (cancel is not None and cancel.is_set())):
                    # Service closes at the EOS token before our loop exhausts. A real normal
                    # DONE obtained by the child's drain establishes the input prefix only.
                    self.router.note(k, ids)
                # Keep current for Service's DONE/timing reads; its next request sets its own thread-local current.
