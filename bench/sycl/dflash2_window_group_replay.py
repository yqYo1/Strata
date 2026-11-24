"""Replay logical CPU weight visits for contiguous committed verifier rows."""

from collections import Counter
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import dflash2_causal_rank_replay as prior

bank, capture, route, census, require = prior.bank, prior.capture, prior.route, prior.census, prior.require
THIS = str(Path(__file__).resolve().relative_to(bank.P))
TILES = (4, 8, 16)
POOL_ROWS = 4


def load_trace():
    ctx = capture.experiment("holdout")
    original = bank.previous.historic(prior.HOLDOUT_COMMIT, ctx.KEY)
    require(original["stage"] == "closed" and original["passed"] and original["complete"]
            and original["ownership_closed"] and not original["active"], "require the closed holdout")
    process, = original["processes"]
    require(process["ownership_closed"] and process["native_exit_code"] == 0
            and not process["fresh_faults"], "the original native process must have closed normally")
    require(ctx.compact(bank.previous.read(ctx.ROOT / "record.json")) == original,
            "retained source-bound routing receipt differs")
    root = Path(original["plans"][0]["root"])
    execution = bank.previous.read(root / "execution.json")
    require(not execution["active"] and bank.previous.read(root / "engine-exit.json")["exit_code"] == 0,
            "original execution is incomplete")
    members = subprocess.run(["ps", "-o", "pid=", "--sid", str(process["native_identity"]["sid"])],
                             capture_output=True, timeout=15)
    require(members.returncode == 1 and not members.stdout.strip(), "original native session is live")
    samples = process["evidence"]["samples"]
    text = (root / "engine.stderr").read_text()
    native, _ = capture.leaves.leaf_reader.evidence(text, samples, 4, False)
    bindings, widths = capture.census_inputs(text, samples, native)
    headers = list(re.finditer(capture.footer.HEADER, text, re.M))
    frame = text[headers[-2].end():headers[-1].start()]
    positions = [(int(p), int(w)) for p, w in re.findall(r"^strata trace: window (\d+) (\d+)$", frame, re.M)]
    require([w for _, w in positions] == widths and len(positions) > 1
            and positions[0][0] == samples[-1]["read_tokens"] - 1, "invalid committed prefix extent")
    meta = original["routing_metadata"]
    require(all(bank.sha(meta[k]) == meta[k + "_sha256"] for k in ("profile", "manifest")),
            "original weight sizes or resident membership changed")
    formats = census.manifest_formats(meta["manifest"])
    membership = census.expected_membership(meta["profile"])
    observed = census.extract(text, bindings, formats, membership)[-1]
    trace, evidence = root / capture.ROUTING_FILE, process["routes"]
    require(trace.stat().st_size == evidence["routing_bytes"] <= original["budgets"]["file_cap"]
            and bank.sha(trace) == evidence["routing_sha256"] and sum(widths) == observed["rows"]
            and len(widths) == evidence["windows"] == observed["windows"],
            "original routing capture length or digest differs")
    return trace, positions, membership, formats, evidence, samples[-1]


def replay(trace, positions, membership, formats):
    """Keep identical chronological committed rows and fixed per-layer residency."""
    original = [0, 0, 0]
    committed = [0, 0, 0]
    states = {tile: [(None, Counter()) for _ in formats] for tile in TILES}
    totals = {tile: dict(jobs_nt4=0, bytes_nt4=0, jobs_once=0, bytes_once=0, entries=0) for tile in TILES}

    def flush(tile, counts, size):
        jobs = sum((n + POOL_ROWS - 1) // POOL_ROWS for n in counts.values())
        total = totals[tile]
        total["jobs_nt4"] += jobs
        total["bytes_nt4"] += jobs * size
        total["jobs_once"] += len(counts)
        total["bytes_once"] += len(counts) * size
        total["entries"] += sum(counts.values())

    row = route.ROUTE_ROW
    suffix = sum(w for _, w in positions) * len(formats) * row.size
    first = positions[0][0]
    with trace.open("rb") as f:
        extent = f.seek(0, 2)
        require(suffix <= extent and extent % row.size == 0, "invalid routing extent")
        f.seek(extent - suffix)
        for i, (position, width) in enumerate(positions):
            # Every following window starts at the previous committed end.
            # Leave the terminal window out of committed-row comparisons.
            kept = positions[i + 1][0] - position if i + 1 < len(positions) else 0
            require(0 <= kept <= width and (kept > 0 or i == len(positions) - 1), "invalid commit advance")
            for layer, (_, _, size) in enumerate(formats):
                offered, active = Counter(), Counter()
                for j in range(width):
                    actual, k, *ids, weights = row.unpack(f.read(row.size))
                    require(actual == layer and k == len(set(ids)) == 10 and weights == route.ZERO_WEIGHTS
                            and all(0 <= e < 512 for e in ids), "unexpected recorded route")
                    cold = [e for e in ids if not (membership[layer] >> e) & 1]
                    offered.update(cold)
                    if j >= kept:
                        continue
                    active.update(cold)
                    for tile in TILES:
                        index = (position + j - first) // tile
                        previous, counter = states[tile][layer]
                        if previous != index:
                            flush(tile, counter, size)
                            counter.clear()
                            states[tile][layer] = (index, counter)
                        counter.update(cold)
                for result, counts in ((original, offered), (committed, active)):
                    result[0] += len(counts)
                    result[1] += sum(counts.values())
                    result[2] += len(counts) * size
        require(f.tell() == extent and not f.read(1), "unexpected route suffix")
    for tile in TILES:
        for layer, (_, counts) in enumerate(states[tile]):
            flush(tile, counts, formats[layer][2])
    require(all(t["entries"] == committed[1] and t["jobs_once"] <= t["jobs_nt4"]
                and t["bytes_once"] <= t["bytes_nt4"] for t in totals.values()),
            "grouped replay changed committed expert entries")
    return original, committed, totals


def main():
    require(sys.executable == bank.basis.reference.model.PYTHON, "use the owned benchmark interpreter")
    revision = prior.forecast.basis.validation.git("rev-parse", "HEAD").decode().strip()
    sources = prior.forecast.verify_sources(revision)
    for name in (THIS, str(Path(prior.__file__).relative_to(bank.P))):
        require((bank.P / name).read_bytes() ==
                prior.forecast.basis.validation.git("show", revision + ":" + name),
                "commit the exact replay before use")
        sources[name] = bank.sha(bank.P / name)
    with (bank.B / "owned-v0141-measurement.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        trace, positions, membership, formats, evidence, sample = load_trace()
        offered, committed, tiled = replay(trace, positions, membership, formats)
        metrics = ("cpu_jobs", "cpu_entries", "logical_job_blob_bytes")
        require(offered == [evidence[k] for k in metrics], "replay differs from original CPU jobs or bytes")
        print(json.dumps(dict(source_commit=revision, sources=sources, trace_commit=prior.HOLDOUT_COMMIT,
            trace_sha256=evidence["routing_sha256"], input_tokens=sample["read_tokens"],
            original_output_tokens=sample["generated_tokens"], original_windows=len(positions),
            offered_target_rows=sum(w for _, w in positions),
            processed_committed_rows=positions[-1][0] - positions[0][0],
            original_offered=dict(zip(metrics, offered)), committed_in_original_windows=dict(zip(metrics, committed)),
            committed_tiles=tiled, scope="Logical visits on one recorded prose trace. "
            "Contiguous committed-row groups assume their routing is already known. "
            "The terminal window is excluded from all committed-row comparisons."), indent=2))


if __name__ == "__main__":
    main()
