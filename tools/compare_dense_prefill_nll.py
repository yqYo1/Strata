#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 yayoi
# SPDX-License-Identifier: MIT
"""Compare dense initial-prefill loss receipts. Never awards quality acceptance.

Each --control/--candidate directory must contain recipe.json (written before
launch), closed.json (external supervisor), nll.bin and nll.bin.meta. The native
child gets SHA256(recipe.json) via STRATA_PREFILL_NLL_RECIPE_SHA256. Do not put
that derived variable into recipe.json itself. See the proposal README schema.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import struct


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "expected SHA256")
    return value


def nonempty(value, label):
    require(isinstance(value, str) and value.strip(), "missing " + label)
    return value


def object_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    require(path.stat().st_size <= 1 << 20, "external receipt exceeds 1 MiB")
    raw = path.read_bytes()
    return json.loads(raw, object_pairs_hook=object_pairs), digest(raw)


def token_input(path):
    require(path.stat().st_size <= 8 << 20, "token-file byte budget")
    raw = path.read_bytes()
    text = raw.decode("ascii").strip()
    fields = text.replace(",", " ").split()
    require(fields and all(re.fullmatch(r"[0-9]+", x) for x in fields), "expected explicit token IDs")
    ids = [int(x) for x in fields]
    require(2 <= len(ids) <= 262144 and all(0 <= x < 2**31 for x in ids), "invalid token input")
    return ids, digest(raw), digest(b"".join(struct.pack("<I", x) for x in ids))


def load_run(root, tokens, token_file_sha, token_ids_sha):
    root = root.resolve()
    recipe, recipe_sha = read_json(root / "recipe.json")
    closed, _ = read_json(root / "closed.json")
    require(recipe["schema"] == "strata-dense-prefill-nll-recipe-v2", "recipe schema")
    require(closed["schema"] == "strata-dense-prefill-nll-closed-v2", "closure schema")
    require(closed["recipe_sha256"] == recipe_sha, "closure recipe mismatch")
    for key in ("snapshot_sha256", "binary_sha256"):
        sha(recipe["source"][key])
    require(bool(re.fullmatch(r"[0-9a-f]{40}", recipe["source"]["commit"])), "source commit")
    sha(recipe["model"]["manifest_sha256"])
    inp = recipe["input"]
    require(inp["count"] == len(tokens) and inp["file_sha256"] == token_file_sha and
            inp["ids_le_u32_sha256"] == token_ids_sha, "input file/count/identity mismatch")
    sha(inp["template_sha256"])
    for key in ("hardware", "compiler", "runtime", "driver"):
        nonempty(recipe["runtime"][key], "runtime " + key)
    command = recipe["command"]
    require(isinstance(command, list) and command and all(isinstance(x, str) for x in command), "command vector")
    for flag in ("--native", "--stream-token", "--tokens-file", "--prefill", "--max-context"):
        require(flag in command, "missing native command flag " + flag)
    require("--serve" not in command and "--prefill-until" not in command, "incompatible command route")
    environment = recipe["environment"]
    require(isinstance(environment, dict) and all(isinstance(k,str) and isinstance(v,str) for k,v in environment.items()), "effective child environment")
    require("STRATA_PREFILL_NLL_RECIPE_SHA256" not in environment, "self-referential recipe digest")
    require(environment["STRATA_PREFILL_NLL"] == str(root / "nll.bin"), "receipt output path mismatch")
    require(environment["STRATA_PREFILL_LAYER_MAJOR"] == "1", "quality gate requires layer-major path")
    for variable in ("STRATA_PREFILL_RELEASE_HEADS", "STRATA_PREFILL_PHASE_AUDIT"):
        require(environment.get(variable, "0") == "0", "incompatible diagnostic flag")
    require("STRATA_LOGPOS" not in environment, "incompatible verifier-window diagnostic")
    settings = recipe["settings"]
    require(settings["dtype"] in ("fp32", "fp16"), "residual dtype")
    require(environment["STRATA_PREFILL_LAYER_MAJOR_R_DTYPE"] == settings["dtype"], "dtype environment mismatch")
    require(environment["STRATA_PREFILL_LAYER_MAJOR_R_GPU"] == str(settings["residual_gpu_rows"]), "placement environment mismatch")
    require(isinstance(settings["residual_gpu_rows"], int) and 0 <= settings["residual_gpu_rows"] <= len(tokens)-1, "placement request")
    require(isinstance(settings["chunk"], int) and settings["chunk"] > 0 and
            settings["context"] >= len(tokens)+1, "chunk/context settings")
    for flag, expected in (("--prefill",settings["chunk"]),("--max-context",settings["context"])):
        require(int(command[command.index(flag)+1]) == expected, "command/settings mismatch")
    nonempty(settings["kv"], "KV setting")
    sha(settings["mtp_manifest_sha256"])
    owner = closed["process"]
    nonempty(owner["boot_id"], "boot identity")
    nonempty(owner["deadline_utc"], "finite process deadline")
    require(isinstance(owner["pid"],int) and owner["pid"] > 0 and isinstance(owner["start_ticks"],int) and
            owner["start_ticks"] > 0 and owner["exit_code"] == 0 and owner["owned_absent"] is True, "process did not close healthy")
    faults = closed["faults"]
    require(faults["fresh_boundaries"] is True and faults["count"] == 0, "fresh fault audit missing or faulted")
    sha(faults["receipt_sha256"])
    observed = closed["observed"]
    for key in ("dtype", "chunk", "context", "residual_gpu_rows"):
        require(observed[key] == settings[key], "actual plan differs from request: " + key)
    require(observed["n_input"] == len(tokens) and observed["rows"] == len(tokens)-1, "observed shape")
    require(observed["head_release"] == 0, "head must remain resident")
    for key in ("residual_host_bytes", "residual_staging_bytes"):
        require(isinstance(observed[key],int) and observed[key] >= 0, "actual placement bytes missing")
    raw_path, meta_path = root / "nll.bin", root / "nll.bin.meta"
    require(raw_path.stat().st_size + meta_path.stat().st_size < 4 << 20 and
            meta_path.stat().st_size < 4096, "raw budget before reading")
    raw = raw_path.read_bytes()
    meta_raw = meta_path.read_bytes()
    require(len(raw)+len(meta_raw) < 4 << 20 and len(meta_raw) < 4096, "raw budget")
    require(digest(raw) == closed["raw_sha256"] and digest(meta_raw) == closed["metadata_sha256"], "output hashes")
    lines = meta_raw.decode("ascii").splitlines()
    require(lines[0] == "strata-dense-prefill-nll-v2", "metadata schema")
    meta = object_pairs([line.split("=", 1) for line in lines[1:]])
    require(meta["recipe_sha256"] == recipe_sha and meta["complete"] == "1", "unpublished/incomplete metadata")
    require(meta["coverage"] == "initial-prefill-all-n_input-minus-one" and
            meta["performance_cohort"] == "excluded", "diagnostic scope")
    require(math.isfinite(float(meta["callback_ms"])) and float(meta["callback_ms"]) >= 0, "diagnostic cost")
    require(len(raw) >= 64 and raw[:8] == b"STNLLV2\n", "raw header")
    version, header, count, vocab, rows, first, head_type, native = struct.unpack_from("<8I", raw, 8)
    require(version == 2 and header == 64 and raw[40:64] == bytes(24), "raw schema/reserved bytes")
    require(count == len(tokens) and rows == count-1 and first == tokens[0] and
            len(raw) == 64+12*rows and all(x < vocab for x in tokens), "raw shape/token0/vocabulary")
    require(int(meta["n_input"]) == count and int(meta["rows"]) == rows and
            int(meta["vocab"]) == vocab and int(meta["raw_bytes"]) == len(raw), "metadata/raw disagreement")
    losses = []
    for position, (target, loss) in enumerate(struct.iter_unpack("<Id", raw[64:])):
        require(target == tokens[position+1], "gold token mismatch at position " + str(position))
        require(math.isfinite(loss) and loss >= 0, "invalid loss at position " + str(position))
        losses.append(loss)
    return dict(root=str(root), recipe=recipe, recipe_sha256=recipe_sha, closed=closed,
                losses=losses, head=[vocab,head_type,native], callback_ms=float(meta["callback_ms"]))


def invariant(run):
    recipe = run["recipe"]
    settings = {k:v for k,v in recipe["settings"].items() if k not in ("dtype","residual_gpu_rows")}
    ignored = {"STRATA_PREFILL_NLL", "STRATA_PREFILL_LAYER_MAJOR_R_DTYPE", "STRATA_PREFILL_LAYER_MAJOR_R_GPU"}
    environment = {k:v for k,v in recipe["environment"].items() if k not in ignored}
    return {"source":recipe["source"], "model":recipe["model"], "input":recipe["input"],
            "runtime":recipe["runtime"], "command":recipe["command"], "settings":settings,
            "environment":environment, "head":run["head"]}


def summarize(values):
    require(bool(values), "empty summary")
    ordered = sorted(values)
    return {"rows":len(values), "mean":statistics.fmean(values), "min":ordered[0], "max":ordered[-1],
            "mean_abs":statistics.fmean(abs(x) for x in values),
            "p50":ordered[(len(ordered)-1)//2], "p99":ordered[int((len(ordered)-1)*.99)]}


def pair(a, b, chunk, boundary):
    delta = [y-x for x,y in zip(a["losses"],b["losses"])]
    n = len(delta)
    bins = {"all": summarize(delta)}
    for i, name in enumerate(("early","middle","late")):
        lo, hi = n*i//3, n*(i+1)//3
        if hi>lo:
            bins[name] = summarize(delta[lo:hi])
    edges = sorted({p for edge in range(chunk,n,chunk) for p in range(max(0,edge-1),min(n,edge+1))})
    if edges:
        bins["chunk_edges"] = summarize([delta[p] for p in edges])
    if 0 < boundary < n:
        bins["placement_edge"] = {"positions":[boundary-1,boundary],
                                  "delta":[delta[boundary-1],delta[boundary]]}
    return {"reference":a["root"], "other":b["root"], "delta_other_minus_reference":bins}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens-file",type=Path,required=True)
    parser.add_argument("--control",type=Path,action="append",required=True)
    parser.add_argument("--candidate",type=Path,required=True)
    parser.add_argument("--placement",choices=("matched","all-gpu"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    require(len(args.control) >= 2, "need at least two independent FP32 controls")
    ids, file_sha, ids_sha = token_input(args.tokens_file)
    controls = [load_run(p,ids,file_sha,ids_sha) for p in args.control]
    candidate = load_run(args.candidate,ids,file_sha,ids_sha)
    runs = controls+[candidate]
    require(len({r["root"] for r in runs}) == len(runs), "duplicate run directory")
    require(len({(r["closed"]["process"]["boot_id"],r["closed"]["process"]["pid"],r["closed"]["process"]["start_ticks"]) for r in runs}) == len(runs), "process repeats must be independent")
    require(all(invariant(r)==invariant(controls[0]) for r in runs), "source/model/input/command/runtime/settings/environment differ outside declared axes")
    require(all(r["recipe"]["settings"]["dtype"] == "fp32" for r in controls) and
            candidate["recipe"]["settings"]["dtype"] == "fp16", "arm dtype")
    prefix = controls[0]["recipe"]["settings"]["residual_gpu_rows"]
    require(all(r["recipe"]["settings"]["residual_gpu_rows"] == prefix for r in controls), "control placement changed")
    if args.placement == "matched":
        require(candidate["recipe"]["settings"]["residual_gpu_rows"] == prefix, "matched-placement request differs")
    else:
        observed = candidate["closed"]["observed"]
        require(observed["residual_gpu_rows"] == len(ids)-1 and observed["residual_host_bytes"] == 0 and
                observed["residual_staging_bytes"] == 0, "all-GPU placement was not admitted")
    chunk = controls[0]["recipe"]["settings"]["chunk"]
    result = {"schema":"strata-dense-prefill-nll-comparison-v2", "status":"COMPARED; QUALITY_ACCEPTANCE_UNSET",
              "physical_context_correctness":"UNQUALIFIED", "task_quality":"UNQUALIFIED",
              "performance_cohort":"excluded", "placement_axis":args.placement,
              "n_input":len(ids), "rows":len(ids)-1, "input_ids_sha256":ids_sha,
              "arms":[{"directory":r["root"], "recipe_sha256":r["recipe_sha256"],
                       "loss":summarize(r["losses"]), "callback_ms":r["callback_ms"]} for r in runs],
              "fp32_repeat_spread":[pair(a,b,chunk,prefix) for a,b in itertools.combinations(controls,2)],
              "fp16_against_each_control":[pair(c,candidate,chunk,prefix) for c in controls],
              "acceptance":"Set a numeric quality margin from FP32 repeatability before viewing candidate outcomes; identical generated tokens are not required."}
    with args.output.open("x") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write("\n")

if __name__ == "__main__":
    main()
