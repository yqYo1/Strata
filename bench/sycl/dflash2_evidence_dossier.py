"""Screen a separate thinking-enabled workload for weak DFlash2 draft agreement."""

import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_snapshot_decode as parent

bank, basis, require, leaves = parent.bank, parent.basis, parent.require, parent.leaves
client = leaves.suite.official.d.client
lg = client.long_generation
KEY = "dflash2_snapshot_evidence_dossier_v1"
ROOT = bank.B / KEY.replace("_", "-")
THIS = str(Path(__file__).resolve().relative_to(bank.P))
WORKLOAD_ID = "dflash2-snapshot-evidence-dossier-v1"
PHASE = "long-generation/" + lg.DOSSIER_WORKLOAD
ORDER = ("incumbent",)
INCUMBENT_REFERENCE = "ea197d7a08a8382326e9faeaa79346b33c502285"
CAP = lg.MAX_OUTPUT_TOKENS
PROTOCOL = dict(order=list(ORDER), workload=lg.DOSSIER_WORKLOAD,
    execution="One independent incumbent process with the official four warmups. Keep stock spec4, "
              "confidence 0.6, KV and expert placement, prompt cache off, and the qualified runtime.",
    evidence="Count and time the same thinking-on body. Require 32768..65536 fresh input tokens, "
             "1024..16384 actual outputs including thought, visible answer and at least 256 drafts. "
             "Record prompt time, client TTFT and sustained decode separately.",
    decision="Weak means at most 0.40 observed draft acceptance. Preserve the measured screen if "
             "acceptance is higher. Independent replications and a matched comparison remain separate.",
    stop="Stop on the first source, request, route, correctness, ownership, budget or GPU fault mismatch.")


def wire_sha(workload):
    body = client.chat_body(lg.evidence_dossier_messages(0, workload), CAP, True,
        request_overrides={"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget_tokens": 0})
    return hashlib.sha256(json.dumps(body).encode()).hexdigest()


def baseline():
    row = parent.baseline()
    prior = bank.previous.historic(INCUMBENT_REFERENCE, parent.KEY)
    case = prior["processes"][4]
    require(case["index"] == 4 and case["arm"] == ORDER[0] and case["passed"] and case["complete"]
            and case["ownership_closed"] and case["native_exit_code"] == 0 and not case["fresh_faults"]
            and all(row[k] == prior[k] for k in ("binary", "native_source_commit", "boot", "runtime_manifest"))
            and case["binary"] == row["binary"] and case["boot_unchanged"],
            "recover the exact closed incumbent regular-code control")
    return dict(row, incumbent_reference=INCUMBENT_REFERENCE,
                incumbent_case_sha256=bank.signature(case), dossier_body_sha256=wire_sha(WORKLOAD_ID))


def command(plan):
    root = Path(plan["root"])
    return ["/bin/sh", str(bank.P / "bench.sh"), "--config", str(root / "config.json"),
            "--port", str(plan["port"]), "--long-generation-only", "--long-generation-workload",
            lg.DOSSIER_WORKLOAD, "--long-generation-max-tokens", str(CAP), "--think", "--runs", "1",
            "--workload-id", WORKLOAD_ID, "--fail-fast-engine-error", "--ready-timeout", "600",
            "--out", str(root / "bench")]


def read_suite(root, workload_id):
    path, = (root / "bench").glob("*/results.json")
    result = bank.previous.read(path)
    rows, args, identity = (result[k] for k in ("results", "params", "comparison_identity"))
    journal = client.verify_request_journal(path.parent)
    generation = rows["long_generation"]
    workload = dict(workload_id=workload_id, nonce=workload_id, nonce_policy="fixed-id")
    require(result["complete"] and not rows["failures"] and not rows["skipped"]
            and args["runs"] == 1 and args["think"] and args["long_generation_only"]
            and generation["complete"] and generation["diagnostic_subset"] and not generation["speed_only"]
            and generation["selected_workloads"] == [lg.DOSSIER_WORKLOAD]
            and generation["thinking"] and generation["max_tokens"] == CAP and generation["reasoning_budget_tokens"] == 0
            and generation["workload_version"] == identity["long_generation_version"] == lg.DOSSIER_VERSION
            and identity["workload"] == args["workload"] == workload
            and identity["context"] == result["machine"]["context"] == 262144
            and identity["client"] == client.client_fingerprint()
            and journal["valid"] and journal["complete"] and not journal["pending"],
            "require the complete official dossier request and exact benchmark identity")
    entries = [json.loads(line) for line in (path.parent / "requests.jsonl").read_text().splitlines()]
    counted, = (e for e in entries if e["kind"] == "count-preflight")
    start, = (e for e in entries if e["kind"] == "start" and e["phase"] == PHASE)
    proof, sha = counted["proof"], wire_sha(workload_id)
    require(counted["phase"] == PHASE and counted["run"] == 0 and counted["index"] + 1 == start["index"]
            and proof["input_case"] == start["input_case"] == lg.DOSSIER_WORKLOAD
            and proof["body_sha256"] == start["request_sha256"] == sha,
            "require the exact count-only proof immediately before the timed body")
    leaves.suite.gen.warmup_reconciliation([e for e in entries if e is not counted], (lg.DOSSIER_WORKLOAD,))
    samples = [dict(e["sample"], phase=e["phase"], run=e["run"], turn=e["turn"])
               for e in entries if e["kind"] == "result"]
    sample, = leaves.suite.gen.collect_samples(generation)
    last = samples[-1]
    tracked = (*leaves.WORK, "decode_ms", "prompt_ms", "ttft_s", "weak_draft_observed",
               "counted_input_tokens", "count_preflight_body_sha256")
    require(sample["controller_eligible"] and (sample["workload"], sample["run"], sample["order"]) ==
            (lg.DOSSIER_WORKLOAD, 0, 0) and all(sample[k] == last[k] for k in tracked)
            and last["phase"] == PHASE and last["request_sha256"] == last["count_preflight_body_sha256"] == sha
            and last["read_tokens"] == last["prompt_tokens"] == last["counted_input_tokens"] ==
                proof["counted_input_tokens"] and last["reused_tokens"] == 0
            and lg.MIN_INPUT_TOKENS <= last["read_tokens"] <= lg.DOSSIER_MAX_INPUT_TOKENS,
            "dossier journal, body or actual consumed input differs")
    require(last["drafts_offered"] >= lg.MIN_DRAFTS and last["weak_draft_observed"] is
            (5 * last["drafts_accepted"] <= 2 * last["drafts_offered"]), "draft classification differs")
    basis.comparison.require_valid_client_ttft(last)
    fields = (*leaves.WORK, "phase", "run", "turn", "decode_ms", "prompt_ms", "ttft_s")
    return dict(samples=[{k: s[k] for k in fields} for s in samples], journal=journal,
                decode_rates={lg.DOSSIER_WORKLOAD: 1000 * last["generated_tokens"] / last["decode_ms"]},
                count_preflight=proof, weak_draft_observed=last["weak_draft_observed"])


def routes(text, samples, arm):
    require(arm == ORDER[0], "unknown dossier arm")
    return dict(leaves.routes(text, samples, "clean", phase=PHASE),
                window_work=parent.confidence.window_counts(samples[-1], "threshold"))


def decision(processes):
    p, = processes
    require(p["arm"] == ORDER[0] and p["passed"] and p["complete"] and p["ownership_closed"]
            and p["native_exit_code"] == 0 and not p["fresh_faults"], "one closed incumbent screen required")
    evidence = p["evidence"]
    sample = evidence["samples"][-1]
    return dict(weak_draft_observed=evidence["weak_draft_observed"],
        acceptance=sample["drafts_accepted"] / sample["drafts_offered"],
        TG=evidence["decode_rates"][lg.DOSSIER_WORKLOAD], independent_replications_qualified=False, adopted=False)


def experiment():
    ctx = parent.experiment()
    pins = ctx.pins
    ctx.KEY, ctx.ROOT, ctx.THIS, ctx.ORDER, ctx.WORKLOAD_ID = KEY, ROOT, THIS, ORDER, WORKLOAD_ID
    ctx.baseline, ctx.protocol, ctx.routes, ctx.command = baseline, PROTOCOL, routes, command
    ctx.evidence_reader.read_complete_suite, ctx.evidence_reader.decide = read_suite, decision

    def dossier_pins(revision):
        result = pins(revision)
        require((bank.P / THIS).read_bytes() == basis.validation.git("show", revision + ":" + THIS),
                "commit the exact dossier controller")
        result[THIS] = bank.sha(bank.P / THIS)
        return result

    ctx.pins = dossier_pins
    return ctx


if __name__ == "__main__":
    parent.clock.main(experiment())
