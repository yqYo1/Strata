"""Screen a separate thinking-enabled workload for weak DFlash2 draft agreement."""

import fcntl
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


def host_evidence():
    host = basis.validation.committed(bank.RESULT, ("dflash2_evidence_dossier_host_v1",))
    require(host["stage"] == "closed" and host["passed"] and host["tests"] == 54 and host["skipped"] == 0
            and not host["native_engine_executed"] and host["python"] == sys.executable
            and bank.sha(sys.executable) == host["python_sha256"]
            and all(bank.sha(bank.P / name) == sha for name, sha in host["sources"].items())
            and host["body_sha256"] == wire_sha(WORKLOAD_ID) and host["output_cap"] == CAP
            and lg.MIN_INPUT_TOKENS <= host["actual_count"] <= lg.DOSSIER_MAX_INPUT_TOKENS,
            "require the checked exact frontend and real-tokenizer dossier body")
    return host


def baseline():
    host = host_evidence()
    row = parent.baseline()
    prior = bank.previous.historic(INCUMBENT_REFERENCE, parent.KEY)
    case = prior["processes"][4]
    require(case["index"] == 4 and case["arm"] == ORDER[0] and case["passed"] and case["complete"]
            and case["ownership_closed"] and case["native_exit_code"] == 0 and not case["fresh_faults"]
            and all(row[k] == prior[k] for k in ("binary", "native_source_commit", "boot", "runtime_manifest"))
            and case["binary"] == row["binary"] and case["boot_unchanged"],
            "recover the exact closed incumbent regular-code control")
    return dict(row, incumbent_reference=INCUMBENT_REFERENCE,
                incumbent_case_sha256=bank.signature(case), dossier_body_sha256=host["body_sha256"],
                dossier_counted_input=host["actual_count"], dossier_host_sha256=bank.signature(host))


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
                proof["counted_input_tokens"] == host_evidence()["actual_count"] and last["reused_tokens"] == 0
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


FAILED_PILOT = "a51a083d5c841307fbb8b658c481de1efc9accb4"
REPLAY_KEY = "dflash2_dossier_continuation_replay_v1"
PILOT_RECORD_SHA256 = "56e11a3a4d17ccd08de1884c465c232e1c38b4aea72c3f2c45e02ac35200d00f"
REPLAY_INPUTS = {
    "execution.json": "648b2f843bed0d01c44a853c89c3b09ea478646bdb29f970b03278ea6a64e96f",
    "engine-start.json": "087c0f2b522e039577b7b33b80eeeb4889e76ce0210a6596e83ce83d0a86b86d",
    "engine-exit.json": "e26e6d849c82ca6b01af130d2e2c6fdd12900ea30355e92c14d75b7fdc7bf986",
    "engine.stderr": "7fa1e4171f584c36ec5c4ae92cebf1cc51714658aaae4ec5b6424a5daafc6d31",
    "validation.log": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "config.json": "972d1210892e9b350df6f69cc788facdd1d8f945fab43f22f1c532ee9bbf926c",
}
REPLAY_JOURNAL = {
    "results.json": "2d4973088cedcca6c7a0ef37dea2ed4b223d1c2d4cc625f471f1b6bc6d2e5b67",
    "requests.jsonl": "bc2fa1c87ec86704ff025fda00acba441c8a839f7f2027967cd54ac1111971e1",
    "journal-state.json": "f0f0a57ece87fa33fce19bc15a4d4ae936cb82100bc16e6db77fdd1f8728c5c4",
}


def replay_continuation():
    """Reconcile the original failed pilot without changing its route decision."""
    from decode_kv_bench import insertion_recipe, native_windows

    root = ROOT / "0"
    compact = bank.previous.historic(FAILED_PILOT, KEY)
    record_path = ROOT / "record.json"
    require(bank.sha(record_path) == PILOT_RECORD_SHA256, "original full pilot receipt changed")
    attempt = bank.previous.read(record_path)
    original, = attempt["processes"]
    committed = basis.validation.committed(bank.RESULT)
    require(committed[KEY] == compact == basis.compact(attempt) and REPLAY_KEY not in committed
            and attempt["stage"] == "failed"
            and not attempt["active"] and attempt["ownership_closed"] and not original["fresh_faults"]
            and original["error"] == "RuntimeError('complete native CPU footers')",
            "recover the exact closed request-footer rejection")
    require(all(bank.sha(root / path) == digest for path, digest in REPLAY_INPUTS.items()),
            "original owned engine evidence changed")
    result_file, = (root / "bench").glob("*/results.json")
    require(all(bank.sha(result_file.parent / path) == digest for path, digest in REPLAY_JOURNAL.items()),
            "original request evidence changed")
    revision = basis.validation.git("rev-parse", "HEAD").decode().strip()
    paths = (THIS, "tools/sycl/decode_kv_bench.py", "tools/sycl/generation_engine.py",
             "tools/strata_tokenizer.py", "tools/strata_bench.py", "serve/server.py")
    require(all((bank.P / path).read_bytes() == basis.validation.git("show", revision + ":" + path)
                for path in paths), "commit the exact continuation reader")
    require(all(bank.sha(bank.P / path) == attempt["sources"][path] for path in paths[-3:]),
            "the original client, server or tokenizer implementation changed")
    require(read_suite(root, WORKLOAD_ID) == original["evidence"], "the original API sample differs")

    cfg = bank.previous.read(root / "config.json")
    tokenizer = Path(cfg["tokenizer"])
    for filename in ("vocab.json", "merges.txt", "token_type.json", "tokenizer.json"):
        path = tokenizer / filename
        require(bank.sha(path) == attempt["model_assets"][str(path)]["sha256"], "original tokenizer changed")
    require(cfg.get("answer_room") is not False and bank.sha(attempt["binary"]["path"]) == attempt["binary"]["sha256"],
            "require the default answer-room policy and the original image")
    execution = bank.previous.read(root / "execution.json")
    owner, = execution["commands"]
    require(not execution["active"] and execution["outcome"] == original
            and owner["exit_code"] == 0 and owner["observation_complete"], "original owned bench did not close")
    gen = leaves.suite.gen
    start = bank.previous.read(root / "engine-start.json")
    end = gen.native_end(root, attempt, dict(config=cfg), attempt["boot"], owner["direct_identity"]["pid"])
    require(start["native"] == end["native"] == original["native_identity"], "owned native identity differs")

    wrap = insertion_recipe(cfg)["REASONING_WRAP_UP"]
    sample, = gen.collect_samples(bank.previous.read(result_file)["results"]["long_generation"])
    require(sample["controller_eligible"] and sample["workload"] == lg.DOSSIER_WORKLOAD,
            "the original dossier client sample was incomplete")
    log = (root / "engine.stderr").read_text()
    evidence = basis.validation.preload_evidence(log, end["native"]["pid"])
    require(evidence["loader"] == attempt["native_validation_probe"]["loader"]["path"]
            and not re.search(r"UR_RESULT_ERROR_|ZE_RESULT_ERROR_|teardown uncertain", log),
            "validation of the original native process differs")
    frames = native_windows(log)
    require(len(frames) == 6, "require four warmups and exactly two generation legs")
    leaves.leaf_reader.evidence("".join(frame + "\n" for frame in frames[:4]),
                                original["evidence"]["samples"][:4], 4, False)
    joined = gen.api_segment_evidence("".join(frame + "\n" for frame in frames[4:]), sample,
                                      insertion_token_counts=(wrap["tokens"],))
    first, second = joined["segments"]
    require(first["generated_tokens"] == CAP - CAP // 4 and second["server_inserted_before"] == wrap["tokens"]
            and joined["server_inserted_tokens"] > 0 and not original["evidence"]["weak_draft_observed"]
            and sum("strata DFlash2: eager, prompt_tail_rows=2046 cold_prefix=0" in frame
                    for frame in frames[4:]) == 2, "the checked answer-room continuation differs")
    receipt = dict(
        source_commit=revision, failed_pilot=FAILED_PILOT, original_gate_passed=False,
        native_source_commit=attempt["native_source_commit"], binary_sha256=attempt["binary"]["sha256"],
        native_exit_code=end["exit_code"], fresh_faults=original["fresh_faults"], replay_only=True,
        original_pilot_record_sha256=PILOT_RECORD_SHA256,
        inputs=dict(**REPLAY_INPUTS, **REPLAY_JOURNAL), wrap=wrap, **joined,
        total_native_prompt_read=sum(row["read_tokens"] for row in joined["segments"]),
        total_native_prefill_ms=sample["prompt_ms"], client_ttft_s=sample["ttft_s"],
        draft_acceptance=sample["drafts_accepted"] / sample["drafts_offered"],
        weak_draft_observed=False, adoption_eligible=False,
    )
    committed[REPLAY_KEY] = receipt
    bank.stage.write(bank.RESULT, committed)
    print({k: receipt[k] for k in ("native_exit_code", "native_generated_tokens", "server_inserted_tokens",
                                  "total_native_prompt_read", "exact_native_decode_tps", "draft_acceptance")}, flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["replay-continuation"]:
        require(sys.executable == basis.reference.model.PYTHON, "use the pinned benchmark interpreter")
        with (bank.B / "owned-v0141-measurement.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            bank.builder.guards.no_model_owners([])
            replay_continuation()
    else:
        parent.clock.main(experiment())
