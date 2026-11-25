"""Compare fixed DFlash2 widths on the same official thinking-on workload."""

from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_snapshot_decode as snapshot
from decode_kv_bench import insertion_recipe, native_windows

bank, basis, require, leaves = snapshot.bank, snapshot.basis, snapshot.require, snapshot.leaves
gen = leaves.suite.gen
model = snapshot.confidence.model
KEY = "dflash2_width_generation_v4"
ROOT = bank.B / KEY.replace("_", "-")
THIS = str(Path(__file__).resolve().relative_to(bank.P))
WIDTH_EVIDENCE = "66810f1301de259d53839d597b68389a7acb9712"
REPLAY_EVIDENCE = "226e2d78cc6470312b862d60c5e4b99a20ceb57b"
REFUSAL_EVIDENCE = "22a7d484bcba16e219fbc1484149a33ad1a8c049"
READER_REFUSAL = "ae25f8272459495f3502e492cfdd2c08a16c769a"
CACHE_SLOTS = 1024
ORDER = ("4", "5", "5", "4")
WRAP_TOKENS = 18
CONTEXT = 262144
MAX_OUTPUT = 16384

# Imported benchmark modules that support the baseline and live result readers.
# The inherited suite pins the remaining controller and native inputs.
VALIDATION_INPUTS = (
    "bench/sycl/client_resources.py",
    "bench/sycl/dflash2_prefill_direct_store.py",
    "bench/sycl/dflash2_prefill_direct_store_speed.py",
    "bench/sycl/dflash2_prefill_kv_stage_model.py",
    "bench/sycl/dflash2_prefill_kv_stage_speed.py",
    "bench/sycl/dflash2_prefill_window_admission.py",
    "bench/sycl/dflash2_prefill_window_bank.py",
    "bench/sycl/dflash2_prefill_window_bank_speed.py",
    "bench/sycl/dflash2_prefill_window_direct.py",
    "bench/sycl/dflash2_prefill_window_speed.py",
    "bench/sycl/dflash2_prefill_window_writer_speed.py",
    "bench/sycl/dflash2_target_snapshot_restore.py",
    "bench/sycl/dflash2_v2_commit_order.py",
    "bench/sycl/dflash2_v2_split_admission.py",
    "bench/sycl/prefill_kv_upload_capacity.py",
)

PROTOCOL = dict(
    order=list(ORDER), workload=leaves.WORKLOAD,
    factor="Use the same target snapshot executable, single GPU, fixed loan-off 1024-slot "
           "placement and a 1024-token chunk on the explicit chunked TTFT strategy. "
           "Change only --spec between arms. Hold draft confidence zero "
           "in all processes to reconcile the actual full-width windows from draft counts.",
    evidence="Four warmups and one fresh thinking-on regular-code request in each process. "
             "Require at least 32768 actual fresh prompt tokens and 1024 native outputs "
             "including thought, a visible answer and valid client/native clocks. "
             "Conserve each native leg, the effective width, draft counts, target rows "
             "and CPU expert jobs. Keep client TTFT and prefill separate.",
    decision="Compare both process pairs. Match native prompt/reuse extents, physical "
             "prefill placement and native output lengths. Width five advances only "
             "with a gain in both matched pairs. Retain unmatched and failed outcomes. "
             "Confidence 0.6, other workloads, RESTORE and adoption have separate gates.",
    stop="Stop on the first input, context, runtime, route, ownership, budget or Xe fault failure.",
)


def baseline():
    row = snapshot.baseline()
    failed = bank.previous.historic(REFUSAL_EVIDENCE, "dflash2_width_generation_v1")
    refusal = bank.previous.historic(REFUSAL_EVIDENCE, "dflash2_width_generation_v1_prefill_refusal")
    require(failed["stage"] == "failed" and failed["ownership_closed"] and not failed["active"]
            and failed["snapshot_arm"] == row["snapshot_arm"]
            and refusal["configuration"] == dict(prefill_strategy="throughput", chunk=1024, prefill_borrow=False)
            and refusal["phase"] == "warm-up/prompt" and refusal["long_generation_requests"] == 0
            and refusal["reaped_and_session_empty"] and refusal["boot_unchanged"] and not refusal["fresh_xe_faults"],
            "recover the closed chunk8192 mixed-scratch admission refusal")
    prior = bank.previous.historic(READER_REFUSAL, "dflash2_width_generation_v2")
    reader = bank.previous.historic(READER_REFUSAL, "dflash2_width_generation_v2_startup_reader_refusal")
    require(prior["stage"] == "failed" and prior["ownership_closed"] and not prior["active"]
            and prior["snapshot_arm"] == row["snapshot_arm"]
            and reader["native_and_bench_exit_code"] == 0 and reader["physical_profile"]["slots"] == CACHE_SLOTS
            and reader["rejected_by"]["expected_slots"] == 1256,
            "recover the closed run with its original startup-reader decision")
    admitted = bank.previous.historic(WIDTH_EVIDENCE, "dflash2_width_admission_v3")
    plan = admitted["plan"]
    require(admitted["stage"] == "closed" and admitted["complete"] and admitted["passed"]
            and admitted["ownership_closed"] and admitted["outcomes"]["host"]["passed"]
            and all(plan[k] == row["snapshot_arm"][k] for k in row["snapshot_arm"])
            and all(plan[k] == row[k] for k in ("driver", "boot", "runtime_manifest"))
            and plan["drafter_assets"] == row["dflash2_assets"]
            and plan["model_assets_sha256"] == bank.signature(row["model_assets"]),
            "recover the exact closed width and placement control")
    commands = {}
    head_routes = {"4": "dpas", "5": "q6-reference"}
    for width in head_routes:
        outcome = admitted["outcomes"][width]
        require(outcome["passed"] and outcome["ownership_closed"] and outcome["native_exit_code"] == 0
                and not outcome["fresh_faults"] and outcome["head_captures"][-1]["route"] == head_routes[width]
                and set(outcome["window_work"]["width_counts"]) == {"1", width},
                "both widths must execute on the checked image")
        single, = outcome["requests"]
        expected = fixed_window_work(dict(prompt_tokens=single["prompt"], generated_tokens=single["generated"],
            drafts_offered=single["offered"], drafts_accepted=single["accepted"]), int(width))
        counts = outcome["window_work"]["width_counts"]
        require(expected["windows"] == sum(counts.values())
                and expected["target_rows"] == sum(int(rows) * count for rows, count in counts.items())
                and expected["unissued_outputs_at_end"] == outcome["window_work"]["committed_after_last_output"],
                "fixed-width accounting must reproduce the actual bounded verifier trace")
        command = plan["cases"][width]["command"]
        require(command[:2] == [plan["binary"]["path"], "--serve"], "observed native image differs")
        commands[width] = command[2:]
    replay = bank.previous.historic(REPLAY_EVIDENCE, "dflash2_dossier_continuation_replay_v1")
    tokenizer = Path(row["configuration"]["tokenizer"])
    for item in ("vocab.json", "merges.txt", "token_type.json", "tokenizer.json"):
        file = tokenizer / item
        require(bank.sha(file) == row["model_assets"][str(file)]["sha256"], "pinned tokenizer changed")
    wrap = insertion_recipe(row["configuration"])["REASONING_WRAP_UP"]
    require(wrap == replay["wrap"] and wrap["tokens"] == WRAP_TOKENS, "checked continuation insertion differs")
    return dict(row, width_evidence=WIDTH_EVIDENCE, width_evidence_sha256=bank.signature(admitted),
                prior_refusal=dict(commit=REFUSAL_EVIDENCE, sha256=bank.signature(refusal)),
                reader_refusal=dict(commit=READER_REFUSAL, sha256=bank.signature(reader)),
                checked_width_args=commands, continuation_wrap=wrap)


def configuration(row, plan):
    width = plan["arm"]
    template = snapshot.confidence.configuration(row, dict(plan, arm="full"))
    args = list(row["checked_width_args"][width])
    strategy = "--prefill-strategy"
    require(args.count(strategy) == 1 and strategy not in template["user_args"]
            and args[args.index(strategy) + 1] == "throughput",
            "recover the explicit strategy from the fixed-width launch")
    args[args.index(strategy) + 1] = "ttft"
    user_args = model.server.user_args.merge(template["user_args"],
        ["--spec", width, "--prefill", "1024", "--expert-cache", str(CACHE_SLOTS), "--no-prefill-borrow"])
    cfg = dict(template, args=args, user_args=user_args)
    require(model.server.engine_args(cfg, exe=cfg["exe"], announce=False) == args,
            "the production server resolver differs from the observed native command")
    return cfg


def environment(row, plan):
    env = snapshot.confidence.environment(row, dict(plan, arm="full"))
    require(env["STRATA_PREFILL_RELEASE_HEADS"] == env["STRATA_SYCL_DECODE_Q6_DPAS_HEAD"] == "1"
            and env["STRATA_COMMIT_SYNC"] == "1"
            and not any(name in env for name in ("STRATA_TRACE", "STRATA_DECODE_TIMING",
                                                  "STRATA_DRAFT_TIMING", "LD_PRELOAD")),
            "keep the checked canonical head, commit ordering and clean environment")
    env.update(STRATA_SYCL_CACHE_SCRATCH_LEASE="0", STRATA_SYCL_CACHE_LOAN_AUDIT="0",
               STRATA_SYCL_Q6_ROUTE_AUDIT="0")
    return env


def read_suite(root, workload_id):
    captured = leaves.read_subset(root, workload_id)
    full, = (root / "bench").glob("*/results.json")
    sample, = gen.collect_samples(bank.previous.read(full)["results"]["long_generation"])
    for name in ("decode_tps", "decode_tps_source"):
        captured["samples"][-1][name] = sample[name]
    return captured


def fixed_window_work(leg, width):
    require(leg["prompt_tokens"] + MAX_OUTPUT + width < CONTEXT,
            "the fixed-width comparison needs room for an unshortened window")
    windows, remainder = divmod(leg["drafts_offered"], width - 1)
    windows += 1
    unused = windows + leg["drafts_accepted"] - leg["generated_tokens"]
    require(not remainder and windows > 1 and 0 <= unused < width,
            "fixed width, accepted prefix or final window accounting differs")
    return dict(windows=windows, target_rows=windows + leg["drafts_offered"],
                unissued_outputs_at_end=unused)


def routes(text, samples, arm):
    width = int(arm)
    leaves.speed.validate_model_text(text, {"arm": "clean"}, cache_slots=CACHE_SLOTS)
    require(not any(mark in text for mark in ("strata trace:", "strata decode timing:",
                                              "strata cache loan: phase=", "strata Q6_ROUTE ")),
            "unexpected timing observer or target cache loan")
    ready = re.findall(r"(?m)^strata verify: window up to (\d+) tokens, ([\d.]+) MiB of device buffers$", text)
    require(len(ready) == 1 and int(ready[0][0]) == width
            and re.search(rf"(?m)^strata verify: phase=decode tokens={width} q6_", text),
            "require an actual decode capture at the selected maximum width")

    frames = native_windows(text)
    require(len(samples) == 5 and len(frames) >= 5, "require four native warmups and a sustained request")
    leaves.leaf_reader.evidence("\n".join(frames[:4]), samples[:4], width, False)
    long_frames = frames[4:]
    joined = gen.api_segment_evidence("\n".join(long_frames), samples[-1],
                                      insertion_token_counts=(WRAP_TOKENS,))
    require(joined["native_generated_tokens"] >= 1024 and samples[-1]["read_tokens"] >= 32768,
            "require actual long input and native generation")
    expert_work, _ = leaves.leaf_reader.evidence("\n".join(long_frames), joined["segments"], width, False)

    # The native argv pins TTFT, whose engine dispatch always uses run_impl.
    # Verify the real scratch geometry and handoff without enabling tracing.
    scratch = re.findall(r"(?m)^strata prefill: FP16 MoE compact requested=1 effective=1, "
                         r"chunk=(\d+), scratch rows=(\d+), accounted=(\d+), shared region=(\d+) bytes$", text)
    require(len(scratch) == 1 and scratch[0][:2] == ("1024", "1024")
            and not any(mark in text for mark in ("strata prefill: layer-major candidate",
                                                  "strata prefill attention bank:",
                                                  "strata prefill heads: target detached")),
            "require the admitted chunked scratch with no layer-major transition")
    placement = []
    for frame, leg in zip(long_frames, joined["segments"], strict=True):
        require(leg["read_tokens"] > 1024 and leg["reused_tokens"] == 0
                and frame.count("strata DFlash2: eager, prompt_tail_rows=2046 cold_prefix=0") == 1,
                "require fresh multi-chunk prefill and a complete DFlash2 tap handoff")
        placement.append(dict(strategy="ttft", chunk=1024, batched_rows=leg["read_tokens"] - 1,
                              scratch_bytes=int(scratch[0][2]), shared_bytes=int(scratch[0][3])))
    work = [fixed_window_work(leg, width) for leg in joined["segments"]]
    return dict(generation=joined, verifier_work=work, expert_work=expert_work, prefill_placement=placement,
                effective_width=width, cache_slots=CACHE_SLOTS, verifier_buffer_mib=float(ready[0][1]))


def comparable_work(process):
    routes = process["routes"]
    keys = ("prompt_tokens", "read_tokens", "reused_tokens", "generated_tokens", "server_inserted_before")
    return ([tuple(leg[key] for key in keys) for leg in routes["generation"]["segments"]],
            routes["prefill_placement"])


def decision(processes):
    require(tuple(p["arm"] for p in processes) == ORDER
            and len({(p["native_identity"]["pid"], p["native_identity"]["start_ticks"]) for p in processes}) == len(ORDER)
            and all(p["passed"] and p["complete"] and p["ownership_closed"] and p["native_exit_code"] == 0
                    and not p["fresh_faults"] for p in processes), "require four closed independent processes")
    def rate(p):
        return p["routes"]["generation"]["exact_native_decode_tps"]
    pairs = []
    for a, b in zip(processes[::2], processes[1::2], strict=True):
        original, candidate = (a, b) if a["arm"] == "4" else (b, a)
        pairs.append(dict(matched_work=comparable_work(original) == comparable_work(candidate),
                          tg_ratio=rate(candidate) / rate(original)))
    rates = {w: [rate(p) for p in processes if p["arm"] == w] for w in ("4", "5")}
    return dict(pairs=pairs, TG=rates, TG_median={w: statistics.median(v) for w, v in rates.items()},
                consistent_fixed_width_gain=all(p["matched_work"] and p["tg_ratio"] > 1 for p in pairs),
                adopted=False)


def experiment():
    ctx = leaves.experiment()
    run = ctx.run
    prior_pins = snapshot.experiment().pins
    ctx.KEY, ctx.ROOT, ctx.THIS, ctx.ORDER = KEY, ROOT, THIS, ORDER
    ctx.baseline, ctx.configuration, ctx.environment = baseline, configuration, environment
    ctx.routes, ctx.protocol = routes, PROTOCOL
    ctx.evidence_reader.read_complete_suite, ctx.evidence_reader.decide = read_suite, decision

    def native_context(row, plan):
        require(plan["arm"] == ORDER[plan["index"]], "fixed-width process order differs")
        return dict(row, **row["snapshot_arm"])

    def pins(revision):
        result = prior_pins(revision)
        for name in (THIS, "tools/sycl/decode_kv_bench.py", "tools/sycl/generation_engine.py",
                     *VALIDATION_INPUTS):
            p = bank.P / name
            require(p.read_bytes() == basis.validation.git("show", revision + ":" + name),
                    "changed fixed-width input: " + name)
            result[name] = bank.sha(p)
        return result

    ctx.pins, ctx.native_context = pins, native_context
    ctx.run = lambda index: run(index, native_context=native_context)
    return ctx


if __name__ == "__main__":
    snapshot.clock.main(experiment())
