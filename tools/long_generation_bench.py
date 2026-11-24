"""Fixed long-output workloads; MTP categories require observed proposal counts."""
from __future__ import annotations

import math

MIN_OUTPUT_TOKENS = 1024
MAX_OUTPUT_TOKENS = 16384
OUTPUT_TOKEN_CHOICES = (16384, 32768, 65536)
CONTEXT_RESERVE = 256
MIN_BODY_CHARS = 512  # presence/substance check, deliberately not a token-count estimate
MIN_INPUT_TOKENS = 32768
TARGET_INPUT_TOKENS = 33792  # calibration is approximate; check actual usage afterwards
MIN_DRAFTS = 256
WORKLOAD_VERSION = 5
THINKING = True
SAMPLING_PROFILES = ("greedy", "qwen-thinking")
DEFAULT_SAMPLING_SEED = 1
STANDARD_INPUT_CASE = "standard-v5-33k"
SPLIT_INPUT_CASE = "split-v1-96k"
INPUT_CASES = (STANDARD_INPUT_CASE, SPLIT_INPUT_CASE)
SPLIT_TARGET_INPUT_TOKENS = 98304
SPLIT_MIN_INPUT_TOKENS = 90114
SPLIT_MAX_INPUT_TOKENS = 117489

WORKLOADS = {
    "regular-code": {
        "expected_mtp": "strong",
        "instruction": (
            "Write a complete Python module containing 80 small unit tests for an integer identity function. "
            "Start with import unittest and def identity(value): return value. Then define class "
            "TestIdentity(unittest.TestCase). Write a separate test_000 through test_079 method, in order, "
            "each with self.assertEqual(identity(N), N), where N is that method's index. "
            "Use the same formatting for every method. Include every method explicitly; do not use loops, "
            "ellipsis or a summary. Finish with the usual unittest.main guard. "
            "Give the full module rather than a short example."
        ),
    },
    "varied-prose": {
        "expected_mtp": "weak",
        "instruction": (
            "日本語で4000字以上の独創的な短編小説を最初から最後まで詳しく書いてください。"
            "舞台は架空の港町の夜市です。旅先で道に迷った一人の時計修理職人が、"
            "古い方位磁針を売る店主と出会い、見知らぬ人に宛てた手紙を預かります。"
            "夜市を歩きながら手紙の届け先を探し、夜明けまでに小さな旅を終える物語にしてください。"
            "主人公の視点で、屋台の食べ物、海辺の天気、町の音や匂い、出会う人との会話を"
            "具体的に描いてください。主人公が何を感じ、なぜ次の行動を選んだかを、"
            "出来事の中で自然に伝えてください。複雑な謎や厳密な仕掛けは必要ありません。"
            "箇条書き、構想、要約、執筆方針の説明ではなく、始まりから結末までの小説本文を"
            "十分な長さで出力してください。"
        ),
    },
}


DOSSIER_WORKLOAD = "evidence-dossier"
DOSSIER_VERSION = "evidence-dossier-v1"
DOSSIER_MAX_INPUT_TOKENS = 65536
DOSSIER_WORKLOAD_SPEC = {
    "expected_mtp": "weak",
    "instruction": (
        "Review the fictional evidence packet. Write an audit of the proposed "
        "efficiency and stability claims, with at least eight materially different "
        "examples from the documents. Show conversions when possible. Cite the "
        "source ID and the replication ID in each example. Identify a direction "
        "reversal, a normalization artifact, a comparison with incompatible "
        "windows, and at least one case where the replication is genuinely "
        "compatible with the primary record. Explain which conclusions remain "
        "supported after those corrections and design a focused follow-up for "
        "each unresolved measurement type. Use the records to make decisions. "
        "Provide the full audit with the evidence and calculations in the response."
    ),
}
WORKLOAD_SPECS = {**WORKLOADS, DOSSIER_WORKLOAD: DOSSIER_WORKLOAD_SPEC}


def evidence_dossier_messages(run, workload_id):
    """Generate a bounded, deterministic packet of fictional methods records."""
    sites = ("Loam", "Quince", "Dune", "Tern", "Sedge", "Aster", "Mica", "Thyme")
    methods = (
        ("pressure", "kilopascals", "pascals", 1000),
        ("flow", "litres per minute", "millilitres per minute", 1000),
        ("mass", "grams", "milligrams", 1000),
        ("delay", "seconds", "milliseconds", 1000),
    )
    documents = []
    for number in range(216):
        site = sites[number % len(sites)]
        outcome, units, small_units, factor = methods[(number // 7) % len(methods)]
        baseline = 28 + (29 * number) % 79
        change = ((13 * number) % 23) - 11
        trial = baseline + change
        minutes = 18 + (number % 7) * 3
        replicates = (number % 4) + 3
        omit = "omitted" if number % 5 == 0 else "included"
        window = minutes - 3 if number % 6 == 0 else minutes
        replicate_trial = trial if number % 11 else baseline - change
        normalization_note = "The summary presents one reading per station for both arms."
        if number % 9 == 0:
            normalization_note = (
                f"The dashboard compares a reference aggregate of {4 * baseline} from four stations with "
                f"a trial aggregate of {8 * trial} from eight stations. It labels both values per station. "
                "The calibration sheet identifies both as sums with no division already applied."
            )
        documents.append(
            f"Record P{number:03d}, site {site}. The protocol measures {outcome} over "
            f"{minutes} minutes. The displayed reference and trial medians are "
            f"{baseline} and {trial} {units}. The table labels these readings as "
            "unadjusted. Empty instrument intervals are excluded from the median "
            "but their duration is included in the log. The protocol requires the "
            "same observation window and the same empty-interval treatment before "
            "combining measurements. A reviewer proposed pooling records with the "
            "same outcome using just the site label. That proposal has not been "
            "approved and the individual calibration sheets are authoritative.\n"
            f"Replication R{number:03d} uses {replicates} consecutive runs, each "
            f"with a {window}-minute observation window. It {omit} the last warmup "
            f"from the result. It records the reference as {baseline * factor} and "
            f"the trial as {replicate_trial * factor} {small_units}. The summary "
            "rounds to whole display units only after aggregation. The operator "
            "says that the baseline console was left in its default configuration. "
            + normalization_note
        )
    return [
        {"role": "system", "content": "Use only the supplied fictional evidence. Explain the audit in the final response."},
        {"role": "user", "content": f"[{DOSSIER_VERSION} run {run} {workload_id}]\n"
         + "\n\n".join(documents) + "\n\nTask:\n" + DOSSIER_WORKLOAD_SPEC["instruction"]},
    ]


def sampling_plan(profile="greedy", seed=None):
    """Keep legacy wire requests; opt in to a separate reproducible distribution.

    Qwen/Qwen3.8-Flash-Next generation_config at de4b8e4d43b917e7706784d8bb445c9af86a3540
    supplies temperature/top-p/top-k. Neutral penalties and positive seeds are
    explicit benchmark controls, not a claim of bit-identical model output.
    """
    if profile not in SAMPLING_PROFILES:
        raise ValueError("unsupported long-generation sampling profile")
    if profile == "greedy":
        if seed is not None:
            raise ValueError("a long-generation seed requires qwen-thinking sampling")
        return dict(profile=profile, version=1, request_parameters={"temperature": 0}, seed_policy="unused")
    seed = DEFAULT_SAMPLING_SEED if seed is None else seed
    if type(seed) is not int or not 1 <= seed <= (1 << 63) - 1:
        raise ValueError("long-generation seed must be an integer in 1..2**63-1")
    return dict(profile=profile, version=1, seed_policy="base-plus-run; same seed for both workloads",
        request_parameters=dict(temperature=1.0, top_p=.95, top_k=20, min_p=0.0,
            repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
            penalty_last_n=0, seed=seed))


def sampling_request(plan, run):
    """Fresh overrides; greedy keeps chat_body's defaults and existing wire bytes."""
    if type(run) is not int or run < 0:
        raise ValueError("long-generation run must be a nonnegative integer")
    if plan["profile"] == "greedy":
        return {}
    parameters = dict(plan["request_parameters"])
    parameters["seed"] += run
    if parameters["seed"] > (1 << 63) - 1:
        raise ValueError("long-generation seed plus run exceeds 2**63-1")
    return parameters


def sample_eligibility(raw, sample, minimum=MIN_OUTPUT_TOKENS,
                       maximum=MAX_OUTPUT_TOKENS, context=None):
    """No stream-piece fallback, short/EOS success or absent clock can pass this gate."""
    if type(maximum) is not int or maximum not in OUTPUT_TOKEN_CHOICES:
        raise ValueError("unsupported long-generation output budget")
    reasons = []
    usage = raw.get("usage") or {}
    generated = usage.get("completion_tokens")
    if not sample.get("ok") or not raw.get("done_received") or raw.get("finish_reason") not in ("stop", "length"):
        reasons.append("response did not finish normally")
    if type(generated) is not int or not minimum <= generated <= maximum:
        reasons.append(f"actual completion usage is outside {minimum}..{maximum} tokens")
    if len(raw.get("text") or "") < MIN_BODY_CHARS:
        reasons.append("thinking did not leave at least 512 characters of answer body")
    if not raw.get("reasoning_text"):
        reasons.append("thinking output is absent")
    prompt = usage.get("prompt_tokens")
    if type(prompt) is not int or prompt < MIN_INPUT_TOKENS:
        reasons.append("actual input is below 32768 tokens or unreported")
    if context is not None and (type(context) is not int or type(prompt) is not int or
                                prompt + maximum + CONTEXT_RESERVE > context):
        reasons.append("actual input leaves insufficient context for the requested output budget")
    if sample.get("reused_tokens") != 0:
        reasons.append("fresh-prefix reuse is nonzero or unreported")
    for field in ("decode_tps", "decode_ms"):
        value = sample.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            reasons.append(f"{field} is absent or invalid")
    if sample.get("decode_tps_source") != "engine":
        reasons.append("engine decode clock is missing")
    speed_reasons = list(reasons)
    offered, accepted = sample.get("drafts_offered"), sample.get("drafts_accepted")
    counters_ok = (type(offered) is int and type(accepted) is int and 0 <= accepted <= offered)
    rate = accepted / offered if counters_ok and offered > 0 else None
    if not counters_ok or offered < MIN_DRAFTS:
        reasons.append("MTP counters are missing, invalid or below 256 offered drafts")
    return {"eligible": not reasons, "agreement_eligible": not reasons,
            "speed_eligible": not speed_reasons, "speed_ineligible_reasons": speed_reasons, "ineligible_reasons": reasons, "acceptance_rate": rate,
            "usage_generated_tokens": generated, "completion_includes_thinking": True}


def classify_pair(workloads, runs):
    """Predeclared conservative bands, independent of the candidate's measured speed."""
    regular = workloads["regular-code"]["requests"]
    reasoning = workloads["varied-prose"]["requests"]
    enough = runs >= 3 and all(len(rows) == runs and all(r["eligible"] for r in rows)
                              for rows in (regular, reasoning))
    strong = enough and all(r["acceptance_rate"] >= .60 for r in regular)
    weak = enough and all(r["acceptance_rate"] <= .40 for r in reasoning)
    return {"qualified": bool(strong and weak), "strong_case_confirmed": bool(strong),
            "weak_case_confirmed": bool(weak), "minimum_runs": 3,
            "strong_min_acceptance": .60, "weak_max_acceptance": .40,
            "scope": "Observed draft agreement only; prompts do not guarantee an MTP regime or prove speedup."}


def run_eligible(sample, speed_only=False):
    """Explicit throughput runs may have zero drafts; agreement stays separate."""
    offered, accepted = sample.get("drafts_offered"), sample.get("drafts_accepted")
    valid = type(offered) is int and type(accepted) is int and 0 <= accepted <= offered
    return bool(valid and sample.get("speed_eligible" if speed_only else "eligible") is True)


def split_sample_eligibility(raw, sample, proof, maximum=MAX_OUTPUT_TOKENS, context=None):
    """Qualify client counts/body only; native split admission remains a separate gate."""
    result = sample_eligibility(raw, sample, maximum=maximum, context=context)
    errors = []
    prompt = (raw.get("usage") or {}).get("prompt_tokens")
    read = (raw.get("timings") or {}).get("prompt_n")
    if (type(prompt) is not int or type(read) is not int or prompt != read or
            not SPLIT_MIN_INPUT_TOKENS <= prompt <= SPLIT_MAX_INPUT_TOKENS):
        errors.append("split input needs matching actual prompt/read counts in 90114..117489")
    if type(context) is not int or context <= 0:
        errors.append("split input needs known context and output room")
    for field in ("prompt_ms", "prompt_tps", "decode_ms", "decode_tps"):
        value = sample.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            errors.append(f"{field} is absent or invalid for split continuation")
    if sample.get("prompt_tps_source") != "engine":
        errors.append("engine prefill clock is missing")
    proof = proof if isinstance(proof, dict) else {}
    sha = proof.get("body_sha256")
    if (not isinstance(sha, str) or len(sha) != 64 or
            any(c not in "0123456789abcdef" for c in sha) or
            raw.get("request_sha256") != sha or sample.get("request_sha256") != sha or
            proof.get("input_case") != SPLIT_INPUT_CASE or
            type(proof.get("counted_input_tokens")) is not int or
            proof.get("counted_input_tokens") != prompt):
        errors.append("timed body/count differs from the split count-only preflight")
    result["speed_ineligible_reasons"].extend(errors)
    result["ineligible_reasons"].extend(errors)
    result.update(eligible=not result["ineligible_reasons"],
        agreement_eligible=not result["ineligible_reasons"],
        speed_eligible=not result["speed_ineligible_reasons"],
        input_case=SPLIT_INPUT_CASE, target_input_tokens=SPLIT_TARGET_INPUT_TOKENS,
        counted_input_tokens=proof.get("counted_input_tokens"), count_preflight_body_sha256=sha,
        native_route_qualified=False, timing_frontend_state="post-count-preflight")
    return result


def dossier_sample_eligibility(raw, sample, proof, maximum=MAX_OUTPUT_TOKENS, context=None):
    """Apply the same counted-body evidence in the journal and the final report."""
    result = sample_eligibility(raw, sample, maximum=maximum, context=context)
    proof = proof if isinstance(proof, dict) else {}
    prompt = (raw.get("usage") or {}).get("prompt_tokens")
    read = (raw.get("timings") or {}).get("prompt_n")
    cached = sample.get("reused_tokens")
    counted = proof.get("counted_input_tokens")
    sha = proof.get("body_sha256")
    errors = []
    if type(context) is not int or context <= 0:
        errors.append("evidence dossier needs known context and output room")
    if (type(prompt) is not int or type(read) is not int or type(cached) is not int or
            not MIN_INPUT_TOKENS <= prompt <= DOSSIER_MAX_INPUT_TOKENS or
            (read, cached) != (prompt, 0)):
        errors.append("evidence dossier needs matching actual prompt/read counts and zero reuse")
    if (not isinstance(sha, str) or len(sha) != 64 or
            any(c not in "0123456789abcdef" for c in sha) or
            raw.get("request_sha256") != sha or sample.get("request_sha256") != sha or
            proof.get("input_case") != DOSSIER_WORKLOAD or
            type(counted) is not int or counted != prompt):
        errors.append("evidence dossier timed body or count differs from preflight")
    result["ineligible_reasons"].extend(errors)
    result["speed_ineligible_reasons"].extend(errors)
    eligible = not result["ineligible_reasons"]
    result.update(eligible=eligible, agreement_eligible=eligible,
        speed_eligible=not result["speed_ineligible_reasons"],
        input_case=DOSSIER_WORKLOAD, counted_input_tokens=counted,
        count_preflight_body_sha256=sha, timing_frontend_state="post-count-preflight",
        weak_draft_observed=bool(eligible and 5 * sample["drafts_accepted"] <= 2 * sample["drafts_offered"]))
    return result
