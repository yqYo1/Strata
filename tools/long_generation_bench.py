"""Fixed long-output workloads; MTP categories require observed proposal counts."""
from __future__ import annotations

import math

MIN_OUTPUT_TOKENS = 1024
MAX_OUTPUT_TOKENS = 1152
MIN_INPUT_TOKENS = 32768
TARGET_INPUT_TOKENS = 33792  # calibration is approximate; check actual usage afterwards
MIN_DRAFTS = 256
WORKLOAD_VERSION = 1

WORKLOADS = {
    "regular-code": {
        "expected_mtp": "strong",
        "instruction": (
            "Write a complete Python module containing 180 small unit tests for an integer identity function. "
            "Start with import unittest and def identity(value): return value. Then define class "
            "TestIdentity(unittest.TestCase). Write a separate test_000 through test_179 method, in order, "
            "each with self.assertEqual(identity(N), N), where N is that method's index. "
            "Use the same formatting for every method. Include every method explicitly; do not use loops, "
            "ellipsis or a summary. Finish with the usual unittest.main guard. "
            "Give the full module rather than a short example."
        ),
    },
    "constraint-reasoning": {
        "expected_mtp": "weak",
        "instruction": (
            "Solve and explain this finite scheduling problem from first principles. Jobs A through H have "
            "durations 3, 5, 2, 7, 4, 6, 3, 5 respectively. There are two identical workers. "
            "A precedes D and E; B precedes E and F; C precedes F; D and E precede G; F precedes H. "
            "Jobs cannot be interrupted and a worker can execute only one job at a time. "
            "Exactly one unit of setup time is required when a worker switches between even-duration and "
            "odd-duration jobs; there is no initial setup. A and C cannot overlap because they share an "
            "external resource. Find a schedule minimizing completion time, derive useful lower bounds, "
            "and explain a reproducible exact enumeration algorithm that proves optimality. "
            "Evaluate several tempting but suboptimal schedules, give explicit start/end/setup times, "
            "and provide executable Python for the solver and its verification tests. "
            "Work through the constraints carefully and write a detailed complete solution of at least "
            "2500 words plus code. Do not give only a final schedule or a brief outline."
        ),
    },
}


def sample_eligibility(raw, sample, minimum=MIN_OUTPUT_TOKENS):
    """No stream-piece fallback, short/EOS success or absent clock can pass this gate."""
    reasons = []
    usage = raw.get("usage") or {}
    generated = usage.get("completion_tokens")
    if not sample.get("ok") or not raw.get("done_received") or raw.get("finish_reason") not in ("stop", "length"):
        reasons.append("response did not finish normally")
    if type(generated) is not int or not minimum <= generated <= MAX_OUTPUT_TOKENS:
        reasons.append("actual completion usage is outside the 1024..1152 token band")
    prompt = usage.get("prompt_tokens")
    if type(prompt) is not int or prompt < MIN_INPUT_TOKENS:
        reasons.append("actual input is below 32768 tokens or unreported")
    if sample.get("reused_tokens") != 0:
        reasons.append("fresh-prefix reuse is nonzero or unreported")
    for field in ("decode_tps", "decode_ms"):
        value = sample.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            reasons.append(f"{field} is absent or invalid")
    if sample.get("decode_tps_source") != "engine":
        reasons.append("engine decode clock is missing")
    offered, accepted = sample.get("drafts_offered"), sample.get("drafts_accepted")
    counters_ok = (type(offered) is int and type(accepted) is int and 0 <= accepted <= offered)
    rate = accepted / offered if counters_ok and offered > 0 else None
    if not counters_ok or offered < MIN_DRAFTS:
        reasons.append("MTP counters are missing, invalid or below 256 offered drafts")
    return {"eligible": not reasons, "ineligible_reasons": reasons, "acceptance_rate": rate,
            "usage_generated_tokens": generated, "completion_includes_thinking": True}


def classify_pair(workloads, runs):
    """Predeclared conservative bands, independent of the candidate's measured speed."""
    regular = workloads["regular-code"]["requests"]
    reasoning = workloads["constraint-reasoning"]["requests"]
    enough = runs >= 3 and all(len(rows) == runs and all(r["eligible"] for r in rows)
                              for rows in (regular, reasoning))
    strong = enough and all(r["acceptance_rate"] >= .60 for r in regular)
    weak = enough and all(r["acceptance_rate"] <= .40 for r in reasoning)
    return {"qualified": bool(strong and weak), "strong_case_confirmed": bool(strong),
            "weak_case_confirmed": bool(weak), "minimum_runs": 3,
            "strong_min_acceptance": .60, "weak_max_acceptance": .40,
            "scope": "Observed draft agreement only; prompts do not guarantee an MTP regime or prove speedup."}
