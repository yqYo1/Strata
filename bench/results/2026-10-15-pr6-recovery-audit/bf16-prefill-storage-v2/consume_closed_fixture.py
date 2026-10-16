#!/usr/bin/env python3
"""Consume Main-owned CLOSED BF16 mechanics runs; never launch or clean a run.

Input closure JSON schema is documented in README.md. This is a new consumer;
historical FP16 checker/receipts remain unchanged. It does not qualify a model.
"""
import argparse
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_file(item):
    path = Path(item["path"])
    require(path.is_absolute(), "artifact path must be absolute")
    require(sha(path) == item["sha256"], f"artifact SHA mismatch: {path}")
    return path


def same_bytes(first, second):
    with first.open("rb") as a, second.open("rb") as b:
        while True:
            aa, bb = a.read(1 << 20), b.read(1 << 20)
            if aa != bb:
                return False
            if not aa:
                return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closure = json.loads(args.closure.read_text())
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / "source-manifest.json").read_text())
    require(closure["source_manifest_sha256"] == sha(here / "source-manifest.json"), "source manifest identity")
    require(closure["source_diff_sha256"] == sha(here / "source.diff") == manifest["source_diff_sha256"], "patch identity")
    expected_sources = {item["path"]: item["proposed_sha256"] for item in manifest["files"]}
    require(closure["compiled_source_sha256"] == expected_sources, "actual compiled source identity")
    checked_file(closure["binary"])
    for key in ("hardware", "compiler", "runtime", "driver", "build_command", "build_log"):
        require(bool(closure[key]), f"missing {key}")
    checked_file(closure["build_log"])
    require(closure["build_exit_code"] == 0, "build did not succeed")
    modes = ("conversion", "fullgpu", "fullhost", "mixedstore")
    runs = closure["runs"]
    require(len(runs) == 4 and {r["mode"] for r in runs} == set(modes), "four independent BF16 modes required")
    results, raws = [], []
    for run in runs:
        mode = run["mode"]
        require(run["exit_code"] == 0 and run["owned_process_closed"] is True, "run not normally closed")
        require(run["fresh_fault_count"] == 0, "fresh fault gate")
        checked_file(run["fault_receipt"])
        for key in ("command", "effective_environment", "boot_id", "owned_pid", "owned_start_time", "started", "ended", "deadline", "log_budget_bytes"):
            require(bool(run[key]), f"run missing {key}")
        require(run["environment_set_before_exec"] is True, "immutable process environment contract")
        stdout = checked_file(run["stdout"])
        lines = [line for line in stdout.read_text().splitlines() if line.startswith('{"version":1,')]
        require(len(lines) == 1, "one complete fixture record required")
        record = json.loads(lines[0])
        require(record["arm"] == "bf16_boundary_storage" and record["mode"] == mode, "explicit BF16 arm/mode")
        require(record["residual_dtype"] == "bf16" and record["residual_element_bytes"] == 2, "BF16 dtype/bytes")
        require(record["status"] == "complete" and record["packed_bits_equal"] is True and record["restored_bits_equal"] is True, "exact boundary mechanics")
        for key in ("quality_qualified", "performance_qualified", "physical_262144_qualified"):
            require(record[key] is False, "fixture cannot qualify model, speed or physical context")
        if mode == "conversion":
            require(record["pack_cases"] == 35 and record["expansion_cases"] == 65536, "conversion coverage")
            require(record["invalid_geometry_or_queue_rejections"] == 8, "API geometry and queue rejection coverage")
            require(record["pack_oracle"] == "literal_bits_and_bf16_from_f32_RNE_NaN_preserving" and record["unpack_oracle"] == "all_65536_exact_shift16", "conversion oracle")
            for key in ("nan_inputs", "signed_zeros", "infinities", "finite_overflows", "subnormal_results", "underflows_to_zero"):
                require(record[key] > 0, f"missing edge coverage {key}")
        else:
            prefix = {"fullgpu": 19, "fullhost": 0, "mixedstore": 8}[mode]
            expected = {"rows": 19, "chunk": 4, "layers": 48, "row_bytes": 20480,
                        "residual_total": 19 * 20480, "residual_gpu_rows": prefix,
                        "residual_gpu": prefix * 20480, "residual_host": (19-prefix) * 20480,
                        "residual_staging": 0 if prefix == 19 else 4 * 20480,
                        "pack_calls": 235, "unpack_calls": 235, "packed_scalars": 47 * 19 * 10240}
            require(all(record[k] == value for k, value in expected.items()), "placement/staging arithmetic")
            env = run["effective_environment"]
            require(env.get("STRATA_PREFILL_BOUNDARY_ROWS") == "0,3,4,7,8,15,16,18", "selected row coverage")
            require("STRATA_PREFILL_BOUNDARY_ROWS_ABSOLUTE" not in env, "relative row fixture")
            audit = checked_file(run["audit"])
            require(env.get("STRATA_PREFILL_BOUNDARY_PATH") == str(audit), "audit path environment")
            data = audit.read_text().splitlines()
            require(any(line.startswith("boundary_audit_v2 ") for line in data), "dtype-aware audit version")
            require("storage=bf16 pack_oracle=bf16_from_f32_RNE_NaN_quiet_preserve_sign_payload unpack_oracle=f32_from_bf16_exact_shift16" in data, "BF16 audit oracle metadata")
            require(data[-1] == "status=complete boundary_memcmp=equal", "audit exact completion")
            wanted = {"output_before_store": 384, "stored_bf16_RNE_packed_bits": 376,
                      "next_input_bf16_exact_expected": 376}
            for phase, count in wanted.items():
                records = [dict(word.split("=", 1) for word in line.split())
                           for line in data if line.startswith("phase=" + phase + " ")]
                require(len(records) == count, f"audit count: {phase}")
                for item in records:
                    require(item["storage"] == "bf16", "audit dtype label")
                    require(item["memcmp"] == ("NA" if phase == "output_before_store" else "equal"), "audit exact memcmp")
                    if phase != "output_before_store":
                        require(item["source"] == ("GPU" if int(item["relative"]) < prefix else "RAM"), "audit placement")
            raw = checked_file(run["audit_raw"])
            require(str(raw) == str(audit) + ".rows.f32" and raw.stat().st_size == 8 * 48 * 10240 * 4, "raw fixture byte count")
            raws.append(raw)
        results.append({"mode": mode, "fixture": record, "stdout": run["stdout"],
                        "exit_code": run["exit_code"], "fault_receipt": run["fault_receipt"],
                        "fresh_fault_count": run["fresh_fault_count"]})
    require(all(same_bytes(raws[0], raw) for raw in raws[1:]), "placement source rows must match byte for byte")
    summary = {"schema": 1, "arm": "bf16_boundary_storage", "status": "mechanics_complete",
               "source_diff_sha256": manifest["source_diff_sha256"], "binary": closure["binary"],
               "closure": {"path": str(args.closure.resolve()), "sha256": sha(args.closure)},
               "runs": results, "placement_source_rows_equal": True,
               "quality_qualified": False, "performance_qualified": False,
               "physical_262144_qualified": False,
               "scope": "Synthetic storage and conversion only; not model recurrence quality or physical admission."}
    # Never overwrite a historical receipt or a previous attempt.
    with args.output.open("x") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
