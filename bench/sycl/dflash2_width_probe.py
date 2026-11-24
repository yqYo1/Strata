"""Run the bounded, loan-off DFlash2 width comparison on the snapshot build."""

import fcntl
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import dflash2_snapshot_decode as snapshot
import dflash2_target_snapshot_build as build
import dflash2_v2_commit_order as native
import dflash2_width_admission as audit

bank, basis, require = snapshot.bank, snapshot.basis, snapshot.require
model = native.model
KEY = "dflash2_width_admission_v1"
ROOT = bank.B / KEY.replace("_", "-")
TEST = bank.P / "bench/sycl/test_dflash2_width_admission.py"
ORDER = ("host", "4", "5")
OUTPUT = 32
SLOTS = "1024"
CHUNK = "1024"
NATIVE_PATHS = (
    "src/program/generate.cpp", "src/core/verify.cpp", "src/core/expert_source.cpp",
    "include/strata/core/cache_scratch_lease.hpp", "include/strata/core/verify_window_policy.hpp",
    "include/strata/kernels/verify_kernels.hpp", "include/strata/kernels/cpu/expert.hpp",
    "src/kernels/sycl/native_mmvq.cpp",
)
CONTROLLERS = (
    Path(__file__).resolve(), Path(audit.__file__).resolve(),
    TEST, bank.P / native.THIS,
)
OBSERVER = dict(
    STRATA_TRACE="1", STRATA_SYCL_Q6_ROUTE_AUDIT="1", STRATA_SYCL_CACHE_LOAN_AUDIT="1",
    STRATA_SYCL_CACHE_SCRATCH_LEASE="0", STRATA_PREFILL_RELEASE_HEADS="0",
    STRATA_PREFILL_RELEASE_MTP_EXPERTS="0", STRATA_SYCL_DECODE_Q6_DPAS_HEAD="1",
)
HOST_LIMITS = dict(wall=60, cpu=30, rss_cap=512 << 20, text_cap=64 << 10,
                   file_cap=128 << 10, total_cap=512 << 10)


def inputs(revision):
    base = snapshot.baseline()
    selected = base["snapshot_arm"]
    image = selected["binary"]
    record = bank.previous.read(build.ROOT / "record.json")
    require(record["binaries"]["strata"] == image
            and bank.signature(bank.builder.compact(record)) == selected["native_build_sha256"]
            and bank.sha(image["path"]) == image["sha256"], "target snapshot image or build record changed")
    native_sources = {name: record["source_files"][name] for name in NATIVE_PATHS}
    require(all(bank.sha(build.ROOT / "source" / name) == digest for name, digest in native_sources.items()),
            "native verifier policy source differs from the compiled projection")

    pins = snapshot.experiment().pins(revision)
    for path in CONTROLLERS:
        name = str(path.relative_to(bank.P))
        require(path.read_bytes() == basis.validation.git("show", revision + ":" + name),
                "uncommitted width probe input: " + name)
        pins[name] = bank.sha(path)
    host_argv = [sys.executable, "-B", str(TEST), "-q"]
    host_env = dict(PATH=str(Path(sys.executable).parent) + ":/usr/bin:/bin", LC_ALL="C.UTF-8")
    cases, launches = {}, {"host": host_env}
    user_args = model.server.user_args
    common = ["--prefill", CHUNK, "--expert-cache", SLOTS, "--no-prefill-borrow"]
    template = snapshot.confidence.configuration(base, dict(root=str(ROOT), arm="full"))
    ids = model.request_ids(Path(template["tokenizer"]))["ids"]
    require(len(ids) == 82, "the checked prompt fixture changed")
    for width in ORDER[1:]:
        root = ROOT / width
        changes = ["--spec", width, *common]
        cfg = dict(template, user_args=user_args.merge(template["user_args"], changes))
        args = model.server.engine_args(cfg, exe=cfg["exe"], announce=False)
        expected = {"--draft": "dflash2", "--draft-fallback": "none", "--draft-conf": "0",
                    "--spec": width, "--expert-cache": SLOTS, "--prefill": CHUNK,
                    "--kv-resident": "20480", "--pool-workers": "5", "--pcie-frac": "0",
                    "--adapt-swaps": "0", "--prompt-cache": "0", "--max-context": str(audit.CONTEXT)}
        require(all(args.count(k) == 1 and args[args.index(k) + 1] == v for k, v in expected.items())
                and args.count("--no-prefill-borrow") == 1 and "--no-token-graph" in args
                and not set(args).intersection(("--mtp", "--mtp-max-t", "--draft-cap", "--spec-min-p")),
                "effective diagnostic width, drafter or placement differs")
        case_env = snapshot.confidence.environment(base, dict(root=str(root), arm="full"))
        env = model.native_environment(base, dict(case_env, **OBSERVER))
        require(env.get("STRATA_COMMIT_SYNC") == "1" and env.get("STRATA_WATCHDOG_S") == "240"
                and env.get("STRATA_SYCL_DECODE_Q6_COLUMNS") == "1"
                and env.get("STRATA_SYCL_DECODE_Q6_CENTERING", "0") == "0"
                and env.get("STRATA_SYCL_DECODE_Q6_NATIVE_HALF", "0") == "0",
                "preserve the checked commit order and canonical head flags")
        cases[width] = dict(command=[image["path"], "--serve", *args],
                            info=dict(model.INFO, spec=width, expert_slots=SLOTS, lookup="0",
                                      pool_workers="5", pcie_frac="0.00", spec_min_p="0.00"),
                            environment_sha256=bank.signature(env))
        launches[width] = env

    left, right = (cases[width]["command"] for width in ORDER[1:])
    require(len(left) == len(right) and [a for a, b in zip(left, right) if a != b] == ["4"]
            and right[left.index("--spec") + 1] == "5", "only the effective width may differ")
    normalized = [bank.signature({k: v.replace(str(ROOT / width), "<case>") for k, v in launches[width].items()})
                  for width in ORDER[1:]]
    require(normalized[0] == normalized[1], "another native setting differs between widths")
    plan = dict(
        source_commit=revision, inputs=dict(files=len(pins), sha256=bank.signature(pins)),
        native_sources=native_sources, **selected, boot=base["boot"], driver=base["driver"],
        runtime_manifest=base["runtime_manifest"], model_assets_sha256=bank.signature(base["model_assets"]),
        validation=base["native_validation_probe"], drafter_assets=base["dflash2_assets"],
        request=dict(ids=ids, output=OUTPUT, wire_sha256=model.restore.signature_bytes(model.wire(ids, OUTPUT))),
        order=list(ORDER), host=dict(command=host_argv, environment_sha256=bank.signature(host_env), limits=HOST_LIMITS),
        cases=cases, native_limits=model.LIMITS,
        scope="Run host accounting checks, then one fresh 82-input/32-output native request per width. "
              "Use spare VRAM via the same smaller prefill chunk and expert cache in both arms. No loans. "
              "Require true full-width replays after the first anchor, complete commits, validated head "
              "dispatch, normal child exit and no new Xe fault. Stop on first failure. Diagnostic only.",
    )
    return plan, launches


def inspect_case(text, plan, pid, requests, responses, width):
    proof = basis.validation.preload_evidence(text, pid)
    require(proof["loader"] == plan["validation"]["loader"]["path"]
            and text.count("strata native head DPAS: ready=1 columns=4 layout=canonical") == 1
            and not re.search(r"(?m)^strata mtp: .*loaded|^strata cache loan: phase=|"
                              r"UR_RESULT_ERROR_|ZE_RESULT_ERROR_|teardown uncertain", text),
            "unexpected validation, cache borrowing, target head or MTP load")
    taps = re.findall(r"(?m)^strata DFlash2: eager, prompt_tail_rows=81 cold_prefix=0 vram_bytes=([1-9]\d*)$", text)
    allocations = re.findall(r"(?m)^strata verify: window up to (\d+) tokens, ([\d.]+) MiB of device buffers$", text)
    require(len(taps) == 1 and len(allocations) == 1 and int(allocations[0][0]) == width,
            "require the expected draft tail and initial verifier allocation")
    sample, = audit.audit_requests(text, requests, responses, width=width)
    require(sample["width_counts"].keys() == {1, width}, "full-width decode graph was not executed")
    return dict(validation=proof, window_work=sample, head_captures=audit.audit_head(text, sample["width_counts"]),
                draft_vram_bytes=int(taps[0]), verifier_buffer_mib=float(allocations[0][1]))


def save(row):
    bank.stage.write(ROOT / "record.json", row)
    result = bank.previous.read(bank.RESULT)
    result[KEY] = row
    bank.stage.write(bank.RESULT, result)


def run_host(plan, env):
    owner = model.owner_module.Owner(ROOT / "host" / "owned")
    entry, _, stderr = owner.run("accounting", plan["host"]["command"], env, bank.P, **HOST_LIMITS)
    closed = model.owner_module.closed(entry)
    results = stderr.read_text()
    passed = bool(closed and entry.get("exit_code") == 0
                  and re.search(r"(?m)^Ran 4 tests in ", results) and results.rstrip().endswith("\nOK"))
    return dict(passed=passed, ownership_closed=closed, exit_code=entry.get("exit_code"),
                error=entry.get("error"), GPU_executed=False)


def main(action):
    require(sys.executable == basis.reference.model.PYTHON and action in ("prepare", *ORDER),
            "use the pinned Python with prepare, host, 4 or 5")
    with (bank.B / "owned-v0141-measurement.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if action == "prepare":
            require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT),
                    "inspect the existing width probe before preparing")
            revision = basis.validation.git("rev-parse", "HEAD").decode().strip()
            plan, _ = inputs(revision)
            for width in ORDER[1:]:
                for folder in ("cache", "tmp"):
                    (ROOT / width / folder).mkdir(parents=True, mode=0o700)
            (ROOT / "host").mkdir(mode=0o700)
            save(dict(plan=plan, stage="prepared", outcomes={}, passed=False, complete=False, ownership_closed=True))
            return

        row = bank.previous.read(ROOT / "record.json")
        require(row == basis.validation.committed(bank.RESULT, (KEY,)), "commit the exact width plan and prior outcomes")
        plan, launches = inputs(row["plan"]["source_commit"])
        earlier = set(ORDER[:ORDER.index(action)])
        require(row["plan"] == plan and row["ownership_closed"] and set(row["outcomes"]) == earlier
                and all(outcome["passed"] for outcome in row["outcomes"].values())
                and not (ROOT / action / "owned").exists() and not (ROOT / action / "execution.json").exists(),
                "frozen width inputs, order or ownership changed")
        if action == "host":
            outcome = run_host(plan, launches[action])
        else:
            requests = [(plan["request"]["ids"], plan["request"]["output"])]
            outcome = native.run_case(plan, ROOT / action, action, launches[action], requests,
                lambda text, pid, responses: inspect_case(text, plan, pid, requests, responses, int(action)),
                expected_info=plan["cases"][action]["info"])
        row["outcomes"][action] = outcome
        complete = bool(outcome["passed"] and action == ORDER[-1])
        row.update(stage=("closed" if complete else "partial") if outcome["passed"] else "failed",
                   passed=complete, complete=complete, ownership_closed=outcome["ownership_closed"])
        save(row)
        print({key: outcome.get(key) for key in ("passed", "ownership_closed", "native_exit_code", "fresh_faults", "error")},
              flush=True)
        require(outcome["passed"], "width probe stage failed")


if __name__ == "__main__":
    action, = sys.argv[1:]
    main(action)
