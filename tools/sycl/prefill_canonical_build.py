"""Prepare and build the isolated canonical upload fixture against the qualified ordinary image."""
import fcntl
from itertools import chain
from pathlib import Path
import re
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
import prefill_attention_bank_capacity as capacity
import prefill_attention_bank_component as component
import prefill_attention_bank_generation as generation
import prefill_canonical_trace as trace
import qsa_cached_subgroup as diagnostics

build, validation, require = capacity.build, capacity.validation, capacity.require
P, RESULT, sha = capacity.P, capacity.RESULT, capacity.sha
signature, read = build.signature, build.read
ROOT = build.B / "prefill-canonical-copy-fixture-v1"
KEY = "prefill_canonical_copy_fixture_build_v1"
SOURCE = "fa2191d384e7b36589b261056dfc5c43eeb4f702"
PARENT = "20f39e0dc1d609d53910a9c96bcbd54f1e878ec2"
HEADER = "include/strata/core/canonical_dense_bank.hpp"
FIXTURE = "tests/sycl/canonical_dense_bank.cpp"
TARGET = "sycl_canonical_dense_bank_test"
OBJECT = "CMakeFiles/" + TARGET + ".dir/" + FIXTURE + ".o"
BINARY = ROOT / TARGET
BUDGET = dict(wall=240, cpu=360, rss_cap=8 << 30, text_cap=16 << 20, file_cap=128 << 20, total_cap=192 << 20)
ORDER = ("trace-rows", "trace-runs", "rows", "runs", "runs", "rows", "rows", "runs")
TEST = "tests/core/test_prefill_canonical_trace.py"


def native_inputs():
    official = validation.committed(RESULT, (generation.KEY,))
    require(official["complete"] and official["ownership_closed"] and not official["active"]
            and len(official["processes"]) == 6 and all(p["ownership_closed"] and p["native_exit_code"] == 0
            and not p["fresh_faults"] for p in official["processes"]), "finish the current official GPU work first")
    native = read(build.ROOT / "record.json")
    original = build.historic(capacity.BUILD_EVIDENCE, build.KEY)
    require(validation.committed(RESULT, (build.KEY,)) == original == build.builder.compact(native)
            and native["stage"] == "built" and native["passed"] and native["ownership_closed"]
            and not native["active"] and native["host_control"]["passed"], "retained parent build changed")
    source, foundation, speed = capacity.basis.basis(expected=native["foundation"])
    retained_component = build.historic(build.COMPONENT_EVIDENCE, build.COMPONENT)
    _, payload = build.stage_image(source, retained_component)
    payload.update(build.stage.source.blobs(P, retained_component["source_commit"], build.OVERLAYS))
    require(build.stage.hashed(payload) == native["source_files"] and foundation == native["foundation"],
            "original compiled source cannot be reconstructed from Git")
    controls = native["controller_sources"]
    require(build.stage.hashed(dict(build.stage.source.blobs(P, native["source_commit"], controls))) == controls
            and native["actual_inputs"]["uncovered"] == 0 and not native["actual_inputs"]["partial_archive_reuse"]
            and native["driver"] == build.builder.guards.driver_guard()
            and native["retained_decode_reference"] == build.builder.decode_reference(), "parent input provenance changed")
    for p, h in native["source_files"].items():
        require(sha(build.ROOT / "source" / p) == h, "materialized parent source changed: " + p)
    for group in (native["configured_files"], *(native["actual_inputs"][k] for k in build.ACTUAL)):
        require(all(sha(p) == h for p, h in group.items()), "actual configured/compiler/link input changed")
    require(all(sha(v["path"]) == v["sha256"] for v in native["binaries"].values()), "parent executable changed")
    for name, raw in build.stage.source.blobs(P, PARENT, (HEADER, FIXTURE, "src/program/generate.cpp")):
        require(build.stage.hashed({name: raw})[name] == native["source_files"][name],
                "candidate base differs from the compiled input: " + name)
    require(native["external_inputs"] == source["external_inputs"]
            and sha(native["external_inputs"]["path"]) == native["external_inputs"]["sha256"],
            "external toolchain inventory changed")
    return native, speed


def ninja(directory, *args):
    return subprocess.check_output(["/usr/bin/ninja", "-t", *args], cwd=directory, text=True, timeout=30)


def commands(native):
    """Substitute only the new fixture TU/header and private output paths in the configured recipes."""
    directory = build.ROOT / "build"
    db = read(directory / "compile_commands.json")
    entry, = (v for v in db if v["file"] == str(build.ROOT / "source" / FIXTURE)
              and shlex.split(v["command"])[-4:] == ["-o", OBJECT, "-c", v["file"]])
    args = shlex.split(entry["command"])
    require(Path(entry["directory"]) == directory and "-fsycl" in args and "-ffp-contract=off" in args,
            "unexpected fixture compiler or working directory")
    compiler = args[0]
    extra = ["-MD", "-MF", str(ROOT / "fixture.d"), "-o", str(ROOT / "fixture.o"), "-c", str(ROOT / FIXTURE)]
    compile_args = [compiler, "-I" + str(ROOT / "include"), *args[1:-4], *extra]
    raw_link = ninja(directory, "commands", TARGET).splitlines()[-1]
    link = shlex.split(raw_link)
    require(link[:3] == [":", "&&", compiler] and link[-2:] == ["&&", ":"] and link.count(OBJECT) == 1
            and link[link.index("-o") + 1] == TARGET and "-Wl,-t" in link, "unexpected CMake fixture link")
    link = link[2:-2]
    archives = [arg for arg in link if arg.endswith(".a")]
    require(len(archives) == len(set(archives)) == 7 and [a for a in link if a.endswith(".o")] == [OBJECT],
            "unexpected object or archives in the fixture link")
    members = archive_members(native, directory, archives)
    link[link.index(OBJECT)] = str(ROOT / "fixture.o")
    link[link.index("-o") + 1] = str(BINARY)
    return dict(compile=compile_args, link=link, cwd=str(directory), archive_members_sha256=signature(members),
                archive_members={k: len(v) for k, v in members.items()})


def archive_members(native, directory, archives):
    deps = {}
    for block in ninja(directory, "deps").strip().split("\n\n"):
        first, *headers = block.splitlines()
        if first.endswith("(VALID)"):
            deps[first.split(": #deps", 1)[0]] = {line.strip() for line in headers}
    actual = native["actual_inputs"]
    header = str(build.ROOT / "source" / HEADER)
    result = {}
    for name in archives:
        archive = directory / name
        require(sha(archive) == actual["produced"][str(archive)], "archive changed: " + name)
        query = ninja(directory, "query", name).splitlines()
        objects = [s.strip() for s in query[2:query.index("  outputs:")] if s.strip().endswith(".o")]
        table = subprocess.check_output(["/usr/bin/ar", "t", str(archive)], text=True, timeout=30).splitlines()
        require(objects and [Path(o).name for o in objects] == table, "archive member order or name changed: " + name)
        for obj in objects:
            path = directory / obj
            require(obj in deps and deps[obj] and header not in deps[obj] and sha(path) == actual["objects"][str(path)],
                    "missing dependency evidence or class definition in linked member: " + obj)
        result[name] = objects
    require(sum(map(len, result.values())) == 106, "unexpected archive membership")
    return result


def trial_case(speed, index):
    mode = ORDER[index]
    traced = mode.startswith("trace-")
    directory = ROOT / str(index)
    env = speed["environment_template"]
    old = env["STRATA_MEASUREMENT_ROOT"]
    env = {k: v.replace(old, str(directory)) for k, v in env.items()}
    probe = speed["native_validation_probe"]
    require(env["ZES_ENABLE_SYSMAN"] == "0" and env["EnableDirectSubmission"] == "0"
            and env["UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD"] == "1" and env["STRATA_SYCL_PREFILL_SAME_QUEUE"] == "1",
            "ordinary queue safety environment changed")
    env.update(TMPDIR=str(directory / "tmp"), XDG_CACHE_HOME=str(directory / "cache"),
               LD_PRELOAD=probe["library"]["path"], STRATA_UR_PROBE_EXPECTED_LOADER=probe["loader"]["path"],
               UR_LOG_VALIDATION="level:warning;flush:warning;output:file," + str(directory / "validation.log"))
    if traced:
        env.update(UR_ENABLE_LAYERS="UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION",
                   UR_LOG_TRACING="level:info;flush:info;output:stderr", LD_DEBUG="libs")
    else:
        require(env["UR_ENABLE_LAYERS"] == "UR_LAYER_PARAMETER_VALIDATION" and "UR_LOG_TRACING" not in env
                and "LD_DEBUG" not in env, "unexpected observer in clean timing")
    args = speed["configuration"]["args"]
    pack = Path(args[args.index("--pack") + 1])
    shards = sorted(p for p in speed["model_assets"] if p.endswith(".gguf"))
    require(len(shards) == 2 and str(pack / "index.txt") in speed["model_assets"], "expected two shards and pack")
    flag = "--prefill-copy-" + mode[6:] if traced else "--time-prefill-copy-" + mode
    cmd = [str(BINARY), flag, str(pack), *shards]
    budget = dict(wall=900, cpu=1500, rss_cap=22 << 30, text_cap=(96 if traced else 2) << 20,
                  file_cap=112 << 20, total_cap=128 << 20)
    return env, dict(mode=mode, command=cmd, environment_sha256=signature(env), budgets=budget)


def inputs(native, speed, revision):
    files = dict(build.stage.source.blobs(P, SOURCE, (HEADER, FIXTURE)))
    sources = build.stage.hashed(files)
    require(all(sources[p] != native["source_files"][p] for p in sources), "missing candidate changes")
    controls = build.stage.source.pin(P, ["tools/sycl", TEST, *capacity.basis.POLICIES])["source_sha256"]
    original_controls = build.stage.hashed(dict(build.stage.source.blobs(P, revision, controls)))
    require(original_controls == controls, "current verifier differs from the pinned controller revision")
    recipe = commands(native)
    env = dict(native["recipe"]["environment"], TMPDIR=str(ROOT / "tmp"), XDG_CACHE_HOME=str(ROOT / "cache"))
    return files, dict(candidate=SOURCE, source_sha256=sources, controller_commit=revision,
                       controller_sha256=signature(controls),
                       parent_receipt_sha256=signature(build.builder.compact(native)), recipe=recipe,
                       environment=env, budgets=BUDGET, host_python=dict(path=sys.executable, sha256=sha(sys.executable)),
                       model_assets_sha256=signature(speed["model_assets"]),
                       validation_probe=speed["native_validation_probe"], runtime_manifest=speed["runtime_manifest"],
                       trials=[trial_case(speed, i)[1] for i in range(len(ORDER))],
                       gate="Match exact calls, addresses, bytes, restored data and owners in traced rows/runs. "
                            "Three independent rows/runs, runs/rows, rows/runs pairs. Four full 48-layer "
                            "sweeps after validation per timing process. First sweep separate. All three "
                            "paired medians of the final three sweeps must improve to advance.")


def save(row):
    build.stage.write(ROOT / "record.json", row)
    public = read(RESULT)
    public[KEY] = row
    build.stage.write(RESULT, public)


def prepare():
    require(not ROOT.exists() and KEY not in validation.committed(RESULT), "inspect existing fixture receipt")
    native, speed = native_inputs()
    revision = build.stage.source.git(P, "rev-parse", "HEAD").decode().strip()
    files, frozen = inputs(native, speed, revision)
    ROOT.mkdir(mode=0o700)
    for name in ("tmp", "cache"):
        (ROOT / name).mkdir(mode=0o700)
    for name, raw in files.items():
        path = ROOT / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(raw)
    save(dict(stage="prepared", frozen=frozen, ownership_closed=True, passed=False, complete=False, processes=[],
              scope="Isolated fixture only. Model and full-context generator objects are outside its link."))


def compiled_inputs(native, frozen):
    external = read(Path(native["external_inputs"]["path"]))
    allowed = {**external, **native["actual_inputs"]["compiled"], **native["actual_inputs"]["generated"]}
    allowed.update({str(build.ROOT / "source" / p): h for p, h in native["source_files"].items()})
    candidates = {str(ROOT / p): h for p, h in frozen["source_sha256"].items()}
    allowed.update(candidates)
    deps = component.dependency_paths((ROOT / "fixture.d").read_text())
    require(set(candidates) <= deps and str(build.ROOT / "source" / HEADER) not in deps
            and all(p in allowed and sha(p) == allowed[p] for p in deps), "unexpected compiler dependency")
    return {p: allowed[p] for p in sorted(deps)}


def link_inputs(native, logs, directory, obj_hash):
    allowed = {**native["actual_inputs"]["links"], **native["actual_inputs"]["produced"],
               **read(Path(native["external_inputs"]["path"])), str(ROOT / "fixture.o"): obj_hash}
    found = {}
    for line in logs.splitlines():
        token = line.strip()
        if not token or len(token) > 4096 or any(c.isspace() for c in token):
            continue
        path = Path(token)
        if not path.is_absolute() and path.suffix not in (".a", ".o") and ".so" not in path.name:
            continue
        path = (directory / path).resolve()
        if not path.exists():
            require(path.is_relative_to(ROOT / "tmp") and path.suffix == ".o", "missing linker input: " + str(path))
            continue
        require(str(path) in allowed and sha(path) == allowed[str(path)], "unfrozen linker input: " + str(path))
        found[str(path)] = allowed[str(path)]
    return found


def compile_fixture():
    row = read(ROOT / "record.json")
    require(row == validation.committed(RESULT, (KEY,)) and row["stage"] == "prepared",
            "commit the prepared recipe before building")
    native, speed = native_inputs()
    _, frozen = inputs(native, speed, row["frozen"]["controller_commit"])
    require(row["frozen"] == frozen and all(sha(ROOT / p) == h for p, h in frozen["source_sha256"].items()),
            "prepared source, archive ancestry, toolchain or recipe changed")
    owner = component.ownership.Owner(ROOT / "owned", lambda: build.stage.write(ROOT / "ownership.json", owner.commands))
    recipe = frozen["recipe"]
    directory = Path(recipe["cwd"])
    try:
        command, _, err = owner.run("reader-tests", [sys.executable, "-m", "unittest", "discover",
            "-s", "tests/core", "-p", Path(TEST).name], frozen["environment"], P, **BUDGET)
        require(component.ownership.closed(command) and command["exit_code"] == 0
                and re.search(r"Ran [1-9]\d* tests? in ", err.read_text()), "trace reader tests missing or failed")
        for label in ("compile", "link"):
            command, out, err = owner.run(label, recipe[label], frozen["environment"], directory, **BUDGET)
            require(component.ownership.closed(command) and command["exit_code"] == 0, "fixture " + label + " failed")
            if label == "compile":
                deps = compiled_inputs(native, frozen)
                row["object_sha256"] = sha(ROOT / "fixture.o")
                row["compiler_dependencies"] = dict(count=len(deps), sha256=signature(deps))
            else:
                linked = link_inputs(native, out.read_text() + "\n" + err.read_text(), directory, row["object_sha256"])
                require(all(str(directory / name) in linked for name in recipe["archive_members"])
                        and str(ROOT / "fixture.o") in linked and any("libsycl" in p for p in linked),
                        "incomplete actual fixture link")
                row["link_inputs"] = dict(count=len(linked), sha256=signature(linked))
        row.update(stage="built", build_passed=True, binary=dict(path=str(BINARY), sha256=sha(BINARY),
                                                                bytes=BINARY.stat().st_size))
    except BaseException as error:
        row.update(stage="failed", passed=False, error=repr(error))
        raise
    finally:
        row["ownership_closed"] = owner.active is None and all(component.ownership.closed(c) for c in owner.commands)
        save(row)


def verify_binary(row, native, speed):
    frozen = row["frozen"]
    _, current = inputs(native, speed, frozen["controller_commit"])
    require(current == frozen and row["build_passed"] and row["ownership_closed"]
            and row["binary"] == dict(path=str(BINARY), sha256=sha(BINARY), bytes=BINARY.stat().st_size)
            and all(sha(ROOT / p) == h for p, h in frozen["source_sha256"].items())
            and sha(ROOT / "fixture.o") == row["object_sha256"], "source, recipe or product changed")
    deps = compiled_inputs(native, frozen)
    logs = (ROOT / "owned/link.stdout").read_text() + "\n" + (ROOT / "owned/link.stderr").read_text()
    linked = link_inputs(native, logs, Path(frozen["recipe"]["cwd"]), row["object_sha256"])
    for key, files in (("compiler_dependencies", deps), ("link_inputs", linked)):
        require(row[key] == dict(count=len(files), sha256=signature(files)), "compiled closure differs: " + key)


def fixture_providers(stderr, speed, env):
    probe = speed["native_validation_probe"]
    allowed = {str(Path(p).resolve()): h for p, h in read(Path(speed["runtime_manifest"]["path"])).items()}
    allowed[str(Path(probe["library"]["path"]).resolve())] = probe["library"]["sha256"]
    loaded = {str(Path(p.strip()).resolve()) for p in re.findall(r"^\s*\d+:\s+calling init:\s*(/[^\n]+)$", stderr, re.M)}
    required = [env["UR_ADAPTERS_FORCE_LOAD"], probe["loader"]["path"], probe["library"]["path"]]
    require(all(str(Path(p).resolve()) in loaded for p in required)
            and any("libsycl.so" in p for p in loaded) and any("libze_intel_gpu.so" in p for p in loaded)
            and all(p in allowed and sha(p) == allowed[p] for p in loaded), "unexpected or missing loaded provider")
    return dict(count=len(loaded), sha256=signature({p: allowed[p] for p in loaded}))


def run_trial():
    row = read(ROOT / "record.json")
    require(row == validation.committed(RESULT, (KEY,)) and row["stage"] in ("built", "measuring"),
            "commit the exact successful previous phase")
    native, speed = native_inputs()
    verify_binary(row, native, speed)
    index = len(row["processes"])
    env, case = trial_case(speed, index)
    require(case == row["frozen"]["trials"][index] and [p["mode"] for p in row["processes"]] == list(ORDER[:index])
            and all(p["passed"] for p in row["processes"]), "wrong order or failed prerequisite")
    cmd = case["command"]
    pack_index = Path(cmd[2]) / "index.txt"
    layouts = trace.pack_layout(pack_index)
    capacity.basis.comparison.shared.models.model_pins(speed["configuration"], speed["model_assets"], full=index == 0)
    directory = ROOT / str(index)
    directory.mkdir(mode=0o700)
    for name in ("cache", "tmp"):
        (directory / name).mkdir(mode=0o700)
    owner = component.ownership.Owner(directory / "owned",
        lambda: build.stage.write(directory / "ownership.json", owner.commands))
    cursor, = (line.split(": ", 1)[1] for line in subprocess.check_output(
        ["journalctl", "-k", "-b", "-n", "0", "--show-cursor", "--no-pager"], text=True, timeout=15).splitlines()
        if line.startswith("-- cursor: "))
    result = dict(index=index, mode=case["mode"], host_probe_accepted=False)
    try:
        command, out, err = owner.run("fixture", cmd, env, P, **case["budgets"])
        require(component.ownership.closed(command) and command["exit_code"] == 0, "native fixture failed")
        stderr, stdout = err.read_text(), out.read_text()
        proof = validation.preload_evidence(stderr, command["pid"])
        diagnostics.diagnostics(stderr, command["pid"])
        warnings = directory / "validation.log"
        require(proof["loader"] == speed["native_validation_probe"]["loader"]["path"]
                and not (warnings.exists() and warnings.stat().st_size), "provider or parameter validation failed")
        completion = re.findall(r"^COMPONENT_ONLY canonical=1538035200 bank=(\d+) persistent=(\d+) "
                                r"rows=777; no model/runtime-quality claim$", stdout, re.M)
        require(len(completion) == 1 and all(int(x) > 0 for x in completion[0]), "incomplete actual loader and restore")
        contiguous = case["mode"].endswith("runs")
        if index < 2:
            result["submissions"] = trace.verify(chain(stderr.splitlines(), stdout.splitlines()), layouts, contiguous)
            result["providers"] = fixture_providers(stderr, speed, env)
        else:
            require("<--- urEnqueue" not in stderr, "API tracing remained active during timing")
            result.update(trace.timing(stdout, contiguous))
        if index == len(ORDER) - 1:
            capacity.basis.comparison.shared.models.model_pins(speed["configuration"], speed["model_assets"], full=True)
        result.update(validation=proof, native_exit_code=0, host_probe_accepted=True)
    except Exception as error:
        result["error"] = repr(error)
    finally:
        validation.finish(result, owner, cursor, speed["boot"])
        result["passed"] &= result.get("driver_after") == native["driver"]
        row["processes"].append(result)
        row.update(stage="measuring" if result["passed"] else "failed", ownership_closed=result["ownership_closed"])
        if result["passed"] and index == len(ORDER) - 1:
            ratios = []
            for a, b in zip(row["processes"][2::2], row["processes"][3::2], strict=True):
                t = {p["mode"]: p["subsequent_median_ms"] for p in (a, b)}
                ratios.append(t["runs"] / t["rows"])
            row.update(stage="closed", passed=True, complete=True,
                       decision=dict(runs_over_rows=ratios, model_eligible=all(r < 1 for r in ratios)))
        save(row)
    require(result["passed"], result.get("error", "GPU, driver or process ownership changed"))


if __name__ == "__main__":
    require(sys.argv[1:] in (["prepare"], ["compile"], ["run"]), "prepare|compile|run")
    with (build.B / "owned-v0141-measurement.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build.builder.guards.no_model_owners([])
        if sys.argv[1] == "prepare":
            prepare()
        elif sys.argv[1] == "compile":
            compile_fixture()
        else:
            run_trial()
