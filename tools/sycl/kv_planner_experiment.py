#!/usr/bin/env python3
"""Main-owned qualification of the cooperative KV planner; repository LICENSE.

Build only the five standalone fixture/runtime/kernel translation units. This
does not replace an engine or select a planner in production inference.
"""
import ast
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
import mapped_rows_experiment as common
import decode_window_bench as retained
import runtime_probe

P, B, RESULT = common.P, common.B, common.RESULT
ROOT, KEY = B / 'kv-planner-validation-v1', 'kv_planner_validation_v1'
TARGET = 'sycl_kv_copy_groups_bench'
TUS = {'tools/sycl/kv_copy_groups_bench.cpp', 'src/kernels/sycl/kv_stream.cpp',
       'src/sycl/runtime.cpp', 'src/sycl/compat.cpp', 'src/sycl/command_profile.cpp'}
require, sha = common.require, common.sha


def context():
    spec = importlib.util.spec_from_file_location('kv_planner_builder', common.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.TARGET, ctx.TUS = ROOT, KEY, TARGET, TUS
    original_check = ctx.check
    def check(row, *, committed=True):
        original_check(row, committed=committed)
        _, proof = retained.retained_native()
        require(proof == row['foundation_proof'], 'retained coherent foundation changed')
    ctx.check = check
    return ctx


def prepare():
    ctx = context()
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'fresh KV validation only')
    native, proof = retained.retained_native()
    speed = json.loads((retained.previous.prefill.ROOT / 'record.json').read_text())
    original = retained.reuse.ROOT / 'record.json'
    recipe = copy.deepcopy(native['recipe'])
    recipe['configure'] = [x.replace(str(original.parent), str(ROOT)) for x in recipe['configure']]
    recipe['configure'][recipe['configure'].index('-S') + 1] = str(P)
    recipe['environment'] = {k: v.replace(str(original.parent), str(ROOT))
        for k, v in recipe['environment'].items()}
    recipe['targets'] = recipe['build_targets'] = [TARGET]
    require(recipe['configure'][recipe['configure'].index('-B') + 1] == str(ROOT / 'build'),
        'new standalone output root required')
    probe, reuse = ctx.probe.current_probe()
    tests = ast.parse((P / 'tools/sycl/test_compile_census.py').read_text())
    count = sum(isinstance(n, ast.FunctionDef) and n.name.startswith('test_') for n in ast.walk(tests))
    require(count == 28, 'review the changed compiler-census suite before reuse')
    row = dict(stage='prepared', project=ctx.source.pin(P, ctx.PREFIXES),
        dependencies=copy.deepcopy(native['dependencies']), toolchain=native['toolchain'] |
            {str(Path(sys.executable).resolve()): sha(sys.executable)},
        external_inputs=native['external_inputs'], fallbacks=native['binaries'], recipe=recipe,
        recipe_origin=dict(path=str(original), sha256=sha(original), source_commit=retained.reuse.SOURCE),
        foundation_proof=proof, runtime_manifest=speed['runtime_manifest'],
        runtime_files=json.loads(Path(speed['runtime_manifest']['path']).read_text()),
        native_validation_probe=probe, runtime_reuse=reuse,
        driver_library=speed['environment_template']['ZE_ENABLE_ALT_DRIVERS'],
        boot=speed['boot'], driver=speed['driver'], cases=[], commands=[],
        active=False, ownership_closed=True, complete=False, passed=False,
        GPU_executed=False, model_executed=False,
        host_command=[sys.executable, '-B', '-m', 'unittest', '-v', 'tools.sycl.test_compile_census'],
        host_tests=28,
        budget=dict(wall=900, cpu=2400, rss_cap=8 << 30, text_cap=16 << 20,
            file_cap=256 << 20, total_cap=1 << 30),
        plan=dict(question='Can cooperative selection reads/hit marks preserve the exact ordered KV policy?',
            origin='Main-owned scalar SYCL policy and independent host oracle;Niko MIT parallel-hit design and Khronos synchronization contract. No XeStrata input.',
            evidence='Rejected-overhead resource profile observes serial resolver service161.75/162.20ms over80calls. One dependent GPU work-item is a mechanism hypothesis,not a physical-saturation or clean-speed conclusion.',
            build='Five fresh TUs only,no engine or native-object reuse. Preserve coherent14508and all12ELFs. Freeze original source,recipe,SDK,controller,generated inputs and actual executable.',
            order='Commit plan;28census controls and configure/compile;commit build;traced small --planner-smoke;commit;quiet --planner-validate. Stop first failure,no retries or changed pins.',
            smoke='INT8,1043blocks/1031slots,page4/head1/dim64:14semantics,refusals/API/null-stream checks and20dynamic graph replays. Same immutable pre-exec runtime safety;pre-main validation and provider/exit/fault checks.',
            full='Three KVformats,795direct oracle records,60graph replays and15physical-tail graph replays. Full map/control/miss/payload/source/input guards;empty,duplicates,overflow,epoch/counter wrap,invalid gaps across256,zero/invalid widths and last cell262143. Component capacity is not full-context model qualification.',
            acceptance='Exact committed source/build/compiler/link census,zero uncovered inputs,12unchanged retained ELFs,current boot/driver,normal0,closed owner,quiet validation,no fresh GPU fault and exact CPU oracle. No performance gate or model adoption follows from correctness.',
            timing='No speed claim in this plan. After correctness,commit a separate paired graph-time plan including resets,observer limits and representative2/4-query shapes before measuring gain.'),
        retention='One named standalone component build for immediate planner correctness/performance decision;no third engine build or overwrite. Review at full validation closure. Legacy13-ELF consumer migration and ended mapped-row fixture/diagnostic-engine retirement remain explicit holds;reconcile before another engine generation.')
    ctx.check(row, committed=False)
    ROOT.mkdir(mode=0o700)
    for leaf in ('tmp', 'cache'): (ROOT / leaf).mkdir(mode=0o700)
    ctx.save(row)
    print('Prepared fresh five-TU KV planner qualification;commit before build.', flush=True)


def run(mode):
    ctx = context(); row = json.loads((ROOT / 'record.json').read_text())
    root = ROOT / mode
    require(not root.exists(), 'one owned validation attempt')
    ctx.check(row)
    require(row['stage'] == 'built' and (not row['cases'] if mode == 'smoke' else
        len(row['cases']) == 1 and row['cases'][0]['passed']), 'ordered smoke/full qualification only')
    root.mkdir(mode=0o700)
    common.write(root / 'attempt.json', dict(mode=mode))
    for name in ('tmp', 'cache'): (root / name).mkdir()
    env = ctx.environment(row, mode, traced=mode == 'smoke')
    env['STRATA_SYCL_KV_COPY_GROUPS'] = '0'
    cursor = next(s.split(': ', 1)[1] for s in common.subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
        text=True, timeout=15).splitlines() if s.startswith('-- cursor: '))
    outcome, owner = dict(mode=mode, complete=False, passed=False, GPU_executed=False), None
    def interact(io, entry):
        deadline = time.monotonic() + (100 if mode == 'smoke' else 570)
        lines = 0
        while True:
            remaining = deadline - time.monotonic()
            require(remaining > 0, 'planner completion deadline')
            line = io.line(min(remaining, 30)); lines += 1
            require(lines <= 1000, 'planner output contract exceeded')
            if line == 'KV_PLAN_WAIT': break
        maps = runtime_probe.executable_mappings(
            Path('/proc', str(entry['pid']), 'maps').read_text())
        allowed = {str(Path(p).resolve()): h for p, h in row['runtime_files'].items()}
        for value in (row['binary'], *row['native_validation_probe'].values()):
            allowed[str(Path(value['path']).resolve())] = value['sha256']
        required = {str(Path(p).resolve()) for p in (env['UR_ADAPTERS_FORCE_LOAD'],
            row['driver_library'], row['native_validation_probe']['loader']['path'],
            row['native_validation_probe']['library']['path'])}
        require(len(maps) >= 8 and maps <= allowed.keys() and required <= maps
            and all(sha(p) == allowed[p] for p in maps), 'actual fixture providers differ')
        outcome['runtime_maps'] = dict(files=len(maps),
            sha256=common.signature({p: allowed[p] for p in sorted(maps)}), unknown=0, missing=0)
        io.send(b'QUIT\n'); io.proc.stdin.close()
        io.exit(30)
    try:
        owner = common.ownership.Owner(root / 'owned')
        entry, out, err = owner.run('fixture', [row['binary']['path'], '--planner-' + ('smoke' if mode == 'smoke' else 'validate')],
            env, P, interaction=interact, wall=120 if mode == 'smoke' else 600, cpu=1800,
            rss_cap=4 << 30, text_cap=16 << 20, file_cap=32 << 20, total_cap=64 << 20)
        require(common.ownership.closed(entry) and entry['exit_code'] == 0, 'owned planner validation failed')
        text = out.read_text(); stderr = err.read_text()
        require('PASS bounded component scope only' in stderr, 'checked fixture completion missing')
        probe_evidence = common.validation.preload_evidence(stderr, entry['pid'])
        require(Path(probe_evidence['loader']).resolve() ==
            Path(row['native_validation_probe']['loader']['path']).resolve(), 'actual preload loader differs')
        records = [json.loads(s) for s in text.splitlines() if s.startswith('{')]
        counts = {name: sum(r.get('phase') == name for r in records) for name in
            ('kv_stream_resolve_component', 'graph_correctness', 'planner_context_graph')}
        if mode == 'smoke':
            require('planner_smoke=PASS direct_cases=14 graph_replays=20' in text
                and counts['graph_correctness'] == 20, 'complete bounded smoke required')
        else:
            require(counts == dict(kv_stream_resolve_component=795, graph_correctness=60,
                planner_context_graph=15) and all(r['oracle_exact'] for r in records),
                'all planned oracle/direct/graph/context records required')
        log = root / 'validation.log'
        require(not log.exists() or log.stat().st_size == 0, 'parameter validation warning/error')
        require(not re.search(r'UR_RESULT_ERROR_|SYCL exception|^FAIL ', stderr, re.M),
            'unexpected runtime error')
        if mode == 'smoke':
            trace = root / 'ur.trace'; traced = trace.read_text()
            names = dict(urCommandBufferCreateExp=5, urCommandBufferFinalizeExp=5,
                urEnqueueCommandBufferExp=20, urCommandBufferReleaseExp=5)
            observed = {}
            for name, count in names.items():
                begin = len(re.findall(r'^\s*---> ' + name + r'\b', traced, re.M))
                end = len(re.findall(r'^\s*<--- ' + name + r'\(.* -> UR_RESULT_SUCCESS;', traced, re.M))
                require(begin == end == count, 'incomplete graph trace: ' + name)
                observed[name] = count
            require(set(re.findall(r'UR_RESULT_ERROR_[A-Z_]+', traced)) <=
                {'UR_RESULT_ERROR_INVALID_NULL_HANDLE'}, 'unexpected traced runtime error')
            outcome['tracing'] = dict(bytes=trace.stat().st_size, sha256=sha(trace), calls=observed)
        outcome.update(complete=True, host_probe_accepted=True, native_exit_code=0, validation=probe_evidence,
            counts=counts, oracle_exact=True, numerical_only=True,
            physical262144_model_qualified=False, performance_qualified=False)
        ctx.check(row)
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        if owner: common.validation.finish(outcome, owner, cursor, row['boot'])
        outcome['GPU_executed'] = bool(owner and any(c.get('pid') for c in owner.commands))
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete']
            and 'error' not in outcome and outcome.get('driver_after') == row['driver'])
        row['cases'].append(outcome)
        done = mode == 'full' and outcome['passed']
        row.update(stage='closed' if done else 'built' if outcome['passed'] else 'failed',
            passed=bool(done), complete=bool(done), active=bool(owner and owner.active),
            ownership_closed=outcome.get('ownership_closed', False),
            GPU_executed=any(c['GPU_executed'] for c in row['cases']))
        ctx.save(row)
    require(outcome['passed'], outcome.get('error', 'planner validation failed'))
    print(json.dumps(outcome), flush=True)


if __name__ == '__main__':
    require(sys.executable == retained.PYTHON, 'Use the qualified model/controller interpreter')
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build', 'smoke', 'full'),
        'prepare|build|smoke|full')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); common.guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare()
        elif sys.argv[1] == 'build': context().compile_fixture()
        else: run(sys.argv[1])
