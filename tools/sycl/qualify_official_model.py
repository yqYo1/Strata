#!/usr/bin/env python3
"""Bounded same-build component progression before official model execution.

Main-authored, repository LICENSE. Reuses maintained provider/owner checks;
never runs the old 2048-call GU batch or selects inference from fixture success.
"""
import copy
import fcntl
import json
from pathlib import Path
import re
import select
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import build_official_model as model
import matrix_shapes as inputs
import qualify_native16_control as prior
import runtime_probe as runtime

P, B, RESULT = model.P, model.B, model.RESULT
ROOT = model.ROOT / 'component-checks-v1'
KEY = 'official_prefill_device_v1'
require, digest, write = model.require, model.digest, model.write


def compact(row):
    return {k: v for k, v in row.items() if k not in ('inputs', 'runtime_files')}


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text())
    data[KEY] = compact(row)
    write(RESULT, data)


def native():
    row = json.loads((model.ROOT / 'record.json').read_text())
    require(row['stage'] == 'built-not-executed' and row['passed'] and row['ownership_closed'] and
            not row['active'] and all(model.supervision.closed(v) for v in row['commands_executed']),
            'closed official build required')
    model.admit(row)
    proof = row['compiled_inputs']
    require(proof['uncovered'] == 0 and digest(proof['path']) == proof['sha256'], 'compiler inventory changed')
    used = json.loads(Path(proof['path']).read_text())
    require(len(used) == proof['files'] and all(digest(p) == sha for p, sha in used.items()), 'compiled input changed')
    require(digest(model.ROOT / 'build/compile_commands.json') == row['compile_database_sha256'], 'compile recipe changed')
    for item in row['binaries'].values():
        require(digest(item['path']) == item['sha256'], 'same-build executable changed')
    return row


def cases(loader):
    topk = 'sycl_qsa_topk_prefill_entry_test'
    return [dict(name='host-canary', target=None, args=[], trace=True),
        dict(name='Topk-properties', target=topk, args=['--properties', loader], trace=True),
        *[dict(name='Topk-smoke' + str(arm), target=topk, args=['--smoke'], topk=arm, trace=True) for arm in (0, 1)],
        dict(name='submission', target='sycl_submission_test', args=['--submission'], trace=True),
        dict(name='native16-smoke', target='sycl_vllm_native16_test', args=['--device', '0', '--suite', 'smoke'], trace=True),
        dict(name='native16-guards', target='sycl_vllm_native16_test', args=['--device', '0', '--suite', 'guards'], trace=True),
        dict(name='GU-progress', target='sycl_moe_gu_swiglu_bench', args=['--progress-public', '0'], gu_arm=0, trace=True),
        *[dict(name='GU-public' + str(arm), target='sycl_moe_gu_swiglu_bench', args=['--smoke-public', str(arm)],
               gu_arm=arm, trace=True) for arm in (0, 1)],
        dict(name='Down-split-original', target='sycl_moe_down_scatter_bench', args=['--split-route-smoke', '1'], trace=True),
        *[dict(name='Topk-numerics' + str(arm), target=topk, args=['--numerics'], topk=arm, trace=False) for arm in (0, 1)]]


def prepare():
    require(KEY not in model.committed(RESULT) and not ROOT.exists(), 'fresh single-use component cohort only')
    built = native()
    rt = runtime.validation.committed(runtime.RESULT)
    runtime.current(rt)
    loader = str(Path('/opt/intel/oneapi/compiler/2026.2/lib/libur_loader.so.0').resolve())
    sdk = model.sdk()
    files = prior.runtime_inputs(dict(built, fixtures=built['binaries']), sdk, rt)
    require(sdk.get(loader) == digest(loader), 'original official loader snapshot changed')
    files[loader] = sdk[loader]
    env = dict(built['environment'], LD_PRELOAD=rt['preload']['library']['path'],
        STRATA_UR_PROBE_EXPECTED_LOADER=loader,
        ZE_ENABLE_ALT_DRIVERS=next(p for p in files if 'libze_intel_gpu.so.' in p),
        STRATA_SYCL_PREFILL_NATIVE16='bf16', STRATA_SYCL_PREFILL_NATIVE16_BACKEND='vllm',
        STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE='auto', STRATA_SYCL_SELECT_RADIX4='1',
        STRATA_SYCL_SELECT_CYCLIC='0', STRATA_SYCL_PREFILL_COMMAND_PROFILE='0',
        STRATA_SYCL_PREFILL_FUSED_COMMAND_PROFILE='0', STRATA_SYCL_PREFILL_DOWN_SPLIT_KERNEL='original')
    revision = runtime.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    pins = inputs.sources(revision, Path(__file__).resolve())
    ROOT.mkdir(mode=0o700)
    write(ROOT / 'runtime-inputs.json', files)
    row = dict(stage='prepared', source_commit=revision, inputs=pins,
        native_source_commit=built['source_commit'], native_build_commit='fcad66e1a',
        binaries=built['binaries'], canary=rt['preload']['canary'], environment=env,
        runtime_files=files, runtime_manifest=dict(path=str(ROOT / 'runtime-inputs.json'),
            sha256=digest(ROOT / 'runtime-inputs.json'), files=len(files)),
        runtime_basis_sha256=digest(runtime.RESULT), boot=built['boot'], driver=built['driver'],
        cases=cases(loader), processes=[], active=False, ownership_closed=True, complete=False, passed=False,
        GPU_executed=False, model_executed=False, inference_adopted=False,
        ordinary_runtime_teardown_qualified=False, physical262144_model_qualified=False,
        budgets=dict(wall=180, cpu=600, rss_cap=4 << 30, text_cap=8 << 20, file_cap=8 << 20, total_cap=64 << 20),
        whole_bytes=64 << 20, whole_wall_seconds=900,
        criteria='13ordered fresh owned processes,stop first failure. Exact source/ELF/providers and immutable '
            'environment,pre-main parameter validation,quiet warnings,no new fault. Traced tiny Top-k off/on, '
            '32dyadic matrix cases,26admission guards,64ordinary public GU calls in16call groups,public GU '
            'threshold cases and original split-Down route;then145Top-k numerical cases per off/on process '
            'with warning-only logging. Never2048call batch. Same-build component checks only;MoE8job '
            'production-loop progress,model speed,decode/handoff,task and physical full context remain separate.',
        retention='Retain compact individual outcomes/case counts and source/native identity. Successful full '
            'API traces,JIT and polling output retire after verification. Seven candidate binaries remain '
            'only for imminent production/model comparison;selected6e40+12fixtures remain untouched.')
    save(row)
    print('Prepared13same-build bounded processes;commit plan before execution.', flush=True)


def current(row):
    require(model.committed(RESULT)[KEY] == compact(row), 'commit exact component cohort plan')
    built = native()
    require(built['binaries'] == row['binaries'] and built['source_commit'] == row['native_source_commit'],
            'official native generation changed')
    require(inputs.sources(row['source_commit'], Path(__file__).resolve()) == row['inputs'], 'component controller changed')
    require(row['cases'] == cases(row['environment']['STRATA_UR_PROBE_EXPECTED_LOADER']), 'component case plan changed')
    require(digest(runtime.RESULT) == row['runtime_basis_sha256'], 'runtime proof changed')
    require(digest(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256'] and
            json.loads(Path(row['runtime_manifest']['path']).read_text()) == row['runtime_files'] and
            all(digest(p) == sha for p, sha in row['runtime_files'].items()), 'runtime inputs changed')
    require(model.guards.driver_guard() == row['driver'], 'driver/fault state changed')


def analyze(case, stdout, pid):
    name = case['name']
    if name.startswith('Topk-'):
        require('DONE public_entry=1 model_adopted=0 decode_changed=0' in stdout and
                'DEVICE name=Intel' in stdout and 'B570' in stdout and 'FAIL' not in stdout, 'Top-k completion failed')
        if name == 'Topk-properties':
            require(stdout.count('PROPERTY ') == 1, 'compiled property census missing')
            return dict(properties=True)
        count = sum(s.startswith('CASE ') for s in stdout.splitlines())
        require(count == (145 if '--numerics' in case['args'] else 1) and
                ('ROUTE early=' + str(case['topk'])) in stdout, 'Top-k route/numeric census differs')
        if '--numerics' in case['args']:
            require('ADMISSION refusals=60 apis=3 output=PASS' in stdout, 'Top-k admission controls missing')
        return dict(numeric_cases=count, early=case['topk'], input_output_guards=True)
    if name == 'GU-progress':
        rows = [json.loads(s) for s in stdout.splitlines() if s.startswith('{')]
        result = [v for v in rows if v.get('kind') == 'public_progress_result']
        require(len(result) == 1, 'public GU progress result missing')
        v = result[0]
        require(v['status'] == 'PASS' and v['calls'] == 64 and v['fallback'] == 64 and
                v['groups'] == 4 and v['calls_per_group'] == 16 and v['max_pending_ordinary_kernels'] == 32 and
                v['max_half_ulp'] == 0 and v['guards'] and v['inputs_immutable'], 'public GU bounded progress failed')
        stages = [v for v in rows if v.get('kind') == 'gu_stage']
        enters = [(v['sequence'], v['stage']) for v in stages if v['phase'] == 'enter']
        returns = [(v['sequence'], v['stage']) for v in stages if v['phase'] == 'return']
        require(len(enters) == 308 and enters == returns, 'public GU stage entry/return census differs')
        return dict(calls=64, groups=4, stage_pairs=len(enters), maximum_pending_ordinary_kernels=32,
                    max_half_ulp=0, guards=True, inputs_immutable=True)
    if name == 'Down-split-original':
        values = [json.loads(s) for s in stdout.splitlines() if s.startswith('{')]
        require('fixture_status=PASS' in stdout and len(values) == 8 and
                [v['rows'] for v in values] == [2333,2333,2334,2334,2335,2335,8192,8192] and
                all(v['kind'] == 'selected_route' and v['status'] == 'PASS' and v['split_rows'] > 0 and
                    bool(v['selected']) == (v['rows'] >= 2334) and v['inputs_immutable'] and
                    v['false_route_no_write'] and v['expected_mask'] == 0 for v in values),
                'split Down route/numeric census differs')
        return dict(cases=8, original_split_kernel=True, inputs_immutable=True)
    return prior.analyze(case, stdout, pid)


def run():
    row = json.loads((ROOT / 'record.json').read_text())
    current(row)
    require(row['stage'] == 'prepared' and not (ROOT / 'owned').exists(), 'fresh ordered cohort only')
    initial, owner, seen = copy.deepcopy(row), None, set()
    cursor = next(v.split(': ', 1)[1] for v in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if v.startswith('-- cursor: '))
    deadline = time.monotonic() + row['whole_wall_seconds']
    def persist():
        write(ROOT / 'execution.json', dict(processes=row['processes'], active=bool(owner and owner.active),
                                          commands=owner.commands if owner else []))
    outcome = dict(host_probe_accepted=False)
    try:
        owner = model.supervision.Owner(ROOT / 'owned', persist)
        for case in row['cases']:
            folder = ROOT / case['name']
            folder.mkdir()
            for sub in ('tmp', 'cache'):
                (folder / sub).mkdir()
            env = dict(row['environment'], TMPDIR=str(folder / 'tmp'), XDG_CACHE_HOME=str(folder / 'cache'),
                STRATA_SYCL_PREFILL_GU_SWIGLU='large' if case.get('gu_arm', 1) else '0',
                STRATA_SYCL_PREFILL_DOWN_SCATTER='large', STRATA_TEST_MAIN_GATES='1',
                STRATA_SYCL_PREFILL_TOPK_EARLY=str(case.get('topk', 0)),
                UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if case['trace'] else 'UR_LAYER_PARAMETER_VALIDATION',
                UR_LOG_TRACING='level:info;flush:info;output:file,' + str(folder / 'ur.trace'),
                UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(folder / 'validation.log'))
            artifact = row['binaries'][case['target']] if case['target'] else row['canary']
            allowed = dict(row['runtime_files'])
            allowed[str(Path(artifact['path']).resolve())] = artifact['sha256']
            mapped_case, gates, env_seen = set(), [], False
            def observe(pid):
                nonlocal env_seen
                mapped = runtime.executable_mappings(Path('/proc', str(pid), 'maps').read_text())
                actual = dict(v.split(b'=', 1) for v in Path('/proc', str(pid), 'environ').read_bytes().split(b'\0') if b'=' in v)
                require(all(actual.get(k.encode()) == v.encode() for k, v in env.items()) and
                        set(actual) - {k.encode() for k in env} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'},
                        'actual native environment differs')
                require(mapped <= set(allowed), 'unplanned executable map: ' + repr(mapped - set(allowed)))
                env_seen = True
                mapped_case.update(mapped)
            def monitor(io, entry):
                next_size = 0
                while not io.exit_watch.poll(0) or not io.eof:
                    io.poll()
                    require(time.monotonic() < deadline, 'whole cohort wall budget')
                    try:
                        observe(io.proc.pid)
                    except (FileNotFoundError, ProcessLookupError):
                        pass
                    except PermissionError:
                        require(io.exit_watch.poll(20), 'live native /proc access denied')
                    if time.monotonic() >= next_size:
                        require(model.supervision.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole cohort byte budget')
                        next_size = time.monotonic() + .25
                    if select.select([io.proc.stdout], [], [], .01)[0]:
                        io.read()
                        while b'\n' in io.pending:
                            line, _, rest = io.pending.partition(b'\n')
                            io.pending[:] = rest
                            text = line.decode()
                            if text.startswith('GATE stage=') or text.startswith('SPLIT_READY '):
                                stage = text.split('=', 1)[1] if text.startswith('GATE ') else text.split(' ', 1)[1]
                                require(stage == ['before-runtime', 'runtime-ready', 'before-exit'][len(gates)], 'unexpected native gate')
                                observe(io.proc.pid)
                                gates.append(stage)
                                io.send(b'continue\n' if text.startswith('GATE ') else b'X')
                io.pending.clear()
                io.exit(1)
            entry, out, err = owner.run(case['name'], [artifact['path'], *case['args']], env, P,
                interaction=monitor, **row['budgets'])
            require(model.supervision.closed(entry) and entry['exit_code'] == 0, 'native process failed: ' + case['name'])
            activation = runtime.validation.preload_evidence(err.read_text(), entry['pid'])
            require(activation['loader'] == env['STRATA_UR_PROBE_EXPECTED_LOADER'], 'wrong actual validation loader')
            require(not (folder / 'validation.log').exists() or (folder / 'validation.log').stat().st_size == 0,
                    'parameter-validation warning')
            details = analyze(case, out.read_text(), entry['pid'])
            if case['target']:
                require(env_seen and env['STRATA_UR_PROBE_EXPECTED_LOADER'] in mapped_case and
                        env['ZE_ENABLE_ALT_DRIVERS'] in mapped_case and
                        '/opt/intel/oneapi/compiler/2026.2/lib/libsycl.so.9.0.0' in mapped_case,
                        'actual native provider observation incomplete')
                if case['trace'] and case['name'] != 'Topk-properties':
                    require('urEnqueue' in (folder / 'ur.trace').read_text(), 'GPU submission trace missing')
            if case['target'] in ('sycl_qsa_topk_prefill_entry_test', 'sycl_moe_down_scatter_bench'):
                require(gates == ['before-runtime', 'runtime-ready', 'before-exit'], 'native gates incomplete')
            runtime.validation.finish(check := dict(host_probe_accepted=True), owner, cursor, row['boot'])
            require(check['passed'] and check['driver_after'] == row['driver'], 'fault/ownership gate failed')
            row['processes'].append(dict(name=case['name'], passed=True, ownership_closed=True,
                pre_main_parameter_validation=True, gates=len(gates), details=details))
            seen.update(mapped_case)
            persist()
            print('Passed ' + case['name'], flush=True)
        current(initial)
        outcome['host_probe_accepted'] = True
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        if owner:
            runtime.validation.finish(outcome, owner, cursor, row['boot'])
        row.update(stage='closed' if outcome.get('passed') and len(row['processes']) == len(row['cases']) else 'failed',
            passed=bool(outcome.get('passed') and len(row['processes']) == len(row['cases'])),
            complete=len(row['processes']) == len(row['cases']), active=bool(owner and owner.active),
            ownership_closed=bool(owner and owner.active is None and all(model.supervision.closed(v) for v in owner.commands)),
            GPU_executed=bool(owner and len(owner.commands) > 1), fresh_faults=outcome.get('fresh_faults'),
            observed_provider_count=len(seen), error=outcome.get('error'))
        save(row)
    require(row['passed'], row.get('error') or 'component progression failed')
    print('13same-build checks passed;production MoE/model checks remain pending.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'run'), 'use prepare|run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model.guards.no_model_owners([])
        (prepare if sys.argv[1] == 'prepare' else run)()
