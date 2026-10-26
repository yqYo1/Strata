#!/usr/bin/env python3
"""Main-owned official-SDK matrix-only ABI gate; repository LICENSE.

No pipeline, repeated-batch timing, model, decode or environment mutation.
"""
import copy
from collections import Counter
import fcntl
import json
from pathlib import Path
import re
import select
import subprocess
import sys
sys.dont_write_bytecode = True
import qualify_native16_control as q

r, P, B, RESULT = q.r, q.P, q.B, q.RESULT
ROOT = B / 'legacy-gemm-abi-v1'
NATIVE = B / 'native16-legacy-gemm-v1'
BUILD_KEY = 'legacy_gemm_abi_build_v1'
KEY = 'legacy_gemm_abi_v1'
require, digest, write = q.require, q.digest, q.write


def compact(row):
    return {k: v for k, v in row.items() if k not in
            ('controller_inputs', 'runtime_files', 'native_build', 'environment')}


def save(row):
    raw = RESULT.read_text(); desired = json.loads(raw)
    q.build.scope(desired)[KEY] = compact(row)
    literal = json.dumps(KEY) + ':'
    value = json.dumps(compact(row), indent=2).replace('\n', '\n      ')
    if KEY in q.build.scope(json.loads(raw)):
        require(raw.count(literal) == 1, 'ambiguous matrix-only result')
        start = raw.index(literal) + len(literal)
        while raw[start].isspace(): start += 1
        _, size = json.JSONDecoder().raw_decode(raw[start:])
        raw = raw[:start] + value + raw[start + size:]
    else:
        anchor = '"fused_command_profile": {'
        require(raw.count(anchor) == 1, 'ambiguous result container')
        raw = raw.replace(anchor, anchor + '\n      ' + literal + ' ' + value + ',', 1)
    require(json.loads(raw) == desired, 'matrix update changed unrelated evidence')
    write(ROOT / 'record.json', row); RESULT.write_text(raw)


def prepare():
    require(not ROOT.exists() and KEY not in q.build.scope(r.validation.committed(RESULT)),
            'single-use matrix-only gate; inspect any existing attempt')
    q.current(json.loads((q.ROOT / 'record.json').read_text()))
    built, sdk, proof = q.native(NATIVE, BUILD_KEY)
    rt = r.validation.committed(r.RESULT); r.current(rt)
    revision = r.validation.git('rev-parse', 'HEAD').decode().strip()
    artifact = proof['fixtures']['sycl_gemm_test']
    env = dict(built['environment'],
        STRATA_SYCL_PREFILL_NATIVE16='bf16', STRATA_SYCL_PREFILL_NATIVE16_BACKEND='vllm',
        STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE='auto', STRATA_SYCL_PREFILL_COMMAND_PROFILE='0',
        STRATA_SYCL_PREFILL_FUSED_COMMAND_PROFILE='0',
        ZE_ENABLE_ALT_DRIVERS='/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.17.39758',
        LD_PRELOAD=rt['preload']['library']['path'],
        STRATA_UR_PROBE_EXPECTED_LOADER=str((q.build.OFFICIAL / 'lib/libur_loader.so.0').resolve()),
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:file,' + str(ROOT / 'ur.trace'),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(ROOT / 'validation.log'),
        TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    files = q.runtime_inputs(built, sdk, rt)
    row = dict(stage='prepared', source_commit=revision,
        controller_inputs=q.gen.benchmark_pins(revision), native_build=proof,
        native_reference=dict(source_commit=built['source_commit'], executable=artifact,
            record=proof['record'], compiled_inputs=proof['compiled_inputs']),
        runtime_sha256=digest(r.RESULT), runtime_files=files, environment=env,
        boot=rt['boot'], driver=rt['driver'], args=['--matrix-only'],
        budgets=dict(wall=30, cpu=120, rss_cap=2 << 30, text_cap=64 << 20,
                     file_cap=64 << 20, total_cap=64 << 20),
        complete=False, passed=False, active=False, ownership_closed=True,
        GPU_executed=False, model_executed=False, inference_adopted=False,
        performance_qualified=False, physical262144_qualified=False,
        criteria='One fresh traced native process,22 CPU-parity cases with output-stride guards: '
            '18 FP16/BF16-to-FP32 GEMMs (three shapes and beta0/.5/1) and4 Q8 strided cases. '
            'All pipeline/ring/separate-queue cases skipped. Require normal owned exit,exact source/'
            'ELF/environment/providers,pre-main validation,no warning or fresh GPU fault. Stop first '
            'failure,no retry/reset. Passing qualifies only this ABI/matrix subset,not model or speed.',
        retention='Retain compact22case/provider/fault/owner outcome. After source recovery retire '
            'normal full API trace,JIT,private receipts and candidate build. Official v4fallback '
            'and original13engine assets remain intact;no automatic model advancement.')
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (ROOT / name).mkdir()
    write(ROOT / 'runtime-inputs.json', files)
    row['runtime_manifest'] = dict(path=str(ROOT / 'runtime-inputs.json'),
        sha256=digest(ROOT / 'runtime-inputs.json'), files=len(files))
    save(row); print('Prepared; commit the exact one-process plan before running.', flush=True)


def current(row, historical=False):
    require(q.build.scope(r.validation.committed(RESULT))[KEY] == compact(row), 'commit exact matrix plan')
    require(q.gen.benchmark_pins(row['source_commit'], match_worktree=not historical) ==
            row['controller_inputs'], 'controller changed or original source unrecoverable')
    require(q.native(NATIVE, BUILD_KEY)[2] == row['native_build'], 'original native proof changed')
    require(digest(r.RESULT) == row['runtime_sha256'], 'qualified runtime record changed')
    rt = r.validation.committed(r.RESULT); r.current(rt)
    require(rt['boot'] == row['boot'] and rt['driver'] == row['driver'], 'boot/driver changed')
    require(digest(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256'] and
        json.loads(Path(row['runtime_manifest']['path']).read_text()) == row['runtime_files'] and
        all(digest(p) == sha for p, sha in row['runtime_files'].items()), 'prospective provider changed')


def api_counts(trace):
    # UR entries end after the function name; arguments occur on RETURN lines.
    entries = re.findall(r'^\s*---> (urEnqueue\w+)\s*$', trace, re.M)
    returns = re.findall(r'^\s*<--- (urEnqueue\w+)\(.*\) -> (UR_RESULT_\w+);$', trace, re.M)
    calls = Counter(entries)
    require(calls == Counter(name for name, _ in returns) and
        all(status == 'UR_RESULT_SUCCESS' for _, status in returns), 'unbalanced or failed enqueue trace')
    require(sum(v for k, v in calls.items() if k.startswith('urEnqueueKernelLaunch')) > 0 and
        calls['urEnqueueUSMMemcpy'] > 0, 'GPU launch/copy trace missing')
    return dict(sorted(calls.items()))


def prepare_replay(row):
    require(row['stage'] == 'failed' and not row['passed'] and not row['complete'] and
        row['outcome']['ownership_closed'] and row['outcome']['fresh_faults'] == [] and
        row['outcome'].get('error') == "RuntimeError('GPU launch/copy trace missing')" and
        'evidence_replay' not in row, 'only the original closed parser failure may be replayed')
    current(row, historical=True)
    paths = [ROOT / n for n in ('execution.json', 'runtime-inputs.json', 'ur.trace',
        'owned/matrix-only.stdout', 'owned/matrix-only.stderr')]
    if (ROOT / 'validation.log').exists(): paths.append(ROOT / 'validation.log')
    row['evidence_replay'] = dict(stage='prepared', passed=False, GPU_executed=False,
        source_commit=r.validation.git('rev-parse', 'HEAD').decode().strip(),
        original_failed_commit='d708f9ab3', input_sha256={str(p): digest(p) for p in paths},
        validation_log_present=(ROOT / 'validation.log').exists(),
        criteria='Offline replay only;preserve original failed stage/outcome. Recover original '
            'controller/native/SDK/runtime and exact existing argv/environment/logs/normal owner '
            'closure;verify22case footer,pre-main activation,no parameter warning,and balanced '
            'successful API enqueue entry/returns. No new process or model/performance advancement.')
    save(row); print('Prepared offline replay; commit its exact inputs first.', flush=True)


def replay(row):
    current(row, historical=True)
    plan = row['evidence_replay']
    require(plan['stage'] == 'prepared' and not plan['passed'] and not plan['GPU_executed'],
        'single-use offline replay')
    q.gen.benchmark_pins(plan['source_commit'])
    require(all(digest(p) == sha for p, sha in plan['input_sha256'].items()) and
        (ROOT / 'validation.log').exists() == plan['validation_log_present'], 'frozen replay input changed')
    original = q.build.scope(json.loads(r.validation.git('show',
        plan['original_failed_commit'] + ':' + str(RESULT.relative_to(P)))))[KEY]
    require(compact({k: v for k, v in row.items() if k != 'evidence_replay'}) == original,
        'original failed outcome changed')
    commands = json.loads((ROOT / 'execution.json').read_text())['commands']
    require(len(commands) == 1, 'exactly one original native process required')
    entry = commands[0]
    require(r.supervision.closed(entry) and entry['exit_code'] == 0 and
        entry['argv'] == [row['native_reference']['executable']['path'], *row['args']] and
        entry['environment'] == row['environment'], 'native ownership/argv/environment mismatch')
    for name, info in entry['logs'].items():
        p = ROOT / 'owned' / name
        require(p.stat().st_size == info['bytes'] and digest(p) == info['sha256'], 'original process log changed')
    text = (ROOT / 'owned/matrix-only.stdout').read_text()
    require(text == 'SYCL oneMKL GEMM: 22 CPU parity cases passed\n'
        'MATRIX_SUBSUITE_PASS; pipeline and separate-queue cases UNRUN\n', 'matrix subset differs')
    activation = r.validation.preload_evidence((ROOT / 'owned/matrix-only.stderr').read_text(), entry['pid'])
    require(activation['loader'] == row['environment']['STRATA_UR_PROBE_EXPECTED_LOADER'] and
        (not plan['validation_log_present'] or not (ROOT / 'validation.log').stat().st_size),
        'activation/validation evidence differs')
    counts = api_counts((ROOT / 'ur.trace').read_text())
    plan.update(stage='closed', passed=True, CPU_parity_cases=22, full_API_entry_counts=counts,
        native_exit_code=0, native_elapsed_seconds=entry['elapsed_seconds'], pre_main_validation=activation,
        original_live_environment_provider_checks='Original committed controller passed its live '
            'environment/map/provider requirements before the trace-parser refusal;not reconstructed maps.',
        source_runtime_recovered=True, original_failure_preserved=True,
        model_executed=False, performance_qualified=False, physical262144_qualified=False)
    save(row); print(json.dumps(plan), flush=True)
    return 0


def run(row):
    current(row)
    require(row['stage'] == 'prepared' and not (ROOT / 'owned').exists(), 'single-use native gate')
    initial = copy.deepcopy(row); owner = None; outcome = dict(host_probe_accepted=False)
    seen = set(); env_seen = False
    cursor = next(v.split(': ', 1)[1] for v in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True,
        timeout=15).splitlines() if v.startswith('-- cursor: '))
    def persist():
        write(ROOT / 'execution.json', dict(commands=owner.commands if owner else [],
            active=bool(owner and owner.active)))
    try:
        owner = r.supervision.Owner(ROOT / 'owned', persist)
        artifact = row['native_reference']['executable']; env = row['environment']
        allowed = dict(row['runtime_files']); allowed[artifact['path']] = artifact['sha256']
        def monitor(io, entry):
            nonlocal env_seen
            while not io.exit_watch.poll(0) or not io.eof:
                io.poll()
                try:
                    mapped = r.executable_mappings(Path('/proc', str(io.proc.pid), 'maps').read_text())
                    actual = dict(v.split(b'=', 1) for v in Path('/proc', str(io.proc.pid),
                        'environ').read_bytes().split(b'\0') if b'=' in v)
                    require(all(actual.get(k.encode()) == v.encode() for k, v in env.items()) and
                        set(actual) - {k.encode() for k in env} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'},
                        'actual native environment differs')
                    env_seen = True
                except (FileNotFoundError, ProcessLookupError): mapped = set()
                except PermissionError:
                    require(io.exit_watch.poll(20), 'live native /proc denied'); mapped = set()
                require(mapped <= set(allowed), 'unplanned executable map: ' + repr(mapped - set(allowed)))
                seen.update(mapped)
                require(r.supervision.artifact_bytes(ROOT) <= 64 << 20, 'whole diagnostic byte budget')
                if select.select([io.proc.stdout], [], [], .01)[0]: io.read(); io.pending.clear()
            io.exit(1)
        entry, out, err = owner.run('matrix-only', [artifact['path'], *row['args']], env, P,
            interaction=monitor, **row['budgets'])
        require(r.supervision.closed(entry) and entry['exit_code'] == 0, 'native matrix process failed')
        activation = r.validation.preload_evidence(err.read_text(), entry['pid'])
        require(env_seen and activation['loader'] == env['STRATA_UR_PROBE_EXPECTED_LOADER'] and
            activation['loader'] in seen and str(q.build.OFFICIAL / 'lib/libsycl.so.9.0.0') in seen and
            {p for p in seen if '/libze_intel_gpu.so.' in p} == {env['ZE_ENABLE_ALT_DRIVERS']} and
            any('/libmkl_sycl_blas.so.' in p for p in seen), 'actual provider evidence missing')
        require(not (ROOT / 'validation.log').exists() or not (ROOT / 'validation.log').stat().st_size,
                'parameter validation warning')
        text = out.read_text()
        require(text.count('SYCL oneMKL GEMM: 22 CPU parity cases passed') == 1 and
            text.count('MATRIX_SUBSUITE_PASS; pipeline and separate-queue cases UNRUN') == 1,
            'matrix checks or subset declaration missing')
        trace = (ROOT / 'ur.trace').read_text()
        calls = api_counts(trace)
        row['details'] = dict(CPU_parity_cases=22, stride_guards=True, pipeline_executed=False,
            full_API_entry_counts=calls, pre_main_validation=activation,
            observed_providers={p: row['runtime_files'][p] for p in sorted(seen & set(row['runtime_files']))})
        current(initial); outcome['host_probe_accepted'] = True
    except BaseException as error: outcome['error'] = repr(error)
    finally:
        if owner: r.validation.finish(outcome, owner, cursor, row['boot'])
        row.update(stage='closed' if outcome.get('passed') else 'failed',
            complete=bool(outcome.get('passed')), passed=bool(outcome.get('passed')), active=False,
            ownership_closed=outcome.get('ownership_closed', True), outcome=outcome,
            GPU_executed=bool(owner and owner.commands))
        persist(); save(row)
    print(json.dumps({k: row.get(k) for k in ('stage', 'complete', 'passed', 'ownership_closed', 'outcome')}), flush=True)
    return int(not row['passed'])


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); r.guards.no_model_owners([])
        if sys.argv[1:] == ['prepare']: prepare(); return 0
        if sys.argv[1:] == ['run']: return run(json.loads((ROOT / 'record.json').read_text()))
        if sys.argv[1:] == ['prepare-replay']: prepare_replay(json.loads((ROOT / 'record.json').read_text())); return 0
        if sys.argv[1:] == ['replay']: return replay(json.loads((ROOT / 'record.json').read_text()))
        raise RuntimeError('use prepare/run or prepare-replay/replay')


if __name__ == '__main__': sys.exit(main())
