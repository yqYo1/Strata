#!/usr/bin/env python3
"""Bounded official-bench model gate for the qualified NEO39758 runtime.

Repository LICENSE; independently authored. No rebuild or inference selection.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import select
import shutil
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import full_context_model as model
import generation_engine as gen
import runtime_probe as runtime

guards, validation, supervision = runtime.guards, runtime.validation, runtime.supervision
P, B = runtime.P, runtime.B
ROOT = B / 'neo-39758-model-v2'
RESULT = runtime.RESULT.with_name('runtime-model-current.json')
write, require, digest = runtime.write, guards.require, runtime.digest


def fingerprint(pins):
    return hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()


def references():
    qualification = validation.committed(runtime.RESULT)
    require(qualification['passed'] and qualification['complete'] and
        qualification['stage'] == 'device-checked' and qualification['ownership_closed'] and
        qualification['fresh_faults'] == [], 'closed updated-runtime device gate required')
    native = runtime.current(qualification)
    base = model.setup()
    speed = validation.committed(RESULT.with_name('phase-prefill-speed-current.json'))
    require(speed['selection']['selected_source'] == native['source_commit'] and
        not speed['selection']['candidate_adopted'], 'preserve selected prefill/decode package')
    preload = qualification['preload']
    probe = dict(library=preload['library'], loader=dict(path=preload['loader'],
        sha256=qualification['runtime_files'][preload['loader']]),
        retained_reference_until_exit=True, runtime_teardown_qualified=False)
    return qualification, native, base, speed, probe


def current(row):
    require(validation.committed(RESULT) == row, 'commit exact model gate first')
    qualification, native, base, speed, probe = references()
    require(native['binary'] == row['binary'] and base['reference'] == row['original_configuration'] and
        probe == row['native_validation_probe'] and digest(runtime.RESULT) == row['runtime_reference']['sha256'],
        'model/build/runtime qualification changed')
    pins = gen.benchmark_pins(row['source_commit'])
    require(len(pins) == row['benchmark_files'] and fingerprint(pins) == row['benchmark_fingerprint'],
        'benchmark/controller changed')
    for name, sha in row['generated_inputs'].items():
        require(digest(name) == sha, 'generated model input changed: ' + name)
    model.model_pins(base['config'], base['models'])
    return qualification, base


def prepare():
    require(not ROOT.exists(), 'fresh model gate only; never retry')
    prior = validation.committed(RESULT)
    require(prior['stage'] == 'preparation-failed' and not prior['GPU_executed'] and
        not prior['native_executed'] and prior['ownership_closed'] and not prior['active'],
        'preserve closed pre-native preparation refusal')
    prior_reference = dict(commit=gen.git('log', '-1', '--format=%H', '--',
        str(RESULT.relative_to(P))).decode().strip(), path=str(RESULT.relative_to(P)), sha256=digest(RESULT))
    qualification, native, base, speed, probe = references()
    revision = gen.git('rev-parse', 'HEAD').decode().strip()
    pins = gen.benchmark_pins(revision)
    model.model_pins(base['config'], base['models'])
    config = copy.deepcopy(base['config'])
    config['args'][config['args'].index('--expert-profile') + 1] = str(ROOT / 'expert-profile.bin')
    config.update(log=str(ROOT / 'engine.stderr'), env={})
    env = dict(speed['environment_template'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'),
        STRATA_MEASUREMENT_ROOT=str(ROOT), STRATA_TRACE='1',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(ROOT / 'validation.log'))
    require('LD_PRELOAD' not in env and 'UR_LOG_TRACING' not in env and
        env['UR_ENABLE_LAYERS'] == 'UR_LAYER_PARAMETER_VALIDATION', 'bounded native validation only')
    for key in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD', 'NEOReadDebugKeys', 'EnableDirectSubmission',
                'STRATA_SYCL_PREFILL_SAME_QUEUE', 'SYCL_CACHE_PERSISTENT', 'ZES_ENABLE_SYSMAN'):
        require(env[key] == qualification['environment'][key], 'qualified safety changed: ' + key)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT / name).mkdir()
    shutil.copyfile(P / 'data/expert-profile.bin', ROOT / 'expert-profile.bin')
    write(ROOT / 'config.json', config)
    write(ROOT / 'plan.json', dict(binary=native['binary'], native_validation_probe=probe))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    row = dict(stage='prepared', source_commit=revision, prior_preparation_failure=prior_reference,
        benchmark_files=len(pins),
        benchmark_fingerprint=fingerprint(pins), binary=native['binary'],
        original_configuration=base['reference'], native_validation_probe=probe,
        runtime_reference=dict(path=str(runtime.RESULT.relative_to(P)), sha256=digest(runtime.RESULT),
            source_commit=qualification['source_commit']), config=config, environment=env,
        generated_inputs={str(ROOT / name): digest(ROOT / name)
            for name in ('config.json', 'plan.json', 'expert-profile.bin')},
        command=['/bin/sh', str(P / 'bench.sh'), '--config', str(ROOT / 'config.json'), '--port', str(port),
            '--prefill-only', '--prefill-tokens', '32768', '--think', '--runs', '1',
            '--profile-request-gates', '--workload-id', 'neo39758_model_20261025',
            '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(ROOT / 'bench')],
        boot=qualification['boot'], driver=qualification['driver'],
        budgets=dict(wall=900, cpu=10800, rss_cap=96 << 30, text_cap=32 << 20,
            file_cap=32 << 20, total_cap=64 << 20),
        criteria='One fresh official bench process: existing short chat warmup,then one fresh>=32768-input thinking-on request with1..24 output tokens. Exact selected6e40 binary/config/math/decode placement;pre-main validation,live native owner/environment/new NEO mapping,complete request journal and native footer conservation,actual BF16 GPU residual/mode10 long path,normal owned benchmark/native exit,no validation errors/fresh GPU faults. Stop first failure,no retries.',
        limits='Execution/state-handoff smoke gate only. No numerical/task quality,decode nonregression,sustained TG,clean speed,physical262144,RESTORE or ordinary runtime destruction qualification;no old/new runtime sample pooling.',
        retention='Retain compact source/command/config/environment,individual request extents/times/hash/outcomes and limits. After verification retire successful logs,trace,caches and generated duplicate inputs with a deletion manifest. No new engine generation.',
        active=False, ownership_closed=True, passed=False, complete=False, adoption_qualified=False,
        samples=[], clean_speed_qualified=False, ordinary_teardown_qualified=False)
    write(ROOT / 'record.json', row)
    write(RESULT, row)
    print('Prepared bounded updated-runtime model gate; commit before execution.', flush=True)


def run(row):
    qualification, base = current(row)
    require(row['stage'] == 'prepared' and not (ROOT / 'owned').exists(), 'single-use model gate')
    initial = copy.deepcopy(row)
    owner = None
    cursor_text = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'],
        text=True, timeout=15)
    cursor = next(line.split(': ', 1)[1] for line in cursor_text.splitlines() if line.startswith('-- cursor: '))

    def persist():
        write(ROOT / 'record.json', dict(row, active=bool(owner and owner.active is not None),
            commands=owner.commands if owner else []))

    def interact(io, entry):
        for stage in ('begin', 'end'):
            while True:
                line = io.line(720)
                if line.startswith('STRATA_PROFILE_GATE '):
                    break
            require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                dict(stage=stage, phase='prefill-only', run=0), 'official request gate differs')
            claim = json.loads((ROOT / 'engine-start.json').read_text())
            native = claim['native']
            actual = supervision.identity(native['pid'])
            require(actual and actual['sid'] == entry['pid'] and actual['start_ticks'] == native['start_ticks'] and
                Path('/proc', str(native['pid']), 'exe').resolve() == Path(row['binary']['path']).resolve(),
                'live native identity differs')
            env = dict(v.split(b'=', 1) for v in Path('/proc', str(native['pid']), 'environ').read_bytes().split(b'\0') if b'=' in v)
            expected = dict(row['environment'], LD_PRELOAD=row['native_validation_probe']['library']['path'],
                STRATA_UR_PROBE_EXPECTED_LOADER=row['native_validation_probe']['loader']['path'])
            require(all(env.get(k.encode()) == v.encode() for k, v in expected.items()), 'effective native environment differs')
            mapped = runtime.executable_mappings(Path('/proc', str(native['pid']), 'maps').read_text())
            require(all(path in qualification['runtime_files'] or path == str(Path(row['binary']['path']).resolve())
                for path in mapped), 'unplanned executable mapping')
            require('/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.17.39758' in mapped, 'updated NEO not mapped')
            row[stage + '_live_check'] = dict(owner_verified=True, environment_verified=True,
                only_pinned_executable_mappings=True, updated_NEO_observed=True)
            persist()
            io.send(b'continue\n')
        io.proc.stdin.close()
        deadline = time.monotonic() + 120
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            require(time.monotonic() < deadline, 'benchmark teardown deadline')
            if select.select([io.proc.stdout], [], [], .05)[0]:
                io.read()
                io.pending.clear()
        io.exit(1)

    try:
        owner = supervision.Owner(ROOT / 'owned', persist)
        entry, _, _ = owner.run('bench', row['command'], row['environment'], P,
            interaction=interact, **row['budgets'])
        require(supervision.closed(entry) and entry['exit_code'] == 0, 'owned official benchmark failed')
        result = json.loads(next((ROOT / 'bench').glob('*/results.json')).read_text())
        samples = result['results']['request_samples']
        row['samples'] = samples
        require(result['complete'] and not result['results']['failures'] and len(samples) == 2 and
            [s['phase'] for s in samples] == ['warm-up/chat', 'prefill-only'] and
            result['results']['prefill_only']['complete'], 'official request completion differs')
        journal, _ = gen.journal_evidence(ROOT)
        require(journal['valid'] and journal['complete'], 'official journal incomplete')
        events = [json.loads(line) for line in next((ROOT / 'bench').glob('*/requests.jsonl')).read_text().splitlines()]
        require([event['sample'] for event in events if event['kind'] == 'result'] ==
            [item['sample'] for item in samples], 'journal/result samples differ')
        starts = [event for event in events if event['kind'] == 'start']
        require(len(starts) == 2 and all(event['thinking'] is True and event['max_tokens'] == 24
            for event in starts), 'thinking/output contract differs')
        row['request_identity'] = [{key: event[key] for key in
            ('phase', 'request_sha256', 'thinking', 'max_tokens')} for event in starts]
        end = gen.native_end(ROOT, row, row, row['boot'], entry['pid'])
        text = (ROOT / 'engine.stderr').read_text()
        row['activation'] = validation.preload_evidence(text, end['native']['pid'])
        warning = ROOT / 'validation.log'
        require(not warning.exists() or warning.stat().st_size == 0, 'native validation warning')
        require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:', text), 'native runtime failure')
        footers = re.findall(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*$', text, re.M)
        require(len(footers) == 2, 'native request footer count')
        for footer, item in zip(footers, samples):
            sample = item['sample']
            require(sample['ok'] and tuple(map(int, footer)) == (sample['prompt_tokens'],
                sample['reused_tokens'], sample['read_tokens'], sample['generated_tokens']) and
                sample['reused_tokens'] == 0 and 1 <= sample['generated_tokens'] <= 24,
                'native/request extent differs')
        short, long = [item['sample'] for item in samples]
        require(0 < short['read_tokens'] < 256 and long['read_tokens'] >= 32768 and long['eligible'], 'short/long extent differs')
        cells = long['read_tokens'] - 1
        route = re.findall(r'^strata prefill: layer-major candidate, tokens=(\d+) chunk=8192 '
            r'residual_dtype=bf16 residual_element_bytes=2 residual_total=(\d+) residual_gpu_rows=(\d+) '
            r'residual_gpu=(\d+) residual_host=0 residual_staging=0 ', text, re.M)
        require(len(route) == 1 and tuple(map(int, route[0])) == (cells, cells * 20480, cells, cells * 20480) and
            len(re.findall(r'^strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows='
                + str(cells) + r' mixed_fp16_bytes=0$', text, re.M)) == 1, 'actual selected long route differs')
        current(initial)
        row.update(host_probe_accepted=True, native_exit_code=0, benchmark_exit_code=0,
            short_and_long_native_footer_conservation=True, actual_BF16_GPU_rows=cells,
            actual_MoE_mode=10, journal_valid=True, journal_complete=True,
            evidence_sha256={name: digest(ROOT / name) for name in ('engine.stderr', 'engine-start.json', 'engine-exit.json')})
    except BaseException as error:
        row['error'] = repr(error)
    finally:
        if owner is not None:
            validation.finish(row, owner, cursor, row['boot'])
        row['passed'] = bool(row.get('passed') and row.get('driver_after') == row['driver'])
        row.update(stage='model-checked' if row['passed'] else 'failed', complete=row['passed'], active=False)
        persist()
        write(RESULT, row)
    print(json.dumps({key: row.get(key) for key in ('stage', 'passed', 'ownership_closed', 'native_exit_code', 'error')}), flush=True)
    return int(not row['passed'])


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'run'), 'prepare|run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare()
        else:
            sys.exit(run(validation.committed(RESULT)))
