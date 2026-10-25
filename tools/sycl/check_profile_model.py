#!/usr/bin/env python3
"""Bounded first whole-model check of our fused-command observer, Main only.

Repository LICENSE. Adapted from our archived2c55ef928 diagnostic controller.
No clean speed, observer-cost, restoration or production-adoption claim.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import select
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import build_profile_engine as native
import command_service
import cpu_capacity as models
import generation_engine as gen
import measurement_owner as owners

r, P, B, RESULT = native.r, native.P, native.B, native.RESULT
ROOT = native.ROOT / 'model-smoke-v1'
TEMPLATE = gen.A / 'phase-prefill-speed-current.json'
HISTORY = gen.A / 'prefill-runtime-comparison-current.json'
require, digest, write = r.require, r.digest, r.write


def scope(container):
    return container['event_reconciliation']['fused_command_profile']


def pins(revision):
    return gen.benchmark_pins(revision)


def compact(row):
    return {k: v for k, v in row.items() if k not in ('source_pins', 'models')}


def save(row):
    container = json.loads(RESULT.read_text())
    scope(container)['model_smoke'] = compact(row)
    write(RESULT, container)


def engine():
    container = gen.committed(RESULT)
    built = json.loads(native.RECORD.read_text())
    require(scope(container)['engine_build'] == native.compact(built) and built['passed']
        and built['ownership_closed'] and not built['active'] and built['stage'] == 'built-not-executed',
        'closed whole-engine build required')
    native.admit(built)
    compiled = built['compiled_inputs']
    require(digest(compiled['manifest']['path']) == compiled['manifest']['sha256'], 'engine dependency manifest changed')
    inputs = json.loads(Path(compiled['manifest']['path']).read_text())
    require(len(inputs) == compiled['files'] and all(digest(p) == h for p, h in inputs.items()),
            'engine compiled input changed')
    binary = built['binary']
    require(digest(binary['path']) == binary['sha256'] and Path(binary['path']).stat().st_size == binary['bytes'],
            'diagnostic engine changed')
    return built


def current(row):
    require(scope(gen.committed(RESULT))['model_smoke'] == compact(row), 'commit frozen smoke plan')
    require(pins(row['source_commit']) == row['source_pins'], 'smoke controller/benchmark source changed')
    built = engine()
    require(built['binary'] == row['binary'] and built['source_commit'] == row['build_source_commit'],
            'diagnostic engine identity differs')
    require(digest(TEMPLATE) == row['template_sha256'] and digest(HISTORY) == row['history_sha256']
        and digest(r.RESULT) == row['runtime_sha256'], 'reference qualification changed')
    for name, sha in row['generated_inputs'].items():
        require(digest(name) == sha, 'generated input changed')
    models.model_pins(row['configuration'], row['models'])
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'boot changed')
    return gen.committed(r.RESULT)


def prepare():
    container = gen.committed(RESULT)
    require(not ROOT.exists() and 'model_smoke' not in scope(container), 'fresh smoke only; no retry')
    built = engine()
    rt = gen.committed(r.RESULT)
    history = gen.committed(HISTORY)['historical_regression_audit']['userland_crossover']
    config = copy.deepcopy(history['configuration']); config['log'] = str(ROOT / 'engine.stderr')
    original = json.loads(gen.git('show', '1e676d554:' + str(TEMPLATE.relative_to(P))))
    model_pins = models.model_pins(config, original['model_assets'])
    env = gen.committed(TEMPLATE)['environment_template']; old = env['STRATA_MEASUREMENT_ROOT']
    env = {k: v.replace(old, str(ROOT)) for k, v in env.items()}
    require('LD_PRELOAD' not in env, 'private native validation preload required')
    env.update(STRATA_TRACE='1', STRATA_SYCL_PREFILL_COMMAND_PROFILE='1',
        STRATA_SYCL_PREFILL_FUSED_COMMAND_PROFILE='1',
        STRATA_PREFILL_LAYER_MAJOR_PROFILE='1', STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE='1',
        STRATA_PREFILL_LAYER_MAJOR_GDN_PROFILE='1', STRATA_PREFILL_CPU_WAIT_PROFILE='1',
        ZE_ENABLE_ALT_DRIVERS=next(p for p in rt['runtime_files'] if '/libze_intel_gpu.so.' in p),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(ROOT / 'validation.log'))
    preload = dict(library=rt['preload']['library'], loader=dict(path=rt['preload']['loader'],
        sha256=rt['runtime_files'][rt['preload']['loader']]))
    revision = gen.git('rev-parse', 'HEAD').decode().strip()
    row = dict(stage='prepared', source_commit=revision, source_pins=pins(revision),
        binary=built['binary'], build_source_commit=built['source_commit'], boot=rt['boot'], driver=rt['driver'],
        configuration=config, environment=env, models=model_pins,
        template_sha256=digest(TEMPLATE), history_sha256=digest(HISTORY), runtime_sha256=digest(r.RESULT),
        preload=preload, generated_inputs={}, active=False, ownership_closed=True,
        passed=False, complete=False, model_qualified=False, observer_cost_qualified=False,
        clean_speed_qualified=False, historical_regression_attributed=False, production_selected=False,
        question='Does the separately built diagnostic engine complete two real-model16K-class requests '
            'with four conservative sampled MoE windows each and checked prefill/decode handoff?',
        criteria='One short warmup and two fresh official bench.sh requests with requested16K input, '
            'actual16384..32767 tokens,0reuse,24output,thinking enabled. Require normal stream and owned closure, '
            'pre-main validation and actual pinned providers,silent validation log,2 layer-major BF16/mode10 routes, '
            'two version3 sparse profiles with4windows and complete clocks/counts/caps, '
            'normal native exit,same boot,no newGPUfault and15 final model hashes. Stop first failure,no retry. '
            'Small smoke only;not task quality,physical262144context,sustained decode or observer overhead. '
            'Failed1baa performance gate remains failed;this independently scoped diagnostic does not adopt it '
            'or advance its prior gated restoration trials. Device service belongs to this diagnostic source package, '
            'not an isolated original6e40/historical counterfactual.',
        retention='Keep source/binary/runtime/configuration,two useful individual requests and compact '
            'profile conservation/clock/owner/provider/fault outcomes. Retire normal logs/JIT/raw profile arrays '
            'after replay. No tensor captures;original selected engine/decode1256placement unchanged.',
        budgets=dict(wall=1200, cpu=36000, rss_cap=104 << 30, text_cap=16 << 20,
                     file_cap=32 << 20, total_cap=64 << 20))
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT / name).mkdir()
    write(ROOT / 'config.json', config)
    write(ROOT / 'plan.json', dict(binary=row['binary'], native_validation_probe=preload))
    row['generated_inputs'] = {str(ROOT / name): digest(ROOT / name) for name in ('config.json', 'plan.json')}
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    row['command'] = ['/bin/sh', str(P / 'bench.sh'), '--config', str(ROOT / 'config.json'), '--port', str(port),
        '--prefill-only', '--prefill-tokens', '16384', '--think', '--runs', '2', '--profile-request-gates',
        '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error', '--ready-timeout', '600',
        '--out', str(ROOT / 'bench')]
    write(ROOT / 'record.json', row); save(row)
    print('Prepared diagnostic model smoke;commit exact plan before execution.', flush=True)


def run(row):
    rt = current(row)
    require(row['stage'] == 'prepared' and not (ROOT / 'owned').exists(), 'single-use model smoke')
    initial = copy.deepcopy(row)
    owner = None
    outcome = dict(complete=False, verified_request_boundaries=0)
    cursor = next(v.split(': ', 1)[1] for v in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if v.startswith('-- cursor: '))
    def persist():
        write(ROOT / 'live.json', dict(outcome, active=bool(owner and owner.active), commands=owner.commands if owner else []))
    def interaction(io, leader):
        for request in range(2):
            for stage in ('begin', 'end'):
                while True:
                    line = io.line(1000)
                    if line.startswith('STRATA_PROFILE_GATE '):
                        break
                require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                    dict(stage=stage, phase='prefill-only', run=request), 'request gate differs')
                require(not (ROOT / 'validation.log').exists() or not (ROOT / 'validation.log').stat().st_size,
                        'parameter validation warning')
                if stage == 'begin':
                    start = json.loads((ROOT / 'engine-start.json').read_text())
                    pid = start['native']['pid']; ident = owners.identity(pid)
                    require(ident and ident['start_ticks'] == start['native']['start_ticks'] and
                        ident['sid'] == leader['pid'] and str(Path('/proc', str(pid), 'exe').resolve()) ==
                        str(Path(row['binary']['path']).resolve()), 'native identity differs')
                    actual = dict(v.split(b'=', 1) for v in Path('/proc', str(pid), 'environ').read_bytes().split(b'\0') if b'=' in v)
                    expected = dict(row['environment'], LD_PRELOAD=row['preload']['library']['path'],
                        STRATA_UR_PROBE_EXPECTED_LOADER=row['preload']['loader']['path'])
                    require(all(actual.get(k.encode()) == v.encode() for k, v in expected.items()) and
                        set(actual) - {k.encode() for k in expected} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'},
                        'native pre-exec environment differs')
                    paths = r.executable_mappings(Path('/proc', str(pid), 'maps').read_text())
                    require(paths <= set(rt['runtime_files']) | {str(Path(row['binary']['path']).resolve())} and
                        row['environment']['ZE_ENABLE_ALT_DRIVERS'] in paths, 'native provider differs')
                    outcome['verified_request_boundaries'] = request + 1
                io.send(b'continue\n')
        io.proc.stdin.close(); deadline = time.monotonic() + 180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); require(time.monotonic() < deadline, 'smoke teardown deadline')
            if select.select([io.proc.stdout], [], [], .05)[0]:
                io.read(); io.pending.clear()
        io.exit(1)
    try:
        owner = owners.Owner(ROOT / 'owned', persist)
        entry, _, _ = owner.run('bench', row['command'], row['environment'], P, interaction=interaction, **row['budgets'])
        require(owners.closed(entry) and entry['exit_code'] == 0, 'official smoke/owner failed')
        rf, = (ROOT / 'bench').glob('*/results.json')
        result = json.loads(rf.read_text()); samples = result['results']['prefill_only']['requests']
        require(result['complete'] and not result['results']['failures'] and len(samples) == 2 and
            outcome['verified_request_boundaries'] == 2, 'official smoke incomplete')
        for sample in samples:
            require(sample['ok'] and sample['eligible'] and sample['stream_complete'] and
                sample['thinking_requested'] and sample['reasoning_observed'] and
                16384 <= sample['actual_prompt_tokens'] == sample['actual_read_tokens'] < 32768 and
                sample['actual_reused_tokens'] == 0 and sample['generated_tokens'] == 24 and
                sample['prompt_tps_source'] == 'engine', 'small diagnostic workload differs')
        sys.path.insert(0, str(P / 'tools'))
        from strata_bench import verify_request_journal
        jf, = (ROOT / 'bench').glob('*/requests.jsonl')
        journal = verify_request_journal(jf.parent)
        events = [json.loads(v) for v in jf.read_text().splitlines()]
        require(journal['valid'] and journal['complete'] and [dict(v['sample'], run=v['run']) for v in events
            if v['kind'] == 'result' and v['phase'] == 'prefill-only'] == samples, 'journal binding differs')
        native_end = gen.native_end(ROOT, json.loads((ROOT / 'plan.json').read_text()),
            dict(config=row['configuration']), row['boot'], entry['pid'])
        log = (ROOT / 'engine.stderr').read_text()
        r.validation.preload_evidence(log, native_end['native']['pid'])
        footers = list(re.finditer(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*', log, re.M))
        require(len(footers) == 3 and int(footers[0][1]) < 8192, 'warmup/request native extent differs')
        profiles = []
        for index, sample in enumerate(samples):
            require(tuple(map(int, footers[index + 1].groups())) == (sample['actual_prompt_tokens'], 0,
                sample['actual_read_tokens'], 24), 'native read/generated extent differs')
            part = log[footers[index].end():footers[index + 1].start()]
            require(part.count('strata prefill: layer-major candidate, tokens=' +
                str(sample['actual_read_tokens'] - 1) + ' chunk=8192 residual_dtype=bf16') == 1 and
                part.count('strata MoE window configuration: requested_mode=10 effective_mode=10') == 1,
                'small diagnostic route differs')
            profile, = command_service.parse_profiles(part, 4)
            require(profile['version'] == 3, 'fused profile version differs')
            profiles.append(dict(version=3, windows=4, records=int(profile['summaries'][0]['records']),
                peak_events=int(profile['summaries'][0]['peak_events']), groups=len(profile['groups']),
                service_complete=True, submit_clock_qualified=profile['summaries'][0]['submit_clock_qualified'],
                invalid=0, complete_device_coverage=False))
        require(not any(v in log for v in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'terminate called', 'teardown uncertain')),
                'native runtime error')
        current(initial); models.model_pins(row['configuration'], row['models'], full=True)
        fields = ('run', 'request_sha256', 'actual_prompt_tokens', 'actual_read_tokens', 'actual_reused_tokens',
                  'generated_tokens', 'prompt_ms', 'thinking_requested')
        outcome.update(complete=True, host_probe_accepted=True,
            requests=[{k: s[k] for k in fields} for s in samples], profiles=profiles,
            journal_sha256=digest(jf), native_identity=native_end['native'], full_model_hashes=15)
    except BaseException as error:
        outcome['error'] = repr(error)
        if (ROOT / 'engine.stderr').exists():
            outcome['error_excerpt'] = (ROOT / 'engine.stderr').read_text()[-4096:]
    finally:
        if owner:
            r.validation.finish(outcome, owner, cursor, row['boot'])
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete']); persist()
        row.update(stage='closed', active=False, ownership_closed=outcome.get('ownership_closed', True),
            complete=outcome['complete'], passed=outcome['passed'], outcome=outcome)
        write(ROOT / 'record.json', row); save(row)
    print(json.dumps({k: outcome.get(k) for k in ('passed', 'complete', 'ownership_closed', 'profiles', 'error')}), flush=True)
    return int(not outcome['passed'])


if __name__ == '__main__':
    with (B / 'owned-v0141-measurement.lock').open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); r.guards.no_model_owners([])
        require(sys.argv[1:] in (['prepare'], ['run']), 'check_profile_model.py prepare|run')
        sys.exit(prepare() if sys.argv[1] == 'prepare' else run(json.loads((ROOT / 'record.json').read_text())))
