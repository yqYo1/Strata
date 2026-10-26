#!/usr/bin/env python3
"""Main-owned bounded on/off loader-control GPU gate; repository LICENSE.

Use original, already qualified fixtures; no build, model or speed claim.
Both arms retain one loader reference and preserve immutable safety settings.
"""
import copy
import fcntl
import json
from pathlib import Path
import re
import select
import subprocess
import sys
sys.dont_write_bytecode = True
import ur_hold_control as h

r, P, B, RESULT = h.r, h.P, h.B, h.RESULT
ROOT = B / 'ur-hold-device-v1'
KEY = 'ur_hold_device_v1'
require, digest, write = h.require, h.digest, h.write


def evidence(text, pid, mode, loader):
    require(mode in ('on', 'off'), 'invalid control mode')
    enabled = int(mode == 'on'); probe = '46' if enabled else 'skipped'
    begin = f'strata UR hold-control: pid={pid} phase=begin mode={mode} loader={loader}'
    end = (f'strata UR hold-control: pid={pid} phase=before-main mode={mode} preinit=37 '
           f'enable_calls={enabled} postinit_probe={probe} retained_ref=1')
    require(text.splitlines().count(begin) == text.splitlines().count(end) == 1 and
        text.index(begin) < text.index(end) and 'hold-control failure:' not in text,
        'native control mode/provider/init proof missing')
    actual = re.findall(r'^strata UR hold-control API: pid=(\d+) operation=([\w-]+) result=(\d+)$', text, re.M)
    expected = ['config-create', 'layers-size', 'layers-query'] + (['enable'] if enabled else []) + ['first-init', 'config-release']
    require(actual == [(str(pid), op, '0') for op in expected], 'loader API sequence differs')
    return dict(pid=pid, mode=mode, loader=loader, preinit=37, enable_calls=enabled,
        postinit_probe=probe, retained_reference_until_exit=True, ordinary_teardown_qualified=False)


def host_gate():
    row = r.validation.committed(RESULT, ('event_reconciliation', 'fused_command_profile', 'ur_hold_host_v1'))
    private = json.loads((B / 'ur-hold-host-v1/record.json').read_text())
    require(row == {k: v for k, v in private.items() if k not in
        ('controller_inputs', 'runtime_files', 'environment', 'native_build')} and
        row['stage'] == 'closed' and row['passed'] and row['complete'] and
        row['ownership_closed'] and not row['active'] and not row['GPU_commands_executed'] and
        len(row['processes']) == 5 and all(v['passed'] and v['ownership_closed'] for v in row['processes']),
        'exact closed five-host gate required')
    require(h.controller(row['source_commit'], worktree=False) == private['controller_inputs'],
        'original host controller unrecoverable')
    return dict(source_commit=row['source_commit'], result_sha256=h.identity(row))


def compact(row):
    return {k: v for k, v in row.items() if k not in ('controller_inputs', 'runtime_files', 'environment')}


def save(row):
    raw = RESULT.read_text(); desired = json.loads(raw); h.q.build.scope(desired)[KEY] = compact(row)
    literal = json.dumps(KEY) + ':'; value = json.dumps(compact(row), indent=2).replace('\n', '\n      ')
    if KEY in h.q.build.scope(json.loads(raw)):
        require(raw.count(literal) == 1, 'ambiguous device scope')
        start = raw.index(literal) + len(literal)
        while raw[start].isspace(): start += 1
        _, size = json.JSONDecoder().raw_decode(raw[start:]); raw = raw[:start] + value + raw[start + size:]
    else:
        anchor = '"fused_command_profile": {'; require(raw.count(anchor) == 1, 'ambiguous result container')
        raw = raw.replace(anchor, anchor + '\n      ' + literal + ' ' + value + ',', 1)
    require(json.loads(raw) == desired, 'device update changed unrelated evidence')
    write(ROOT / 'record.json', row); RESULT.write_text(raw)


def reviewed():
    row = r.validation.committed(r.validation.INDEX, ('UR_parameter_validation_hold_control_20261026',))
    review = row['native_review']
    require(row['status'] == 'reviewed_device_gate_pending' and review['status'] == 'complete' and
        review['matched_empty_layers_plan_accepted'] is True and
        row['proposal']['sha256'] == h.native()[0]['native_source']['sha256'],
        'accepted frozen-source contract review required')
    return dict(report=row['report']['sha256'], native_review=review['sha256'],
        matched_empty_layers_plan_accepted=True)


def prepare():
    require(not ROOT.exists() and KEY not in h.q.build.scope(r.validation.committed(RESULT)), 'single-use device gate')
    helper, proof = h.native(); host = host_gate(); review = reviewed()
    rt = r.validation.committed(r.RESULT); full = r.current(rt)
    revision = r.validation.git('rev-parse', 'HEAD').decode().strip()
    env = dict(r.validation.committed(r.SETTINGS)['environment'])
    require('LD_PRELOAD' not in env, 'no constructor preload in prospective ldd')
    cases = []
    for mode in ('on', 'off'):
        for label, target, args in (('submission', 'sycl_submission_test', ['--submission']),
                ('graph0', 'sycl_graph_envelope_test', ['--profile=0']),
                ('graph1', 'sycl_graph_envelope_test', ['--profile=1'])):
            cases.append(dict(label=mode + '-' + label, mode=mode, kind=label,
                binary=full['fixtures'][target], args=args))
    files = {p: sha for p, sha in rt['runtime_files'].items() if p != rt['preload']['library']['path']}
    for item in [helper['library'], *[c['binary'] for c in cases]]:
        files[str(Path(item['path']).resolve())] = item['sha256']
        out = subprocess.check_output(['/usr/bin/ldd', item['path']], env=env, text=True, timeout=15)
        require('not found' not in out, 'unresolved fixture dependency')
        for line in out.splitlines():
            path = next((v for v in line.split() if v.startswith('/')), None)
            if path:
                path = str(Path(path).resolve()); sha = digest(path)
                require(path not in files or files[path] == sha, 'prospective provider differs'); files[path] = sha
    env.update(LD_PRELOAD=helper['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=rt['preload']['loader'])
    neo = next(p for p in files if 'libze_intel_gpu' in p)
    row = dict(stage='prepared', source_commit=revision, controller_inputs=h.controller(revision),
        helper=proof, host_gate=host, review=review, original_build=full['binary'],
        runtime_sha256=digest(r.RESULT), environment=env, environment_sha256=h.identity(env),
        runtime_files=files, NEO=neo, cases=cases, processes=[], boot=rt['boot'], driver=rt['driver'],
        budgets=dict(wall=90, cpu=120, rss_cap=4 << 30, text_cap=16 << 20, file_cap=32 << 20, total_cap=64 << 20),
        whole_bytes=256 << 20, active=False, ownership_closed=True, complete=False, passed=False,
        GPU_executed=False, model_executed=False, performance_qualified=False,
        criteria='Six ordered fresh processes,on then off for submission/graph0/graph1. Same original6e40 '
            'fixtures and held loader ref;bounded full API traces plus silent validation log. Exact source/'
            'ELFs/native-env/live maps including NEO and loader;all numerical,guard,profiling and checked '
            'fixture cleanup markers. Stop first mismatch/fault/deadline. Tracing-mode environment lists '
            'differ deliberately;this gate supplies no speed evidence. Clean later A/B must use identical '
            'empty layer env and config-only validation,retained ref in both;ordinary UR teardown unqualified.',
        retention='Keep compact six-case results and provider/native identities until upcoming matched '
            'validation-cost comparison. Retire normal full traces/private output after verification. '
            'Original13 assets and official v4fallback stay unchanged.')
    ROOT.mkdir(mode=0o700); write(ROOT / 'runtime-inputs.json', files)
    row['runtime_manifest'] = dict(path=str(ROOT / 'runtime-inputs.json'), sha256=digest(ROOT / 'runtime-inputs.json'), files=len(files))
    save(row); print('Prepared six-process device gate;commit plan before run.', flush=True)


def current(row):
    require(r.validation.committed(RESULT, ('event_reconciliation', 'fused_command_profile', KEY)) == compact(row),
        'commit exact device plan')
    require(h.controller(row['source_commit']) == row['controller_inputs'] and
        h.native()[1] == row['helper'] and host_gate() == row['host_gate'] and reviewed() == row['review'],
        'controller/helper/host/review changed')
    rt = r.validation.committed(r.RESULT)
    require(digest(r.RESULT) == row['runtime_sha256'] and r.current(rt)['binary'] == row['original_build'] and
        rt['boot'] == row['boot'] and rt['driver'] == row['driver'], 'selected runtime changed')
    m = row['runtime_manifest']
    require(h.identity(row['environment']) == row['environment_sha256'] and digest(m['path']) == m['sha256'] and
        json.loads(Path(m['path']).read_text()) == row['runtime_files'] and
        all(digest(p) == sha for p, sha in row['runtime_files'].items()), 'provider/environment changed')


def run(row):
    current(row); require(row['stage'] == 'prepared' and not (ROOT / 'owned').exists(), 'single-use device run')
    initial = copy.deepcopy(row); owner = None; outcome = dict(host_probe_accepted=False)
    cursor = next(v.split(': ', 1)[1] for v in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True,
        timeout=15).splitlines() if v.startswith('-- cursor: '))
    def persist(): write(ROOT / 'execution.json', dict(commands=owner.commands if owner else [],
        active=bool(owner and owner.active), processes=row['processes']))
    try:
        owner = r.supervision.Owner(ROOT / 'owned', persist)
        for case in row['cases']:
            current(initial)
            root = ROOT / case['label']; root.mkdir()
            for name in ('tmp', 'cache'): (root / name).mkdir()
            env = dict(row['environment'], STRATA_UR_HOLD_VALIDATION=case['mode'],
                UR_ENABLE_LAYERS='UR_LAYER_TRACING' + (',UR_LAYER_PARAMETER_VALIDATION' if case['mode'] == 'on' else ''),
                UR_LOG_TRACING='level:info;flush:info;output:file,' + str(root / 'ur.trace'),
                UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root / 'validation.log'),
                TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'))
            seen = set(); env_seen = False
            def monitor(io, entry):
                nonlocal env_seen
                while not io.exit_watch.poll(0) or not io.eof:
                    io.poll()
                    try:
                        maps = r.executable_mappings(Path('/proc', str(io.proc.pid), 'maps').read_text())
                        actual = dict(v.split(b'=', 1) for v in Path('/proc', str(io.proc.pid), 'environ').read_bytes().split(b'\0') if b'=' in v)
                        require(all(actual.get(k.encode()) == v.encode() for k, v in env.items()) and
                            set(actual) - {k.encode() for k in env} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'}, 'native environment differs')
                        env_seen = True
                    except (FileNotFoundError, ProcessLookupError): maps = set()
                    except PermissionError:
                        require(io.exit_watch.poll(20), 'live /proc denied'); maps = set()
                    require(maps <= set(row['runtime_files']), 'unexpected code mapping: ' + repr(maps - set(row['runtime_files'])))
                    seen.update(maps); require(r.supervision.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole byte budget')
                    if select.select([io.proc.stdout], [], [], .01)[0]: io.read(); io.pending.clear()
                io.exit(1)
            row['GPU_executed'] = True
            entry, out, err = owner.run(case['label'], [case['binary']['path'], *case['args']], env, P,
                interaction=monitor, **row['budgets'])
            require(r.supervision.closed(entry) and entry['exit_code'] == 0, 'device case failed: ' + case['label'])
            proof = evidence(err.read_text(), entry['pid'], case['mode'], env['STRATA_UR_PROBE_EXPECTED_LOADER'])
            require(env_seen and {row['NEO'], env['STRATA_UR_PROBE_EXPECTED_LOADER']} <= seen,
                'actual native environment/NEO/UR provider not observed')
            require(not (root / 'validation.log').exists() or not (root / 'validation.log').stat().st_size,
                'parameter validation warning')
            require((root / 'ur.trace').exists() and 'urEnqueue' in (root / 'ur.trace').read_text(), 'enqueue trace missing')
            text = out.read_text(); markers = [v for v in text.splitlines() if v.startswith('case ')]
            if case['kind'] == 'submission':
                require(len(markers) == 10 and 'fixture_status=PASS mode=submission' in text, 'submission checks incomplete')
            else:
                marker = 'numerical=PASS guards=PASS aliases=PASS checked_teardown=PASS support=' + (
                    'SUPPORTED' if case['kind'] == 'graph1' else 'NOT_ENABLED')
                require(marker in text, 'graph checks incomplete')
            probe = dict(host_probe_accepted=True); r.validation.finish(probe, owner, cursor, row['boot'])
            require(probe['passed'] and probe['driver_after'] == row['driver'], 'fresh fault/closure failed')
            row['processes'].append(dict(label=case['label'], native_pid=entry['pid'], exit_code=0,
                passed=True, ownership_closed=True, elapsed_seconds=entry['elapsed_seconds'], proof=proof,
                live_environment_observed=env_seen, observed_executable_maps=len(seen),
                trace_bytes=(root / 'ur.trace').stat().st_size, parameter_warnings=0,
                submission_cases=len(markers) if case['kind'] == 'submission' else None))
            persist(); print('Passed device ' + case['label'], flush=True)
        current(initial); outcome['host_probe_accepted'] = True
    except BaseException as error: outcome['error'] = repr(error)
    finally:
        if owner: r.validation.finish(outcome, owner, cursor, row['boot'])
        row.update(stage='closed' if outcome.get('passed') else 'failed', passed=bool(outcome.get('passed')),
            complete=bool(outcome.get('passed')), active=False, ownership_closed=outcome.get('ownership_closed', True), outcome=outcome)
        persist(); save(row)
    print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'complete', 'ownership_closed', 'outcome')}), flush=True)
    return int(not row['passed'])


def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); r.guards.no_model_owners([])
        if sys.argv[1:] == ['prepare']: prepare(); return 0
        if sys.argv[1:] == ['run']: return run(json.loads((ROOT / 'record.json').read_text()))
        raise RuntimeError('use prepare or run')


if __name__ == '__main__': sys.exit(main())
