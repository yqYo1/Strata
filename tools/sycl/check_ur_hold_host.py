#!/usr/bin/env python3
"""Finite host-only gate for the matched loader observer; repository LICENSE.

Five fresh processes, no device/queue/enqueue/model tests or environment mutation.
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
ROOT = B / 'ur-hold-host-v1'
KEY = 'ur_hold_host_v1'
require, digest, write = h.require, h.digest, h.write


def compact(row):
    return {k: v for k, v in row.items() if k not in
            ('controller_inputs', 'runtime_files', 'environment', 'native_build')}


def save(row):
    raw = RESULT.read_text(); desired = json.loads(raw); h.q.build.scope(desired)[KEY] = compact(row)
    literal = json.dumps(KEY) + ':'; value = json.dumps(compact(row), indent=2).replace('\n', '\n      ')
    if KEY in h.q.build.scope(json.loads(raw)):
        require(raw.count(literal) == 1, 'ambiguous host-control scope')
        start = raw.index(literal) + len(literal)
        while raw[start].isspace(): start += 1
        _, size = json.JSONDecoder().raw_decode(raw[start:]); raw = raw[:start] + value + raw[start + size:]
    else:
        anchor = '"fused_command_profile": {'; require(raw.count(anchor) == 1, 'ambiguous container')
        raw = raw.replace(anchor, anchor + '\n      ' + literal + ' ' + value + ',', 1)
    require(json.loads(raw) == desired, 'host update changed unrelated evidence')
    write(ROOT / 'record.json', row); RESULT.write_text(raw)


def prepare():
    require(not ROOT.exists() and KEY not in h.q.build.scope(r.validation.committed(RESULT)), 'single-use host gate')
    built, proof = h.native(); rt = r.validation.committed(r.RESULT); r.current(rt)
    revision = r.validation.git('rev-parse', 'HEAD').decode().strip()
    env = dict(r.validation.committed(r.SETTINGS)['environment'])
    require('LD_PRELOAD' not in env, 'no preload during prospective ldd query')
    target = dict(path=str(Path('/usr/bin/true').resolve()), sha256=digest('/usr/bin/true'))
    files = {p: sha for p, sha in rt['runtime_files'].items() if p != rt['preload']['library']['path']}
    for seed in (target, built['library']):
        files[seed['path']] = seed['sha256']
        output = subprocess.check_output(['/usr/bin/ldd', seed['path']], env=env, text=True, timeout=15)
        require('not found' not in output, 'unresolved prospective dependency')
        for line in output.splitlines():
            name = next((v for v in line.split() if v.startswith('/')), None)
            if name:
                name = str(Path(name).resolve()); sha = digest(name)
                require(name not in files or files[name] == sha, 'prospective provider differs'); files[name] = sha
    env.update(LD_PRELOAD=built['library']['path'], STRATA_UR_PROBE_EXPECTED_LOADER=rt['preload']['loader'])
    cases = [dict(label='on', mode='on', exit_code=0), dict(label='off', mode='off', exit_code=0),
        dict(label='wrong-provider', mode='off', exit_code=125, reason='unexpected loader provider'),
        dict(label='forbidden-layer', mode='off', exit_code=125, reason='unexpected enabled layers'),
        dict(label='bad-mode', mode='invalid', exit_code=125, reason='explicit on/off mode required')]
    row = dict(stage='prepared', source_commit=revision, controller_inputs=h.controller(revision),
        native_build=proof, native_reference=dict(source_commit=built['source_commit'], library=built['library']),
        runtime_sha256=digest(r.RESULT), runtime_files=files, environment=env,
        environment_sha256=h.identity(env), target=target, cases=cases, processes=[],
        boot=rt['boot'], driver=rt['driver'], budgets=dict(wall=30, cpu=30, rss_cap=1 << 30,
            text_cap=4 << 20, file_cap=4 << 20, total_cap=64 << 20),
        complete=False, passed=False, active=False, ownership_closed=True,
        GPU_commands_executed=False, model_executed=False, performance_qualified=False,
        criteria='Five ordered fresh owned host processes;normal on/off exits plus three expected '
            'constructor refusals125. Check exact source/library/providers/argv,pre-main mode/first-init/'
            'held-reference proof,zero parameter warnings and no queue/enqueue API trace. Stop first '
            'unexpected result. Fast host /proc observations are recorded,not required as GPU-provider '
            'qualification. No GPU/model or ordinary teardown advancement from this host gate alone.',
        retention='Keep compact per-case outcomes and native identities. Full normal host traces/private '
            'receipts retire after verification;helper stays only for named imminent GPU/validation-cost '
            'comparison. Original selected13assets and official v4fallback unchanged.')
    ROOT.mkdir(mode=0o700); write(ROOT / 'runtime-inputs.json', files)
    row['runtime_manifest'] = dict(path=str(ROOT / 'runtime-inputs.json'),
        sha256=digest(ROOT / 'runtime-inputs.json'), files=len(files))
    save(row); print('Prepared;commit exact five-process host plan first.', flush=True)


def current(row):
    require(h.q.build.scope(r.validation.committed(RESULT))[KEY] == compact(row), 'commit exact host plan')
    require(h.controller(row['source_commit']) == row['controller_inputs'] and h.native()[1] == row['native_build'],
        'controller/native build changed')
    require(digest(r.RESULT) == row['runtime_sha256'] and h.identity(row['environment']) == row['environment_sha256'],
        'runtime/host environment changed')
    rt = r.validation.committed(r.RESULT); r.current(rt)
    require(rt['boot'] == row['boot'] and rt['driver'] == row['driver'], 'boot/driver changed')
    require(digest(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256'] and
        json.loads(Path(row['runtime_manifest']['path']).read_text()) == row['runtime_files'] and
        all(digest(p) == sha for p, sha in row['runtime_files'].items()), 'frozen provider changed')


def run(row):
    current(row); require(row['stage'] == 'prepared' and not (ROOT / 'owned').exists(), 'single-use host run')
    initial = copy.deepcopy(row); owner = None; outcome = dict(host_probe_accepted=False)
    cursor = next(v.split(': ', 1)[1] for v in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True,
        timeout=15).splitlines() if v.startswith('-- cursor: '))
    def persist(): write(ROOT / 'execution.json', dict(commands=owner.commands if owner else [],
        active=bool(owner and owner.active), processes=row['processes']))
    try:
        owner = r.supervision.Owner(ROOT / 'owned', persist)
        for case in row['cases']:
            root = ROOT / case['label']; root.mkdir()
            for name in ('tmp', 'cache'): (root / name).mkdir()
            env = dict(row['environment'], STRATA_UR_HOLD_VALIDATION=case['mode'],
                UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if case['mode'] == 'on'
                    or case['label'] == 'forbidden-layer' else 'UR_LAYER_TRACING',
                UR_LOG_TRACING='level:info;flush:info;output:file,' + str(root / 'ur.trace'),
                UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root / 'validation.log'),
                TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'))
            if case['label'] == 'wrong-provider': env['STRATA_UR_PROBE_EXPECTED_LOADER'] = '/not/the/qualified/loader'
            seen = set(); env_seen = False
            def monitor(io, entry):
                nonlocal env_seen
                while not io.exit_watch.poll(0) or not io.eof:
                    io.poll()
                    try:
                        maps = r.executable_mappings(Path('/proc', str(io.proc.pid), 'maps').read_text())
                        actual = dict(v.split(b'=', 1) for v in Path('/proc', str(io.proc.pid),
                            'environ').read_bytes().split(b'\0') if b'=' in v)
                        require(all(actual.get(k.encode()) == v.encode() for k, v in env.items()) and
                            set(actual) - {k.encode() for k in env} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'},
                            'actual host environment differs'); env_seen = True
                    except (FileNotFoundError, ProcessLookupError): maps = set()
                    except PermissionError:
                        require(io.exit_watch.poll(20), 'live host /proc denied'); maps = set()
                    require(maps <= set(row['runtime_files']), 'unplanned host executable map: ' + repr(maps - set(row['runtime_files'])))
                    seen.update(maps); require(r.supervision.artifact_bytes(ROOT) <= 64 << 20, 'host byte budget')
                    if select.select([io.proc.stdout], [], [], .01)[0]: io.read(); io.pending.clear()
                io.exit(1)
            entry, out, err = owner.run(case['label'], [row['target']['path']], env, P,
                interaction=monitor, **row['budgets'])
            require(r.supervision.closed(entry) and entry['exit_code'] == case['exit_code'] and
                not out.stat().st_size, 'host case failed: ' + case['label'])
            text = err.read_text()
            require(not (root / 'validation.log').exists() or not (root / 'validation.log').stat().st_size,
                'unexpected parameter warning')
            trace = (root / 'ur.trace').read_text() if (root / 'ur.trace').exists() else ''
            require(not re.search(r'ur(?:Enqueue|QueueCreate)', trace), 'host gate submitted GPU work')
            if case['exit_code'] == 0:
                mode = case['mode']; enabled = int(mode == 'on'); probe = '46' if enabled else 'skipped'
                begin = f"strata UR hold-control: pid={entry['pid']} phase=begin mode={mode} loader={env['STRATA_UR_PROBE_EXPECTED_LOADER']}"
                end = f"strata UR hold-control: pid={entry['pid']} phase=before-main mode={mode} preinit=37 enable_calls={enabled} postinit_probe={probe} retained_ref=1"
                require(text.splitlines().count(begin) == text.splitlines().count(end) == 1 and
                    text.index(begin) < text.index(end) and 'hold-control failure:' not in text,
                    'pre-main mode/provider/init evidence missing')
                operations = re.findall(r'^strata UR hold-control API: pid=\d+ operation=([\w-]+) result=(\d+)$', text, re.M)
                expected = ['config-create', 'layers-size', 'layers-query'] + (['enable'] if enabled else []) + ['first-init', 'config-release']
                require(operations == [(name, '0') for name in expected], 'loader API sequence differs')
            else:
                require(text == f"strata UR hold-control failure: pid={entry['pid']} {case['reason']}\n",
                    'wrong constructor refusal')
            row['processes'].append(dict(label=case['label'], native_pid=entry['pid'], exit_code=entry['exit_code'],
                passed=True, ownership_closed=True, elapsed_seconds=entry['elapsed_seconds'],
                live_environment_observed=env_seen, observed_executable_maps=len(seen),
                GPU_commands_executed=False, retained_reference=True if case['exit_code'] == 0 else False))
            persist(); print('Passed host ' + case['label'], flush=True)
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
