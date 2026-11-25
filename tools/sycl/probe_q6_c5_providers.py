#!/usr/bin/env python3
"""Identify the clean C5 runtime at the pre-kernel device gate."""

import fcntl
import json
from pathlib import Path
import select
import subprocess
import sys

sys.dont_write_bytecode = True
import check_q6_c5_component as check

P, B, RESULT = check.P, check.B, check.RESULT
ROOT = B / 'q6-c5-clean-provider-probe-v1'
KEY = 'q6_c5_clean_provider_probe_v1'
THIS = str(Path(__file__).resolve().relative_to(P))
require, sha, write = check.require, check.sha, check.write
validation, guards, ownership = check.validation, check.guards, check.ownership


def inputs():
    failed = validation.committed(RESULT, (check.KEY,))
    refusal = validation.committed(RESULT, (check.KEY + '_clean_provider_refusal',))
    require(failed['stage'] == 'failed' and failed['outcomes'][-1]['mode'] == 'time0'
            and not failed['outcomes'][-1]['passed'] and refusal['direct_child_reaped'] and refusal['session_empty']
            and not refusal['kernel_execution_observed'] and refusal['same_fixture'] == failed['binary'],
            'recover the closed clean timing refusal')
    active = check.device_plan()
    require(all(failed[k] == v for k, v in active.items()), 'same fixture and runtime required')
    env = dict(active['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    for name in ('LD_PRELOAD', 'STRATA_UR_PROBE_EXPECTED_LOADER', 'UR_ENABLE_LAYERS',
                 'UR_LOG_TRACING', 'UR_LOG_VALIDATION'):
        env.pop(name)
    return dict(binary=active['binary'], boot=active['boot'], driver=active['driver'],
        environment=env, runtime_manifest=active['runtime_manifest'],
        required=[active['validation_probe']['loader']['path'], env['UR_ADAPTERS_FORCE_LOAD'], env['ZE_ENABLE_ALT_DRIVERS']],
        source_execution_sha256=refusal['execution_sha256'])


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text())
    data[KEY] = row
    write(RESULT, data)


def prepare():
    require(not ROOT.exists() and KEY not in validation.committed(RESULT), 'fresh provider observation required')
    source = check.build.base.source.pin(P, (THIS,))
    row = dict(stage='ready', source_commit=source['revision'], controller_sha256=source['source_sha256'][THIS],
        parent_reader_sha256=sha(P / check.THIS), **inputs(),
        budgets=dict(wall=180, cpu=180, text_cap=1 << 20, file_cap=1 << 20, total_cap=32 << 20),
        scope='Inspect the clean runtime at the two fixture gates. Refuse entry before any kernel or timing work.')
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    save(row)


def probe():
    row = json.loads((ROOT / 'record.json').read_text())
    require(row == validation.committed(RESULT, (KEY,)) and row['stage'] == 'ready', 'commit the provider observation plan')
    require(sha(P / THIS) == row['controller_sha256'] and sha(P / check.THIS) == row['parent_reader_sha256']
            and (P / THIS).read_bytes() == validation.git('show', row['source_commit'] + ':' + THIS),
            'the observation controller changed')
    require(all(row[name] == value for name, value in inputs().items()), 'fixture or runtime configuration changed')
    require(not (ROOT / 'execution.json').exists(), 'inspect the existing provider attempt')
    pins = json.loads(Path(row['runtime_manifest']['path']).read_text())
    env = row['environment']
    outcome = dict(phases=[], host_probe_accepted=False)
    cursor = next(line.split(': ', 1)[1] for line in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if line.startswith('-- cursor: '))

    def persist():
        write(ROOT / 'execution.json', dict(active=bool(owner.active), commands=owner.commands, outcome=outcome))

    def interact(io, command):
        pid = command['pid']
        for phase, tag in enumerate(('STRATA_Q6_DPAS_GATE ', 'STRATA_Q6_DPAS_DEVICE_GATE ')):
            line = io.line(100)
            while not line.startswith(tag):
                line = io.line(100)
            proc = Path('/proc', str(pid))
            identity = ownership.identity(pid)
            require(line == tag + 'pid=' + str(pid) and str((proc / 'exe').resolve()) == row['binary']['path']
                    and identity and identity['start_ticks'] == command['direct_identity']['start_ticks'], 'actual fixture identity differs')
            actual = dict(v.split('=', 1) for v in (proc / 'environ').read_text().split('\0') if '=' in v)
            changed = sorted(k for k in actual.keys() | env.keys() if actual.get(k) != env.get(k))
            mapped = set(check.benchmark_runtime.code_maps(pid))
            unknown = sorted(mapped - (set(pins) | {row['binary']['path']}))
            expected = {str(Path(p).resolve()) for p in row['required']}
            outcome['phases'].append(dict(stage=phase, changed_environment_names=changed, mapped_files=len(mapped),
                missing=sorted(expected - mapped), extra=[dict(path=p, sha256=sha(p), bytes=Path(p).stat().st_size) for p in unknown]))
            persist()
            require(phase or not changed and not unknown, 'initial process environment or runtime changed')
            io.send(b'stop\n' if phase else b'continue\n')
        io.proc.stdin.close()
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            if select.select([io.proc.stdout], [], [], .02)[0]:
                io.read()
                io.pending.clear()
        io.exit(1)

    owner = ownership.Owner(ROOT / 'owned', persist)
    try:
        command, out, err = owner.run('provider', [row['binary']['path'], '--time-c5', '0'], env, P,
                                     interaction=interact, **row['budgets'])
        output = out.read_text()
        outcome['native_exit_code'] = command.get('exit_code')
        require(ownership.closed(command) and command['exit_code'] == 2
                and len(outcome['phases']) == 2 and 'Q6 DPAS component failed: native gate' in err.read_text()
                and 'ADMISSION ' not in output and 'SAMPLE ' not in output, 'expected pre-kernel stop differs')
        outcome['host_probe_accepted'] = True
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        validation.finish(outcome, owner, cursor, row['boot'])
        outcome['passed'] = outcome['passed'] and outcome.get('driver_after') == row['driver']
        row.update(stage='closed' if outcome['passed'] else 'failed', outcome=outcome)
        persist()
        save(row)
    require(outcome['passed'], outcome.get('error', 'provider observation failed'))
    print(json.dumps(outcome['phases'], ensure_ascii=False))


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['probe']), 'select prepare or probe')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare()
        else:
            probe()
