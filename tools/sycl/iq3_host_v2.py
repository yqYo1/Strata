#!/usr/bin/env python3
"""Fresh host-only admission of the unchanged IQ3 model-library outputs.

Repository LICENSE. V1's artifact-budget observation stays failed. This plan
changes only the owned output scope, not native bytes, arithmetic or routing.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
import build_iq3_model as build

P, B, RESULT, base = build.P, build.B, build.RESULT, build.base
ROOT = B / 'decode-iq3-host-v2'
KEY = 'decode_iq3_host_v2'
SELF = Path(__file__).resolve()
require, digest, write = build.require, build.digest, build.write


def compact(row):
    out = {k:v for k,v in row.items() if k not in ('sources', 'commands')}
    out['sources'] = dict(files=len(row['sources']), aggregate_sha256=hashlib.sha256(
        json.dumps(row['sources'], sort_keys=True).encode()).hexdigest())
    return out


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def proof():
    native = build.context().verify()
    failed = native['host_result']
    evidence = native['host_failure_evidence']
    require(not failed['passed'] and not failed['complete'] and not failed['GPU_executed'] and
            evidence['error'] == "RuntimeError('artifact cap')" and not evidence['observation_complete'] and
            evidence['direct_child_reaped'] and evidence['session_empty'] and
            not evidence['survivors'] and not evidence['active'] and not evidence['cleanup'],
            'preserve the failed V1 observation and prove its processes are terminal')
    return native


def prepare():
    require(not ROOT.exists() and KEY not in base.committed(RESULT), 'one fresh versioned host plan')
    native = proof()
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    row = dict(stage='prepared', source_commit=revision, sources=base.controls(revision, controller=SELF),
        build_record=dict(path=str(build.ROOT / 'record.json'), sha256=digest(build.ROOT / 'record.json')),
        binaries=native['binaries'], native_source_commit=native['source_commit'],
        environment=dict(native['recipe']['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache')),
        cases=[dict(kind=kind, arm=arm, target=target, args=args) for arm in (0, 1)
            for kind, target, args in [('exact', 'iq_single_exact_test', ['--dispatch']),
                                       ('pool', 'cpu_service_pool_test', ['0'])]],
        plan='Same five admitted native outputs and same384single/80pair arithmetic,80entry-call,'
            'prefill exclusion and28real-pool-case criteria. Fresh exact0,pool0,exact1,pool1 children;'
            'PAIR0/1 fixed before exec,AUDIT1,SINGLE_EXACT0,CPU_SERVICE0. The owned commands directory'
            'is under this private host root,so its parent byte budget excludes the compiled model tree.'
            'Stop first failure;no retry or reinterpretation of V1;no native rebuild/GPU/model/speed claim.',
        budgets=dict(wall=60, cpu=60, rss_cap=512 << 20, text_cap=64 << 10,
                     file_cap=64 << 10, total_cap=1 << 20),
        results=[], commands=[], passed=False, complete=False, active=False, ownership_closed=True,
        GPU_executed=False, native_rebuilt=False)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (ROOT / name).mkdir()
    save(row)
    print('Prepared host-v2 with unchanged native outputs and isolated1MiB output root;commit before run.', flush=True)


def unchanged(row):
    require(base.committed(RESULT)[KEY] == compact(row) and
            base.controls(row['source_commit'], controller=SELF) == row['sources'], 'commit and freeze host-v2 inputs')
    require(digest(row['build_record']['path']) == row['build_record']['sha256'], 'original build record drift')
    native = proof()
    require(native['binaries'] == row['binaries'] and native['source_commit'] == row['native_source_commit'],
            'reuse only the exact admitted native/fixture bytes')


def run():
    row = json.loads((ROOT / 'record.json').read_text())
    require(row['stage'] == 'prepared' and not (ROOT / 'commands').exists(), 'one fresh host-v2 attempt')
    unchanged(row)
    owner, sums = None, []
    def persist():
        write(ROOT / 'execution.json', dict(active=bool(owner and owner.active),
            commands=owner.commands if owner else []))
    try:
        owner = base.supervision.Owner(ROOT / 'commands', persist=persist)
        require(base.supervision.artifact_bytes(ROOT) < row['budgets']['total_cap'], 'private output preflight')
        deadline = time.monotonic() + 300
        for case in row['cases']:
            env = dict(row['environment'], STRATA_IQ3_PAIR=str(case['arm']))
            entry, out, err = owner.run(case['kind'] + '-' + str(case['arm']),
                [row['binaries'][case['target']]['path'], *case['args']], env, ROOT,
                **dict(row['budgets'], wall=min(60, deadline - time.monotonic())))
            require(base.supervision.closed(entry) and entry['exit_code'] == 0 and not err.stat().st_size,
                    'host-v2 child failed or incomplete: ' + err.read_text(errors='replace')[-1500:])
            content = out.read_text().strip()
            if case['kind'] == 'exact':
                expected = ('IQ single exact PASS: 384 single and 80 paired cases, bitwise GGML parity and row canaries\n'
                    f'IQ3 production dispatch PASS arm={case["arm"]} calls=80 nt1_rows=20480 pair_rows={case["arm"] * 10240} prefill_pair_rows=0')
                require(content == expected, 'complete exact arithmetic/phase/route footer required')
            else:
                match = re.fullmatch(r'cpu_service_pool PASS observer=0 formats=7 nt=1\.\.4 cases=28 checksum_fnv1a64_le=([0-9a-f]{16})', content)
                require(match is not None, 'complete real-pool footer required'); sums.append(match[1])
            row['results'].append(dict(kind=case['kind'], arm=case['arm'], passed=True,
                exit_code=0, stdout_sha256=digest(out), ownership_closed=True))
        require(sums == [build.component()['correctness']['pool_checksum']] * 2, 'same-build pool bits differ')
        frozen = json.loads((ROOT / 'record.json').read_text()); unchanged(frozen)
        row.update(stage='closed', passed=True, complete=True, pool_checksum=sums[0])
    except BaseException as error:
        row.update(stage='failed', error=repr(error)); raise
    finally:
        row.update(commands=owner.commands if owner else [], active=bool(owner and owner.active),
            ownership_closed=owner is None or (owner.active is None and
                all(base.supervision.closed(e) for e in owner.commands)))
        save(row)
    print('All four host-v2 arithmetic/phase/pool children pass;V1 remains failed;no GPU/model qualification.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'run'), 'usage: iq3_host_v2.py prepare|run')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        {'prepare': prepare, 'run': run}[sys.argv[1]]()
