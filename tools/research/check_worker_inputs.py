#!/usr/bin/env python3
"""Bounded host qualification for the report-worker filesystem checker.

Main-authored; repository LICENSE. Pin source and recipe before four real
read-only namespace controls. This does not launch or qualify an Orca worker.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / 'tools/sycl'))
import measurement_guards as guards
import measurement_owner as ownership
import ur_validation as validation

B = validation.B
ROOT = B / 'worker-filesystem-controls-v1'
INDEX = validation.INDEX
KEY = 'worker_filesystem_controls_v1'
SOURCES = ('tools/research/check_worker_inputs.py', 'tools/research/worker_input_check.py',
    'tools/research/test_worker_input_check.py', 'tools/sycl/measurement_guards.py',
    'tools/sycl/measurement_owner.py', 'tools/sycl/ur_validation.py', 'AGENTS.md', 'docs/ARTIFACT_RETENTION.md')
require, digest = guards.require, validation.digest


def write(path, row):
    Path(path).write_text(json.dumps(row, ensure_ascii=False, indent=2) + '\n')


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(INDEX.read_text()); data[KEY] = row; write(INDEX, data)


def current(row):
    require(validation.committed(INDEX, (KEY,)) == row, 'commit exact filesystem control plan/result')
    for path, sha in row['sources'].items():
        require(digest(P / path) == sha and hashlib.sha256(subprocess.check_output(
            ['git', 'show', row['source_commit'] + ':' + path], cwd=P)).hexdigest() == sha,
            'filesystem-control source changed')
    require(all(digest(v['path']) == v['sha256'] for v in row['programs'].values()), 'test interpreter/bwrap changed')


def prepare():
    require(not ROOT.exists() and KEY not in validation.committed(INDEX), 'fresh filesystem controls only')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip()
    sources = {p: digest(P / p) for p in SOURCES}
    require(all(hashlib.sha256(subprocess.check_output(['git', 'show', revision + ':' + p], cwd=P)).hexdigest() == h
                for p, h in sources.items()), 'commit checker/controller/tests first')
    bwrap = shutil.which('bwrap'); require(bwrap is not None, 'real bwrap required')
    programs = {k: dict(path=str(Path(v).resolve()), sha256=digest(Path(v).resolve()))
        for k, v in (('python', sys.executable), ('bwrap', bwrap))}
    row = dict(stage='prepared', source_commit=revision, sources=sources, programs=programs,
        command=[sys.executable, '-B', '-m', 'unittest', '-v', 'tools.research.test_worker_input_check'],
        environment=dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
            PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(ROOT / 'tmp')),
        budget=dict(wall=90, cpu=90, rss_cap=1 << 30, text_cap=1 << 20, file_cap=2 << 20, total_cap=8 << 20),
        question='Does one checker inspect every declared source and refuse missing/hash-mismatched/writable mounts without changing bytes?',
        acceptance='Exactly four real bwrap controls,no skips,normal0,closed owners. All9read-only inputs pass;missing/hashbad fail8of9;RWfails0of9;no source byte changes. No GPU/network/Orca tasks.',
        passed=False, complete=False, active=False, ownership_closed=True,
        GPU_executed=False, Orca_worker_qualified=False,
        consumer='Fresh Orca Luna research tasks require this source-pinned helper plus separate actual worker-side IPC/lifecycle proof.',
        retention='Keep compact source/recipe/case outcome;retire routine test output and temporary namespaces after verification.')
    ROOT.mkdir(mode=0o700); (ROOT / 'tmp').mkdir(); save(row)


def check():
    row = json.loads((ROOT / 'record.json').read_text()); current(row)
    require(row['stage'] == 'prepared' and not (ROOT / 'attempt.json').exists(), 'one host attempt only')
    write(ROOT / 'attempt.json', dict(source_commit=row['source_commit']))
    owner = ownership.Owner(ROOT / 'owned')
    try:
        entry, out, err = owner.run('filesystem-controls', row['command'], row['environment'], P, **row['budget'])
        text = out.read_text() + err.read_text()
        row['exit_code'] = entry['exit_code']
        require(ownership.closed(entry) and entry['exit_code'] == 0 and 'Ran 4 tests' in text
                and '\nOK\n' in text and 'skipped' not in text, 'all four real namespace controls required')
        row.update(stage='closed', passed=True, complete=True, tests=4, skipped=0,
            cases=['all-nine-readonly-pass', 'missing-input-refused', 'wrong-hash-refused', 'writable-mount-refused'])
    except BaseException as error:
        row.update(stage='failed', passed=False, complete=False, error=repr(error))
        raise
    finally:
        row['active'] = owner.active is not None
        row['ownership_closed'] = owner.active is None and all(ownership.closed(c) for c in owner.commands)
        row['passed'] = bool(row['passed'] and row['ownership_closed'])
        save(row)
    print('Four real filesystem controls pass;Orca IPC/lifecycle still require a fresh worker proof.', flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['check']), 'use prepare|check')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else check()
