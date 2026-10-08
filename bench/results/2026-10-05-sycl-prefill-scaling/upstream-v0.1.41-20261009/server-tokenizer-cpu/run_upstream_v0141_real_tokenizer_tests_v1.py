"""Run the five previously skipped CPU tokenizer tests after quiet GPU timing ends."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

b = Path(__file__).parent
rw = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/fix-sycl-server-env-v0141-2026-10-09')
raw = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/bench-upstream-v0.1.41-20261009')
sequence = b / 'upstream-v0141-tuned-matched32k-comparison-sequence-v1/record.json'
gate = json.loads(sequence.read_text())
assert not gate['active'] and gate['passed'], 'quiet comparison must finish first'
for step in gate['steps']:
    terminal = json.loads(Path(step['receipt']).read_text())
    assert not terminal['active'] and terminal['completed'] and terminal['healthy']
    assert terminal['exit_code'] == 0 and not any(terminal['cleanup'].values())
    for key in ['inferior', 'debugger']:
        identity = terminal[key]
        stat = Path('/proc') / str(identity['pid']) / 'stat'
        if stat.exists():
            current = stat.read_text().rsplit(')', 1)[1].split()
            assert int(current[19]) != identity['start_ticks'], 'owned GPU process still alive'

python = b / 'upstream-refresh-20261007/test-venv/bin/python'
tokenizer = Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s/tokenizer')
assert python.is_file()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
for rel in ['serve/test_detok.py', 'serve/server.py', 'tools/strata_tokenizer.py']:
    assert (rw / rel).read_bytes() == (raw / rel).read_bytes(), rel
fixture_sha = {name: sha(tokenizer / name) for name in ['vocab.json', 'merges.txt', 'token_type.json']}
labels = [
    'serve.test_detok.Detok.test_text_with_split_characters',
    'serve.test_detok.Detok.test_random_ids_and_broken_bytes',
    'serve.test_detok.Detok.test_constant_cost',
    'serve.test_detok.HeapBpe.test_pack_vocab',
    'serve.test_detok.HeapBpe.test_speed_pack',
]
argv = [str(python), '-m', 'unittest', '-v', *labels]
out = b / 'upstream-v0141-real-tokenizer-cpu-tests-v1'
assert not out.exists()
out.mkdir(mode=0o700)
env = dict(os.environ)
env.update(STRATA_TOKENIZER=str(tokenizer), PYTHONDONTWRITEBYTECODE='1')
record = {
    'active': True, 'passed': False,
    'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope': 'Five previously skipped upstream CPU tokenizer cases using the actual model fixture. This does not rerun the593-case suite or launch the model/GPU.',
    'argv': argv, 'cwd': str(rw),
    'environment': {key: env[key] for key in ['STRATA_TOKENIZER', 'PYTHONDONTWRITEBYTECODE']},
    'fixture_sha256': fixture_sha,
    'test_source_sha256': {rel: sha(rw / rel) for rel in ['serve/test_detok.py', 'serve/server.py', 'tools/strata_tokenizer.py']},
    'quiet_sequence_sha256': sha(sequence),
    'gpu_execution': False, 'timeout_seconds': 180,
}
started = time.monotonic()
child = None


def save():
    record['elapsed_seconds'] = time.monotonic() - started
    tmp = out / 'record.json.tmp'
    tmp.write_text(json.dumps(record, indent=2) + '\n')
    tmp.replace(out / 'record.json')


save()
try:
    with (out / 'stdout').open('w') as stdout, (out / 'stderr').open('w') as stderr:
        child = subprocess.Popen(argv, cwd=rw, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        record['pid'] = child.pid
        save()
        try:
            record['exit_code'] = child.wait(timeout=180)
        except subprocess.TimeoutExpired:
            record['timed_out'] = True
            os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
            record['exit_code'] = child.returncode
    text = (out / 'stderr').read_text()
    count = re.search(r'Ran (\d+) tests in', text)
    record['tests_run'] = int(count[1]) if count else None
    record['skipped'] = 0 if 'skipped=' not in text and '... skipped' not in text else None
    assert record['exit_code'] == 0 and record['tests_run'] == 5 and record['skipped'] == 0
    assert all(label.rsplit('.', 1)[-1] in text for label in labels)
    assert all(sha(tokenizer / name) == value for name, value in fixture_sha.items())
    record['passed'] = True
except BaseException as error:
    record['error'] = repr(error)
finally:
    record['active'] = False
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for name in ['stdout', 'stderr']:
        path = out / name
        if path.exists():
            record[name] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    save()
print(json.dumps({key: record.get(key) for key in ['passed', 'tests_run', 'skipped', 'elapsed_seconds', 'exit_code', 'error']}))
if not record['passed']:
    raise SystemExit(1)
