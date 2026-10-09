"""Read an already closed native experiment; do not recollect or rewrite its receipt."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import importlib.util
import json
import time
import types

B = Path(__file__).parent
OUT = B / 'gprofng-native-protocol-smoke-v2-r1'
SOURCE = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-gprofng-native-smoke-20261010/bench/results/2026-10-10-gprofng-native-smoke/check_native_protocol_v2.py')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
lock = (B / 'owned-v0141-measurement.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
started = time.monotonic()
receipt_path = OUT / 'record.json'
receipt_sha = sha(receipt_path)
receipt = json.loads(receipt_path.read_text())
assert not receipt['active'] and not receipt['passed'] and not receipt['cleanup'] and not receipt['survivors']
assert receipt['protocol'][-1] == 'BYE'
assert all(s.get('exit_code') == 0 for s in receipt['steps'] if 'argv' in s)
spec = importlib.util.spec_from_file_location('closed_collector_reader', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for row in receipt['owners'].values():
    assert not module.same(row)
evidence = {}
valid = module.Supervisor.evidence(types.SimpleNamespace(out=OUT, record=evidence))
assert not valid
assert evidence['profile_validation']['collector_errors'] == [{'attributes': {'kind': 'cerror', 'id': '9'}, 'text': 'itimer could not be set'}]
assert evidence['profile_validation']['data_file_sizes'] == {'data.frameinfo': 0}
assert not evidence['profile_validation']['ptimer_profiles']
assert evidence['collector_xml']['tail_nul_bytes'] == 63455
assert sha(receipt_path) == receipt_sha
result = {
    'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope': 'CPU-only postprocessing of the single closed native collection. No collector/target rerun, GPU, model or original receipt change.',
    'extractor_sha256': sha(Path(__file__)), 'reader_sha256': sha(SOURCE),
    'original_receipt_sha256': receipt_sha, 'original_passed': False,
    'original_controller_error': receipt['error'], 'original_protocol_complete': True,
    'original_owned_processes_absent': True, 'collector_usable': False,
    'independent_collector_rejection': 'cerror9 itimer could not be set; no ptimer schema and empty frameinfo. Native target avoids Python and application timers.',
    **evidence,
}
raw = (OUT / 'result.er/log.xml').read_bytes()
text = raw.rstrip(b'\0\r\n\t ').decode('utf-8') + '\n'
compact = B / 'gprofng-native-v2-rejection.xml.txt'
with compact.open('x') as stream:
    stream.write(text)
result['compact_xml_sha256'] = sha(compact)
result['elapsed_seconds'] = time.monotonic() - started
assert result['elapsed_seconds'] < 5
with (B / 'gprofng-native-v2-closed-evidence-v1.json').open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(json.dumps({k: result[k] for k in ['original_passed', 'original_protocol_complete', 'collector_usable', 'independent_collector_rejection', 'elapsed_seconds']}))
