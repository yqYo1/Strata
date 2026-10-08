"""Wait for this owned physical test, then run the guarded quiet comparison."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

BASE = Path(__file__).parent
FULL = BASE / 'owned-upstream-v0141-integrated-full256k-diagnostic-r2/record.json'
DEFAULT = BASE / 'upstream-v0141-matched32k-comparison-sequence-v1/record.json'
CONTROLLER = BASE / 'run_owned_upstream_v0141_full256k_v2.py'
SEQUENCE = BASE / 'run_upstream_v0141_tuned_comparison_sequence_v3.py'
TUNED = BASE / 'upstream-v0141-tuned-matched32k-comparison-sequence-v1/record.json'
SOURCES = {
    CONTROLLER: 'a34cb70ed479a4bea5f9727cb19461d32014d9679d3c4642a6e9cc380d2651a1',
    SEQUENCE: '0d2f9b13b8288b28833ae138ee57b0ff42de52fe0464496be966a94ca38316ff',
    BASE / 'run_owned_upstream_v0141_tuned32k_v3.py': '80fe9ba0ee4127ded95c0fd5a09e25f4ad754badde381253746ea022b04c11ac',
}
sys.path.insert(0, '/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05/sycl/tools')
from owned_gdb import process_identity


def physical_gate(value):
    if value['active']:
        return False
    assert value['completed'] and value['healthy'] and value['math_gate_passed'], 'physical correctness failed'
    assert value['physical256k_sequence_completed'] and value['capacity_sequence_completed'], 'physical lifecycle incomplete'
    assert value['prior_four_fresh32k_heads_state_output_passed'], 'separate four fresh numerical reads incomplete'
    assert value['prior_four_fresh32k_numerical_subset_sha256']=='95ae2a0a7e61f2240f6138f21fdcb35c1a70ec54300b22a7b99fd6b92873b4a7'
    assert value['exit_code'] == 0 and not value['exit_signal']
    assert not value['new_fault_messages'] and not any(value['cleanup'].values())
    return True


def check_sources():
    for path, expected in SOURCES.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
    assert hashlib.sha256(DEFAULT.read_bytes()).hexdigest() == '01383f1d0118250de479af457a0d40ffa0d321943ea543c54d34cf165393caa9'


def main():
    check_sources()
    first = json.loads(FULL.read_text())
    assert first['active'] and first['binary_sha256'] == '86972697ecb1903750d202f8be29a7a1da351eb3415678c3e3b6af790804d1c8'
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert first['boot_id'] == boot
    argv = ['/usr/bin/python3', str(CONTROLLER), 'integrated', 'diagnostic', '2']
    identities = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmd = proc.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
        except (OSError, UnicodeError):
            continue
        if cmd == argv:
            identities.append(process_identity(int(proc.name)))
    assert len(identities) == 1 and identities[0], 'exact owned physical controller not found'
    identity = identities[0]
    assert not TUNED.parent.exists()
    out = BASE / 'upstream-v0141-full-then-tuned-sequence-v2'
    out.mkdir(mode=0o700)
    started = time.monotonic()
    record = {
        'active': True, 'passed': False,
        'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Wait only for the already running exact physical test; start a quiet comparison only after its terminal correctness and process-absence gates. No change to its deadline, process, source, GPU or services.',
        'phase': 'waiting_for_owned_physical', 'wait_deadline_seconds': 11000,
        'full_started_utc': first['started_utc'], 'full_controller_identity': identity,
        'source_sha256': {str(p): h for p, h in SOURCES.items()},
        'supervisor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    child = None

    def save():
        record['elapsed_seconds'] = time.monotonic() - started
        temp = out / 'record.json.tmp'
        temp.write_text(json.dumps(record, indent=2) + '\n')
        temp.replace(out / 'record.json')

    save()
    try:
        while True:
            assert time.monotonic() - started < 11000, 'physical wait deadline reached; do not restart it'
            value = json.loads(FULL.read_text())
            assert value['started_utc'] == first['started_utc'] and value['boot_id'] == boot
            record['full_elapsed_seconds'] = value['elapsed_seconds']
            record['full_active_request'] = value.get('active_request')
            ready = physical_gate(value)
            roles = [identity, value.get('inferior'), value.get('debugger')]
            live = []
            for old in roles:
                if old:
                    now = process_identity(old['pid'])
                    if now and now['start_ticks'] == old['start_ticks']:
                        live.append(old)
            if ready and not live:
                break
            assert live or not value['active'], 'physical controller disappeared without terminal receipt'
            save()
            time.sleep(5)
        check_sources()
        assert not TUNED.parent.exists()
        record['full_terminal_receipt_sha256'] = hashlib.sha256(FULL.read_bytes()).hexdigest()
        record['phase'] = 'tuned_comparison'
        save()
        with (out / 'tuned.stdout').open('w') as stdout, (out / 'tuned.stderr').open('w') as stderr:
            child = subprocess.Popen(['/usr/bin/python3', str(SEQUENCE)], stdout=stdout, stderr=stderr, start_new_session=True)
            record['tuned_controller_identity'] = process_identity(child.pid)
            record['tuned_started_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            end = time.monotonic() + 3700
            save()
            while child.poll() is None:
                assert time.monotonic() < end, 'tuned sequence own deadline did not finish'
                save()
                time.sleep(5)
            record['tuned_exit_code'] = child.returncode
            child = None
        assert record['tuned_exit_code'] == 0
        terminal = json.loads(TUNED.read_text())
        assert terminal['passed'] and not terminal['active'] and len(terminal['steps']) == 4
        record['tuned_terminal_receipt_sha256'] = hashlib.sha256(TUNED.read_bytes()).hexdigest()
        record['phase'] = 'complete'
        record['passed'] = True
    except BaseException as error:
        record['error'] = repr(error)
        if child and child.poll() is None:
            # Only this supervisor's quiet sequencer; the physical test is untouched.
            os.kill(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=90)
            except subprocess.TimeoutExpired:
                record['tuned_cleanup_pending'] = process_identity(child.pid)
    finally:
        record['active'] = False
        record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
    print(json.dumps(record, ensure_ascii=False))
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
