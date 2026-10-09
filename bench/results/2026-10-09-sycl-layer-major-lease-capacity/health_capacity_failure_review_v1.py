#!/usr/bin/env python3
"""Root-owned read-only admission after the pinned C capacity failure. No reset."""
from pathlib import Path
import ast
import datetime
import fcntl
import hashlib
import json
import os
import re
import subprocess
import textwrap
import types

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
C = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-layer-major-streamed-kv-v0141-20261009')
O = C.parent / 'sync-upstream-2026-10-05'
OUT = B / 'streamed-kv-capacity-failure-safety-review-v1'


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def current_identity(pid):
    p = Path('/proc') / str(pid) / 'stat'
    if not p.exists(): return None
    fields = p.read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=int(fields[19]), state=fields[0])


lock = (B / 'owned-v0141-measurement.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
active = subprocess.run(['ps', '-C', 'strata', '-C', 'gdb', '-C', 'ninja', '-C', 'icpx', '-C', 'zstd', '-o', 'comm='], capture_output=True, text=True)
assert not active.stdout.strip(), active.stdout
boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert boot == '0a908c16-e292-4299-80ff-082da39f4bb9'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C, text=True).strip() == '6cdf81078128488e5797afb9e92824b7ebe51124'
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C).strip()
binary = C / 'build-sycl-layer-major-streamed-kv-v0141-v1/strata'
assert sha(binary) == '66d5bee60f4bbdb6f710b395ae11a1e6e14a61e04db417bcd6092d4f7956377e'
assert sha(C / 'sycl/src/prefill/prefill.cpp') == '92b849df3e19ad77a8a56927cae48cd91d4be0f73f4edc1ff189d2177c0cd8c1'
failed_path = B / 'owned-layer-major-streamed-kv-v0141-code32k-streamed-diagnostic-r1/record.json'
assert sha(failed_path) == '9f76562cc4ced2a60363e7b01b31cfd6da4bbfde8abc62c45c0abe737027f0a1'
failed = json.loads(failed_path.read_text())
assert failed['boot_id'] == boot
assert failed['active'] is False and failed['completed'] is False and failed['healthy'] is False and failed['math_gate_passed'] is False
assert failed['cleanup'] == dict(forced=True, inferior_survived=False, gdb_survived=False)
assert not failed['new_fault_messages'] and len(failed['requests']) == 2
a, b = failed['requests']
assert a['name'] == 'A-read0' and a['math_gate_passed'] and len(a['ids']) == len(a['logprobs']) == 64
assert all(a['validation'].values()) and all(a['head_and_live_state_comparison'].values())
assert all(a['qualified_physical_reference_output_comparison'].values())
assert not a['live_prefill_comparison']['different_live_parts']
error = 'ERR ExpertCache: 1363148800 slots x 1 B = 1.27 GiB, but only 0.99 GiB of VRAM is free (8.94 GiB of 9.93 GiB total). Lower --expert-cache.'
assert b['name'] == 'B-read1' and not b['ids'] and not b['logprobs'] and b['protocol'] == ['RESUME 0', error]
assert failed['error'] == repr(RuntimeError(error))
for role in ['inferior', 'debugger']:
    old = failed[role]; now = current_identity(old['pid'])
    assert now is None or now['start_ticks'] != old['start_ticks']
trace = failed_path.parent / 'debugger/inferior.stderr.zst'
assert sha(trace) == '5cb974ea27da9e354f2fb63ea7da038d519646a922ca1d97050ad540775f6aec'
cursor = next(x['argv'][x['argv'].index('--after-cursor') + 1] for x in failed['steps'] if x['label'] == 'kernel-after')
helper = O / 'sycl/tools/recover-xe.sh'
assert sha(helper) == '2f90005f79d2309747d917cff51dc53d8ee6192d68cb1feca074f84af8b0796c'
probe = Path('/home/yayoi/.local/bin/strata-xe-health')
assert sha(probe) == '7375211a3a97a2b284215dcdad6ba652fc9af2fa2e1a92bc47361e3893931d3b'
assert not OUT.exists()
OUT.mkdir(mode=0o700); probes = OUT / 'probes'; probes.mkdir()
source = helper.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
m = types.ModuleType('read_only_health')
exec(compile(source, str(helper), 'exec'), m.__dict__)
# Add a byte budget to the original owned-process polling loop, retaining its
# TERM/KILL deadlines, process-group ownership, survivor checks and error status.
cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Runner')
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
run = textwrap.dedent(ast.get_source_segment(source, method))
old_loop = 'while child.poll() is None and time.monotonic() < deadline:'
assert run.count(old_loop) == 1
run = run.replace(old_loop, 'while child.poll() is None and time.monotonic() < deadline and sum(p.stat().st_size for p in self.output.iterdir() if p.is_file()) < 64 * 1024**2:')
exec(compile(run, str(Path(__file__)), 'exec'), m.__dict__)
m.Runner.run = m.run
r = m.Runner(probes)
record = dict(active=True, passed=False, healthy=False, gpu_executed=False,
              started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), boot_id=boot,
              failed_receipt_sha256=sha(failed_path), source_head=failed['commit'], binary_sha256=sha(binary),
              preserved_failure_status=dict(completed=False, healthy=False, math_gate_passed=False),
              owned_failed_processes_absent=True, failed_first_A_math_passed=True,
              prior_capacity_failure_exact=True, prior_kernel_after_cursor=cursor,
              full_trace_archive_sha256=sha(trace), helper_sha256=sha(helper), probe_sha256=sha(probe),
              controller_sha256=sha(Path(__file__)), raw_log_budget_bytes=64 * 1024**2,
              recovery_reset_service_actions=False, steps=[])


def save():
    record['steps'] = r.calls
    tmp = OUT / 'record.json.tmp'; tmp.write_text(json.dumps(record, indent=2) + '\n'); tmp.replace(OUT / 'record.json')


save()
try:
    gap = r.run('kernel-gap', ['/usr/bin/journalctl', '-k', '--after-cursor', cursor, '--no-pager', '-o', 'json'], seconds=5)
    rows = [json.loads(x) for x in gap.splitlines() if x.startswith('{')]
    faults = [x['MESSAGE'] for x in rows if ((('0000:05:00.0' in x.get('MESSAGE', '') or re.search(r'\bxe\b', x.get('MESSAGE', ''))) and m.FAULT.search(x.get('MESSAGE', ''))) or ('strata' in x.get('MESSAGE', '') and 'segfault' in x.get('MESSAGE', '')))]
    record['preflight_fault_messages'] = faults
    assert not faults
    assert r.run('embedding-state', ['/usr/bin/systemctl', '--user', 'show', 'llama-server-qwen3embed.service', '-p', 'ActiveState'], seconds=5).strip() == 'ActiveState=inactive'
    os.environ.update(NEOReadDebugKeys='1', EnableDirectSubmission='0')
    record['gpu_executed'] = True; save()
    m.check_health(types.SimpleNamespace(bdf='0000:05:00.0'), r, probe)
    record['health_kernel_cursor'] = next(x['argv'][x['argv'].index('--after-cursor') + 1] for x in r.calls if x['label'] == 'health-kernel')
    assert all(x['exit_code'] == 0 and not x['timed_out'] and not x['still_alive'] for x in r.calls)
    assert all(current_identity(x['pid']) is None for x in r.calls)
    record['passed'] = record['healthy'] = True
except BaseException as e:
    record['error'] = repr(e)
finally:
    record['active'] = False
    record['log_bytes'] = sum(p.stat().st_size for p in probes.iterdir() if p.is_file())
    record['log_budget_exceeded'] = record['log_bytes'] >= record['raw_log_budget_bytes']
    if record['log_budget_exceeded']: record['passed'] = record['healthy'] = False
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(); fcntl.flock(lock, fcntl.LOCK_UN); lock.close()
print(json.dumps(record, indent=2))
raise SystemExit(0 if record['passed'] else 1)
