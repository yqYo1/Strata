#!/usr/bin/env python3
"""Main's accounting scope for one already identified native Orca worker.

Authored independently from the installed systemd D-Bus signature and local
systemd.scope/resource-control manuals. Repository LICENSE. No launch wrapper,
agent code, credentials, GPU operation, or modification of another application.
Memory/accounting semantics: https://docs.kernel.org/admin-guide/cgroup-v2.html
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shlex
import subprocess
import sys
import time

P = Path(__file__).resolve().parents[2]
THIS = 'tools/research/orca_scope.py'
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/review')
REPORT_ROOTS = (B, B.with_name('research'))
LIMITS = dict(wall_seconds=600, CPU_seconds=180, memory_bytes=4 << 30,
              captured_text_bytes=8 << 20, tasks=256)


def require(ok, message):
    if not ok: raise RuntimeError(message)


def control(root):
    """Keep Main's inputs/evidence outside the worker's writable report root."""
    path = root.with_name(root.name + '-control')
    require(path.is_dir() and not path.is_symlink(), 'existing private Main control directory required')
    return path


def run(args):
    result = subprocess.run(args, check=True, capture_output=True, timeout=20)
    require(len(result.stdout) + len(result.stderr) < 65536, 'bounded control response required')
    return result.stdout.decode()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_pin():
    original = subprocess.check_output(['git', '-C', str(P), 'show', 'HEAD:' + THIS], timeout=20)
    require(original == (P / THIS).read_bytes(), 'commit the exact accounting controller first')
    return hashlib.sha256(original).hexdigest()


def stat(pid):
    text = Path(f'/proc/{pid}/stat').read_text()
    tail = text[text.rindex(') ') + 2:].split()
    return dict(pid=pid, parent=int(tail[1]), sid=int(tail[3]), start_ticks=int(tail[19]))


def identity(expected):
    pid = expected['pid']
    require(type(pid) is int and pid > 1 and pid != os.getpid(), 'named owned native PID required')
    actual = stat(pid)
    require(actual['start_ticks'] == expected['start_ticks'] and actual['sid'] == expected['sid']
            and Path(f'/proc/{pid}').stat().st_uid == os.geteuid(), 'native identity/owner changed')
    argv = Path(f'/proc/{pid}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    exe = Path(f'/proc/{pid}/exe').resolve()
    require(argv == expected['argv'] and exe == Path(expected['executable']['path']).resolve()
            and sha(exe) == expected['executable']['sha256']
            and exe.name == 'codex', 'plain pinned native Codex process required')
    return actual


def children(root):
    census = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try: census[int(p.name)] = stat(int(p.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError): pass
    selected = {root}
    while True:
        added = {p for p, row in census.items() if row['parent'] in selected} - selected
        if not added: break
        selected.update(added)
    require(root in census and len(selected) <= LIMITS['tasks'], 'bounded live native family required')
    require(all(Path(f'/proc/{pid}').stat().st_uid == os.geteuid() for pid in selected),
            'foreign process in native family')
    return [census[p] for p in sorted(selected)]


def properties(unit):
    names = ['ControlGroup', 'MemoryMax', 'MemorySwapMax', 'CPUAccounting',
             'MemoryAccounting', 'RuntimeMaxUSec', 'TasksMax', 'ActiveState']
    args = ['systemctl', '--user', 'show', unit]
    for name in names: args += ['--property', name]
    props = dict(line.split('=', 1) for line in run(args).splitlines())
    manager = ['busctl', '--user', 'call', 'org.freedesktop.systemd1',
               '/org/freedesktop/systemd1', 'org.freedesktop.systemd1.Manager']
    path = shlex.split(run([*manager, 'GetUnit', 's', unit]))
    require(len(path) == 2 and path[0] == 'o', 'actual scope object path required')
    # systemctl prints this duration in human units; use the typed D-Bus value.
    runtime = shlex.split(run(['busctl', '--user', 'get-property',
        'org.freedesktop.systemd1', path[1], 'org.freedesktop.systemd1.Scope', 'RuntimeMaxUSec']))
    require(len(runtime) == 2 and runtime[0] == 't', 'typed runtime limit required')
    props['RuntimeMaxUSec'] = runtime[1]
    return props


def cgroup(row):
    props = properties(row['unit'])
    require(props['ActiveState'] == 'active'
            and int(props['MemoryMax']) == LIMITS['memory_bytes']
            and int(props['MemorySwapMax']) == 0
            and int(props['RuntimeMaxUSec']) == LIMITS['wall_seconds'] * 1000000
            and int(props['TasksMax']) == LIMITS['tasks']
            and props['CPUAccounting'] == props['MemoryAccounting'] == 'yes',
            'actual scope limits/accounting differ')
    relative = Path(props['ControlGroup'])
    require(relative.is_absolute() and relative.name == row['unit']
            and '..' not in relative.parts, 'owned scope cgroup required')
    root = Path('/sys/fs/cgroup') / str(relative).lstrip('/')
    require(root.resolve().is_relative_to(Path('/sys/fs/cgroup')), 'cgroup path escaped')
    identity(row['identity'])
    line = Path(f'/proc/{row["identity"]["pid"]}/cgroup').read_text().splitlines()
    require(line == ['0::' + str(relative)], 'native process is outside its accounting scope')
    return root, props


def attach(root, expected_path, unit):
    require(re.fullmatch(r'strata-research-[a-z0-9-]+\.scope', unit), 'fresh named research scope required')
    private = control(root)
    require(expected_path.is_relative_to(private) and not (private / 'scope.json').exists(),
            'private Main identity input and one fresh scope attempt required')
    controller = source_pin()
    expected = json.loads(expected_path.read_text())
    require(expected['argv'][expected['argv'].index('--cd') + 1] == str(root), 'named report cwd required')
    require('--no-daemon' in expected['argv'] and '--strict-config' in expected['argv'],
            'owned native runtime, without a shared daemon, required')
    identity(expected)
    family = children(expected['pid'])
    row = dict(unit=unit, identity=expected, initial_family=family, limits=LIMITS,
               controller_sha256=controller, identity_file_sha256=sha(expected_path),
               started_wall=time.time(), started_monotonic=time.monotonic(),
               source='Main-owned;installed systemd Manager D-Bus and local scope/resource-control manuals')
    # The caller closes this exact worker if creation or verification fails.
    (private / 'scope.json').write_text(json.dumps(row, indent=2) + '\n')
    for member in family:
        require(stat(member['pid']) == member, 'native family changed before CPU limit')
        resource.prlimit(member['pid'], resource.RLIMIT_CPU,
                         (LIMITS['CPU_seconds'], LIMITS['CPU_seconds']))
        require(resource.prlimit(member['pid'], resource.RLIMIT_CPU) ==
                (LIMITS['CPU_seconds'], LIMITS['CPU_seconds']), 'actual native CPU limit differs')
    props = [('PIDs', 'au', [str(len(family)), *[str(p['pid']) for p in family]]),
             ('Description', 's', ['Strata bounded report-only research worker']),
             ('CPUAccounting', 'b', ['true']), ('MemoryAccounting', 'b', ['true']),
             ('MemoryMax', 't', [str(LIMITS['memory_bytes'])]),
             ('MemorySwapMax', 't', ['0']), ('TasksMax', 't', [str(LIMITS['tasks'])]),
             ('RuntimeMaxUSec', 't', [str(LIMITS['wall_seconds'] * 1000000)])]
    args = ['busctl', '--user', 'call', 'org.freedesktop.systemd1',
            '/org/freedesktop/systemd1', 'org.freedesktop.systemd1.Manager',
            'StartTransientUnit', 'ssa(sv)a(sa(sv))', unit, 'fail', str(len(props))]
    for name, signature, values in props: args += [name, signature, *values]
    args += ['0']
    row['start_transient_reply'] = run(args).strip()
    group, actual = cgroup(row)
    settled = children(expected['pid'])
    require({int(n) for n in (group / 'cgroup.procs').read_text().split()} >=
            {p['pid'] for p in settled}, 'native family escaped accounting')
    row['settled_family'] = settled
    row['actual_properties'] = actual
    # The pids controller counts threads as well as processes. Native Codex's
    # code-mode host can exhaust64even when the process-family count is small.
    row['initial_pids'] = {name: (group / name).read_text().strip()
                           for name in ('pids.current', 'pids.peak', 'pids.max', 'pids.events')}
    require(int(row['initial_pids']['pids.max']) == LIMITS['tasks']
            and int(row['initial_pids']['pids.peak']) <= LIMITS['tasks']
            and dict(line.split() for line in row['initial_pids']['pids.events'].splitlines())['max'] == '0',
            'thread/process quota was already reached before task injection')
    row['accounting_ready'] = True
    (private / 'scope.json').write_text(json.dumps(row, indent=2) + '\n')
    print('Owned native scope attached;task execution and acceptance remain unqualified.')


def sample(root, session=None):
    private = control(root)
    row = json.loads((private / 'scope.json').read_text())
    require(row['controller_sha256'] == source_pin() and row['limits'] == LIMITS
            and row.get('accounting_ready'), 'original accounting inputs changed/incomplete')
    group, props = cgroup(row)
    cpu = dict(line.split() for line in (group / 'cpu.stat').read_text().splitlines())
    events = dict(line.split() for line in (group / 'memory.events').read_text().splitlines())
    pids = {name: (group / name).read_text().strip()
            for name in ('pids.current', 'pids.peak', 'pids.max', 'pids.events')}
    peak = int((group / 'memory.peak').read_text())
    current = int((group / 'memory.current').read_text())
    pid = row['identity']['pid']
    status = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines())
    high_water = status['VmHWM'].split()
    require(len(high_water) == 2 and high_water[1] == 'kB', 'kernel native RSS high-water required')
    native_peak = int(high_water[0]) * 1024
    family_ticks, observed_RSS = 0, 0
    for child in (group / 'cgroup.procs').read_text().split():
        try:
            text = Path(f'/proc/{child}/stat').read_text()
            tail = text[text.rindex(') ') + 2:].split()
            family_ticks += sum(max(0, int(v)) for v in tail[11:15])
            child_status = dict(line.split(':', 1) for line in
                Path(f'/proc/{child}/status').read_text().splitlines())
            observed_RSS += int(child_status['VmRSS'].split()[0]) * 1024
        except (FileNotFoundError, ProcessLookupError): pass
    scope_CPU = int(cpu['usage_usec']) / 1000000
    # Includes pre-scope native work; deliberately double-counts post-move work.
    # Read after worker_done when tool children have settled. It is a budget
    # upper bound, not a CPU attribution or throughput measurement.
    CPU_bound = scope_CPU + family_ticks / os.sysconf('SC_CLK_TCK')
    uptime = float(Path('/proc/uptime').read_text().split()[0])
    native_age = uptime - row['identity']['start_ticks'] / os.sysconf('SC_CLK_TCK')
    wall = max(time.monotonic() - row['started_monotonic'], native_age)
    files = []
    for folder in (root, private):
        for p in folder.rglob('*'):
            require(not p.is_symlink(), 'report capture must not follow symlinks')
            if p.is_file() and p.name != 'resource-snapshot.json':
                files.append(dict(path=str(p), bytes=p.stat().st_size))
            require(len(files) <= 256, 'report file-count budget exceeded')
    if session is not None:
        sessions = Path.home() / '.codex/sessions'
        require(session.resolve().is_relative_to(sessions.resolve()) and session.suffix == '.jsonl',
                'only this native session capture may be counted')
        with session.open('rb') as stream:
            raw = stream.readline((1 << 20) + 1)
        require(len(raw) <= 1 << 20, 'bounded native session metadata required')
        meta = json.loads(raw)
        require(meta['type'] == 'session_meta' and meta['payload']['cwd'] == str(root),
                'session belongs to another task')
        files.append(dict(path=str(session), bytes=session.stat().st_size))
    snapshot = dict(unit=row['unit'], sampled_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        wall_seconds=wall, CPU_seconds_conservative_bound=CPU_bound,
        scope_CPU_seconds=scope_CPU,
        cgroup_memory_peak_bytes=peak, cgroup_memory_current_bytes=current,
        native_process_peak_RSS_bytes=native_peak, observed_current_tree_RSS_bytes=observed_RSS,
        aggregate_tree_RSS_peak_qualified=False,
        pids=pids,
        memory_events={k: int(v) for k, v in events.items()}, captured_files=files,
        native_session_counted=session is not None, actual_properties=props,
        meaning='Kernel-reported native VmHWM is the native-process RSS high-water. '
                'The current tree RSS is one observation,not its peak. Scope memory is charged '
                'memory:process migration/shared mappings prevent treating it as an RSS bound. '
                'No task/worker outcome or filesystem-isolation qualification follows.')
    # Conservatively include this new snapshot itself in the capture total.
    captured = sum(v['bytes'] for v in files)
    snapshot['captured_text_bytes_upper_bound'] = captured + 65536
    require(0 <= current <= peak <= LIMITS['memory_bytes']
            and CPU_bound <= LIMITS['CPU_seconds']
            and native_peak <= LIMITS['memory_bytes'] and observed_RSS <= LIMITS['memory_bytes']
            and wall <= LIMITS['wall_seconds']
            and int(pids['pids.max']) == LIMITS['tasks']
            and 0 <= int(pids['pids.current']) <= int(pids['pids.peak']) <= LIMITS['tasks']
            and dict(line.split() for line in pids['pids.events'].splitlines())['max'] == '0'
            and captured + 65536 <= LIMITS['captured_text_bytes']
            and all(events.get(k) == '0' for k in ('oom', 'oom_kill', 'oom_group_kill')),
            'resource budget failed;close only this owned worker and preserve failure')
    raw = json.dumps(snapshot, indent=2) + '\n'
    require(len(raw.encode()) <= 65536, 'bounded snapshot required')
    (private / 'resource-snapshot.json').write_text(raw)
    print(json.dumps({k: snapshot[k] for k in ('unit', 'wall_seconds', 'CPU_seconds_conservative_bound',
        'native_process_peak_RSS_bytes', 'cgroup_memory_peak_bytes',
        'captured_text_bytes_upper_bound', 'native_session_counted')}))


if __name__ == '__main__':
    require(len(sys.argv) >= 3 and sys.argv[1] in ('attach', 'sample'),
            'attach REPORT_ROOT IDENTITY_JSON UNIT | sample REPORT_ROOT [NATIVE_SESSION_JSONL]')
    root = Path(sys.argv[2]).resolve()
    require(any(root.is_relative_to(base) and root != base for base in REPORT_ROOTS)
            and root.is_dir(), 'private existing report root required')
    if sys.argv[1] == 'attach':
        require(len(sys.argv) == 5, 'exact named identity and scope required')
        attach(root, Path(sys.argv[3]).resolve(), sys.argv[4])
    else:
        require(len(sys.argv) in (3, 4), 'one optional exact native session capture')
        sample(root, Path(sys.argv[3]).resolve() if len(sys.argv) == 4 else None)
