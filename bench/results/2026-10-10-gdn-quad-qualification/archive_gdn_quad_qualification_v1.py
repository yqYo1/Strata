from pathlib import Path
import hashlib, json, shutil, subprocess, re, fcntl

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-gdn-quad-pipeline-20261010')
A = W / 'bench/results/2026-10-10-gdn-quad-qualification'
lock = (B / 'owned-v0141-measurement.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not list(A.iterdir()), 'Only resume the empty root-owned archive staging directory'

def identity(p):
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}

def save(n, d):
    (A / n).write_text(json.dumps(d, indent=2, sort_keys=True) + '\n')

records, raw = {}, []
for n in ['gdn-quad-cpu-build-v1', 'gdn-quad-cpu-build-v2', 'gdn-quad-initial-gates-v1', 'gdn-quad-prefix-carry-gates-v1']:
    p = B / n
    d = json.loads((p / 'record.json').read_text())
    assert not d['active'] and not d['survivors'] and not d['cleanup'] and d['exit_pin_gate_passed']
    if n.endswith('build-v1'):
        assert not d['complete'] and not d['passed']  # preserve original failure
    else:
        assert d['complete'] and d['passed']
    records[n] = d
    shutil.copyfile(p / 'record.json', A / (n + '-record.json'))
    raw.extend(identity(x) for x in p.iterdir() if x.is_file())
for n in ['build_gdn_quad_cpu_v1.py', 'build_gdn_quad_cpu_v2.py', 'run_gdn_quad_initial_gates_v1.py', 'run_gdn_quad_prefix_carry_v1.py']:
    shutil.copyfile(B / n, A / n)
for n in ['compile_commands.json', 'CMakeCache.txt']:
    p = W / 'build-sycl-gdn-quad-v2' / n
    assert identity(p)['sha256'] == records['gdn-quad-cpu-build-v2']['files'][n]['sha256']
    shutil.copyfile(p, A / n)
for n in ['compiler.stdout', 'configure.stdout', 'direct-imports.stdout', 'target-commands.stdout']:
    shutil.copyfile(B / 'gdn-quad-cpu-build-v2' / n, A / n)
for n in ['host-contract.stdout', 'host-fail-stop.stderr', 'gpu-short.stdout']:
    shutil.copyfile(B / 'gdn-quad-initial-gates-v1' / n, A / n)
for n in ['prefix-32768.stdout', 'prefix-262144.stdout']:
    shutil.copyfile(B / 'gdn-quad-prefix-carry-gates-v1' / n, A / n)

trace = B / 'gdn-quad-initial-gates-v1/gpu-short.stderr'
lines = trace.read_text().splitlines()
handles, properties, excerpts = {}, {}, {}
for i, line in enumerate(lines, 1):
    m = re.search(r'SUCCESS .* in zeKernelCreate\(.*pKernelName="([^"]+)".*phKernel=(0x[0-9a-f]+)', line)
    if m:
        handles[m.group(2)] = m.group(1)
    m = re.search(r'SUCCESS .* in zeKernelGetProperties\(hKernel=(0x[0-9a-f]+),', line)
    if m and m.group(1) in handles:
        name = handles[m.group(1)]
        fields = {k: int(v) for k, v in re.findall(r'(requiredSubgroupSize|maxSubgroupSize|maxNumSubgroups|localMemSize|privateMemSize|spillMemSize)=([0-9]+)', line)}
        if name not in properties:
            properties[name] = {'fields': fields, 'trace_line': i, 'handle': m.group(1)}
        else:
            assert properties[name]['fields'] == fields
for start, end in [(1, 20), (1098, 1134), (1230, 1250), (len(lines)-49, len(lines))]:
    for i in range(start, end+1):
        excerpts[i] = lines[i-1]
assert properties['_ZTSN6strata7prefill22GdnRecQuadPipelineSG32E']['fields']['spillMemSize'] == 5056
assert properties['_ZTSN6strata7prefill26GdnLegacyPipelineReferenceE']['fields']['spillMemSize'] == 0
(A / 'gpu-short-diagnostic-excerpts.txt').write_text('Raw identity: ' + json.dumps(identity(trace)) + '\n' + '\n'.join(f'{i}: {line}' for i, line in sorted(excerpts.items())) + '\n')
save('observed-kernel-properties.json', {'raw_trace': identity(trace), 'kernels': properties, 'method': 'successful zeKernelCreate name/handle correlated with successful properties; all repeated fields agree', 'runtime_cost_known': False, 'native_isa_available': False})
host = {}
for n in ['host-contract.trace', 'host-fail-stop.trace']:
    p = B / 'gdn-quad-initial-gates-v1' / n
    ls = p.read_text().splitlines()
    hits = [{'line': i, 'text': s} for i, s in enumerate(ls, 1) if re.search(r'/dev/(?:dri|accel)/', s)]
    assert not hits
    host[n] = {'raw_trace': identity(p), 'device_path_mentions': hits, 'first_lines': ls[:3], 'last_lines': ls[-3:], 'scope': 'observable strace open/openat/openat2/ioctl trace; source early path also reviewed, not proof of all possible hardware probing'}
save('host-device-open-summary.json', host)
for n in ['gdn-quad-cpu-build-v1', 'gdn-quad-cpu-build-v2']:
    p = B / n / 'build.stdout'
    ls = p.read_text().splitlines()
    selected = set(range(min(4, len(ls)))) | set(range(max(0, len(ls)-8), len(ls)))
    for i, s in enumerate(ls):
        if re.search(r'fatal error|FAILED:|error:|oneMKL|mkl.*not|not.*mkl', s, re.I):
            selected.update(range(max(0, i-3), min(len(ls), i+5)))
    (A / (n + '-build-excerpts.txt')).write_text('Raw identity: ' + json.dumps(identity(p)) + '\n' + '\n'.join(f'{i+1}: {ls[i]}' for i in sorted(selected)) + '\n')
journal = []
for start, end in [('2026-10-10 08:14:35', '2026-10-10 08:15:00'), ('2026-10-10 08:19:20', '2026-10-10 08:20:20')]:
    argv = ['journalctl', '-k', '--since', start, '--until', end, '--no-pager']
    p = subprocess.run(argv, text=True, capture_output=True, timeout=15)
    assert p.returncode == 0
    hits = [s for s in p.stdout.splitlines() if re.search(r'\b(?:xe|i915|drm)\b|gpu.*(?:hang|fault|reset)|devcoredump', s, re.I)]
    journal.append({'argv': argv, 'exit_code': p.returncode, 'stdout_sha256': hashlib.sha256(p.stdout.encode()).hexdigest(), 'gpu_related_visible_entries': hits, 'stderr': p.stderr, 'scope': 'entries visible to uid, not comprehensive driver health proof'})
save('post-run-visible-kernel-journal.json', journal)
save('raw-file-identities.json', raw)
save('qualification-summary.json', {
    'source_head': '9edf4a5941f029357b2f2d0b19c19bdff4638c06',
    'binary_sha256': records['gdn-quad-cpu-build-v2']['binary_sha256'],
    'receipt_identities': {n: identity(B / n / 'record.json') for n in records},
    'all_closed': True, 'no_cleanup_or_survivors': True,
    'build_v1_original_status': {'complete': False, 'passed': False, 'reason': 'oneMKL include paths omitted in controller minimal environment'},
    'build_v2': {'passed': True, 'fresh_directory': True, 'common_translation_units_same_flags': 158},
    'short': {'passed': True, 'cases': 24, 'positive_candidate_calls': 72, 'explicit_pre_denial_fallback': 1, 'full_state_active_FP32_FP16_bitwise': True, 'guards': True},
    'prefixes': records['gdn-quad-prefix-carry-gates-v1']['prefixes'],
    'model_full_context_lifecycle_passed': False, 'performance_eligible': False, 'adopted': False,
    'observed_spill': {'quad': 5056, 'legacy': 0, 'runtime_cost': 'unknown'},
    'next_gate': 'quiet repeated >=32768 synthetic component comparison, then separate real-model/full physical context/decode gates'
})
(A / 'README.md').write_text('''# GDN quad initial qualification on Arc B570

Root built source 9edf4a5941f029357b2f2d0b19c19bdff4638c06 using IntelLLVM 2026.1, precise FP, SPIR64, default subgroup 32 and per-kernel splitting. All 158 common translation-unit flags match the earlier production recipe. Build v1 failed because the minimal environment omitted oneMKL include paths; its original incomplete/failed receipt is preserved. Corrected v2 used a fresh directory and passed.

Host-only and injected fail-stop probes passed with expected exits 0 and 2, with no destructor marker in the latter and no GPU device-path mentions in the observable strace. The GPU diagnostic suite passed 24 cases, 72 positive SG32 candidate calls and one explicit pre-denial fallback, with bitwise full state and active FP32/FP16 output plus guards. Synthetic 32768 and 262144 prefixes with 2048-token chunk buffers passed 16 and 128 candidate calls, comparing every carry. All runs closed normally with no cleanup or survivors.

These recurrence/norm checks are not model context-capacity, cache/restore, cancellation or inference-lifecycle tests. They are not timings and do not justify adoption. The exact candidate reports local/private/spill memory 1024/0/5056, versus 1536/0/0 for reference. Compiler spill allocation is not measured traffic or latency. Experimental GRF-256 remains disabled; quiet repeated >=32K service comparisons and native-code investigation remain open.

Original controllers and receipts, environments, complete commands, compiler/runtime/source/binary identities and individual correctness outputs are committed. Repetitive successful API histories and compiler warnings are represented by raw path/hash/byte manifests and compact excerpts. Full short API history remains outside Git for the current native-artifact investigation; retention is reviewed when that consumer finishes. Production defaults remain unchanged.
''')
shutil.copyfile(Path(__file__), A / Path(__file__).name)
save('archive-file-identities.json', {str(p.relative_to(A)): {'sha256': identity(p)['sha256'], 'bytes': p.stat().st_size} for p in A.rglob('*') if p.is_file()})
for argv in [['git', 'diff', '--check'], ['git', 'add', 'bench/results/2026-10-10-gdn-quad-qualification'], ['git', 'commit', '-m', 'test(sycl): record quad synthetic GPU qualification'], ['git', 'push', 'origin', 'perf/sycl-gdn-quad-pipeline-20261010']]:
    p = subprocess.run(argv, cwd=W, text=True, capture_output=True)
    print(p.stdout, p.stderr, flush=True)
    assert p.returncode == 0, argv
print(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip())
print('Archive bytes', sum(p.stat().st_size for p in A.rglob('*') if p.is_file()))
