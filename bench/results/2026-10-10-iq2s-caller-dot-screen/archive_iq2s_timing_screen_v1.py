"""Preserve closed CPU screen evidence without duplicating numerical payload logs."""
from pathlib import Path
import csv, datetime, fcntl, hashlib, json, re, shutil, statistics
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-iq2s-actual-cohort-timing-20261010')
P = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-iq2s-nt1-index-spread-20261010')
A = W / 'bench/results/2026-10-10-iq2s-caller-dot-screen'
def identity(p):
    p = Path(p)
    return dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2) + '\n')
def copy(p, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, dest)
    assert identity(p) == identity(dest)
def check_closed(r):
    assert r['passed'] and r['complete'] and not r['active']
    assert not r['cleanup'] and not r['survivors']
    assert not r['gpu_work_submitted'] and not r['inference_run']
    assert all(c['exit_code'] == 0 for c in r['commands'])
with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not A.exists()
    A.mkdir(parents=True)
    names = ['iq2s-timing-release-build-v2', 'iq2s-timing-asan-build-v2',
             'iq2s-timing-correctness-r1-v1', 'iq2s-timing-asan-r1-v1',
             'iq2s-timing-linked-v1'] + [f'iq2s-timing-timing-r{i}-v1' for i in (1,2,3)]
    receipts = {}
    for name in names:
        r = json.loads((B / name / 'record.json').read_text())
        check_closed(r)
        receipts[name] = identity(B / name / 'record.json')
        copy(B / name / 'record.json', A / name / 'record.json')
    write(A / 'receipt-identities.json', receipts)
    for name in names[:2]:
        d = B / name
        r = json.loads((d / 'record.json').read_text())
        build = Path(r['binary']['path']).parent
        assert identity(r['binary']['path']) == {k:r['binary'][k] for k in ('bytes','sha256')}
        for f in ('compile_commands.json', 'CMakeCache.txt'):
            copy(build / f, A / name / f)
        for f, x in r['logs'].items():
            assert identity(d / f) == x
            lines = (d / f).read_text().splitlines()
            excerpt = lines if len(lines) <= 10 else lines[:5] + ['[successful repetitive build lines omitted; full identity in receipt]'] + lines[-5:]
            (A / name / (f + '.excerpt')).write_text('\n'.join(excerpt) + ('\n' if excerpt else ''))
    for p in sorted((B / 'iq2s-timing-linked-v1').glob('*.txt')):
        copy(p, A / 'linked' / p.name)
    copy(B / 'research-source-snapshots/iq2s-actual-timing-source-v2/manifest.json', A / 'source-pins.json')
    for f in ('build_iq2s_timing_v2.py', 'run_iq2s_timing_v1.py', 'run_iq2s_actual_cohort_correctness_v1.py',
              'run_native_service_calibration_cpu_v4.py', 'build_iq2s_actual_cohort_cpu_v1.py'):
        copy(B / f, A / 'controllers' / f)
    base = P / 'bench/results/2026-10-10-iq2s-actual-cohort-qualification/iq2s-actual-cohort-correctness-v1/complete-numerical-records.csv'
    assert identity(base)['sha256'] == '59c9013d1b6392279f9aad3a4b07ab3e625fa9a6bad2d07879d50d8a0dca2f17'
    release = B / 'iq2s-timing-correctness-r1-v1/owned-sample/stdout.csv'
    assert identity(release) == identity(base)
    prefix = lambda lines: [s for s in lines if s.split(',')[0] in ('ID', 'EXTENT', 'INDEX_ID', 'INDEX_SPLIT', 'INDEX_COMPLETE')]
    common = prefix(base.read_text().splitlines())
    projected = {}
    for name in names[3:4] + names[5:]:
        src = B / name / 'owned-sample/stdout.csv'
        lines = src.read_text().splitlines()
        assert prefix(lines) == common
        dest = A / name / 'samples-and-metadata.csv'
        dest.write_text('\n'.join(s for s in lines if s not in common) + '\n')
        samples = [s for s in lines if s.startswith('INDEX_TIMING,')]
        assert len(samples) == 252
        projected[name] = dict(original=identity(src), projection=identity(dest), samples=252,
                               recorded=216, warmup=36, shared_numerical_rows_equal=True)
    write(A / 'numerical-deduplication.json', dict(
        complete_numerical_rows=identity(base), existing_path=str(base),
        existing_commit='906cac75a6f3f48a85e059c64c45393b96a2edc4',
        new_correctness_identical=True, timing_prefixes_identical=True, projected=projected))
    d = B / 'iq2s-timing-correctness-r1-v1'
    trace = (d / 'syscalls.txt').read_text()
    assert not re.search(r'/dev/(dri|nvidia|kfd)|lib(?:sycl|ur_|ze_loader|mkl|igc|intelocl)', trace, re.I)
    (A / 'runtime-audit.txt').write_text('\n'.join(s for s in trace.splitlines() if 'execve(' in s or 'exited with 0' in s) + '\n')
    write(A / 'runtime-file-identities.json', {str(p.relative_to(d)):identity(p) for p in (d / 'syscalls.txt', d / 'owned-sample/stdout.csv', d / 'owned-sample/stderr.txt')})
    summary = {}
    for co in (0,1):
        for st in ('stream192','hot8'):
            cell = f'{co}/{st}'
            rows = [json.loads((B/f'iq2s-timing-timing-r{i}-v1/record.json').read_text())['timing_validation']['process_cell_summaries'][cell] for i in (1,2,3)]
            summary[cell] = dict(
                process_paired_ratio_medians={arm:[r[arm]['median'] for r in rows] for arm in ('register/trait','register/direct','direct/trait')},
                process_arm_median_ms={arm:[r['arm_median_ms'][arm] for r in rows] for arm in ('trait','direct','register')},
                process_position_median_ms=[r['arm_position_median_ms'] for r in rows])
    write(A / 'screen-decision.json', dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_commit='156bef6ddbf59c7f017f46b21fd80bda316f3661',
        decision='REJECT current register-index candidate for production integration',
        reason='All twelve process/cell median paired register/trait time ratios exceed 1 (range 1.228966..1.317891); register/direct exceeds 1.198521..1.229937.',
        unit='New process; within-process rows are paired observations, not independent trials',
        summaries=summary, model_performance=False, adoption=False, full_physical_256K_lifecycle=False))
    (A / 'REVIEW.md').write_text('''# IQ2_S caller-dot screen on Ryzen 5 5600X

Reject the current register-index candidate for integration. In three fresh, serialized processes, the median paired candidate/trait time ratios are 1.229–1.318 across the two splits and the two working sets. Candidate/direct ratios are 1.199–1.230. Ratios above one mean a longer elapsed time. Keep the existing production path. This closes this candidate, not the optimization goal.

The screen uses actual Gate/Up IQ2_S weights from 384 experts /15 layers selected from one dependent 32K route capture, with synthetic finite activations and prepared production-custom Q8_K inputs. The two disjoint 192-ID sets come from the same route population; they are not independent prompts. Stream192 visits 201,523,200 GU-weight bytes per split; hot8 visits 8,396,800 bytes. Full owned blobs also include untimed Down weights (756,940,800 bytes total). No GPU work or model inference ran.

Each process records 216 samples plus 36 warmups: two splits ×two strata ×three arms ×18 rounds, with three warmup rounds per cell. All 756 raw arm records, including warmups, are retained across the three process projections. Samples use the same bound row pointers, inputs and output slots. The clock includes the switch, indirect ABI calls, row loop and output stores. Quantization, allocation, finish, numerical checks, hash/log output and pool work are outside the clock. Their presence between arms can influence later conditions. Expert order and split/stratum order are fixed; the seed rotates only arm order. This is neither a cache-cold/DRAM-bandwidth result nor a kernel-only or model-speed result.

All three process/cell paired medians reject the candidate. Preserve individual observations and position medians because outliers and order effects exist (for example stream192/first split/process1 has slower third-position arms). Do not pool 648 recorded samples as independent trials or claim a population confidence interval from three processes. The direct control also costs more than the trait in median ratios; do not select it.

Numerical admission covered every selected GU row and the finish value: 245,760 row pairs /1,474,560 dot calls per correctness prefix, exact IEEE bits and finite outputs. Entry MXCSR is 0x1f80: nearest-even, FTZ/DAZ off; final status flags may change, control bits do not. All sixteen feature variables are absent. Full production ASan/UBSan also executed the timing loops with exact outputs and no diagnostics. Its durations are correctness evidence only. New Release correctness CSV is byte-identical to the committed parent qualification CSV, so numerical payload rows are referenced once instead of duplicated.

Fresh Release and full ASan/UBSan builds pass after the vector-order hash compile fix. Thirty-eight common compile units retain normalized parent flags (declared private/sanitizer flags excluded); no production source changed. The linked candidate/control/trait and custom quantizer/dispatch bodies are retained and were fully reviewed. Candidate local sign scratch remains 32 bytes, direct control 40 bytes; the candidate has no old index-array spill, but removing that spill did not make it faster. No register-pressure or LUT bottleneck cause is asserted without additional evidence.

Build ownership now includes the whole inherited session, including Ninja child process groups. Every captured build/test/tool returned normally with no forced cleanup or survivors. The source, controller, compiler flags, executable and all 1,152 selected weight extents were pinned; extent hashes were verified before and after every process. The logged correctness process was observed from pre-exec to exit for scoped GPU/runtime opens/imports; no matches occurred. This is the specified observation scope, not proof of every possible future call path.

The original first compile failure and original research handoffs remain unchanged in earlier commits. Complete successful syscall/loader history and repeated numerical prefixes have no remaining full-history question; identities, controller checks, compact audit and individual timing samples suffice. Reclaim those success logs only after this archive is committed and all consumer assignments are closed. Retain failure evidence and sources required for the next comparison. No production adoption or physical 262,144-token lifecycle qualification is claimed.
''')
    copy(Path(__file__), A / Path(__file__).name)
    write(A / 'archive-identity.json', {str(p.relative_to(A)):identity(p) for p in sorted(A.rglob('*')) if p.is_file()})
    print(json.dumps(dict(archive=str(A), files=sum(p.is_file() for p in A.rglob('*')), bytes=sum(p.stat().st_size for p in A.rglob('*') if p.is_file()), decision='REJECT')))
