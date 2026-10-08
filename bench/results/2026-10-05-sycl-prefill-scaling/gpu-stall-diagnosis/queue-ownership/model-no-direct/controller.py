from pathlib import Path
import types, json, datetime, hashlib

root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
source = root / 'sycl/tools/recover-xe.sh'
module = types.ModuleType('xe_audit')
exec(compile(source.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0], str(source) + ':python', 'exec'), module.__dict__)
base = Path(__file__).parent
out = base / 'model-queue-fixed-no-direct'
out.mkdir(); out.chmod(0o700)
frozen = base / 'strata-queue-owned'
record = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              scope='Short real normal-MTP/checkpoint/ordinary shutdown, direct submission disabled, context 128; not performance/full capacity/direct-on causal proof',
              boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              binary_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
              binary=str(frozen), healthy=False, output=str(out),
              controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
r = module.Runner(out)
error = None
try:
    env = module.health_environment()
    env['LD_LIBRARY_PATH'] += ':/opt/intel/oneapi/mkl/2026.1/lib'
    env.update(NEOReadDebugKeys='1', EnableDirectSubmission='0', LogAllocationType='1', LogAllocationStdout='1')
    env.pop('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD', None)
    record['environment'] = {k: env.get(k) for k in ('SYCL_CACHE_PERSISTENT', 'UR_ADAPTERS_FORCE_LOAD', 'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD', 'NEOReadDebugKeys', 'EnableDirectSubmission', 'LogAllocationType', 'LogAllocationStdout', 'LD_LIBRARY_PATH')}
    dependencies = r.run('link-dependencies', ['/usr/bin/ldd', str(frozen)], env=env)
    assert 'not found' not in dependencies, 'link dependency unavailable; GPU not started'
    cursor = module.journal_cursor(r, 'kernel-before')
    try:
        r.run('controller', ['/usr/bin/python3', str(root / 'bench/results/2026-10-05-sycl-upstream-arc/serve_check.py'), '--engine', str(frozen), '--out', str(out / 'normal-mtp.json')], seconds=150, env=env)
    except BaseException as exc:
        error = repr(exc)
    log = r.run('kernel-after', ['/usr/bin/journalctl', '-k', '--after-cursor', cursor, '--no-pager', '-o', 'json'])
    rows = []
    for line in log.splitlines():
        try: d = json.loads(line)
        except ValueError: continue
        if 'xe ' in d.get('MESSAGE', '') or '0000:05:00.0' in d.get('MESSAGE', ''): rows.append(d)
    (out / 'kernel-xe.json').write_text(json.dumps(rows, indent=2) + '\n')
    record['new_fault_messages'] = [d.get('MESSAGE', '') for d in rows if module.FAULT.search(d.get('MESSAGE', ''))]
    if (out / 'normal-mtp.json').exists():
        measured = json.loads((out / 'normal-mtp.json').read_text())
        ref = json.loads((root / 'bench/results/2026-10-05-sycl-prefill-scaling/layer-major/aot-validation/confirmation/normal-mtp.json').read_text())
        equality = [dict(name=c['name'], same_ids=c['ids'] == b['ids'], same_printed_logprobs=c['logprobs'] == b['logprobs']) for c, b in zip(measured['requests'], ref['requests'])]
        record.update(exit_code=measured['exit_code'], requests=equality,
                      all_four_requests_equal=len(equality) == 4 and all(x['same_ids'] and x['same_printed_logprobs'] for x in equality))
        stderr = (out / 'normal-mtp.log').read_text()
        record['shutdown_messages'] = [line for line in stderr.splitlines() if 'SYCL shutdown' in line]
        allocations = [line for line in measured.get('startup', []) if 'Type: SEMAPHORE_BUFFER ' in line or 'Type: RING_BUFFER ' in line]
        for request in measured['requests']:
            allocations.extend(line for line in request['protocol'] if 'Type: SEMAPHORE_BUFFER ' in line or 'Type: RING_BUFFER ' in line)
        record['direct_submission_allocations'] = allocations
        record['healthy'] = error is None and not record['new_fault_messages'] and record['all_four_requests_equal'] and record['exit_code'] == 0
    if error: record['error'] = error
except BaseException as exc:
    record['error'] = repr(exc)
record.update(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), steps=r.calls)
(out / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({k: v for k, v in record.items() if k not in ('steps', 'environment')}, indent=2))
