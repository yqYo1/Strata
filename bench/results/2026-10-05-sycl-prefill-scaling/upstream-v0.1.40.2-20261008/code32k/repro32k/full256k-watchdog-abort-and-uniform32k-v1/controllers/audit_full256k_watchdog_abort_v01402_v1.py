"""Freeze bounded raw evidence of r5's host watchdog abort; do not infer a driver reset."""
from pathlib import Path
import datetime, hashlib, json, re

base = Path(__file__).parent
run = base / 'owned-full-kv-access-v01402-full256k-diagnostic-r5'
out = base / 'full256k-watchdog-abort-v01402-review-v1'
out.mkdir(mode=0o700)
def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
r = json.loads((run/'record.json').read_text())
assert sha(run/'record.json') == 'e7e541f91ef053acebaa6b9ed2bc3139010e269f0b671d23f423869e2c7a882f'
assert not r['active'] and not r['completed'] and not r['healthy']
assert 'SIGABRT' in r['error'] and not r['new_fault_messages']
assert not r['cleanup']['inferior_survived'] and not r['cleanup']['gdb_survived']
mi = run/'debugger/failure.mi.txt'
decoded = ''.join(json.loads(l[1:]) for l in mi.read_text().splitlines() if l.startswith('~'))
stacks = {}
for tid in (82, 80, 79, 78, 1):
    m = re.search(r'\nThread '+str(tid)+r' .*?(?=\nThread |\Z)', decoded, re.S)
    assert m
    stacks[str(tid)] = m.group().split('\nrax ')[0]
assert 'main::{lambda(std::stop_token)#1}' in stacks['82'] and '__GI_abort' in stacks['82']
assert 'Stager::wait' in stacks['1']
(out/'selected-stacks.txt').write_text('\n'.join(stacks.values()))
stderr = run/'debugger/inferior.stderr'
size = stderr.stat().st_size
assert size == 54921377495
spans = [('last-submissions',54498901426,54499950000),
         ('watchdog-report',54907989404,54907993404),
         ('terminal-status',size-32768,size)]
excerpts = []
with stderr.open('rb') as f:
    for name, start, end in spans:
        f.seek(start); data=f.read(end-start)
        assert len(data)==end-start
        p=out/(name+'.raw.txt');p.write_bytes(data)
        excerpts.append(dict(name=name,start=start,end=end,bytes=len(data),sha256=sha(p)))
submissions=(out/'last-submissions.raw.txt').read_text(errors='replace')
watchdog=(out/'watchdog-report.raw.txt').read_text(errors='replace')
assert 'no progress for 60 s' in watchdog and 'layer 33' in watchdog and '196608' in watchdog
assert 'mode 0' in watchdog and '5 of 5 workers parked' in watchdog
assert 'zeCommandListAppendMemoryCopy' in submissions and 'UR_RESULT_SUCCESS' in submissions
pending={}
for h in ('0x672b5d0','0x6724940','0x6723120'):
    matches=[x for x in submissions.splitlines() if '<--- urEnqueueUSMMemcpy(' in x and h in x]
    assert matches
    pending[h]=matches[-1]
(out/'pending-dma-submissions.json').write_text(json.dumps(pending,indent=2)+'\n')
health=base/'post-full256k-abort-v01402-health-v1/record.json'
hr=json.loads(health.read_text())
assert hr['healthy'] and not hr['active'] and hr['terminal_controller_receipt_sha256']==sha(run/'record.json')
record=dict(active=False,passed=True,adopted=False,
    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    scope='Evidence extraction and host-watchdog localization only; underlying pending event cause remains unresolved.',
    controller_sha256=sha(__file__),terminal_receipt_sha256=sha(run/'record.json'),
    debugger_failure_sha256=sha(mi),raw_stderr_bytes=size,raw_stderr_full_hash_pending=True,
    raw_excerpts=excerpts,health_receipt_sha256=sha(health),
    observed={'abort_thread':82,'abort_call':'main watchdog -> release_gpu_waits -> fflush -> abort',
              'watchdog_seconds':60,'stage':'second fresh full prefill, chunk196608, layer33',
              'main_thread':'Stager::wait waiting on CPU host-buffer readiness',
              'three_staging_workers':'UR event execution status polling; L0 counter events return NOT_READY',
              'last_copy_size_bytes':2176000,'last_submissions_succeeded':True,
              'new_kernel_fault_messages':[],'post_exit_device_runtime_health_passed':True,
              'owned_process_survivors':False},
    limits=['Unpatched driver/runtime counter event values were not captured; distinguish stalled DMA, dependency cycle and incorrect status only with more evidence.',
            'Successful appends do not prove GPU completion, and submitted status does not establish an unflushed queue.',
            'No attribution to the separate DPCT ODR inconsistency or the indexer-spare repair.',
            'First full capacity/image passes; repeated full, actual disk restore, clipped restore, refusals and later32K were not completed.'],
    minimum_performance_input_tokens=32768,diagnostic_timing_excluded=True,
    no_reset_rebind_reboot_service_package_global_change=True)
(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(passed=True,record_sha256=sha(out/'record.json'),health_passed=True,underlying_cause_resolved=False),indent=2))
