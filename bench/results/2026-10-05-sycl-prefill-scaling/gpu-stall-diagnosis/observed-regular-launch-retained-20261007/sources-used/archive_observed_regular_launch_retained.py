"""Freeze the retained-cache comparison before the next upstream integration."""
from pathlib import Path
import datetime
import hashlib
import json

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
exec(compile((base / 'archive_lazy_full_and_dequant.py').read_text().split('\nfull = json.loads', 1)[0], str(base / 'archive_lazy_full_and_dequant.py'), 'exec'))
case = 'owned-observed-regular-launch-2048-retained'
result = json.loads((base / case / 'record.json').read_text())
health = json.loads((base / 'post-observed-regular-launch-2048-retained-health/record.json').read_text())
flags = json.loads((base / (case + '-launch-flags.json')).read_text())
assert not result['active'] and not result['completed'] and not result['requests']
assert result['active_request'] == '2048-repeat-1' and not result['new_fault_messages']
assert not result['cleanup']['inferior_survived'] and not result['cleanup']['gdb_survived']
assert result['environment']['STRATA_PREFILL_RELEASE_CACHE'] == '0'
assert result['environment']['STRATA_PREFILL_RELEASE_DRAFT'] == '0'
assert health['healthy'] and health['started_utc'] > result['finished_utc']
assert result['boot_id'] == health['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for key in ['inferior', 'debugger']:
    owner = result[key]
    now = process_identity(owner['pid'])
    assert not now or now['start_ticks'] != owner['start_ticks']
for step in health['steps']:
    assert step['exit_code'] == 0 and not step['timed_out'] and not step['still_alive']
    assert not Path('/proc', str(step['pid'])).exists()
assert flags['counts']['native_launches'] == flags['counts']['successful_UR_enqueues'] == 107755
assert flags['counts'].get('cooperative_UR_enqueues', 0) == 0
assert flags['controller_sha256'] == digest(base / 'audit_observed_launch_flags.py')
assert flags['record_sha256'] == digest(base / case / 'record.json')
assert digest(base / 'run_owned_observed_regular_launch_2048_retained.py') == result['source_sha256'][str(base / 'run_owned_observed_regular_launch_2048_retained.py')]
directory = parent / 'observed-regular-launch-retained-20261007'
assert not directory.exists()
directory.mkdir()
save_folder(base / case, directory, case)
save_folder(base / 'post-observed-regular-launch-2048-retained-health', directory, 'health')
for name in ['archive_observed_regular_launch_retained.py', 'run_owned_observed_regular_launch_2048_retained.py', 'probe_after_observed_regular_launch_2048_retained.py', 'audit_observed_launch_flags.py']:
    copy(base / name, directory, Path('sources-used') / name)
copy(base / (case + '-launch-flags.json'), directory, Path('launch-flags.json'))
phases = []
release = []
with (base / case / 'debugger/inferior.stderr').open(errors='replace') as stream:
    for line in stream:
        if line.startswith('strata prefill sync:'):
            phases.append(line.strip())
            if len(phases) > 12:
                phases.pop(0)
        if 'released the verify window' in line or line.startswith('strata prefill memory:'):
            release.append(line.strip())
assert phases[-1].startswith('strata prefill sync: mark 71177 phase dequant done')
(directory / 'last-phases-and-memory.txt').write_text('\n'.join(phases + release) + '\n')
assessment = dict(
    scope='Same private44-property executable and longpath settings, with both main/MTP prefill leases disabled; first of two2048 requests stalls; no adoption, performance or fullcapacity result',
    recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    binary_sha256=result['binary_sha256'],
    release_main=False, release_MTP=False,
    completed_requests=0, stop_layer=17, stop_chunk_token=256,
    last_completed_phase='dequant', last_phase_mark=71177,
    watchdog_no_progress_seconds=60, stop='SIGABRT',
    successful_native_UR_enqueue_associations=107755, observed_cooperative_flags=0,
    new_xe_faults=False, owned_cleanup=result['cleanup'], post_cleanup_logged_health_passed=True,
    inference='Neither cooperative enqueue flags nor main/MTP prefill retirement is necessary for this wait. The verifier release message only polls verifier queues; it does not establish completion of the distinct prefill queue.',
    release_message_source=dict(path='sycl/src/core/verify.cpp', sha256=digest(root / 'sycl/src/core/verify.cpp'), function='Verifier::release_gpu_waits', queue_predicate='ext_oneapi_empty()', queues=['cs_', 'copy_']),
    API_success_proves_device_completion=False, all262144_cell_suite_passed=False,
    production_changed=False, reset_rebind_reboot_service_package_or_global_changes=False)
(directory / 'assessment.json').write_text(json.dumps(assessment, indent=2) + '\n')
(directory / 'README.md').write_text('''# Retained decode backing comparison on 2026-10-07

On Arc B570 10 GiB / Ryzen 5600X / 128 GiB RAM, the same private executable
33bef89f as the preceding regular-launch comparison retains both main and
MTP decode backing and recorded graphs during prefill. Only
STRATA_PREFILL_RELEASE_CACHE and STRATA_PREFILL_RELEASE_DRAFT change to 0;
the context-4096, int8-KV, FP16 expert path and phase waits are retained.
This is a diagnostic comparison, not a proposed full-context memory policy.

The first of two repeated 2048-token inputs stops at layer 17 of the chunk
starting at token 256. No request returns an output. The last completed
phase is dequant, mark 71177. After 60 seconds without progress the watchdog
raises SIGABRT. Owned GDB records the stop and cleanup removes both owned
processes. The trace contains 107,755 successful native/UR enqueue
associations and zero cooperative enqueue flags. API success is submission
evidence and does not establish device completion.

The failure therefore also occurs without main/MTP prefill retirement.
Removing cooperative properties and retaining decode backing are each
insufficient to prevent this wait. The exact failing resource or operation
remains unidentified. The shutdown message saying the verify-window GPU
finished polls the verifier's own queues through ext_oneapi_empty(); it does
not demonstrate that the separate prefill queue finished.

No new xe fault is recorded. A fresh bounded logged H2D/kernel/D2H test
then passes all three rounds of 16,384 exact words on the same boot, without
a reset or service/driver change. No diagnostic time is treated as clean
throughput, and the complete repeated 262144-cell validation remains open.
The terminal record, exact launch audit, bounded logs and full private-log
hashes are preserved. The previous archive stays frozen at its earlier
point, when this controller had not yet run.
''')
report = manifest(directory)
(base / 'observed-regular-launch-retained-archive.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
