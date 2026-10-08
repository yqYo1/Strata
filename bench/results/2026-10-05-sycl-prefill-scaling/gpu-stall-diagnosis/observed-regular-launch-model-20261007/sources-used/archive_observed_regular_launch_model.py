"""Freeze the private44-property model pass/failure and zero-launch audits."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import sys

base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
exec(compile((base/'archive_lazy_full_and_dequant.py').read_text().split('\nfull = json.loads',1)[0],str(base/'archive_lazy_full_and_dequant.py'),'exec'))
directory=parent/'observed-regular-launch-model-20261007'
assert not directory.exists()
directory.mkdir()
build=json.loads((base/'observed-regular-launch-build/record.json').read_text())
short=json.loads((base/'owned-observed-regular-launch-short/record.json').read_text())
failed=json.loads((base/'owned-observed-regular-launch-2048-repeat/record.json').read_text())
health=json.loads((base/'post-observed-regular-launch-2048-repeat-health/record.json').read_text())
short_flags=json.loads((base/'owned-observed-regular-launch-short-launch-flags.json').read_text())
failed_flags=json.loads((base/'owned-observed-regular-launch-2048-repeat-launch-flags.json').read_text())
query=json.loads((base/'remaining-cooperative-query/query-record.json').read_text())
assert build['passed'] and not build['active'] and build['production_inputs_unchanged'] and build['removed_property_declarations']==44
assert short['healthy'] and short['completed'] and not short['active'] and short['exit_code']==0
assert len(short['requests'])==4 and all(all(r['equality'].values()) for r in short['requests'])
assert short['release_pairs']==dict(release=6,restore=6)
assert not failed['active'] and not failed['healthy'] and not failed['completed'] and not failed['requests']
assert failed['active_request']=='2048-repeat-1' and not failed['new_fault_messages']
assert health['healthy'] and health['started_utc']>failed['finished_utc']
assert short['boot_id']==failed['boot_id']==health['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for result in [short,failed]:
    assert not result['cleanup']['inferior_survived'] and not result['cleanup']['gdb_survived']
    for key in ['inferior','debugger']:
        old=result[key];current=process_identity(old['pid']);assert not current or current['start_ticks']!=old['start_ticks']
assert short_flags['all_41_reviewed_classes_absent_from_cooperative_enqueues'] and short_flags['remaining_cooperative_combinations']==7
assert failed_flags['all_41_reviewed_classes_absent_from_cooperative_enqueues'] and failed_flags['remaining_cooperative_combinations']==0
assert failed_flags['counts']['native_launches']==failed_flags['counts']['successful_UR_enqueues']==258136
assert query['passed'] and not query['active'] and not query['new_fault_messages'] and query['results'][-1]==dict(stage='PASS',queried=7,exceeds=4,kernels_submitted=0)
for result in [health,query]:
    for step in result['steps']:
        assert step['exit_code']==0 and not step['timed_out'] and not step['still_alive'] and not Path('/proc',str(step['pid'])).exists()
for source in build['sources']:
    assert digest(root/source['source'])==source['original_source_sha256']
    assert digest(Path(source['candidate_source']))==source['candidate_source_sha256']
for name,record in [('run_owned_observed_regular_launch_short.py',short),('run_owned_observed_regular_launch_2048_repeat.py',failed)]:
    assert digest(base/name)==record['source_sha256'][str(base/name)]
assert digest(base/'audit_observed_launch_flags_short_v1.py')==short_flags['controller_sha256']
assert digest(base/'audit_observed_launch_flags.py')==failed_flags['controller_sha256']
assert digest(base/'build_observed_regular_launch_candidate.py')==build['controller_sha256']

def save_nonbinary_folder(source,prefix):
    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.name in ['probe','query'] or path.suffix in ['.bin','.o','.a','.d'] or '__pycache__' in path.parts:
            continue
        target=Path(prefix)/path.relative_to(source)
        if path.stat().st_size<=300000:
            copy(path,directory,target)
        else:
            dest=directory/target.with_name(path.name+'.tail');dest.parent.mkdir(parents=True,exist_ok=True)
            with path.open('rb') as stream:stream.seek(max(0,path.stat().st_size-65536));dest.write_bytes(stream.read())
            metadata=dict(private_path=str(path),private_bytes=path.stat().st_size,private_sha256=digest(path),saved_tail=str(dest.relative_to(directory)),saved_tail_bytes=dest.stat().st_size)
            dest.with_name(dest.name+'.metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')

for name in ['observed-regular-launch-build','owned-observed-regular-launch-short','owned-observed-regular-launch-2048-repeat','post-observed-regular-launch-2048-repeat-health','remaining-cooperative-query']:
    save_nonbinary_folder(base/name,name)
for name in ['build_observed_regular_launch_candidate.py','run_owned_observed_regular_launch_short.py','run_owned_observed_regular_launch_2048_repeat.py','probe_after_observed_regular_launch_2048_repeat.py','audit_observed_launch_flags_short_v1.py','audit_observed_launch_flags.py','build_remaining_cooperative_query.py','run_remaining_cooperative_query.py','prefill_cooperative_query.cpp','run_owned_observed_regular_launch_2048_retained.py','archive_observed_regular_launch_model.py']:
    copy(base/name,directory,Path('sources-used')/name)
for name in ['observed-regular-launch-preparation-targets.json','owned-observed-regular-launch-short-launch-flags.json','owned-observed-regular-launch-2048-repeat-launch-flags.json','observed-regular-launch-2048-main-stack.txt']:
    copy(base/name,directory,Path(name))
assessment=dict(scope='Private44-property integrated correctness and terminal stall records; no adoption, speed or full capacity claim',recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                binary_sha256=build['candidate_binary_sha256'],production_inputs_unchanged=True,
                changed_sources=8,removed_property_declarations=44,reviewed_kernel_classes=41,
                short=dict(requests=4,all_heads_IDs_logprobs_equal=True,release_pairs=short['release_pairs'],normal_exit=True,new_xe_faults=False),
                long=dict(requests_completed=0,stage='layer40/token256',last_completed_phase='dequant',last_phase_mark=170571,watchdog_seconds=60,stop='SIGABRT',owned_cleanup=failed['cleanup'],new_xe_faults=False,observed_successful_UR_kernel_enqueues=258136,observed_cooperative_UR_kernel_enqueues=0),
                post_cleanup_logged_health_pass=True,remaining_short_cooperative_query=dict(queried=7,exceed_zero_dynamic_local_bound=4,kernels_submitted=0),
                inference='The traced long path stalls despite zero observed cooperative UR kernel enqueues. Incorrect cooperative settings are independently measured in other paths, but removing them is insufficient to prevent this wait. Cache/recorded-graph lifetime, queue/runtime completion and source/consumer ownership remain open.',
                next_control=dict(controller='sources-used/run_owned_observed_regular_launch_2048_retained.py',scope='Same privatebinary and longpath settings exceptmain/MTP release disabled; two repeated2048inputs, strict frozen head/IDs/LP and zero MTP release/restores required',GPU_executed=False),
                source_rewrite_adopted=False,all262144_cell_suite_passed=False,clean_speed_measured=False,manual_reset_rebind_reboot_service_or_package_change=False)
(directory/'assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
(directory/'README.md').write_text('''# Observed regular-launch model checks on 2026-10-07

A private executable changes only 44 use_root_sync declarations across eight
source files for 41 manually reviewed kernel classes. Original arithmetic,
kernel names, ranges, caller waits and every other archive member remain.
The offline receipt checks all source/object/archive/binary input hashes;
production inputs are unchanged. The binary SHA is
33bef89feb49a97c152c114c48dd002612215e9b44000f334891b24a3804237b.
Selected kernels own separate rows/elements or use only subgroup/work-group
coordination. Other verifier persistent/global functions are retained.

On Arc B570 10 GiB / Ryzen 5600X / 128 GiB RAM, kernel 7.0.0-38-generic,
NEO 26.31.39395.14 and oneAPI 2026.1.1, four context-128 normal-MTP requests
match frozen updated-control IDs, printed logprobs and every first-head byte.
Six MTP release/restore pairs complete; the owned process exits normally with
no new xe fault. These full Level Zero/UR logging and validation checks are
not clean throughput or full-capacity results.

The subsequent three-repeat 2048-input test stops on its first request,
at layer 40 of the chunk starting at token 256. The last completed phase is
dequant, mark 170571. GU/down APIs return success; the following
zeCommandListHostSynchronize has no successful return. After 60 seconds of
no progress the engine watchdog raises SIGABRT. GDB captures the stop and
main-thread sample; owned cleanup removes both inferior and debugger.
The ordinary-launch candidate is not sufficient to prevent this wait.

The complete traced interval has 258,136 successful native/UR kernel enqueue
associations and zero cooperative UR enqueue flags, with complete native
group/local shapes. No selected class retains a cooperative flag. Successful
API returns are submission evidence, not proof that those kernels complete.
This accounting covers the traced enqueue path, not a general assertion
about every recorded command-buffer operation or all possible configurations.
It narrows the diagnosis beyond the cooperative flag; it does not prove a
particular cache, recorded-graph or runtime cause.

No new xe fault appears. A fresh logged H2D/kernel/D2H health probe then passes
three rounds of 16,384 exact words on the same boot, without reset/rebind,
reboot, service or package changes. The main sample is in the vDSO clock path;
its truncated unwind is not used as proof of a particular CSR object.

The successful short trace still contains seven other cooperative kernel/
local-size combinations. A separate unchanged-object query submits no kernels
and finds four above their optimistic zero-extra-local limits: original
attention chunk 1980 versus 144 groups, add-streams broadcast and to_f16
1200 versus 144, and original attention merge 720 versus 144. Recurrence
192 versus 288, serial convolution 80 versus 288 and top-k 30 versus 36
do not exceed that queried bound. Counts within it do not validate additional
dynamic local storage or all synchronization requirements. Those paths still
need per-function correction, independently of the zero-flag long stall.

The retained main/MTP-backing controller is prepared but has not run. It uses
the same private binary, long-path settings and strict whole-head/ID/logprob
checks, disables only both releases, and requires two repeated requests with
zero MTP release/restore pairs. This is a diagnostic comparison, not an adopted
memory policy. Full 262144-cell CLI/normal-MTP serving, clipped-tail, refusal,
later-valid requests and PP1000/TG70 remain incomplete.

Receipts, exact source copies/diffs, both executed parser versions, bounded
log tails, full private-log hashes and GDB state are included. Binaries, whole
output arrays and large logs stay private. The manifest covers every file
except itself.
''')
print(json.dumps(manifest(directory),indent=2))
