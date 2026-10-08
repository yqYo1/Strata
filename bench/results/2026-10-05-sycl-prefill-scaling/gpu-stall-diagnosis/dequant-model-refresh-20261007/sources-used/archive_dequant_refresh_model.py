"""Preserve integrated no-root dequant short parity and paired two-chunk comparison."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent=root/'bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis'
helpers=(base/'archive_lazy_full_and_dequant.py').read_text().split('\nfull = json.loads(',1)[0]
exec(compile(helpers,str(base/'archive_lazy_full_and_dequant.py'),'exec'))
build=json.loads((base/'dequant-no-root-refresh-build/record.json').read_text())
short=json.loads((base/'owned-dequant-no-root-refresh-short/record.json').read_text())
control=json.loads((base/'dequant-refresh-2048-control/record.json').read_text())
candidate=json.loads((base/'dequant-refresh-2048-candidate/record.json').read_text())
assert build['passed'] and build['production_inputs_unchanged']
assert short['healthy'] and not short['active'] and len(short['requests'])==4 and not short['new_fault_messages']
for key in ['inferior','debugger']:
 old=short[key];now=process_identity(old['pid']);assert not now or now['start_ticks']!=old['start_ticks']
for receipt in [control,candidate]:
 assert not receipt['active'] and not receipt['job']['survivors']
 for old in receipt['job']['owned']:
  now=process_identity(old['pid']);assert not now or now['start_ticks']!=old['start_ticks']
assert control['passed'] and len(control['requests'])==1
assert short['binary_sha256']==candidate['binary_sha256']==build['candidate_binary_sha256']
assert control['binary_sha256']==build['production_binary_sha256']
assessment=dict(candidate_adopted=False,integrated_short_model_parity=True,multichunk_candidate_passed=candidate['passed'],
                original_prompt_ms=float(control['requests'][0]['done'].split()[3]),
                original_decode_ms=float(control['requests'][0]['done'].split()[4]),
                paired_requests_per_binary=1,scope='Two-chunk 2048-token normal-MTP API-log-disabled comparison with existing phase waits/progress; not unsynchronized throughput, repeated statistical proof or full256K capacity',
                capacity_suite_passed=False,math_changed=False,kernel_bodies_changed=False,new_fault_messages=candidate['new_fault_messages'])
if candidate['passed']:
 req=candidate['requests'][0];assert all(req['equality'].values())
 assessment.update(candidate_prompt_ms=float(req['done'].split()[3]),candidate_decode_ms=float(req['done'].split()[4]),whole_head_sha256=req['head_sha256'])
 assessment['prompt_time_change_percent']=100*(assessment['candidate_prompt_ms']/assessment['original_prompt_ms']-1)
 assessment['prompt_speedup']=assessment['original_prompt_ms']/assessment['candidate_prompt_ms']
else:assessment['failure']=candidate.get('error')
( base/'dequant-refresh-assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
dest=parent/'dequant-model-refresh-20261007';assert not dest.exists();dest.mkdir()
for name in ['build_dequant_no_root_refresh.py','run_owned_dequant_refresh_short.py','run_dequant_refresh_2048_timing.py','profile_supervisor.py','archive_dequant_refresh_model.py']:
 copy(base/name,dest,Path('sources-used')/name)
for folder in ['dequant-no-root-refresh-build','owned-dequant-no-root-refresh-short','dequant-refresh-2048-control','dequant-refresh-2048-candidate']:
 save_folder(base/folder,dest,folder)
copy(base/'dequant-refresh-assessment.json',dest,Path('assessment.json'))
summary=(f"The updated control's prompt is {assessment['original_prompt_ms']/1000:.4f} seconds, "
         f"versus {assessment['candidate_prompt_ms']/1000:.4f} seconds in the candidate "
         f"({assessment['prompt_time_change_percent']:+.3f}% time change, {assessment['prompt_speedup']:.5f}x)."
         if candidate['passed'] else 'The candidate does not pass the paired multichunk gate; see its failure record.')
(dest/'README.md').write_text("""# Integrated dequant launch-property check after upstream refresh

On B570 10 GiB / Ryzen 5600X / 128 GiB RAM with the recorded oneAPI, NEO and
Level Zero versions, a private candidate replaces exactly one kernel archive
object in the updated ceff2d8a control. Its SHA is 9f43b89e…02766ecf. Exactly
the iq_dequant_f16 and iq_dequant_gu_f16 wrapper property lists drop
use_root_sync. Kernel names, formulas, ND-ranges, device capability checks,
other objects, phase waits and all production source/link inputs are retained.
The candidate source hash a5ece3f8 matches the earlier 144-pair actual-wrapper
check; this archive adds integrated model checks after shared-header refresh.

The first fully Level Zero/UR logged context-128 normal-MTP process completes
all four requests with exact IDs, every logprob and all 248,320 first-head
floats versus the updated control. Six MTP release/restore pairs complete;
normal exit and owned cleanup succeed with no new xe fault.

A following pair disables API logging and validation, retaining the same
phase waits/progress and context-4096/int8 KV/layer-major-1 settings. Each
binary receives one 2048-token request split into two 1024-token chunks,
followed by four normal-MTP outputs. """+summary+"""

These are single samples with synchronization/progress still present, not
unsynchronized throughput or repeated statistical evidence. See the
[assessment](assessment.json) and terminal receipts for exact whole-head,
ID/logprob and exit/cleanup results. Large diagnostic logs and whole heads
remain private with hashes and bounded tails. No GPU reset, service change
or production kernel change was performed. The candidate remains unadopted;
repeated full262144-cell CLI/serve, clipped-tail, refusal and later-valid gates
are incomplete. PP1000/TG70 is not established.
""")
report=manifest(dest);(base/'dequant-refresh-model-archive.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(archive=report,assessment=assessment),indent=2))
