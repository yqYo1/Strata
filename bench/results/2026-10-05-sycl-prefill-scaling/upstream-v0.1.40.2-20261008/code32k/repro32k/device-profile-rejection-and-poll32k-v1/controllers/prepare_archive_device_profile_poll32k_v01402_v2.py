"""Correct a CPU-only archive check for dual RESUME/REUSED protocol reports."""
from pathlib import Path
import ast,datetime,hashlib,json
base=Path(__file__).parent
old=base/'archive_device_profile_abort_and_poll32k_v01402_v1.py'
assert hashlib.sha256(old.read_bytes()).hexdigest()=='fe0ee4f75763661f320ba97b59dfdff3cc4d5e211a8b3280a91ecb009e24a2f8'
text=old.read_text()
before="all(v['input_tokens']==32768 and v['resume_tokens']==[0] for v in fresh)"
after="all(v['input_tokens']==32768 and v['resume_tokens'] and all(n==0 for n in v['resume_tokens']) for v in fresh)"
assert text.count(before)==1;text=text.replace(before,after)
before="             'prepare_owned_poll_backoff_code32k_v01402_v1.py','run_owned_poll_backoff_code32k_v01402_v1.py',Path(__file__).name]:"
after="             'prepare_owned_poll_backoff_code32k_v01402_v1.py','run_owned_poll_backoff_code32k_v01402_v1.py',\n             'archive_device_profile_abort_and_poll32k_v01402_v1.py','prepare_archive_device_profile_poll32k_v01402_v2.py',Path(__file__).name]:"
assert text.count(before)==1;text=text.replace(before,after)
before="out=parent/'device-profile-rejection-and-poll32k-v1';out.mkdir()"
after=before+"\ncopy(base/'archive-device-profile-poll32k-cpu-check-v2.json',out/'checks/archive-device-profile-poll32k-cpu-check-v2.json')"
assert text.count(before)==1;text=text.replace(before,after)
ast.parse(text)
new=base/'archive_device_profile_abort_and_poll32k_v01402_v2.py';assert not new.exists();new.write_text(text)
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
assert not (root/'bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008/code32k/repro32k/device-profile-rejection-and-poll32k-v1').exists()
p=base/'archive-device-profile-poll32k-cpu-check-v2.json';assert not p.exists()
p.write_text(json.dumps({'active':False,'passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'old_controller_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'new_controller_sha256':hashlib.sha256(new.read_bytes()).hexdigest(),
 'old_cpu_rejection':'The completed engine reports both RESUME0 and REUSED0. The archive asserted exactly one zero, before creating any public output. All numerical and freshness gates were already passed.',
 'correction':'Require at least one report, every reported reuse count equal to zero, and all32768 tokens; do not drop protocol fields or rerun the GPU test.',
 'gpu_rerun':False,'raw_terminal_receipt_unchanged':True},indent=2)+'\n')
print(new,hashlib.sha256(new.read_bytes()).hexdigest())
