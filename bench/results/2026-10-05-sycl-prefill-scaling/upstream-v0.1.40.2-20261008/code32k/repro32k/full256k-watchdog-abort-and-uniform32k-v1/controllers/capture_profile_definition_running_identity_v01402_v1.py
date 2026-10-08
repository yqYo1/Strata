"""Prove the actual r6 executable independently of an inherited stale metadata field."""
from pathlib import Path
import datetime,hashlib,json,os
base=Path(__file__).parent
run=base/'owned-profile-definition-v01402-code32k-diagnostic-r6'
r=json.loads((run/'record.json').read_text())
assert r['active'] and not r.get('error')
assert r['binary_sha256']=='7f054f8338c6552a5ae6dba548d2020ff2e3bd2661f4a41bb5a2289c56d07891'
assert r['argv'][0].endswith('/build-sycl-dpct-profile-definition-objects-v1-20261008/strata')
pid=r['inferior']['pid'];proc=Path('/proc',str(pid))
def ticks():return int((proc/'stat').read_text().rsplit(')',1)[1].split()[19])
assert ticks()==r['inferior']['start_ticks']
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();assert boot==r['boot_id']
exe=os.readlink(proc/'exe');sha=hashlib.sha256((proc/'exe').read_bytes()).hexdigest()
assert exe==r['argv'][0] and sha=='dd5efb9002167481d180f370b86fb6978d724f37745f7e4a923432296725ad34'
assert ticks()==r['inferior']['start_ticks']
record=dict(active=False,passed=True,captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            scope='Read-only actual executable/PID/start-ticks/boot identity, captured during the existing r6 run. No device call or new inference.',
            boot_id=boot,pid=pid,start_ticks=r['inferior']['start_ticks'],exe=exe,actual_binary_sha256=sha,
            controller_sha256=hashlib.sha256((base/'run_owned_profile_definition_code32k_v01402_v2.py').read_bytes()).hexdigest(),
            inherited_raw_binary_sha256=r['binary_sha256'],
            metadata_error='Raw binary_sha256 was assigned before selecting the uniform-header binary. Actual argv, preflight binary hash assertion and /proc/PID/exe prove dd5. Preserve the raw receipt and attach this correction; do not rerun for metadata alone.')
p=run/'actual-executable-identity-v1.json';assert not p.exists();p.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'identity_passed':True,'actual_binary_sha256':sha,'record_sha256':hashlib.sha256(p.read_bytes()).hexdigest()},indent=2))
