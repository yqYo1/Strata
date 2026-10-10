import fcntl, hashlib, importlib.util, json
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
HERE=Path(__file__).resolve().parent
OWNER=B/'xe-llvm-v711-source-v3/direct_owner.py'
spec=importlib.util.spec_from_file_location('owner',OWNER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ENV=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')
with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=B/'xe-llvm-owner-lifecycle-root-v1';m.require(not out.exists(),'immutable output');out.mkdir()
    r=dict(active=True,passed=False,complete=False,commands=[],gpu_executed=False,owner_sha256=m.sha(OWNER),fixture_sha256=m.sha(HERE/'children.py'),controller_sha256=m.sha(__file__),scope='Host-only long-build owner lifecycle: more than256 observed cumulative children, bounded live pidfds and normal closure')
    def save(): (out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
    owner=m.Owner(out/'commands',save);r['commands']=owner.commands;save()
    try:
        e,so,se=owner.run('children',['/usr/bin/python3',str(HERE/'children.py')],ENV,HERE,wall=30,cpu=30,text_cap=1<<20,total_cap=8<<20,rss_cap=1<<30)
        m.require(m.closed(e) and e['exit_code']==0,'normal closure without intervention')
        r['observed_cumulative_identities']=len(e['owners']);r['peak_live_identities']=e['peak_live_owned_identities']
        m.require(r['observed_cumulative_identities']>256 and r['peak_live_identities']<64,'actual cumulative/live distinction exercised')
        r['stdout']=so.read_text();r['complete']=True
    except BaseException as exc:r['error']=repr(exc)
    finally:
        r['active']=owner.active is not None;r['passed']=r['complete'] and not r['active'] and not r.get('error');save()
    print(json.dumps({k:r.get(k) for k in ['passed','error','observed_cumulative_identities','peak_live_identities','stdout']}))
    raise SystemExit(0 if r['passed'] else 1)
