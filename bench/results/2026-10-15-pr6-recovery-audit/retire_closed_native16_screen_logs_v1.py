#!/usr/bin/env python3
"""Main only: retire checked duplicate screen capture and closed successful logs."""
from pathlib import Path
import datetime, fcntl, hashlib, importlib.util, json, subprocess
A=Path(__file__).resolve().parent
P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
OWNER=P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def require(ok,why):
    if not ok: raise RuntimeError(why)
def main():
    manifest=A/'closed-native16-screen-log-retention-v1.json'
    require(not manifest.exists(),'fresh manifest only; no automatic retry')
    require(sha(OWNER)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f','owner pin')
    spec=importlib.util.spec_from_file_location('retention_owner',OWNER)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    cohort=[('own-native16-quality-screen-v2-v1.json','main-own-native16-quality-screen-root-v2-v1'),
      ('own-native16-coarse-profile-v1.json','main-own-native16-coarse-profile-root-v1'),
      ('own-native16-fp32-coarse-profile-v1.json','main-own-native16-fp32-coarse-profile-root-v1'),
      ('own-native16-paired-gemm-build-v1.json','main-r803-paired-gemm-v1-build'),
      ('own-native16-paired-gemm-smoke-v1.json','main-r803-paired-gemm-v1-smoke'),
      ('own-native16-paired-gemm-measure-v1.json','main-r803-paired-gemm-v1-measure')]
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
      fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
      boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
      selected=[];receipts=[];screen=None
      for name,dirname in cohort:
        path=A/name;raw=path.read_bytes()
        committed=subprocess.run(['git','show','HEAD:'+str(path.relative_to(P))],cwd=P,capture_output=True,check=True).stdout
        require(raw==committed,'receipt not committed byte-identically '+name)
        r=json.loads(raw);require(r['passed'] and r['complete'] and not r['active'] and r['boot_id']==boot and not r['fresh_faults'] and not r['driver_dump_exists'],'closed successful sameboot cohort '+name)
        root=B/dirname;require(root.is_dir() and root.resolve()==root,'exact root '+dirname)
        receipts.append(dict(path=str(path),sha256=sha(path),original_status='PASS unchanged'))
        if name=='own-native16-quality-screen-v2-v1.json':screen=r
        for cmd in r['commands']:
          require(mod.closed(cmd) and cmd.get('exit_code')==0,'normal owner closure '+name)
          for identity in cmd.get('owners',[]):
            now=mod.identity(identity['pid']);require(now is None or now['start_ticks']!=identity['start_ticks'],'owned identity still live')
          for leaf,proof in cmd.get('logs',{}).items():
            require(Path(leaf).name==leaf and Path(leaf).suffix in ('.stdout','.stderr'),'closed log leaf')
            f=root/'owned'/leaf
            require(f.is_file() and not f.is_symlink() and f.stat().st_size==proof['bytes'] and sha(f)==proof['sha256'],'log file proof '+str(f))
            selected.append(dict(path=str(f),bytes=proof['bytes'],sha256=proof['sha256'],retained_evidence=name,reason='closed success; exact commands/env/status/individual outputs/fault findings committed; verbose stdout/stderr no current consumer'))
      require(screen is not None,'screen receipt')
      raw_a=screen['cases']['fp32-a']['raw'];raw_b=screen['cases']['fp32-b']['raw']
      require(raw_a['sha256']==raw_b['sha256'] and raw_a['bytes']==raw_b['bytes'],'FP32 raw captures not byte duplicates')
      for proof in [raw_a,raw_b,screen['cases']['bf16']['raw']]:
        f=Path(proof['path']);require(f.is_file() and not f.is_symlink() and f.stat().st_size==proof['bytes'] and sha(f)==proof['sha256'],'canonical raw proof')
      selected.append(dict(path=raw_b['path'],bytes=raw_b['bytes'],sha256=raw_b['sha256'],retained_evidence='own-native16-quality-screen-v2-v1.json',identical_retained_capture=raw_a['path'],reason='FP32-b all64 fullvocabulary rows byte-identical to retained FP32-a; controls and all individual results committed; no need for second raw copy'))
      require(len({v['path'] for v in selected})==len(selected),'duplicate deletion paths')
      out=dict(schema='strata-closed-native16-screen-log-retention-v1',owner='Main',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),boot_id=boot,receipts=receipts,complete=False,deleted=[],planned_count=len(selected),planned_logical_bytes=sum(v['bytes'] for v in selected),kept_current_consumers=[dict(raw=screen['cases'][arm]['raw'],metadata=screen['cases'][arm]['metadata'],consumer='Main R810/R804 same-input component/model comparison: reference FP32-a and current custom BF16; selected fullvocabulary logits, not universal quality evidence',review_at='next native16 model comparison closure; by 2026-10-16T23:00:00Z') for arm in ['fp32-a','bf16']],unmodified=['source/models/services/GPU state/currentv2build/oldseparatequeuecountercontrol/oldFP32NLLcontrols/all sourceproposals/privateR803 executable and buildrecord/all result statuses'],limits='Only explicit normally closed owner logs and one content-identical capture; no active run or last required fixture; no physical free-space claim')
      manifest.write_text(json.dumps(out,indent=2)+'\n')
      for item in selected:
        f=Path(item['path']);require(f.is_relative_to(B),'deletion escaped private base')
        require(f.is_file() and not f.is_symlink() and f.stat().st_size==item['bytes'] and sha(f)==item['sha256'],'changed deletion target')
        f.unlink();out['deleted'].append(item);manifest.write_text(json.dumps(out,indent=2)+'\n')
      out['complete']=True;out['deleted_logical_bytes']=sum(v['bytes'] for v in out['deleted']);manifest.write_text(json.dumps(out,indent=2)+'\n')
      print(json.dumps(dict(complete=True,files=len(out['deleted']),logical_bytes=out['deleted_logical_bytes'],kept_raw_bytes=2*raw_a['bytes'],manifest=str(manifest))))
if __name__=='__main__':main()
