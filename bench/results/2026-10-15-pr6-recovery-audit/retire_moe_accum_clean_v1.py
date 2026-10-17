#!/usr/bin/env python3
"""Retire closed clean-comparison text and restored-from-Git research copies."""
import datetime
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
A=Path(__file__).resolve().parent
P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/verify-moe-window')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    dst=A/'moe-accum-clean-retention-v1.json';excerpts=A/'moe-accum-clean-excerpts-v1.json'
    assert not dst.exists() and not excerpts.exists()
    helper=A/'direct_owner_transient_artifacts_v1.py'
    assert sha(helper)=='bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    spec=importlib.util.spec_from_file_location('clean_retire_owner',helper);owned=importlib.util.module_from_spec(spec);spec.loader.exec_module(owned)
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cohort_path=A/'moe-accum-model-clean-cohort-v1.json';cohort=json.loads(cohort_path.read_text())
        assert cohort['passed'] and cohort['complete'] and cohort['active_sample'] is None and len(cohort['completed'])==6
        receipt_hashes={str(cohort_path):sha(cohort_path)};evidence={};deletions={}
        def select(path,reason,replacement,restore=None):
            path=Path(path);assert path.is_file() and not path.is_symlink()
            row={'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'reason':reason,'replacement':replacement}
            if restore:row['restore_from_git']=restore
            deletions[str(path)]=row
        for item in cohort['completed']:
            f=Path(item['path']);assert sha(f)==item['sha256'];s=json.loads(f.read_text());receipt_hashes[str(f)]=sha(f)
            assert s['passed'] and s['complete'] and s['ownership_closed'] and not s['active'] and not s['fresh_faults'] and not s['driver_dump_exists']
            for c in s['commands']:
                assert owned.closed(c) and c['exit_code']==0
                for old in c['owners']+[c['direct_identity']]:
                    now=owned.identity(old['pid']);assert now is None or now['start_ticks']!=old['start_ticks']
            root=B/f"main-moe-accum-model-profile-window{item['window']}-tokens65536-clean-r{item['repetition']}-v3"
            kept=[]
            for name in ('engine.stderr','journal-before.stdout','journal-final.stdout','server.stderr','bench.stderr','prefill-only.stderr'):
                path=root/'owned'/name
                if path.exists():
                    lines=path.read_text(errors='replace').splitlines()
                    useful=[v for v in lines if v.startswith(('strata prefill:','strata MoE','strata serve: prompt','-- cursor:'))]
                    if name!='engine.stderr':useful=lines if path.stat().st_size<4096 else useful
                    kept.append({'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'selected_lines':useful})
            evidence[str(f)]={'receipt_sha256':sha(f),'original_status':'PASS','ownership_closed':True,'fresh_faults':s['fresh_faults'],'window_lines':s['window_lines'],'configuration':s['window_configuration_lines'],'logs':kept}
            for path in root.rglob('*'):
                if path.is_file() and path.suffix in ('.stdout','.stderr','.log','.jsonl'):
                    select(path,'Closed successful text/protocol/progress; individual receipt plus reviewed excerpts retain useful evidence.',str(excerpts))
            for path in [root/'result.json',*root.glob('stock-results/*/results.json')]:
                if path.exists():
                    original=json.loads(path.read_text());expected=s['result'] if path==root/'result.json' else s['benchmark_result'];assert original==expected
                    select(path,'Exact structured duplicate already preserved in immutable individual receipt.',str(f))
        wave=json.loads((A/'orca-research-wave1013-1015-v1.json').read_text());snapshot=Path(wave['source_snapshot']);rev=wave['source_commit']
        subprocess.run(['git','merge-base','--is-ancestor',rev,'origin/feature/sycl'],cwd=W,check=True)
        for relative,digest in wave['source_sha256'].items():
            path=snapshot/relative;assert sha(path)==digest
            restored=subprocess.run(['git','show',rev+':'+relative],cwd=W,capture_output=True,check=True).stdout
            assert hashlib.sha256(restored).hexdigest()==digest
            select(path,'Reviewed immutable research copy; source retained in pushed Git commit.',str(A/'orca-research-wave1013-1015-v1.json'),{'commit':rev,'path':relative})
        excerpts.write_text(json.dumps({'schema':'strata-moe-accum-clean-excerpts-v1','samples':evidence},ensure_ascii=False,indent=2)+'\n')
        manifest={'schema':'strata-moe-accum-clean-retention-v1','recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'complete':False,'receipt_hashes_unchanged':receipt_hashes,'excerpts':{'path':str(excerpts),'sha256':sha(excerpts)},'files':list(deletions.values()),'deleted':0,'logical_bytes_removed':0,'scope':'No source/model/binary/fixture-input/service/GPU changes. Snapshot is a verified Git-restorable copy; Main and review workers finished consuming it. Original result status and bytes unchanged.'}
        def save():dst.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        save()
        try:
            for row in manifest['files']:
                path=Path(row['path']);assert sha(path)==row['sha256'] and path.stat().st_size==row['bytes'];path.unlink();manifest['deleted']+=1;manifest['logical_bytes_removed']+=row['bytes'];save()
            for f,digest in receipt_hashes.items():assert sha(f)==digest
            manifest['complete']=True
        finally:save()
        print(json.dumps({k:manifest[k] for k in ('complete','deleted','logical_bytes_removed')}),flush=True)
if __name__=='__main__':main()
