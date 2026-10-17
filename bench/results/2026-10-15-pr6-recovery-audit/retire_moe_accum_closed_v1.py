#!/usr/bin/env python3
"""Retire reviewed closed component/model logs; preserve immutable receipts."""
import fcntl, hashlib, importlib.util, json, pathlib, re, subprocess
A=pathlib.Path(__file__).resolve().parent
P=A.parents[2]
B=pathlib.Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
    dest=A/'moe-accum-closed-artifact-retention-v1.json';assert not dest.exists()
    names=[f'moe-accum-device-{m}-v1.json' for m in ('build','smoke','numeric','short-weights','short-down','legacy-plan','negative-source','negative-slot','unwind','hot65536','balanced65536')]
    names+=['moe-accum-plan-host-v2.json','moe-accum-engine-syntax-v1.json','moe-accum-engine-build-v1.json','moe-accum-model-profile-window2-tokens8192-v1.json']
    names += [f'moe-accum-model-profile-window{m}-tokens{n}-v2.json' for m in (2,8) for n in (32768,65536)]
    receipts=[]
    for name in names:
        f=A/name;x=json.loads(f.read_text());assert x['ownership_closed'] and not x.get('active',False),name
        assert not x.get('fresh_faults',[]) and not x.get('driver_dump_exists',False) and not x.get('driver_dump_entries',[])
        receipts.append((f,x))
    ownerpath=A/'direct_owner_transient_artifacts_v1.py';assert sha(ownerpath)=='bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    spec=importlib.util.spec_from_file_location('retention_owned',ownerpath);owned=importlib.util.module_from_spec(spec);spec.loader.exec_module(owned)
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        entries=[];excerpts={};seen_logs=set();receipt_pins={str(f):sha(f) for f,_ in receipts}
        for f,x in receipts:
            for c in x['commands']:
                assert owned.closed(c)
                for old in c['owners']+[c['direct_identity']]:
                    now=owned.identity(old['pid']);assert now is None or now['start_ticks']!=old['start_ticks']
            roots=set()
            for c in x['commands']:
                for arg in c.get('argv',[]):
                    if isinstance(arg,str) and arg.startswith(str(B)+'/main-moe-accum-'):
                        roots.add(B/arg[len(str(B))+1:].split('/',1)[0])
            if x.get('root'):roots.add(pathlib.Path(x['root']))
            if f.name.startswith('moe-accum-device-') and x.get('mode'):roots.add(B/f"main-moe-accum-device-{x['mode']}-v1")
            special={'moe-accum-plan-host-v2.json':'main-moe-accum-plan-host-v2','moe-accum-engine-syntax-v1.json':'main-moe-accum-engine-syntax-v1','moe-accum-engine-build-v1.json':'main-moe-accum-engine-build-v1'}
            if f.name in special:roots.add(B/special[f.name])
            for root in roots:
                for log in sorted((root/'owned').glob('*')):
                    if log in seen_logs or log.suffix not in ('.stdout','.stderr') or not log.is_file():continue
                    seen_logs.add(log);data=log.read_bytes();text=data.decode('utf-8',errors='replace')
                    selected=[line for line in text.splitlines() if re.search(r'(fixture_status=|expected_kernels=|strata MoE (?:window|accumulator window)|strata prefill: layer-major candidate|strata trace: read |strata serve: prompt |invalid choice: 8192|AssertionError: 2)',line)]
                    completed=re.findall(r'strata trace: layer-major layer (\d+) of 48: completed (\d+) chunks',text)
                    if completed:
                        assert [int(i) for i,_ in completed]==list(range(48))
                        selected.append('Main verified all48 completed-layer progress entries in order; chunks='+','.join(sorted({v for _,v in completed})))
                    excerpts[str(log)]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'selected_lines':selected}
                    entries.append({'path':str(log),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'reason':'Closed build/component/model test; commands, outcomes, API counts, environment/runtime/owner/fault and useful progress evidence retained in immutable receipts and compact excerpts.'})
        eligibility=json.loads((A/'orca-research-round1011-eligibility-v1.json').read_text());source_root=pathlib.Path(eligibility['snapshot_path']);commit='c05c2545d90fb3bedcfff51400919977b6c276ca';w=pathlib.Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/verify-moe-window')
        for rel,h in eligibility['prerequisites'][0]['evidence']['paths_sha256'].items():
            src=source_root/rel;assert sha(src)==h
            data=subprocess.check_output(['git','show',commit+':'+rel],cwd=w);assert hashlib.sha256(data).hexdigest()==h
            entries.append({'path':str(src),'sha256':h,'bytes':src.stat().st_size,'reason':'R1011/R1012 audit complete; exact source remains in pushed Git commit.','restore_from_git':commit+':'+rel})
        for n in (1005,1006,1008,1009,1011):
            for suffix in ('launch','ready','retry','skip-update','startup-read','check1'):
                f=A/f'orca-research-round{n}-{suffix}-v1.json'
                if f.exists():entries.append({'path':str(f),'sha256':sha(f),'bytes':f.stat().st_size,'reason':'Research closed/released/reviewed; startup-update refusal before Task input and successful retry/provider/settlement recorded in compact indexed receipts.'})
        assert len({e['path'] for e in entries})==len(entries)
        excerpt_file=A/'moe-accum-closed-artifact-excerpts-v1.json';assert not excerpt_file.exists();excerpt_file.write_text(json.dumps({'receipts':receipt_pins,'excerpts':excerpts},indent=2)+'\n')
        manifest={'owner':'Main','status':'planned','recipe_sha256':sha(__file__),'entries':entries,'total_bytes':sum(e['bytes'] for e in entries),'immutable_receipt_pins':receipt_pins,'excerpts_path':str(excerpt_file),'excerpts_sha256':sha(excerpt_file),'active_run_touched':False,'binaries_models_profiles_wrappers_build_objects_touched':False,'first_inventory_attempt':{'status':'refused_before_mutation','reason':'Repeated build-root logs referenced by several fixture receipts; deduplicated by exact path in this recipe.','deleted_files':0}}
        dest.write_text(json.dumps(manifest,indent=2)+'\n')
        for e in entries:
            f=pathlib.Path(e['path']);assert sha(f)==e['sha256'];f.unlink()
        for directory in sorted((q for q in source_root.rglob('*') if q.is_dir()),key=lambda q:len(q.parts),reverse=True):directory.rmdir()
        source_root.rmdir()
        assert all(sha(f)==h for f,h in receipt_pins.items()) and all(not pathlib.Path(e['path']).exists() for e in entries)
        manifest.update(status='verified_deleted',receipt_bytes_and_statuses_unchanged=True,all_deleted_paths_absent=True);dest.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'retired_files':len(entries),'retired_logical_bytes':manifest['total_bytes'],'all_receipts_unchanged':True}))
if __name__=='__main__':main()
