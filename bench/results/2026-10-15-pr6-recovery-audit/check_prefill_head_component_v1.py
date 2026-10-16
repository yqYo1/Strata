#!/usr/bin/env python3
"""One serial isolated NativeHead GPU roundtrip with full first-check UR logs."""
import fcntl
import json
import os
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
sys.path.insert(0,str(A))
from analyze_prefill_window_profile_v2 import B,P,load_owner,require,sha
from run_prefill_window_profile_v4 import DEVICE,FAULT,boot_id


def main():
    root=B/'main-prefill-head-component-root-v1';target=A/'prefill-head-component-v1.json'
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not root.exists() and not target.exists(),'fresh component experiment')
        bp=A/'prefill-head-component-build-v1.json';build=json.loads(bp.read_text())
        product=A/'prefill-head-release-build-v1.json';native=json.loads(product.read_text())
        require(build['passed'] and build['complete'] and not build['active'] and native['passed']
                and native['complete'] and not native['active'],'complete build gates')
        require(sha(Path(build['binary']))==build['binary_sha256'] and
                sha(A/'prefill_head_roundtrip_fixture_v1.cpp')==build['source_sha256'],'fixture identity')
        require(all(sha(P/n)==h for n,h in native['source_pins'].items()),'exact reviewed product source')
        require(all(sha(Path(n))==h for n,h in build['linked_inputs'].items()),'exact linked archives')
        review=A/'research-prefill-head-release-actual-review-round683-v1.txt'
        require(sha(review)=='10ccdc9752c98c7cd9ab1eb2cba922bd388a72fec81ab99bb9aaa6c000b495f3','independent source review')
        latest_path=A/'prefill-window-profile-long-v4.json';latest=json.loads(latest_path.read_text())
        require(latest['passed'] and latest['complete'] and not latest['active'] and not latest['fresh_faults']
                and latest['boot_unchanged'] and not latest['driver_dump_exists'] and latest['boot_id']==boot_id(),
                'latest complete GPU closure')
        module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        for c in latest['commands']+latest['control_commands']+native['commands']+build['commands']:
            require(module.closed(c),'prior command closure')
            for old in c['owners']:
                now=module.identity(old['pid'])
                require(now is None or now['start_ticks']!=old['start_ticks'],'prior owned process still present')
        baseline=json.loads((A/'prefill-only-clean-iq1-c4096-v1.json').read_text())
        shards=sorted(baseline['model_identity'])
        require(len(shards)==2 and all(Path(n).stat().st_size==v['bytes'] and
                Path(n).stat().st_mtime_ns==v['mtime_ns'] for n,v in baseline['model_identity'].items()),'model identities')
        require((DEVICE/'driver').resolve().name=='xe' and not (DEVICE/'devcoredump').exists(),'driver/dump gate')
        root.mkdir(mode=0o700)
        env=dict(latest['environment'])
        env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_TRACING='level:info;flush:info;output:stderr',XDG_CACHE_HOME=str(root/'cache'))
        require(not any(k=='LD_PRELOAD' or k.startswith(('UNITRACE_','PTI_')) or 'TOKEN' in k or 'API_KEY' in k for k in env),
                'no profiler/private environment')
        require(env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD']=='1' and env['EnableDirectSubmission']=='0'
                and env['SYCL_CACHE_PERSISTENT']=='0' and env['STRATA_SYCL_PREFILL_SAME_QUEUE']=='1',
                'unchanged safety environment')
        r=dict(passed=False,complete=False,active=True,GPU_executed=False,commands=[],
            recipe_sha256=sha(Path(__file__)),build_receipt_sha256=sha(bp),product_build_sha256=sha(product),
            independent_review_sha256=sha(review),environment=env,binary_sha256=build['binary_sha256'],
            model_identity=baseline['model_identity'],boot_id=boot_id(),
            limits=dict(wall_seconds=180,RSS_bytes=12<<30,raw_text_bytes=64<<20,job_bytes=128<<20),
            scope='Isolated actual Q6 NativeHead521472000B, two exact snapshot/restore cycles and full logit-bit comparison. No MTP/verifier graphs, prompt, speed, quality, physical262144 or RESTORE qualification.',
            retention=dict(consumer='Main first checked real NativeHead restoration and owner comparison',
                byte_budget=64<<20,next_review='After extracting verified result/UR errors and normal ownership closure; retire successful full API log.'))
        def persist():(root/'record.json').write_text(json.dumps(r,indent=2)+'\n')
        owner=module.Owner(root/'owned',persist);r['commands']=owner.commands;persist()
        host_env=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')
        def utility(label,args):
            c,out,err=owner.run(label,args,host_env,P,wall=60,cpu=60,rss_cap=2<<30,
                text_cap=4<<20,total_cap=128<<20,file_cap=4<<20)
            require(module.closed(c) and c['exit_code']==0,label+' failed: '+err.read_text()[-1000:])
            return out.read_text()
        cursor=None
        try:
            text=utility('journal-before',['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'])
            match=re.search(r'-- cursor: (\S+)',text);require(match,'journal cursor');cursor=match[1]
            r['journal_before_cursor']=cursor;persist()
            r['GPU_executed']=True;persist()
            c,out,err=owner.run('actual-head-component',[build['binary'],*shards],env,P,
                wall=180,cpu=180*len(os.sched_getaffinity(0)),rss_cap=12<<30,
                text_cap=64<<20,total_cap=128<<20,file_cap=64<<20)
            r['component_exit_code']=c.get('exit_code');r['stdout']=out.read_text()
            stderr=err.read_text()
            r['UR_error_lines']=[line for line in stderr.splitlines() if re.search(r'(?:UR|ZE)_RESULT_ERROR_|head component (?:fail-stop|exception)',line)]
            require(module.closed(c) and c['exit_code']==0,'actual component failed: '+stderr[-4000:])
            require(not r['UR_error_lines'],'runtime error in first-check API log')
            pattern=r'head_cycle (\d+) packed_bytes=(\d+) weights_equal=1 logits_equal=1 free_before=(\d+) free_detached=(\d+) free_restored=(\d+) total=(\d+) address_changed=(\d+)'
            values=[list(map(int,m)) for m in re.findall(pattern,r['stdout'])]
            require(len(values)==2 and [v[0] for v in values]==[0,1] and all(v[1]==521472000 for v in values),
                    'both complete actual cycles')
            r['cycles']=[dict(cycle=v[0],packed_bytes=v[1],weights_bitwise_equal=True,logits_bitwise_equal=True,
                free_before=v[2],free_detached=v[3],free_restored=v[4],total=v[5],address_changed=bool(v[6]),
                observed_free_delta=v[3]-v[2]) for v in values]
            require(re.search(r'head_component PASS cycles=2 weights_fnv1a64=[0-9a-f]{16} logits_fnv1a64=[0-9a-f]{16}',r['stdout']),
                    'final checked teardown marker')
            r.update(passed=True,complete=True)
        except BaseException as e:r['error']=repr(e);raise
        finally:
            try:
                if cursor:
                    entries=[json.loads(s) for s in utility('journal-final',['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if s.strip()]
                    r['fresh_faults']=[e for e in entries if FAULT.search(str(e.get('MESSAGE','')))]
                    r['driver_dump_exists']=(DEVICE/'devcoredump').exists();r['boot_unchanged']=boot_id()==r['boot_id']
                    if r['fresh_faults'] or r['driver_dump_exists'] or not r['boot_unchanged']:r['passed']=False
            finally:
                r['active']=owner.active is not None;r['finished_utc']=module.utc();persist()
                target.write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps(dict(passed=r['passed'],cycles=len(r.get('cycles',[])),fresh_faults=len(r.get('fresh_faults',[])))))


if __name__=='__main__':main()
