#!/usr/bin/env python3
"""MAIN ONLY proposed controller. Build, correctness, timing are strictly serial.

Use only after Main has closed other build/test/GPU/service/cleanup jobs. The shared
lock coordinates cooperative controllers; it does not prove outside work is idle.
No product edits, model loading, GPU calls, cache integration or cleanup.
"""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys

from build_probe import A,P,sha
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D=A.parent
OWNER=P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--core',type=int,required=True)
    p.add_argument('--multi',action='store_true',help='After nt1, separate nt2/3/4 custom arms')
    p.add_argument('--main-serialized-owner-confirmed',action='store_true',required=True)
    a=p.parse_args()
    pins=json.loads((A/'pins.json').read_text())
    if sha(OWNER)!=pins['owner_sha256']:raise RuntimeError('owner tool changed')
    spec=importlib.util.spec_from_file_location('owner',OWNER)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    out=a.out_dir.resolve()
    if out.exists():raise RuntimeError('new output directory required')
    if a.core not in os.sched_getaffinity(0):raise RuntimeError('core unavailable')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Same provider receipt already reviewed by Main for the IQ3_S probe.
        old=json.loads((D/'iq3s-rowpair-probe-v2.json').read_text())
        if not (old['passed'] and old['complete'] and not old['active']):raise RuntimeError('old run is not closed')
        for c in old['commands']:
            for i in c['owners']:
                now=m.identity(i['pid'])
                if now is not None and now['start_ticks']==i['start_ticks'] and now['state']!='Z':
                    raise RuntimeError('prior probe process still active')
        out.mkdir(mode=0o700,parents=True)
        env=dict(old['environment']);env['PYTHONDONTWRITEBYTECODE']='1'
        env['ASAN_OPTIONS']='detect_leaks=1:halt_on_error=1'
        env['UBSAN_OPTIONS']='halt_on_error=1:print_stacktrace=1'
        # Keep existing packed prefetch default (2048) in both builds. Included
        # production custom kernels read this only once at startup.
        env.pop('STRATA_IQ_PREFETCH',None)
        def topology():
            values={}
            for rel in ['topology/thread_siblings_list','cpufreq/scaling_governor','cpufreq/scaling_cur_freq','cache/index3/size','cache/index3/shared_cpu_list']:
                q=Path('/sys/devices/system/cpu')/('cpu'+str(a.core))/rel
                if q.exists():values[rel]=q.read_text().strip()
            return values
        r=dict(active=True,passed=False,complete=False,product_applied=False,GPU_executed=False,
               model_weights_modified=False,environment=env,core=a.core,topology_before=topology(),
               boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
               proposal_manifest_sha256=sha(A/'manifest.json'),source_pins=pins['source_sha256'],
               compiler=pins['compiler'],archive_sha256=pins['archive_sha256'],commands=[],correctness={},timings={},
               limits='Synthetic legal packed IQ GU/Q8_K only. Component evidence; no model speed, adoption, RAM cache hit-rate or quality qualification. TSC ticks are not core cycles.')
        def persist():(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
        owned=m.Owner(out/'owned',persist);r['commands']=owned.commands;persist()
        def run(label,args,wall=600):
            e,stdout,stderr=owned.run(label,args,env,P,wall=wall,cpu=wall*len(os.sched_getaffinity(0)),
                text_cap=64<<20,rss_cap=4<<30,total_cap=512<<20,file_cap=64<<20)
            if not m.closed(e) or e['exit_code']!=0:raise RuntimeError((label,e.get('error'),stderr.read_text()[-6000:]))
            return stdout,stderr
        try:
            for sanitized in [False,True]:
                tag='asan-ubsan' if sanitized else 'clean';folder=out/tag
                cmd=[sys.executable,str(A/'build_probe.py'),'--out-dir',str(folder)]
                if sanitized:cmd.append('--sanitize')
                run('build-'+tag,cmd)
                exe=folder/'fixture'
                stdout,stderr=run('correctness-'+tag,['/usr/bin/taskset','-c',str(a.core),str(exe),'--correctness'],1200)
                rows=[json.loads(s) for s in stdout.read_text().splitlines()]
                if not (rows[-1]['kind']=='correctness_close' and rows[-1]['pass']):raise RuntimeError('incomplete correctness')
                r['correctness'][tag]=dict(rows=rows,stderr=stderr.read_text(),binary_sha256=sha(exe),stdout_sha256=sha(stdout),build_receipt=json.loads((folder/'build-receipt.json').read_text()))
                persist()
            exe=out/'clean/fixture'
            for nt in ([1,2,3,4] if a.multi else [1]):
                for experts,inner in [(1,8),(16,2),(64,1)]:
                    if sha(exe)!=r['correctness']['clean']['binary_sha256']:raise RuntimeError('binary changed')
                    stdout,stderr=run(f'timing-e{experts}-nt{nt}',[
                        '/usr/bin/taskset','-c',str(a.core),str(exe),'--timing',str(experts),str(nt),'12',str(inner),
                        '--correctness-approved-by-main'],1200)
                    rows=[json.loads(s) for s in stdout.read_text().splitlines()]
                    samples=[x for x in rows if x['kind']=='timing_sample']
                    if not (rows[-1]['kind']=='timing_close' and rows[-1]['complete'] and len(samples)==432):raise RuntimeError('incomplete timing')
                    if any(x['migration_observed'] or x['cpu_start']!=a.core or x['cpu_end']!=a.core for x in samples):raise RuntimeError('migration')
                    per_format={}
                    for ty in [18,22,21]:
                        s=[x for x in samples if x['type']==ty]
                        medians={arm:statistics.median(x['thread_cpu_ns_per_output_row'] for x in s if x['arm']==arm) for arm in sorted({x['arm'] for x in s})}
                        # Paired ABBA ratios retain round/order context; medians
                        # across all unrelated samples alone do not decide a win.
                        ratios=[]
                        comparisons=[('expanded_exact','ggml'),('expanded_exact','existing_exact'),('existing_exact','ggml')] if nt==1 else [('expanded_custom','packed_custom'),('packed_custom','ggml'),('expanded_custom','ggml')]
                        for rd in range(12):
                            for pair_id,(candidate,control) in enumerate(comparisons):
                                pair=[x for x in s if x['round']==rd and x['pair']==pair_id]
                                values={}
                                for clock in ['thread_cpu_ns_per_output_row','wall_ns_per_output_row']:
                                    ct=statistics.mean(x[clock] for x in pair if x['arm']==candidate)
                                    bt=statistics.mean(x[clock] for x in pair if x['arm']==control)
                                    values[clock+'_ratio']=ct/bt
                                ratios.append(dict(round=rd,pair=pair_id,numerator=candidate,denominator=control,**values))
                        per_format[str(ty)]=dict(median_thread_cpu_ns_per_output_row=medians,paired_ratios=ratios)
                    r['timings'][f'e{experts}-nt{nt}']=dict(rows=rows,per_format=per_format,stderr=stderr.read_text(),stdout_sha256=sha(stdout))
                    persist()
            for path,h in {**pins['source_sha256'],**pins['archive_sha256']}.items():
                if sha(path)!=h:raise RuntimeError('input changed: '+path)
            r.update(passed=True,complete=True,topology_after=topology())
        except BaseException as e:r['error']=repr(e);raise
        finally:r['active']=False;persist()

if __name__=='__main__':main()
