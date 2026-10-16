#!/usr/bin/env python3
"""Exercise exact submission coverage and adversarial ownership transitions."""
import datetime
import fcntl
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

if sys.flags.optimize:
    raise RuntimeError('Run the host verifier without Python optimization.')
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    dst=A/'moe-window-contract-host-v2.json'
    if dst.exists():raise RuntimeError('Preserve the original receipt.')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=time.monotonic()
        old=A/'moe-window-contract-host-v1.json'
        assert sha(old)=='f907025529a17b719a4b4eee62545d5f76e956ce734f7e320d384444f30d193a'
        v1=json.loads(old.read_text());assert v1['passed'] and v1['complete']
        module=A/'moe_window_contract_v2.py'
        spec=importlib.util.spec_from_file_location('moe_window_contract_v2',module)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        def counts_for(n,kind):
            if kind=='ten_hot':return (n,)*10+(0,)*502
            if kind=='balanced':
                q,r=divmod(n*10,512);return tuple(q+(i<r) for i in range(512))
            q,r=divmod(n*8,510)
            return (n,n)+tuple(q+(i<r) for i in range(510))
        def prepared(n=16384,kind='ten_hot',final=True):
            s=m.Lifetime(m.base.schedule(n,n,262144-n)[0],final)
            for i in range(2):s.step('attention_store',i)
            for i in range(2):s.step('prepare',i)
            s.step('route_readback',counts=counts_for(n,kind))
            return s
        def issue_all(s):
            e0=None
            for job in s.jobs:
                e=job[0]
                if e!=e0:s.step('dequantize',e);e0=e
                span=s.step('expert_job',job)
                assert span['device_offset']==0 and span['rows']<=8192
                assert span['host_offset']+span['rows']<=s.window.rows*10
                assert span['down_output_row']==span['host_offset']
        samples=[]
        for n in (8193,16383,16384):
            for kind in ('ten_hot','balanced','two_hot'):
                for final in (False,True):
                    s=prepared(n,kind,final);issue_all(s)
                    assert len(s.dequantized)==sum(c>0 for c in counts_for(n,kind))
                    s.step('combine')
                    for i in range(2):s.step('write_final',i)
                    s.step('checked_drain');s.step('reuse_host_source')
                    for i in range(2):s.step('emit',i)
                    assert s.stage=='done' and not s.region_live and not s.host_source_live
                    assert [(e['row'],e['count'],e['position']) for e in s.emissions]==[
                        (p.row,p.count,p.position) for p in s.window.pieces]
                    samples.append(dict(rows=n,synthetic_distribution=kind,final_layer=final,
                        passed=True,expert_dequantizations=len(s.dequantized),
                        GEMM_pairs=len(s.jobs),routed_rows=s.rows_submitted,
                        emissions=s.emissions))
        negative=[]
        def reject(name,fn,exception=ValueError):
            try:fn()
            except exception:negative.append(dict(case=name,passed=True));return
            raise AssertionError(name+' was accepted')
        s=prepared()
        reject('combine_without_any_expert_job',lambda:s.step('combine'))
        reject('job_without_dequantization',lambda:s.step('expert_job',s.jobs[0]))
        s.step('dequantize',0)
        reject('duplicate_dequantization',lambda:s.step('dequantize',0))
        reject('skip_first_job',lambda:s.step('expert_job',s.jobs[1]))
        s.step('expert_job',s.jobs[0])
        reject('repeat_job',lambda:s.step('expert_job',s.jobs[0]))
        reject('combine_with_missing_jobs',lambda:s.step('combine'))
        reject('host_H2D_owner_reused_early',lambda:s.step('reuse_host_source'))
        reject('queue_changed_during_window',lambda:s.step('change_queue'))
        for bad,name in [((16384,)*512,'route_mass_overflow'),
                         ((16385,)+(0,)*511,'expert_exceeds_window'),
                         ((-1,)+(0,)*511,'negative_expert_count'),
                         ((1,)*511,'missing_expert_bin')]:
            q=m.Lifetime(m.base.schedule(16384,16384)[0])
            for i in range(2):q.step('attention_store',i)
            for i in range(2):q.step('prepare',i)
            reject(name,lambda q=q,bad=bad:q.step('route_readback',counts=bad))
        s=prepared();issue_all(s);s.step('combine');s.step('write_final',0)
        reject('first_MTP_while_second_injection_live',lambda:s.step('emit',0))
        reject('partial_write_region_release',lambda:s.step('checked_drain'))
        reject('failed_retirement_preserves_host_and_device_owners',lambda:s.cancel(False),RuntimeError)
        assert s.invalid and s.region_live and s.host_source_live
        reject('resume_invalid_window',lambda:s.step('write_final',1))
        c=prepared();c.cancel(True)
        assert c.invalid and not c.region_live and not c.host_source_live
        # Source-list rebasing is essential even when global offset is larger
        # than the entire small device list. It is not a device address offset.
        offset=163840-8192
        span=m.base.source_upload_span(offset,8192,163840)
        assert offset>span['device_capacity'] and span['device_offset']==0
        result=dict(schema='strata-moe-window-contract-host-v2',passed=True,complete=True,
            recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            execution_tree=subprocess.check_output(['git','rev-parse','HEAD'],cwd=A,text=True).strip(),
            inputs={str(f):sha(f) for f in (module,Path(__file__),m.BASE,old)},
            cases=samples,negative_cases=negative,
            elapsed_host_check_seconds=time.monotonic()-start,
            scope='Extends frozen v1 scheduling/memory checks with exact synthetic expert/job coverage, once-per-expert dequantization, immutable host source, and callback/retirement state checks. No actual tensors, numerical math, C++ engine, GPU, queue, DMA, allocation, fault recovery or physical-context/RESTORE execution.',
            distributions_are_synthetic=True,GPU_executed=False,native_compiler_executed=False,
            inference_implemented=False,performance_adopted=False)
        dst.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(passed=True,expert_phase_cases=len(samples),
            negative_cases=len(negative),elapsed_host_check_seconds=result['elapsed_host_check_seconds'])))


if __name__=='__main__':main()
