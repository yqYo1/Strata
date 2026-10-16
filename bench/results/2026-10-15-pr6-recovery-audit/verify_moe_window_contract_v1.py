#!/usr/bin/env python3
"""Verify Main's host contract against independent interval/order invariants."""
import datetime
import fcntl
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

if sys.flags.optimize:
    raise RuntimeError('Run the host verifier without Python optimization.')
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    dst = A / 'moe-window-contract-host-v1.json'
    if dst.exists():
        raise RuntimeError('Preserve the original result; use a fresh version.')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic()
        module_path = A/'moe_window_contract_v1.py'
        spec = importlib.util.spec_from_file_location('moe_window_contract', module_path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m
        spec.loader.exec_module(m)
        samples = []
        sizes = [1,8191,8192,8193,16383,16384,16385,24576,32767,32768,
                 32769,39261,65535,65536,65537,131071,131072,262143,262144]
        for n in sizes:
            for g in sorted({0,n,(n//8192)*8192,min((n//8192)*8192,32768)}):
                for pos in sorted({0,262144-n}):
                    windows = m.schedule(n,g,pos)
                    pieces = [p for w in windows for p in w.pieces]
                    expected = [(r,min(8192,n-r),pos+r) for r in range(0,n,8192)]
                    assert [(p.row,p.count,p.position) for p in pieces] == expected
                    assert sum(w.rows for w in windows) == n
                    for w in windows:
                        assert 0 < w.rows <= 16384
                        assert all(0 < p.count <= 8192 for p in w.pieces)
                        assert all((p.row < g) == w.gpu for p in w.pieces)
                        assert not w.grouped or (w.gpu and len(w.pieces)==2)
                        if w.grouped:
                            for final in (False,True):
                                state=m.Lifetime(w,final)
                                for i in range(2):state.step('attention_store',i)
                                for i in range(2):state.step('prepare',i)
                                state.step('route_readback');state.step('combine')
                                for i in range(2):state.step('write_final',i)
                                state.step('checked_drain')
                                for i in range(2):state.step('emit',i)
                                assert state.stage=='done' and not state.region_live
                    samples.append(dict(rows=n,gpu_rows=g,position=pos,passed=True,
                        causal_chunks=len(pieces),grouped_windows=sum(w.grouped for w in windows),
                        fallback_chunks=sum(not w.grouped for w in windows)))
        plan = m.memory_plan()
        spans = list(plan['region'].values())
        for left,right in zip(spans,spans[1:]):
            assert left.end <= right.offset
        assert all(s.offset%256==0 and s.payload>0 and s.end<=1180698624 for s in spans)
        assert plan['region_used']==sum(s.reserved for s in spans)
        assert plan['prep_used']<=plan['region']['down_bf16'].payload
        assert plan['emb_mixed_then_bo_bytes']==8192*2560*4
        expert_cases=[]
        for window in (8193,16383,16384):
            for ne in (0,1,127,128,8191,8192,8193,window):
                if ne>window:continue
                jobs=m.expert_jobs(ne,window)
                assert sum(c for r,c in jobs)==ne
                assert all(c>0 and c<=8192 for r,c in jobs)
                assert [r for r,c in jobs]==list(range(0,ne,8192))
                for r,c in jobs:
                    span=m.source_upload_span(window*10-ne+r,c,window*10)
                    assert span['device_offset']==0 and c<=span['device_capacity']
                expert_cases.append(dict(window_rows=window,expert_rows=ne,
                                         jobs=jobs,passed=True))
        negative=[]
        def reject(name, fn, exception=ValueError):
            try:fn()
            except exception:negative.append(dict(case=name,passed=True));return
            raise AssertionError(name+' was accepted')
        w=m.schedule(16384,16384)[0]
        reject('prepare_before_second_PLE_attention',lambda:m.Lifetime(w).step('prepare',0))
        reject('publish_post_attention',lambda:m.Lifetime(w).step('emit',0))
        reject('nonzero_context_overflow',lambda:m.schedule(262144,32768,1))
        reject('residual_boundary_inside_chunk',lambda:m.schedule(39261,32769))
        reject('expert_rows_exceed_window',lambda:m.expert_jobs(16385,16384))
        reject('source_job_exceeds_device_list',lambda:m.source_upload_span(0,8193,163840))
        reject('source_job_exceeds_host_list',lambda:m.source_upload_span(163839,2,163840))
        s=m.Lifetime(w,True)
        for i in range(2):s.step('attention_store',i)
        for i in range(2):s.step('prepare',i)
        s.step('route_readback');s.step('combine');s.step('write_final',0)
        reject('MTP_before_second_HC_write',lambda:s.step('emit',0))
        reject('release_region_before_second_HC_write',lambda:s.step('checked_drain'))
        s.step('write_final',1);s.step('checked_drain')
        reject('second_callback_before_first',lambda:s.step('emit',1))
        s.step('emit',0);s.step('emit',1)
        reject('duplicate_callback',lambda:s.step('emit',1))
        f=m.Lifetime(w)
        reject('failed_drain_retains_owners',lambda:f.cancel(False),RuntimeError)
        assert f.invalid and f.region_live
        reject('resume_after_fault',lambda:f.step('attention_store',0))
        result=dict(schema='strata-moe-window-contract-host-v1',passed=True,complete=True,
            recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            execution_tree=subprocess.check_output(['git','rev-parse','HEAD'],cwd=A,text=True).strip(),
            input_hashes={str(f):sha(f) for f in (module_path,Path(__file__),A/'research-review-round968-v1.json')},
            cases=samples,expert_cases=expert_cases,negative_cases=negative,
            memory={**{k:v for k,v in plan.items() if k not in ('region','prep')},
                'region':{n:asdict(v) for n,v in plan['region'].items()},
                'prep':{n:asdict(v) for n,v in plan['prep'].items()}},
            elapsed_host_check_seconds=time.monotonic()-start,
            scope='Abstract scheduling, bounds, reservation arithmetic and ownership-state verification only. Not the C++ engine, kernel math, device allocation, actual DMA/queue retirement, fault recovery or full physical context/RESTORE validation.',
            GPU_executed=False,native_compiler_executed=False,
            inference_implemented=False,performance_adopted=False)
        dst.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(passed=True,schedule_cases=len(samples),
            expert_cases=len(expert_cases),negative_cases=len(negative),
            region_used=plan['region_used'],region_spare=1180698624-plan['region_used'],
            elapsed_host_check_seconds=result['elapsed_host_check_seconds'])))


if __name__=='__main__':
    main()
