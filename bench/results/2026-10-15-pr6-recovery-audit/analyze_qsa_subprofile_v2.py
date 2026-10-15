#!/usr/bin/env python3
"""Summarize closed bounded own-source model profile; no GPU action."""
import fcntl
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0,str(A/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner,require,sha


def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=A/'qsa-subprofile-model-v2.json'
        comparison=A/'qsa-subprofile-model-comparison-v2.json'
        target=A/'qsa-subprofile-analysis-v2.json'
        require(not target.exists(),'fresh profile analysis')
        r=json.loads(source.read_text());c=json.loads(comparison.read_text())
        require(r['passed'] and r['complete'] and not r['active'] and c['passed'],'closed qualified selected-row diagnostic')
        module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        require(all(module.closed(command) for command in r['commands']),'all commands closed')
        for command in r['commands']:
            for old in command['owners']:
                now=module.identity(old['pid'])
                require(now is None or now['start_ticks']!=old['start_ticks'],'owned identity absent')
        profile=r['coarse_profile'];header=profile['header'];row=profile['rows'][-1]
        duration=float(header['wrapper_ms']);require(math.isfinite(duration) and duration>0,'valid denominator')
        wall=dict(zip(header['wall_order'].split(','),row['wall_ms']))
        queue=dict(zip(header['queue_order'].split(','),row['queue_ms']))
        samples=dict(zip(header['queue_order'].split(','),row['samples']))
        request=r['benchmark_result']['results']['prefill_only']['requests'][0]
        result=dict(passed=True,complete=True,active=False,GPU_executed=False,
            references={source.name:sha(source),comparison.name:sha(comparison)},recipe_sha256=sha(Path(__file__)),
            binary_sha256=r['binary_sha256'],request_sha256=request['request_sha256'],
            actual_prompt_tokens=request['actual_prompt_tokens'],actual_read_tokens=request['actual_read_tokens'],
            wrapper_ms=duration,engine_prompt_ms=request['prompt_ms'],diagnostic_prompt_tps=request['prompt_tps'],
            host=profile['host'],wall_ms=wall,queue_interval_ms=queue,samples=samples,
            queue_interval_to_wrapper_ratio={k:v/duration for k,v in queue.items()},
            requested_upload_bytes=row['upload_bytes'],requested_input_bytes_host_device=row['input_bytes_host_device'],
            requested_store_bytes_host_device=row['store_bytes_host_device'],
            residual_input_queue_ms_host_device=row['input_queue_host_device_ms'],
            residual_store_queue_ms_host_device=row['store_queue_host_device_ms'],
            dense_lifecycle=r['dense_lifecycle'],performance_qualified=False,
            scope='Instrumented n1 same-request profile, not clean performance. Queue intervals are nonexclusive and include host submission gaps; requested copy bytes are not physical PCIe traffic. Ratios are attribution clues, not exclusive fractions or an Amdahl speedup bound. No task-quality, full physical context or RESTORE qualification.',finished_utc=module.utc())
        fine=r['qsa_profile'];q=fine['rows'][-1];sections=fine['header']['sections'].split(',')
        result['qsa_subinterval_ms']=dict(zip(sections,q['queue_ms']))
        result['qsa_host_wall_ms']=dict(zip(sections,q['host_wall_ms']))
        result['qsa_dispatch']={k:v for k,v in q.items() if isinstance(v,int) and k!='layer'}
        result['qsa_subinterval_to_wrapper_ratio']={k:v/duration for k,v in result['qsa_subinterval_ms'].items()}
        with target.open('x') as out:json.dump(result,out,indent=2,allow_nan=False);out.write('\n')
        print(json.dumps({k:result[k] for k in ('diagnostic_prompt_tps','wrapper_ms','engine_prompt_ms','host','wall_ms','queue_interval_ms','queue_interval_to_wrapper_ratio','qsa_subinterval_ms','qsa_host_wall_ms','qsa_dispatch')}),flush=True)


if __name__=='__main__':main()
