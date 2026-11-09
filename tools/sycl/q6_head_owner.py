#!/usr/bin/env python3
"""Read the current real-head fixture protocol and seal its fault/closure proof.

The earlier packed protocol is recoverable from484d403e6 and its successful
evidence0840eccd2. The current fixture shares canonical weights in both arms.
These pure readers serve the production-linked model fixture/entry gate.
Main-authored;repository LICENSE.
"""
import math
import re
import sys
sys.dont_write_bytecode=True
from measurement_guards import require
import ur_validation as validation

def parse_native(text,dpas):
    require(type(dpas) is int and dpas in (0,1),'explicit owner arm')
    lines=text.splitlines()
    captures=f'CAPTURE graphs=3 nodes_each=2 dpas_decode_calls={dpas} dpas_prompt_calls=0 dpas_C1_calls=0'
    require(lines.count(captures)==3 and sum(v.startswith('CAPTURE ') for v in lines)==3,'all graph families recaptured')
    end=f'OWNER_CHECK dpas={dpas} numeric_cases=4 blocked_cases=4 restore_cycles=2 cancel_cases=1 graph_replays=12 canonical_immutable=PASS extra_weights=absent model=0 Verifier_owner=0'
    require(lines.count(end)==1 and lines.count('HEAD_OWNER_EXIT passed=1 normal=1 provider_teardown=unqualified')==1
        and lines[-1]=='HEAD_OWNER_EXIT passed=1 normal=1 provider_teardown=unqualified','complete owner/exit proof')
    numeric=[]
    for line in lines:
        if not line.startswith('NUMERIC '):continue
        m=re.fullmatch(r'NUMERIC cycle=(-?\d+) seed=(\d+) outputs=993280 oracle_rows=543 max_abs_pair_error=([0-9.eE+-]+) C1=PASS prompt=reference guards=PASS',line)
        require(m is not None,'numeric row schema')
        cycle,seed,value=int(m[1]),int(m[2]),float(m[3])
        require(math.isfinite(value) and value>=0,'finite numerical summary')
        numeric.append(dict(cycle=cycle,seed=seed,max_abs_pair_error=value,outputs=993280,oracle_rows=543))
    require([(v['cycle'],v['seed']) for v in numeric]==[(-1,7),(-1,19),(0,31),(1,32)],'complete fresh-data numerics')
    expected_blocked=['BLOCKED phase='+p+' canonical=PASS dpas=PASS' for _ in range(2)
        for p in ('detached','restored_not_published')]
    require([v for v in lines if v.startswith('BLOCKED ')]==expected_blocked,'publication/blocked owner proof')
    reclaimed=[f'RECLAIM cycle={c} overwritten_device_bytes=521472000 tail=PASS' for c in range(2)]
    require([v for v in lines if v.startswith('RECLAIM ')]==reclaimed,'temporary reuse/write extent')
    retired=[f'RETIREMENT cycle={c} canonical_span=retired extra_weight_span=absent' for c in range(2)]
    require([v for v in lines if v.startswith('RETIREMENT ')]==retired,'registered pointer retirement')
    admission=[v for v in lines if v.startswith('HEAD admitted=')]
    require(len(admission)==1,'one actual admission')
    match=re.fullmatch(r'HEAD admitted=1 dpas=([01]) layout=canonical canonical_bytes=521472000 extra_device_weight_bytes=0 extra_host_weight_bytes=0 max_alloc_bytes=(\d+)',admission[0])
    require(match and int(match[1])==dpas and int(match[2])>=521472000,
        'admitted canonical head/zero extra weights/single-allocation bound')
    timings=[]
    for line in lines:
        if not line.startswith('TRANSITION '):continue
        m=re.fullmatch(r'TRANSITION phase=(\w+) cycle=(-?\d+) wall_ns=(\d+) thread_cpu_ns=(\d+) scope=whole_wrapper',line)
        require(m is not None,'transition clock schema')
        timings.append(dict(phase=m[1],cycle=int(m[2]),wall_ns=int(m[3]),thread_cpu_ns=int(m[4])))
    require([(v['phase'],v['cycle']) for v in timings]==[('load',-1)]+[(p,c) for c in range(2)
        for p in ('prepare','retire_readers','detach','restore','resume')],'all head transition clocks')
    return dict(dpas=dpas,numeric=numeric,transitions=timings,restore_cycles=2,blocked_cases=4,
        graph_replays=12,graphs_each_capture=3,nodes_each_graph=2,canonical_bytes=521472000,
        layout='canonical',extra_device_weight_bytes=0,extra_host_weight_bytes=0,max_alloc_bytes=int(match[2]),
        canonical_immutable=True,
        scope='Actual NativeHead and selector with fixture graphs;not production Verifier/model ownership or speed.')

def finish_gpu_probe(outcome,scope,cursor,boot):
    # The shared finalizer intentionally rejects a caller that omits its
    # acceptance field. Never infer acceptance from normal process exit alone.
    outcome['host_probe_accepted']=bool('error' not in outcome and outcome.get('passed') and outcome.get('assets_pre_post_match')
        and outcome.get('native_exit_code')==0 and outcome.get('owner') and outcome.get('validation'))
    validation.finish(outcome,scope,cursor,boot)

if __name__=='__main__':
    raise SystemExit('Import parse_native/finish_gpu_probe for model admission;the closed CLI is archived at484d403e6.')
