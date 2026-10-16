#!/usr/bin/env python3
"""Analyze closed paired diagnostics, exact issuer shapes and read-only sensors."""
import fcntl
import hashlib
import importlib.util
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def numeric_line(line):
    return {k:float(v) if '.' in v else int(v) for k,v in
            re.findall(r'([a-z_][a-z_0-9]*)=([0-9]+(?:\.[0-9]+)?)',line)}


def distribution(values):
    return dict(n=len(values),min=min(values),median=statistics.median(values),
                max=max(values)) if values else dict(n=0)


def sensors_summary(data):
    summaries={}
    for name,rows in [('all_observations',data['samples']),('interior_layer_progress',[
        row for row in data['samples']
        if (m:=re.search(r'layer-major layer (\d+) of 48: completed',row.get('last_layer_progress','')))
        and int(m[1])<47])]:
        d=dict(samples=len(rows),read_wall_s=sum(row['monotonic_end']-row['monotonic_start'] for row in rows))
        for key in ('gt0_act_freq','gt0_cur_freq','temp2_input','temp3_input'):
            vals=[int(row['values'][key]) for row in rows if key in row['values']]
            d[key]=distribution(vals)
            if key=='gt0_act_freq':
                d['gt0_nonzero_act_freq']=distribution([v for v in vals if v>0])
                d['gt0_zero_act_freq_samples']=sum(v==0 for v in vals)
        d['throttle_reasons']={}
        d['sensor_errors']={}
        for row in rows:
            reason=row['values'].get('gt0_throttle_reasons','UNAVAILABLE')
            d['throttle_reasons'][reason]=d['throttle_reasons'].get(reason,0)+1
            for key,error in row['errors'].items():
                d['sensor_errors'][key]=d['sensor_errors'].get(key,0)+1
        watts=[];invalid=[]
        for left,right in zip(rows,rows[1:]):
            if 'energy1_input' not in left['values'] or 'energy1_input' not in right['values']:
                continue
            dt=(right['monotonic_start']+right['monotonic_end']-left['monotonic_start']-left['monotonic_end'])/2
            de=int(right['values']['energy1_input'])-int(left['values']['energy1_input'])
            if de<0 or not 0<dt<=3:
                invalid.append(dict(delta_microjoule=de,delta_seconds=dt));continue
            watts.append(de/1e6/dt)
        d['approx_card_watts_from_adjacent_energy_reads']=distribution(watts)
        d['invalid_energy_intervals']=invalid
        summaries[name]=d
    return dict(summaries=summaries,stopped_at_cap=data['stopped_at_cap'],
                units=dict(temp2_input='millidegree_C; label in original readings',
                           temp3_input='millidegree_C; label in original readings',
                           energy1_input='microjoule'),
                interpretation='1Hz non-atomic samples; interior selection uses last observed completed layer <47, excludes the first layer and can include transition delays. Power is approximate whole-card power, not process power. No utilization/throughput attribution.')


def shape_summary(census):
    assert census['complete'] and census['issuer_exact_ne_verified']
    n=census['tokens'];assert census['configured_chunk']==8192
    hist={}
    for row in census['layers']:
        assert row['tokens']==n and row['chunks']==(n+8191)//8192
        assert row['gu_logical_nk']==[1280,2560] and row['down_logical_nk']==[2560,640]
        assert sum(ne*count for ne,count,gu,down in row['histogram'])==n*10
        for ne,count,gu,down in row['histogram']:
            assert gu==down==(count if ne else 0)
            hist[ne]=hist.get(ne,0)+count
    calls=sum(count for ne,count in hist.items() if ne)
    rows=sum(ne*count for ne,count in hist.items())
    assert rows==48*n*10
    ranges=[(1,15),(16,31),(32,63),(64,127),(128,255),(256,511),(512,1023),(1024,8192)]
    buckets=[dict(ne_range=[lo,hi],expert_chunk_calls=sum(c for ne,c in hist.items() if lo<=ne<=hi),
                  routed_rows=sum(ne*c for ne,c in hist.items() if lo<=ne<=hi)) for lo,hi in ranges]
    policies={}
    for name in ('current_auto','prospective_half_wide_above128'):
        padded_rows=0;gu_wg=0;down_wg=0;logical_threads=0
        for ne,count in hist.items():
            if not ne:continue
            if ne<128:bm,bn,threads=16,64,64
            elif name=='current_auto':bm,bn,threads=128,128,128
            else:bm,bn,threads=64,128,64
            gm=(ne+bm-1)//bm;g1=gm*((1280+bn-1)//bn);g2=gm*((2560+bn-1)//bn)
            padded_rows+=gm*bm*count;gu_wg+=g1*count;down_wg+=g2*count
            logical_threads+=(g1+g2)*threads*count
        policies[name]=dict(nominal_padded_rows_per_GEMM_kind=padded_rows,
            nominal_M_padding_factor=padded_rows/rows,GU_workgroups=gu_wg,down_workgroups=down_wg,
            launched_work_items=logical_threads,measured_kernel_time=False)
    return dict(expert_chunk_calls=calls,MoE_GEMM_calls=2*calls,routed_rows_per_GEMM_kind=rows,
        zero_expert_chunk_bins=hist.get(0,0),buckets=buckets,
        logical_MoE_GEMM_FLOP=2*rows*(1280*2560+2560*640),
        logical_dequantized_weight_write_bytes_if_once_per_nonzero_expert_chunk=2*calls*(1280*2560+2560*640),
        logical_dequantized_write_bytes_are_not_physical_VRAM_traffic=True,
        policies=policies,policy_limit='Derived grid/padding counts only; no occupancy, realized DPAS instruction count, bandwidth saturation or candidate speed measured.')


def main():
    dst=A/'qsa-bf16-prefix-tilepack-profile-summary-v2.json';assert not dst.exists()
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pair=A/'qsa-bf16-prefix-tilepack-profile-pair-v2.json';p=load(pair)
        assert p['passed'] and p['complete'] and p['active_controller'] is None and len(p['completed'])==2
        parser=A/'analyze_qsa_bf16_prefix_profile_v1.py'
        assert sha(parser)=='53e078c8af454403b8e878991d5c5799f2a7d0a711a6c989f367e34020b4b0c7'
        spec=importlib.util.spec_from_file_location('qualified_profile_parser',parser)
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        owner_path=A/'direct_owner_transient_artifacts_v1.py'
        assert sha(owner_path)=='bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
        spec=importlib.util.spec_from_file_location('qualified_owner',owner_path)
        owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
        cases=[];censuses=[]
        for item in p['completed']:
            f=Path(item['path']);assert sha(f)==item['sha256'];r=load(f)
            assert r['passed'] and r['complete'] and r['ownership_closed'] and not r['active']
            assert r['fresh_faults']==[] and r['boot_unchanged'] and not r['driver_dump_exists']
            for c in r['commands']:
                assert owner.closed(c) and c['exit_code']==0
                for identity in c['owners']+[c['direct_identity']]:
                    now=owner.identity(identity['pid'])
                    assert now is None or now['start_ticks']!=identity['start_ticks']
            vectors={'wall_ms','queue_ms','samples','input_bytes_host_device','store_bytes_host_device',
                     'input_queue_host_device_ms','store_queue_host_device_ms'}
            coarse=helper.parse_table(r['coarse_profile_lines'],vectors,{'samples','input_bytes_host_device','store_bytes_host_device'})
            fine=helper.parse_table(r['qsa_profile_lines'],{'queue_ms','host_wall_ms','samples'},{'samples'})
            assert fine['header']['wrapper_ms']==coarse['header']['wrapper_ms']
            req,=r['benchmark_result']['results']['prefill_only']['requests']
            assert req['ok'] and req['eligible'] and req['actual_prompt_tokens']>=32768 and req['actual_reused_tokens']==0
            wait=numeric_line(r['cpu_wait_lines'][0]);host=numeric_line(r['cpu_host_lines'][0])
            assert wait['available']==wait['complete']==1 and wait['open_waits']==wait['wait_throws']==0
            assert host['available']==1 and host['open_regions']==0
            assert math.isclose(wait['main_cpu_s'],wait['wait_cpu_s']+wait['outside_wait_cpu_s'],abs_tol=1e-6)
            c=r['shape_census'];censuses.append(c);shapes=shape_summary(c)
            assert c['tokens']==req['actual_read_tokens']-1
            assert host['gemm_submit_count']>=shapes['MoE_GEMM_calls']
            ct=coarse['rows'][-1];ft=fine['rows'][-1]
            cases.append(dict(arm=item['arm'],receipt=dict(path=str(f),sha256=sha(f)),
                binary_sha256=r['binary_sha256'],request_sha256=req['request_sha256'],
                actual_prompt_tokens=req['actual_prompt_tokens'],batched_tokens=c['tokens'],
                diagnostic_prompt_ms=req['prompt_ms'],wrapper_ms=float(coarse['header']['wrapper_ms']),
                queue_interval_ms=dict(zip(coarse['header']['queue_order'].split(','),ct['queue_ms'])),
                qsa_interval_ms=dict(zip(fine['header']['sections'].split(','),ft['queue_ms'])),
                requested_upload_bytes=ct['upload_bytes'],
                requested_residual_input_bytes_host_device=ct['input_bytes_host_device'],
                requested_residual_store_bytes_host_device=ct['store_bytes_host_device'],
                cpu=dict(wait=wait,host=host),
                non_MoE_GEMM_submit_count=host['gemm_submit_count']-shapes['MoE_GEMM_calls'],
                shapes=shapes,sensors=sensors_summary(r['sysfs_observation'])))
        assert len({c['binary_sha256'] for c in cases})==len({c['request_sha256'] for c in cases})==1
        result=dict(schema='strata-tilepack-paired-profile-analysis-v2',passed=True,complete=True,GPU_executed=False,
            recipe_sha256=sha(__file__),pair_receipt=dict(path=str(pair),sha256=sha(pair)),cases=cases,
            shape_census_identical=censuses[0]==censuses[1],performance_adoption=False,
            physical_VRAM_PCIe_throughput_collected=False,GPU_execution_unit_utilization_collected=False,
            limits='One instrumented process per arm; queue intervals are nonexclusive and include host gaps. Sensor sampling is host-side and non-atomic. Grid/padding counts and requested bytes are derived, not hardware counters. No general quality/full-model262144/RESTORE or speed adoption.')
        dst.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(summary=str(dst),shape_census_identical=result['shape_census_identical'],
            cases=[{k:c[k] for k in ('arm','wrapper_ms','queue_interval_ms','qsa_interval_ms','non_MoE_GEMM_submit_count')} for c in cases])))


if __name__=='__main__':
    main()
