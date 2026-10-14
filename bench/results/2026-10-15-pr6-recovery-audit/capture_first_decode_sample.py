#!/usr/bin/env python3
"""Extract one completed engine request; does not qualify its still-live parent job."""
import hashlib
import json
import re
from pathlib import Path
from check_gemm_orientation import B, D, sha


def fields(line):
    return dict(re.findall(r'(\w+)=([^ ]+)', line))


def main():
    root=B/'main-primary-decode-ledger-fp32-fixed-normal-root-v1'
    parent=json.loads((root/'record.json').read_text())
    text=(root/'owned/serve.stderr').read_text()
    start=text.index('strata decode joined begin ')
    footer=text.index('strata decode joined close ',start)
    end=text.index('\n',footer)+1
    block=text[start:end]
    lines=block.splitlines()
    coverage=fields(next(s for s in lines if s.startswith('coverage ')))
    owners=[fields(s) for s in lines if s.startswith('owner ')]
    services=[fields(s) for s in lines if s.startswith('service ')]
    emitted=int(coverage['E']);rows=int(coverage['R']);cpu_rows=int(coverage['cpu_rows'])
    cpu_seconds=sum(int(s['cpu_ns']) for s in services)/1e9
    role_seconds={name:sum(int(s['cpu_ns']) for s in services if int(s['role']) in roles)/1e9
                  for name,roles in [('GU',[0,3]),('quant',[1,4]),('Down',[2,5])]}
    sample=dict(scope='Closed first request transcript extracted while the owned parent may still be running. '
                      'Engine completeness flags are retained; Main full protocol/counter/ownership/fault validation is pending.',
        source_pins=parent['source_pins'],binary_sha256=parent['binary_sha256'],input_sha256=parent['input_sha256'],
        parent_recipe_sha256=parent['recipe_sha256'],parent_root=str(root),boot_id=parent['boot_id'],
        parent_active_at_extraction=parent['active'],parent_qualified=False,
        direct_identity=next(c['direct_identity'] for c in parent['commands'] if c['label']=='serve'),
        extraction_recipe_sha256=sha(Path(__file__)),transcript_sha256=hashlib.sha256(block.encode()).hexdigest(),
        transcript_lines=lines,request=parent['requests'][0],
        arithmetic=dict(verifier_rows_per_output=rows/emitted,
            cpu_expert_row_fraction=cpu_rows/int(coverage['route_entries']),
            cpu_job_mean_nt=cpu_rows/int(coverage['cpu_jobs']),
            cpu_logical_blob_bytes_per_output=int(owners[0]['logical_blob'])/emitted,
            cpu_service_seconds=cpu_seconds,cpu_role_service_seconds=role_seconds,
            cpu_service_seconds_per_output=cpu_seconds/emitted,
            target70_cpu_service_slots_if_cost_unchanged=70*cpu_seconds/emitted,
            six_service_slots_conditional_output_rate=6*emitted/cpu_seconds,
            target70_cpu_logical_blob_GB_per_s=70*int(owners[0]['logical_blob'])/emitted/1e9,
            target70_cpu_useful_expert_GFLOP_per_s=70*9830400*cpu_rows/emitted/1e9),
        physical_DRAM_bytes_measured=False,service_is_not_wall_critical_path=True,
        cold_graph_cost_included_in_decode=True,performance_qualified=False,quality_qualified=False,
        physical_262144_qualified=False,adopted=False)
    path=D/'decode-ledger-first-request-v1.json'
    with path.open('x') as stream:
        stream.write(json.dumps(sample,indent=2)+'\n')
    print(json.dumps(sample['arithmetic']),flush=True)


if __name__=='__main__':
    main()
