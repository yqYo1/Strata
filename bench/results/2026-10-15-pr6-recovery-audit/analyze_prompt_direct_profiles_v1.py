#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
from analyze_device_profile_v1 import parse

A=Path(__file__).resolve().parent


def main():
    records=[]
    for n,direct in [(1,False),(2,True)]:
        path=A/f'model-device-profile-v{n}.json';d=json.loads(path.read_text())
        assert d['passed'] and d['complete'] and not d['active'] and d['collected']
        assert not d['fresh_faults'] and d['boot_unchanged'] and not d['driver_dump_exists']
        req=d['requests'][0];assert req['reference_ids_equal'] and req['done']['generated']==128
        table=parse(next(f['text'] for f in d['provider_files'] if Path(f['path']).name=='device_timing.txt'))
        token='launch<1, true, false>' if direct else 'launch<1, false, false>'
        attention=[k for k in table['kernels'] if token in k['kernel']]
        assert len(attention)==1 and attention[0]['calls']==108
        records.append(dict(receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            binary_sha256=d['binary_sha256'],source_pins=d['source_pins'],environment=d['environment'],
            generated_ids=req['generated_ids'],coverage=req['ledger']['coverage'],attention=attention[0],table=table))
    old,new=records
    assert old['source_pins']==new['source_pins'] and old['binary_sha256']==new['binary_sha256']
    assert old['generated_ids']==new['generated_ids'] and old['coverage']==new['coverage']
    changes={k:[old['environment'].get(k),new['environment'].get(k)] for k in old['environment'].keys()|new['environment'].keys()
             if old['environment'].get(k)!=new['environment'].get(k)}
    assert set(changes)=={'STRATA_SYCL_PROMPT_DIRECT','XDG_CACHE_HOME'}
    ratio=new['attention']['total_ns']/old['attention']['total_ns']
    # Original receipts retain the complete provider tables and source/environment.
    # Keep this comparison compact instead of archiving duplicate full tables.
    for record in records:
        table=record.pop('table')
        record['table_summary']={k:v for k,v in table.items() if k!='kernels'}
        record['kernel_entries']=len(table['kernels'])
        record.pop('source_pins');record.pop('environment')
        ids=record.pop('generated_ids')
        record['generated_ids_sha256']=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    result=dict(passed=True,records=records,environment_changes=changes,attention_device_ratio=ratio,
        attention_duration_reduction_fraction=1-ratio,adopted=False,clean_performance_qualified=False,
        quality_qualified=False,physical_262144_qualified=False,restore_qualified=False,
        scope='Same binary/input/output/work, two sequential profiler processes, one per setting. Int8 prompt kernel counts108/arm. Diagnostic candidate signal; no variance study, clean end-to-end improvement or decode speed claim.',
        next='Matched bench.sh repetitions with32K+agent sessions, then existing strong/weak long-generation thinking workloads before adoption. CPU recurrence observer remains separate.')
    (A/'prompt-direct-profile-pair-v1.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,attention_device_ratio=ratio,duration_reduction_percent=100*(1-ratio))))


if __name__=='__main__':main()
