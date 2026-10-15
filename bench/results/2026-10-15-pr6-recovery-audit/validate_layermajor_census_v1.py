"""Independent completed-record validation; never interprets missing data as zero."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def need(ok, message):
    if not ok:
        raise RuntimeError('layer-major census: '+message)


def validate_census(*, stderr_path, expected_tokens):
    prefix='strata lm census: '
    lines=[line[len(prefix):] for line in Path(stderr_path).read_text().splitlines() if line.startswith(prefix)]
    need(len(lines)==1,'exactly one completed request report required')
    need(len((prefix+lines[0]+'\n').encode())<=1<<20,'serialized report exceeds cap')
    d=json.loads(lines[0])
    need(d.get('version')==1 and d.get('complete') is True,'report incomplete')
    need(d['tokens']==expected_tokens and d['position']==0 and d['configured_chunk']==4096,'request geometry')
    need(d.get('issuer_exact_ne_verified') is True and d.get('hardware_cache_residency_measured') is False,'issuer/residency scope')
    need(len(d['layers'])==48,'layer count')
    table=Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s/native_experts.txt')
    metadata={}
    for line in table.read_text().splitlines():
        if line and not line.startswith('#'):
            fields=line.split()
            layer,gu,dn,offset,blob=map(int,fields[:5])
            need(layer not in metadata,'duplicate model metadata layer')
            metadata[layer]=(gu,dn,blob)
    need(set(metadata)==set(range(48)),'model format table geometry')
    width_bytes={18:(256,98),21:(256,110),22:(256,82),23:(256,136),20:(32,18),42:(64,18)}
    totals=defaultdict(lambda:dict(calls=0,rows=0,histogram=defaultdict(int),packed_bytes=0,half_output_bytes=0))
    chunks=(expected_tokens+4095)//4096
    for index,r in enumerate(d['layers']):
        need(r['layer']==index,'duplicate/unordered layer')
        gu,dn,blob=metadata[index]
        need((r['gu_type'],r['down_type'],r['blob_bytes'])==(gu,dn,blob),'actual model format mismatch')
        gu_row=2560//width_bytes[gu][0]*width_bytes[gu][1]
        dn_row=640//width_bytes[dn][0]*width_bytes[dn][1]
        need(r['gu_row']==gu_row and r['down_row']==dn_row,'native row stride')
        need(r['up_offset']==640*gu_row and r['down_offset']==1280*gu_row,'native tensor offset')
        need(blob==1280*gu_row+2560*dn_row,'native blob extent')
        need(r['gu_tensor_bytes']==1280*gu_row and r['down_tensor_bytes']==2560*dn_row,'tensor extent')
        need(r['configured_slots']==r['configured_resident']==512,'configured residency')
        need(r['tokens']==expected_tokens and r['chunks']==chunks,'token/chunk coverage')
        need(r['gu_logical_nk']==[1280,2560] and r['down_logical_nk']==[2560,640],'logical GEMM shape')
        need(r['histogram_columns']==['ne','routed_experts','gu_calls','down_calls'],'histogram schema')
        seen=set();mass=rows=calls=0
        for item in r['histogram']:
            need(len(item)==4 and all(type(v) is int and v>=0 for v in item),'histogram types/ranges')
            ne,count,gcalls,dcalls=item
            need(ne<=4096 and ne not in seen and count>0,'exact ne bins')
            seen.add(ne)
            need(gcalls==dcalls==(count if ne else 0),'actual GU/Down count equality')
            mass+=count;rows+=ne*count;calls+=gcalls
            if ne:
                for role,typ in [('gu',gu),('down',dn)]:
                    t=totals[role+'/'+str(typ)]
                    t['calls']+=count;t['rows']+=ne*count;t['histogram'][ne]+=count
        need(mass==512*chunks and rows==expected_tokens*10,'routing conservation')
        need(calls==r['gu_calls']==r['down_calls'],'issuer call conservation')
        need(rows==r['gu_rows']==r['down_rows'],'issuer row conservation')
        totals['gu/'+str(gu)]['packed_bytes']+=calls*1280*gu_row
        totals['down/'+str(dn)]['packed_bytes']+=calls*2560*dn_row
        totals['gu/'+str(gu)]['half_output_bytes']+=calls*1280*2560*2
        totals['down/'+str(dn)]['half_output_bytes']+=calls*2560*640*2
    return dict(passed=True,complete=True,raw=d,by_role_format=dict(totals),
                model_format_metadata_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
                actual_physical_262144_qualified=False,
                scope='Logical issued shapes/materialization bytes. No hardware cache hit/DRAM/PCIe/device exclusive time claim.')
