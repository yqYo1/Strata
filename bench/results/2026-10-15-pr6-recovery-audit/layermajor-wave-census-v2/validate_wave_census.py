"""Source-only bounded wave-summary validation; existing v1 model gate reused."""
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from validate_layermajor_census_v1 import validate_census, need


def validate_wave_fields(d):
    need(type(d.get('version')) is int and d['version']==1 and d.get('complete') is True,'completed v1 fields required')
    need(type(d.get('wave_schema_version')) is int and d['wave_schema_version']==1 and d.get('wave_grouping')=='request_relative_chunk_ordinal' and
         d.get('issuer_expert_ids_verified') is True,'wave/issuer schema')
    n,position,chunk=d['tokens'],d['position'],d['configured_chunk']
    need(all(type(x) is int for x in (n,position,chunk)) and 0<n<=262144 and 0<=position<=262144-n and 1<=chunk<=8192,'wave request geometry')
    chunks=(n+chunk-1)//chunk
    need(len(d['layers'])==48,'wave layer count')
    for index,layer in enumerate(d['layers']):
        need(all(type(layer.get(k)) is int and 0<=layer[k]<=2**64-1
                 for k in ('layer','chunks','tokens','gu_calls','down_calls')),'wave layer scalar types/ranges')
        need(layer['layer']==index and layer['chunks']==chunks and layer['tokens']==n,'wave layer coverage')
        need(layer['gu_calls']==layer['down_calls'],'wave source incidences')
        waves=layer.get('waves')
        need(isinstance(waves,list) and len(waves)==2,'two wave widths required')
        for w,q in zip(waves,(2,4)):
            need(w.get('width_chunks')==q,'wave ordering')
            keys=('wave_count','partial_wave_count','partial_chunks','partial_tokens','union_sum','union_min','union_max',
                  'active_incidences','duplicate_incidences','expert_rows','max_wave_ne')
            need(all(type(w.get(k)) is int and 0<=w[k]<=2**64-1 for k in keys),'wave scalar types/ranges')
            count=(chunks+q-1)//q;tail=chunks%q
            need(w['wave_count']==count and w['partial_wave_count']==int(tail!=0) and w['partial_chunks']==tail,'wave full/partial counts')
            need(w['partial_tokens']==(n-(chunks-tail)*chunk if tail else 0),'partial final token coverage')
            need(w['active_incidences']==layer['gu_calls'] and w['expert_rows']==10*n,'wave row/incidence conservation')
            u=w['union_sum'];inc=w['active_incidences']
            need(max(10*count,(inc+q-1)//q)<=u<=min(inc,512*count),'wave union bounds')
            need(10<=w['union_min']<=w['union_max']<=512 and count*w['union_min']<=u<=count*w['union_max'],'wave union min/max')
            need(w['duplicate_incidences']==inc-u,'repeat-incidence identity')
            hist=w.get('expert_wave_rows_hist')
            need(isinstance(hist,list) and len(hist)==16 and all(type(x) is int and 0<=x<=2**64-1 for x in hist),'16 wave row bins')
            need(sum(hist)==u and 1<=w['max_wave_ne']<=min(q*chunk,n),'expert-wave cardinality/max ne')
            lo=hi=0
            for b,c in enumerate(hist):
                lower=1<<b;upper=min((1<<(b+1))-1,q*chunk)
                if b==15:upper=32768
                need(not c or lower<=min(q*chunk,n),'impossible wave row bin')
                lo+=c*lower;hi+=c*upper
            need(lo<=10*n<=hi,'histogram weighted row bounds')
            highest=max(i for i,c in enumerate(hist) if c)
            need(w['max_wave_ne'].bit_length()-1==highest,'maximum ne/bin mismatch')
        need(waves[1]['union_sum']<=waves[0]['union_sum']<=2*waves[1]['union_sum'],'nested q2/q4 partition bounds')
    return dict(passed=True,wave_schema_version=1,scope='Exact observer summaries; no measured hardware reuse/latency/speed or quality claim.')


def validate_wave_census(*,stderr_path,expected_tokens):
    # Preserve all existing actual-model metadata and v1 exact-ne checks. Its
    # model-specific position0/chunk4096 scope is intentionally not weakened.
    result=validate_census(stderr_path=stderr_path,expected_tokens=expected_tokens)
    result['wave_validation']=validate_wave_fields(result['raw'])
    return result
