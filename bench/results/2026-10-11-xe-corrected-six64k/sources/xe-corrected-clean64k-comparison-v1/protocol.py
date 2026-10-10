"""Exact already-qualified common GEN64 protocol functions; CPU/model-independent."""
import math
from direct_owner import require

def expected_pp(n,chunk):
    require(n in (32768,65536) and chunk in (4096,8192),'admitted protocol geometry')
    return list(range(chunk,n,chunk))+[n-1]

def validate(result,n=65536,chunk=8192):
    require(n in (32768,65536),'admitted protocol geometry')
    lines=result['protocol']; done=[v.split() for v in lines if v.startswith('DONE ')]
    require(len(done)==1 and len(done[0])>=16,'one full DONE')
    d=done[0]; generated=int(d[1])
    require(0<generated<=64 and len(result['ids'])==len(result['logprobs'])==generated,'complete output count')
    require(d[5] in ('length','stop') and (d[5]!='length' or generated==64),'honest earlystop/length')
    require(int(d[2])==n and int(d[8])==0 and int(d[14])==n,'complete fresh prompt/read_n')
    require(0<=int(d[6])<=int(d[7]),'draft counts')
    require(all(math.isfinite(float(d[i])) and float(d[i])>0 for i in (3,4)),'positive finite phase times')
    for tag in ('RESUME','REUSED'):
        require([int(v.split()[1]) for v in lines if v.startswith(tag+' ')]==[0],'fresh '+tag)
    pp=[v.split() for v in lines if v.startswith('PP ')]
    require([int(v[1]) for v in pp]==expected_pp(n,chunk),'exact chunk/end positions')
    require(all(len(v)==5 and int(v[2])==n and all(math.isfinite(float(x)) and float(x)>=0 for x in v[3:]) for v in pp),'PP fullinput totals/finite')
    require(all(0<=v<248320 for v in result['ids']),'output ID domain')
    # Require every token immediately accompanied by LP5; no duplicate/lost LP.
    outputs=[v for v in lines if v.startswith(('T ','LP '))]
    require(len(outputs)==2*generated and all(outputs[2*i].startswith('T ') and outputs[2*i+1].startswith('LP ') for i in range(generated)),'T/LP pairing')
    for value in result['logprobs']:
        f=value.split(); require(len(f)==7 and math.isfinite(float(f[1])),'finite LP5')
        for v in f[2:]:
            token,lp=v.split(':'); require(0<=int(token)<248320 and math.isfinite(float(lp)),'LP top ID/finite')
    return dict(prompt_tokens=n,generated_tokens=generated,actual_batched_positions=n-1,
                prompt_ms=float(d[3]),decode_ms=float(d[4]),finish_reason=d[5],
                mtp_counts=list(map(int,d[6:8])),fresh_complete_finite=True,
                prefill_chunk=chunk,prefill_tps=n*1000/float(d[3]),
                decode_performance_eligible=(generated==64 and d[5]=='length'),
                decode_tps=(generated*1000/float(d[4]) if generated==64 and d[5]=='length' else None))
