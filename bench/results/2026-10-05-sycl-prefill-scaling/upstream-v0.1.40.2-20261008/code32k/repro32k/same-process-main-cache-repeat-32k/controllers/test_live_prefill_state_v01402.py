"""Pinned page-layout checks, without GPU or model execution."""
from pathlib import Path
import datetime, hashlib, json, tempfile
from compare_live_prefill_state_v01402 import hash_live,padding_ranges

base=Path(__file__).parent;out=base/'live-prefill-state-v01402-host-check';out.mkdir(mode=0o700)
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
source=root/'build-sycl-event-ack-registered-copy-v3-20261008/source'
snapshot=source/'sycl/src/core/conversation_snapshot.cpp';qsa=source/'include/strata/kernels/qsa.hpp'
text=snapshot.read_text();assert 'const int64_t cells = ((upto + s.page_size - 1) / s.page_size) * s.page_size;' in text
assert 'const int64_t pooled = index && upto > 0 ? upto / s.idx_block + 1 : 0;' in text
assert 's.page_size = 4;' in qsa.read_text()
record={'active':True,'passed':False,'gpu_access':False,'cases':[],
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Checks the pinned diagnostic comparison against actual snapshot/page layout contracts. Rounded future cells alone may differ; every live byte and the moving pooled spare row remain a hard gate. Not GPU mapping/model arithmetic proof.',
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [snapshot,qsa,Path(__file__),base/'compare_live_prefill_state_v01402.py']}}
def check(name,condition):
    record['cases'].append({'name':name,'passed':bool(condition)});assert condition,name
try:
    with tempfile.TemporaryDirectory(dir=out) as directory:
        p=Path(directory)/'state.bin'
        for kind in range(4):
            i=6+kind;row=256 if kind<2 else 8;size=8*2*row
            original=bytes((17+j*43+(j//7)*11)%256 for j in range(size));p.write_bytes(original)
            ignored=padding_ranges(i,size,7);h=hash_live(p,0,size,ignored)
            check(f'kind{kind}-two-tail-cells',ignored==[(4*2*row+3*row,4*2*row+4*row),(4*2*row+7*row,8*2*row)])
            for head,(start,end) in enumerate(ignored):
                v=bytearray(original);v[start:end]=bytes(x^0x5a for x in v[start:end]);p.write_bytes(v)
                check(f'kind{kind}-future-head{head}-accepted',hash_live(p,0,size,ignored)==h)
                v=bytearray(original);v[start-1]^=1;p.write_bytes(v)
                check(f'kind{kind}-preceding-live-head{head}-rejected',hash_live(p,0,size,ignored)!=h)
            v=bytearray(original);v[3*row]^=1;p.write_bytes(v)
            check(f'kind{kind}-same-cell-in-previous-page-is-live',hash_live(p,0,size,ignored)!=h)
            check(f'kind{kind}-full-occupancy-has-no-exclusion',padding_ranges(i,size,8)==[])
        check('recurrent-history-tail-counter-have-no-exclusion',all(padding_ranges(i,8,7)==[] for i in range(6)))
        check('moving-pooled-spare-remains-live',padding_ranges(10,(7//4+1)*128*4,7)==[])
        for name,index,size,upto in [('wrong-data-size',6,3,7),('wrong-pooled-size',10,512,7),('nonpositive-extent',6,2048,0)]:
            try:padding_ranges(index,size,upto)
            except ValueError:check(name+'-rejected',True)
            else:check(name+'-rejected',False)
    record['passed']=True
except BaseException as error:
    record['error']=repr(error);raise
finally:
    record['active']=False;(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'passed':record['passed'],'cases':len(record['cases'])},indent=2))
