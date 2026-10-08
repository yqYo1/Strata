"""Track successful UR command-buffer references without touching the GPU."""
from pathlib import Path
import collections, datetime, hashlib, json, re
base=Path(__file__).parent
job=base/'owned-main-vmm-full-ram-repeat32k-v01402-code32k-diagnostic-r1'
r=json.loads((job/'record.json').read_text())
assert r['healthy'] and r['math_gate_passed'] and not r['active'] and r['exit_code']==0
resources=json.loads((base/'main-cache-repeat32k-v01402-resource-analysis/record.json').read_text())
assert resources['passed'] and not resources['active']
out=base/'main-cache-repeat32k-v01402-graph-analysis';out.mkdir(mode=0o700)
apis=['urCommandBufferCreateExp','urCommandBufferRetainExp','urCommandBufferReleaseExp','urCommandBufferFinalizeExp']
pattern=re.compile(rb'<--- ('+b'|'.join(x.encode() for x in apis)+rb')\(.*?\) -> UR_RESULT_SUCCESS;')
create=re.compile(rb'\.phCommandBuffer = 0x[0-9a-f]+ \((0x[0-9a-f]+)\)')
handle=re.compile(rb'\.hCommandBuffer = (0x[0-9a-f]+)')
refs={};counts=collections.defaultdict(collections.Counter);examples={};markers=[];request=0;phase='startup'
result={'active':True,'passed':False,'gpu_access':False,
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'job_record_sha256':hashlib.sha256((job/'record.json').read_bytes()).hexdigest(),
        'scope':'Successful UR command-buffer creation/retention/release/finalization references around three full32K rereads. Native LevelZero command-list objects can outlive released UR buffers in runtime pools. This is not resident heap attribution, speed or a complete SYCL correctness proof.'}
def save():(out/'record.json').write_text(json.dumps(result,indent=2)+'\n')
save()
try:
    with (job/'debugger/inferior.stderr').open('rb') as stream:
        for number,line in enumerate(stream,1):
            if line.startswith(b'strata prefill memory:'):
                phase=line.decode().split(': ',1)[1].split(';',1)[0]
                if phase=='before decode cache release':request+=1
                markers.append({'line':number,'request':request,'phase':phase,'live_ur_command_buffers':len(refs),'total_references':sum(refs.values())})
            elif b'<--- urCommandBuffer' in line:
                m=pattern.search(line)
                if not m:continue
                api=m[1].decode();h=(create if api=='urCommandBufferCreateExp' else handle).search(line)
                assert h,(number,line[:1500])
                pointer=h[1].decode();key=f'read-{request}/{phase}'
                counts[key][api]+=1
                if api=='urCommandBufferCreateExp':
                    assert pointer not in refs,(number,pointer);refs[pointer]=1
                elif api=='urCommandBufferRetainExp':
                    assert pointer in refs,(number,pointer);refs[pointer]+=1
                elif api=='urCommandBufferReleaseExp':
                    assert pointer in refs,(number,pointer);refs[pointer]-=1
                    if refs[pointer]==0:del refs[pointer]
                else:assert pointer in refs,(number,pointer)
                examples.setdefault(key,{}).setdefault(api,{'line':number,'raw':line.decode().rstrip()})
    assert request==3 and len(markers)==24 and not refs
    result.update(passed=True,markers=markers,counts={k:dict(v) for k,v in counts.items()},first_examples=examples,
                  command_buffer_references_at_exit=refs,
                  limitations=['Returned UR reference-count operations are recorded, not internal C++ graph implementation ownership. Queue drain and private compiled-source destruction order are audited separately.',
                               'Neither object counts nor requested allocation sizes explain runtime-private resident heaps or prove that reported free VRAM stops declining after more requests.'])
except BaseException as error:result.update(error=repr(error),passed=False,partial_references=refs)
finally:
    result.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print(json.dumps({k:result.get(k) for k in ['passed','error','markers','counts','command_buffer_references_at_exit']},indent=2))
if not result['passed']:raise SystemExit(1)
