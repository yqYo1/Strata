"""Offline numerical/memory evidence from one completed32K diagnostic."""
from pathlib import Path
import array, datetime, hashlib, json, math, re

base=Path(__file__).parent
job=base/'owned-main-vmm-full-ram-chunk12k-v01402-code32k-diagnostic-r1'
record=json.loads((job/'record.json').read_text())
assert not record['active'] and record['completed'] and record['exit_code']==0
assert not record['new_fault_messages'] and not record['math_gate_passed']
out=base/'main-cache-chunk12k-v01402-diagnostic-analysis'
out.mkdir(mode=0o700)
baseline=json.loads((base/'owned-event-ack-no-root-prefill-default-v01402-code32k-diagnostic-r1/record.json').read_text())['requests'][0]
request=record['requests'][0]
x=array.array('f');x.frombytes((job/'first-head.bin').read_bytes())
y=array.array('f');y.frombytes(Path(baseline['first_head']['file']).read_bytes())
assert len(x)==len(y)==248320 and all(math.isfinite(v) for v in x)
deltas=[abs(a-b) for a,b in zip(x,y)]
analysis={'active':False,'passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scope':'One32K larger-chunk diagnostic is rejected mathematically. Only offline numerical/source/allocation-log analysis follows; no clean timing/adoption or isolated root-cause conclusion.',
          'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'job_record_sha256':hashlib.sha256((job/'record.json').read_bytes()).hexdigest(),
          'execution':{'normal_exit':True,'owned_cleanup_complete':True,'new_xe_faults':[],
                       'input_tokens':32768,'chunk_tokens':12288,'generated_tokens':64,
                       'mtp_counts':list(map(int,request['protocol'][-1].split()[6:8]))},
          'comparison':request['comparison_to_default_counter_control'],
          'first_changed_id':next(([i,a,b] for i,(a,b) in enumerate(zip(request['ids'],baseline['ids'])) if a!=b),None),
          'first_changed_logprob':next(([i,a,b] for i,(a,b) in enumerate(zip(request['logprobs'],baseline['logprobs'])) if a!=b),None),
          'head':{'floats':len(x),'changed_values':sum(a!=b for a,b in zip(x,y)),
                  'max_abs':max(deltas),'rms':math.sqrt(math.fsum(v*v for v in deltas)/len(deltas)),
                  'new_argmax':max(range(len(x)),key=x.__getitem__),
                  'baseline_argmax':max(range(len(y)),key=y.__getitem__)}}
log=job/'debugger/inferior.stderr'
assert log.stat().st_size==record['engine_log_bytes']
allocations=[];frees=[];phases=[];live={};last_clock=None;phase='startup'
allocation=re.compile(rb'in zeMemAllocDevice\(.*?size=(\d+).*?pptr=(0x[0-9a-f]+)\)')
free=re.compile(rb'in zeMemFree\(.*?ptr=(0x[0-9a-f]+)\)')
clock=re.compile(rb'^\[([^\]]+)\]')
memory=re.compile(rb'^strata prefill memory: (.*?); free=(\d+) total=(\d+) bytes')
with log.open('rb') as stream:
    for line_number,line in enumerate(stream,1):
        if line.startswith(b'['):
            m=clock.match(line)
            if m:last_clock=m[1].decode()
        if b'SUCCESS (ZE_RESULT_SUCCESS) in zeMemAllocDevice(' in line:
            m=allocation.search(line);assert m,line[:1000]
            size=int(m[1]);pointer=m[2].decode()
            # Loader capability probes can reuse an address after freeing it.
            assert pointer not in live,(pointer,line_number)
            live[pointer]=size
            allocations.append({'line':line_number,'clock':last_clock,'phase':phase,'bytes':size,'pointer':pointer})
        elif b'SUCCESS (ZE_RESULT_SUCCESS) in zeMemFree(' in line:
            m=free.search(line);assert m,line[:1000]
            pointer=m[1].decode();size=live.pop(pointer,None)
            frees.append({'line':line_number,'clock':last_clock,'phase':phase,'device_bytes':size,'pointer':pointer})
        elif line.startswith(b'strata prefill memory:'):
            m=memory.match(line)
            if m:
                phase=m[1].decode()
                phases.append({'line':line_number,'nearest_previous_api_clock':last_clock,
                               'phase':phase,'free_bytes':int(m[2]),'total_bytes':int(m[3]),
                               'logged_device_alloc_live_bytes':sum(live.values())})
summary={}
for a in allocations:
    s=summary.setdefault(a['phase'],{'alloc_calls':0,'requested_device_bytes':0,'large_allocations':[]})
    s['alloc_calls']+=1;s['requested_device_bytes']+=a['bytes']
    if a['bytes']>=1<<20:s['large_allocations'].append(a)
analysis.update(memory_phases=phases,allocation_summary=summary,
                logged_device_allocations_at_end=[{'pointer':p,'bytes':n} for p,n in live.items()],
                memory_limitations=['Reported VRAM free changes need not equal requested zeMemAllocDevice sizes: physical mappings, internal runtime allocations, pooling/residency and allocation rounding are distinct.',
                                    'Debug/validation run only; no clean timing or matched memory attribution. The nearest earlier API timestamp is not the timestamp of the phase marker.',
                                    'Changing chunks changes GEMM/callback boundaries; this run does not isolate the cause of numerical divergence.'])
(out/'device-allocations.json').write_text(json.dumps(allocations,indent=2)+'\n')
(out/'device-frees.json').write_text(json.dumps(frees,indent=2)+'\n')
(out/'record.json').write_text(json.dumps(analysis,indent=2)+'\n')
print(json.dumps({'head':analysis['head'],'first_changed_id':analysis['first_changed_id'],
                  'mtp_counts':analysis['execution']['mtp_counts'],'phases':phases,
                  'large_device_allocations_after_main_release':[{k:v for k,v in a.items() if k!='pointer'} for a in allocations if a['phase']!='startup' and a['bytes']>=1<<20]},indent=2))
