"""Strict offline transcript/metadata validator for the isolated component fixture."""
import csv, re
ORDERS=((0,1,2,3),(1,2,3,0),(2,3,0,1),(3,0,1,2))
N=1638400; IB=921728; OB=3276928; OW=1638464
def fields(row):
    result={}
    for item in row[1:]:
        assert '=' in item,('non-key-value',row)
        key,value=item.split('=',1);assert key not in result,('duplicate-field',row)
        result[key]=value
    return result
def number(value):
    assert value.isdecimal(),value
    result=int(value);assert result<2**64
    return result
def pointer(value):
    assert re.fullmatch(r'0x[0-9a-f]+',value),value
    result=int(value,16);assert 0<result<2**64
    return result
def expect(got,**values):
    for key,value in values.items():assert got.get(key)==str(value),(key,got.get(key),value)
def validate(path,mode):
    assert mode in ('host-only','qualify','service')
    rows=list(csv.reader(path.read_text().splitlines()));assert rows and all(rows)
    at=0
    def take(tag):
        nonlocal at
        assert at<len(rows) and rows[at][0]==tag,('ordered-record',at,tag,rows[at] if at<len(rows) else 'EOF')
        row=rows[at];at+=1;return row
    def kv(tag):return fields(take(tag))
    def progress(phase,scope):
        expect(kv('PROGRESS'),phase=phase,id=scope,planned_completion_boundary='wait_and_throw')
    schema=kv('SCHEMA');expect(schema,version=1,device_clock='event_backend_ns',host_clock='steady_clock_ns',event_source='existing_wrapper_exact_submit',zero_service='valid',synthetic_component=1,model=0,adoption=0)
    expect(kv('SOURCE'),base='9e54ed18c0e374289293838b026e3e69b1c1fe70',receipt_header_bytes=2756,receipt_header_sha256='5a194e5ad480ae36c63e164c1ae0de51b828919a7aade112ddd29e303c3e0ef0',iq_kernel_bytes=269005,iq_kernel_sha256='cc5080af0c9e341a69edc423c272726922c95985aa7d3125cd1b344a06ad2b12',pins='reference_requires_owner_verification')
    blocks=0 if mode=='host-only' else (8 if mode=='service' else 1)
    config=kv('CONFIG');expect(config,mode=mode,type=20,rows=2560,cols=640,n=N,groups=6400,local_size=32,input_count=64,output_slots=2,calls_per_cell=64,cells=4,blocks=blocks,warm_passes=2,warm_calls=256,prephase_calls=128,final_replay_calls=128,planned_cell_calls=blocks*256,receipt_capacity=64,rotation='(i+block*9)%64',USM_bytes=65544448,USM_budget_bytes=96*1024*1024,host_payload_bytes=73020032,host_budget_bytes=256*1024*1024,default_mode='reject')
    assert number(config['host_fixed_accounted_bytes'])+73020032<256*1024*1024
    for b in range(8 if mode=='host-only' else blocks):
        expect(kv('ORDER'),block=b,**{'cell'+str(i):v for i,v in enumerate(ORDERS[b%4])})
    expect(kv('HOST_PASS'),packing_blocks=51200,oracle_active_words=N,roundtrips=14,boundaries=6,private_rejections=7,plan_batches=32,replay_plans=64,metadata_adversaries=13,nested_rejection=1,no_queue=1,no_device=1,generic_invalid='source_only')
    pattern=kv('INPUT_PATTERN');expect(pattern,packed_bytes=921600,full_bytes=IB,scale_count=14,code_count=16,all_products_finite=1,identical_cohort_bytes=1)
    for key in ('fnv1a64_full','oracle_fnv1a64_full'):assert re.fullmatch('[0-9a-f]{16}',pattern[key])
    if mode=='host-only':
        expect(kv('PASS'),mode=mode,no_queue=1,no_device=1,synthetic_component=1,model=0,adoption=0)
        assert at==len(rows);return dict(passed=True,mode=mode,no_queue=True,records=len(rows),pattern=pattern)
    queue=kv('QUEUE');pointer(queue['queue']);qid=queue['queue']
    expect(queue,backend='level_zero',in_order=1,enable_profiling=1,queue_profiling=1,fp16=1,usm_device=1,vendor='8086',device='e20c',pci='0000:05:00.0',root=1)
    resolution=number(queue['resolution_ns'])
    assert take('FP16_CONFIG')==['FP16_CONFIG','advertised_capabilities_only=1','flag=2','flag=4','flag=8','flag=16','flag=32','flag=64']
    src=[];dst=[];alloc=[]
    for i in range(64):
        d=kv('INPUT');expect(d,index=i,allocation='device',queue=qid,total_bytes=IB,packed_bytes=921600,guard_each_bytes=64,alignment=64,fnv1a64_full=pattern['fnv1a64_full'])
        base=pointer(d['base']);view=pointer(d['view']);assert base%64==0 and view==base+64
        src.append(d['view']);alloc.append((base,IB))
    for i in range(2):
        d=kv('OUTPUT');expect(d,slot=i,allocation='device',queue=qid,total_bytes=OB,active_words=N,guard_each_words=32,alignment=64)
        base=pointer(d['base']);view=pointer(d['view']);assert base%64==0 and view==base+64
        dst.append(d['view']);alloc.append((base,OB))
    for i,(a,na) in enumerate(alloc):
        assert a+na<=2**64
        for b,nb in alloc[:i]:assert a+na<=b or b+nb<=a
    expect(kv('ALLOCATION_RESULT'),input_allocations=64,output_allocations=2,distinct=1,nonoverlap=1,view_alignment=64)
    progress('initial_upload',0)
    scope=0;call_count=0;batch_count=0;timed_samples=[];timed_batches=[]
    call_keys=set('phase timed batch block position cell predecessor scope serial arm regime input_index output_slot queue src dst type n submit start end service_ns queue_delay_ns host_submit_ns metadata_match event_distinct'.split())
    def batch(phase,block,pos,cell,pred,replay,index=0):
        nonlocal scope,call_count,batch_count
        scope+=1;progress(phase,scope);count=2 if replay else 64
        timed=int(phase=='service');calls=[];prior_end=None;last={}
        for i in range(count):
            d=kv('CALL');assert set(d)==call_keys,('CALL keys',set(d)^call_keys)
            arm=('generic' if i==0 else 'private') if replay else ('private' if cell%2 else 'generic')
            input_index=index if replay else (0 if cell<2 else (i+block*9)%64)
            slot=(index+i)%2 if replay else i%2
            expect(d,phase=phase,timed=timed,batch=scope,block=block,position=pos,cell=cell,predecessor=pred,scope=scope,serial=i+1,arm=arm,regime='oracle_replay' if replay else ('same_address' if cell<2 else 'rotation64'),input_index=input_index,output_slot=slot,queue=qid,src=src[input_index],dst=dst[slot],type=20,n=N,metadata_match=1,event_distinct=1)
            submit,start,end=[number(d[k]) for k in ('submit','start','end')]
            assert submit<=start<=end and (prior_end is None or prior_end<=start)
            assert number(d['service_ns'])==end-start and number(d['queue_delay_ns'])==start-submit
            number(d['host_submit_ns']);prior_end=end;calls.append(d);last[slot]=(input_index,arm);call_count+=1
        progress('readback',scope)
        d=kv('BATCH');expect(d,phase=phase,timed=timed,batch=scope,block=block,position=pos,cell=cell,predecessor=pred,planned=count,attempted=count,successful_returns=count,receipts=count,queried=count,dq_waits=1,readback_waits=1,completion='known',async_errors=0,receipt_invalid=0,oracle_last_occupants_only=1,last_slot0_input=last[0][0],last_slot0_arm=last[0][1],last_slot1_input=last[1][0],last_slot1_arm=last[1][1],output_words_checked=2*OW,input_bytes_checked=IB if replay else 64*IB,all_guards_checked=1,input_immutable=1,result='PASS')
        window,wait,wall=[number(d[k]) for k in ('host_submit_window_ns','host_wait_ns','host_wall_ns')]
        assert window+wait==wall and sum(number(c['host_submit_ns']) for c in calls)<=window
        batch_count+=1
        if timed:timed_samples.extend(calls);timed_batches.append(d)
    def replay(phase):
        expect(kv('REPLAY_BEGIN'),phase=phase,inputs=64,arms=2,calls=128,active_words_per_arm=64*N,timing_samples=0)
        for i in range(64):batch(phase,i,0,0,-1,True,i)
        progress('readback',scope)
        expect(kv('REPLAY_END'),phase=phase,calls=128,oracle='full_independent_RNE',input_immutable=1,full_guards=1,both_slots_exercised=1,closing_full_cohort_scan=1,closing_readback_waits=1)
    replay('prephase');progress('warm_two_passes',0)
    expect(kv('WARM'),passes=2,inputs_per_arm_per_pass=64,arms=2,calls=256,dq_waits=1,receipt_scope='inactive',timing_samples=0,completion='known',async_errors=0)
    predecessor=-1
    for b in range(blocks):
        for pos,cell in enumerate(ORDERS[b%4]):
            batch('service' if mode=='service' else 'qualification_cell',b,pos,cell,predecessor,False)
            predecessor=cell
    replay('final_replay')
    waits=260+8*blocks;wrappers=512+256*blocks
    expect(kv('FINAL_COUNTS'),completed_queue_waits=waits,expected_queue_waits=waits,successful_wrappers=wrappers,expected_wrappers=wrappers,upload_waits=1,warm_waits=1,replay_DQ_waits=128,replay_readback_waits=128,replay_closing_waits=2,cell_DQ_waits=4*blocks,cell_readback_waits=4*blocks,result='PASS')
    progress('USM_release',scope)
    expect(kv('USM_RELEASED'),allocations=66,bytes=65544448,completion='known',events_released=1)
    expect(kv('PASS'),mode=mode,cell_batches=4*blocks,cell_calls=256*blocks,replay_calls=256,warm_calls=256,completed_queue_waits=waits,successful_wrappers=wrappers,USM_freed=1,queue_teardown=1,synthetic_component=1,model=0,adoption=0)
    assert at==len(rows) and batch_count==128+4*blocks and call_count==256+256*blocks
    return dict(passed=True,mode=mode,records=len(rows),pattern=pattern,queue=queue,profiling_resolution_ns=resolution,receipt_calls=call_count,batches=batch_count,successful_wrappers=wrappers,completed_waits=waits,full_pre_and_post_oracle_replay=True,all_input_and_output_guards=True,exact_metadata_and_distinct_event=True,raw_timed_samples=timed_samples,timed_batches=timed_batches,performance_scope='Instrumented direct helper only, fixed untimed readbacks condition cache state; no model speed or cold-cache claim',model=False,adopted=False)
