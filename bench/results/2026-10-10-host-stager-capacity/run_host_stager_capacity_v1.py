from pathlib import Path
import fcntl, hashlib, json, math, statistics, subprocess, types, traceback, datetime
B=Path(__file__).parent
O=B/'host-stager-capacity-v1-owned'
def ident(p):
    p=Path(p);return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not O.exists();O.mkdir()
    source=B/'host_stager_capacity_v1.cpp'
    assert ident(source)['sha256']=='42d61ab146c2333cd9d2b85a9cf381f5d022fab695f0950a259a8c58c98cef2d'
    planpath=B/'host-stager-capacity-plan-v1.json';plan=json.loads(planpath.read_text())
    metadata=B/'host-stager-capacity-metadata-v1.json'
    assert ident(metadata)['sha256']==plan['metadata_sha256']
    parent=B/'run_gdn_gate_factor_probe_v2.py'
    assert ident(parent)['sha256']=='7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f'
    owner_source=parent.read_text().split('\ndef parse_probe(',1)[0].replace('assert rss <= 2 << 30','assert rss <= 1 << 30')
    m=types.ModuleType('hoststagerowner');exec(compile(owner_source,str(parent),'exec'),m.__dict__);m.W=O;o=m.Owner(O)
    binary=O/'host-stager-capacity'
    env=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8')
    r=dict(active=True,complete=False,passed=False,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),controller=ident(Path(__file__)),source=dict(path=str(source),**ident(source)),plan=dict(path=str(planpath),**ident(planpath)),metadata=dict(path=str(metadata),**ident(metadata)),parent_owner=ident(parent),modified_owner_sha256=hashlib.sha256(owner_source.encode()).hexdigest(),commands=o.commands,cells=[],scope=plan['scope'],gpu_work_submitted=False,model_inference=False,adopted=False,source_root_review='All source/handoff read; static disjoint arena; mutex/generation completion happens-before, all worker departures before full byte verification and acknowledgment, no worker/source lifetime escapes.',limits=plan['bounds'],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    def save():(O/'record.json').write_text(json.dumps(r,indent=2)+'\n')
    o.persist=save
    def run(label,args,expected=0,wall=60):
        command,so,se=o.run(label,args,env,wall=wall,text_cap=4<<20,file_cap=8<<20)
        assert m.completed(command) and command['normal_exit'] and command['exit_code']==expected,(label,command)
        return command,so,se
    expected_hashes=None
    def parse_cell(so,se,workers,stage):
        nonlocal_holder=None
        assert not se.read_bytes()
        x=json.loads(so.read_text());assert x['success'] and not x['gpu'] and x['not_pinned'] and x['not_production_stager_or_dma']
        assert x['workers']==workers and x['seed']==1 and x['ring_slots']==16 and x['corpus_blobs']==192 and x['blob_bytes']==1971200
        assert x['arena_bytes']==410022912 and x['alignment_bytes']==64
        assert x['logical_source_payload_bytes_per_sample']==x['logical_destination_payload_bytes_per_sample']==378470400
        assert x['logical_read_plus_write_bytes_per_sample']==756940800
        measured=1 if stage=='qualify' else 7
        assert x['samples']==measured and x['warmups']==1 and x['timing_claim']==(stage=='measure')
        assert x['total_fully_byte_verified_jobs_including_warmup']==(measured+1)*192 and x['final_immutable_source_full_hash_checks']==192
        assert len(x['expected_source_fnv1a64'])==192 and len(set(x['expected_source_fnv1a64']))==192
        assert 0<x['process_work_ns_before_output']<60*10**9
        assert len(x['individual_samples'])==measured+1
        for i,s in enumerate(x['individual_samples']):
            assert s['warmup']==(i==0) and s['jobs_fully_byte_verified']==192 and s['tail_canary_checks']==384
            assert len(s['batch_wall_ns'])==12 and all(v>0 for v in s['batch_wall_ns'])
            assert s['copy_issue_wait_batch_wall_ns_sum']==sum(s['batch_wall_ns'])>0
            assert len(s['job_copy_ns'])==192 and all(v>=0 for v in s['job_copy_ns'])
            assert s['worker_copy_span_ns_sum']==sum(s['job_copy_ns'])
            assert s['correctness_ack_ns']>0 and s['job_fnv1a64']==x['expected_source_fnv1a64']
            assert len(s['worker_jobs'])==workers and all(v>=0 for v in s['worker_jobs']) and sum(s['worker_jobs'])==192
            if stage=='measure':
                a=378470400/s['copy_issue_wait_batch_wall_ns_sum']
                assert math.isfinite(s['timed_batch_payload_GBps']) and math.isclose(s['timed_batch_payload_GBps'],a,rel_tol=1e-5)
                assert math.isclose(s['timed_batch_logical_read_plus_write_GBps'],2*a,rel_tol=1e-5)
            else:assert 'timed_batch_payload_GBps' not in s and 'timed_batch_logical_read_plus_write_GBps' not in s
        return x
    try:
        run('compile',['/usr/bin/g++',*plan['build_flags'],str(source),'-o',str(binary)],wall=120)
        r['binary']=dict(path=str(binary),**ident(binary));save()
        _,so,_=run('dependencies',['/usr/bin/readelf','-d',str(binary)],wall=30)
        r['dependencies']=so.read_text();assert not any(v in r['dependencies'] for v in ('libsycl','libze_loader','libur_adapter','libcuda','libhip'))
        negatives=[[],['0'],['2'],['5'],['7'],['-1'],['01'],['1','-1'],['1','18446744073709551616'],['1','--qualify','--qualify'],['1','1','extra'],['1','1','--qualify','extra'],['1','abc'],['1','1x'],['1','01']]
        r['negative_parser_checks']=[]
        for i,args in enumerate(negatives):
            _,so,se=run('parser-negative-'+str(i),[str(binary),*args],expected=1,wall=10)
            assert not so.read_bytes();failure=json.loads(se.read_text());assert failure['success'] is False and failure['no_capacity_result_valid'] is True and failure['stage']=='parse'
            r['negative_parser_checks'].append(dict(argv=args,result=failure));save()
        for w in plan['workers']:
            command,so,se=run('qualify-w'+str(w),['/usr/bin/taskset','-c','0-5',str(binary),str(w),'1','--qualify'])
            x=parse_cell(so,se,w,'qualify')
            if expected_hashes is None:expected_hashes=x['expected_source_fnv1a64']
            assert x['expected_source_fnv1a64']==expected_hashes
            r['cells'].append(dict(stage='qualify',workers=w,result=x,stdout_identity=ident(so),stderr_identity=ident(se),command_index=len(o.commands)-1));save()
        for repeat,order in enumerate(plan['order_per_repeat'],1):
            for w in order:
                label='measure-w'+str(w)+'-r'+str(repeat)
                _,so,se=run(label,['/usr/bin/taskset','-c','0-5',str(binary),str(w),'1'])
                x=parse_cell(so,se,w,'measure');assert x['expected_source_fnv1a64']==expected_hashes
                samples=[378470400/s['copy_issue_wait_batch_wall_ns_sum'] for s in x['individual_samples'] if not s['warmup']]
                r['cells'].append(dict(stage='measure',workers=w,repeat=repeat,result=x,stdout_identity=ident(so),stderr_identity=ident(se),command_index=len(o.commands)-1,recomputed_payload_rates_GBps=samples,median_payload_GBps=statistics.median(samples)));save()
                print(json.dumps(dict(stage='measure',workers=w,repeat=repeat,median_payload_GBps=statistics.median(samples))),flush=True)
        assert len(r['cells'])==16 and sum(len(c['result']['individual_samples']) for c in r['cells'] if c['stage']=='measure')==96
        assert ident(source)=={k:r['source'][k] for k in ('bytes','sha256')} and ident(binary)=={k:r['binary'][k] for k in ('bytes','sha256')}
        assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==r['boot_id']
        r['summary']=[]
        for w in plan['workers']:
            values=[c['median_payload_GBps'] for c in r['cells'] if c['stage']=='measure' and c['workers']==w]
            assert len(values)==3
            r['summary'].append(dict(workers=w,independent_processes=3,median_of_process_medians_payload_GBps=statistics.median(values),process_median_range_payload_GBps=[min(values),max(values)],logical_read_plus_write_GBps=2*statistics.median(values)))
        r.update(passed=True,complete=True,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    except BaseException as e:r['error']=type(e).__name__+': '+str(e);r['traceback']=traceback.format_exc()
    finally:r['active']=o.active is not None;save();print(json.dumps({k:r.get(k) for k in ('passed','complete','active','error','summary')}),flush=True)
    if not r['passed']:raise SystemExit(1)
