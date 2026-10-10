from pathlib import Path
import csv,fcntl,hashlib,json,math,os,subprocess,sys,types,traceback,importlib.util
B=Path(__file__).parent
stage=sys.argv[1];mode=sys.argv[2];workers=int(sys.argv[3]);repeat=int(sys.argv[4]);assert stage in ('qualify','measure') and mode in ('sequential','random') and workers in (1,16,64) and repeat in (1,2,3);assert mode!='sequential' or workers in (1,16);assert stage!='qualify' or repeat==1
SHARD=Path('/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf')
OUT=B/f'storage-capacity-v2-{stage}-{mode}-w{workers}-r{repeat}'
def ident(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def stat_id(p):
 s=p.stat();return dict(dev=s.st_dev,ino=s.st_ino,size=s.st_size,mode=s.st_mode,mtime_sec=s.st_mtime_ns//1000000000,mtime_nsec=s.st_mtime_ns%1000000000,ctime_sec=s.st_ctime_ns//1000000000,ctime_nsec=s.st_ctime_ns%1000000000)
def counters():
 pool=Path('/proc/spl/kstat/zfs/rpool/iostats');d={}
 for line in pool.read_text().splitlines():
  parts=line.split()
  if len(parts)==3 and parts[1]=='4' and parts[0].isidentifier():d[parts[0]]=int(parts[2])
 return dict(host_monotonic_ns=__import__('time').monotonic_ns(),ZFS_pool_iostats=d,block_leaf_nvme0n1=list(map(int,Path('/sys/block/nvme0n1/stat').read_text().split())),block_partition_nvme0n1p3=list(map(int,Path('/sys/class/block/nvme0n1p3/stat').read_text().split())))
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 buildpath=B/'storage-capacity-v2-build/record.json';build=json.loads(buildpath.read_text());assert build['passed'] and build['complete'] and not build['active'];src=Path(build['source']['path']);binary=Path(build['binary']['path']);assert ident(src)=={k:build['source'][k] for k in ('bytes','sha256')};assert ident(binary)=={k:build['binary'][k] for k in ('bytes','sha256')}
 qualifiers={}
 if stage=='measure':
  qp=B/f'storage-capacity-v2-qualify-{mode}-w{workers}-r1/record.json';q=json.loads(qp.read_text());assert q['passed'] and q['complete'] and not q['active'] and q['binary']==build['binary'];qualifiers[str(qp)]=ident(qp)
 metadata=B/'storage-capacity-metadata-v1.json';md=json.loads(metadata.read_text());assert md['page_size']==4096 and md['fields']['/sys/module/zfs/parameters/zfs_dio_enabled']=='1';assert Path('/sys/module/zfs/parameters/zfs_dio_enabled').read_text().strip()=='1' and Path('/sys/module/zfs/parameters/zfs_dio_strict').read_text().strip()=='0'
 properties=subprocess.check_output(['zfs','get','-Hp','-o','property,value','recordsize,compression,compressratio,primarycache,secondarycache,checksum,copies','rpool/USERDATA/yayoi'],text=True);assert properties==md['commands']['dataset']['stdout']
 parser=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-native-service-capacity-20261010/tools/gguf_reader.py');spec=importlib.util.spec_from_file_location('storage_gguf_extent',parser);gmod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=gmod;spec.loader.exec_module(gmod);gg=gmod.GGUFFile(SHARD);assert len(gg.tensors)==1;t=gg.tensors[0];assert t.name=='per_layer_token_embd.weight' and t.type_id==20 and t.shape==[160,320001536] and gg.data_start+t.offset==192 and t.expected_bytes()==28800138240 and SHARD.stat().st_size==28800138432
 with SHARD.open('rb') as f:header=f.read(192)
 fixture=dict(path=str(SHARD),**stat_id(SHARD),GGUF_parser=dict(path=str(parser),**ident(parser)),header_bytes=192,header_sha256=hashlib.sha256(header).hexdigest(),table_offset=192,table_bytes=28800138240,tensor_shape=t.shape,type_id=t.type_id)
 parent=B/'run_gdn_gate_factor_probe_v2.py';assert ident(parent)['sha256']=='7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f';source=parent.read_text().split('\ndef parse_probe(',1)[0];assert source.count("assert rss <= 2 << 30, 'owned RSS limit'")==1;source=source.replace("assert rss <= 2 << 30, 'owned RSS limit'","assert rss <= 1 << 30, 'owned RSS limit'");m=types.ModuleType('storagecapacityowner');exec(compile(source,str(parent),'exec'),m.__dict__);assert not OUT.exists();OUT.mkdir();m.W=OUT;o=m.Owner(OUT)
 env=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8');seed=2026101001;args=[str(binary),str(SHARD),mode,str(workers),str(seed)]+(['--qualify'] if stage=='qualify' else [])
 r=dict(active=True,complete=False,passed=False,stage=stage,mode=mode,workers=workers,repeat=repeat,seed=seed,fixture=fixture,binary=build['binary'],source=build['source'],build_receipt=dict(path=str(buildpath),**ident(buildpath)),controller=ident(Path(__file__)),parent_owner=ident(parent),modified_owner_sha256=hashlib.sha256(source.encode()).hexdigest(),commands=o.commands,metadata=dict(path=str(metadata),**ident(metadata)),qualifiers=qualifiers,environment=env,argv=args,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),gpu_work_submitted=False,model_inference=False,adopted=False,scope='SameoriginalPLEfile read-only blockingpread capacitycontrol; not actualPleReader/model, absoluteSSDpeak or physicalNAND service.',limits=dict(wall_seconds=180,RSS_session_bytes=1<<30,AS_each_bytes=16<<30,CPU_each_seconds=[120,121],text_bytes=4<<20,file_bytes=8<<20),counter_scope='Pool/block global counters around WHOLEPROCESS including startup and untimed buffered finalreferences; cannotlabel measured-window or puremedia traffic. No unrelatedtraffic subtraction.')
 def save():(OUT/'record.json').write_text(json.dumps(r,indent=2)+'\n')
 o.persist=save
 try:
  r['counters_before']=counters();command,stdout,stderr=o.run('storage-cell',args,env,wall=180,text_cap=4<<20,file_cap=8<<20);r['counters_after']=counters();save();lines=stdout.read_text().splitlines();r['output_rows']=[json.loads(line) for line in lines];r['stderr']=dict(path=str(stderr),**ident(stderr));save();assert m.completed(command) and command['exit_code']==0,command;assert len(lines)==1 and not stderr.read_bytes();x=r['output_rows'][0];assert x['success'] and x['stat_unchanged'] and x['mode']==mode and x['workers']==workers and x['seed']==seed
  count=max(16,workers) if stage=='qualify' else (27464 if mode=='sequential' else 65536);bs=1048576 if mode=='sequential' else 4096
  assert x['requests']==x['completed_requests']==count and x['block_bytes']==bs and x['logical_bytes']==count*bs and x['pread_syscalls']==count+x['eintr_retries'];assert x['qualification_only']==(stage=='qualify') and x['timing_claim']==(stage=='measure') and x['warmup_requests']==0
  assert x['final_buffer_checks']==workers and x['final_buffer_checked_bytes']==workers*bs and 1<=x['actual_peak_inflight_requests']<=workers and x['latency_clock_calls']==2*count
  assert math.isfinite(x['wall_seconds_observed']) and x['wall_seconds_observed']>0 and 0<=x['latency_median_ns']<=x['latency_p95_ns']<=x['latency_p99_ns']<=x['latency_sum_ns']
  if stage=='measure':
   assert math.isclose(x['logical_GBps'],count*bs/x['wall_seconds_observed']/1e9,rel_tol=1e-12) and math.isclose(x['IOPS'],count/x['wall_seconds_observed'],rel_tol=1e-12)
  else:assert 'logical_GBps' not in x and 'IOPS' not in x
  assert x['table_offset']==192 and x['table_bytes']==28800138240 and x['aligned_begin']==4096 and x['aligned_end']==28800135168 and x['sequential_begin']==1048576
  assert 4096<=x['min_requested_offset']<x['max_requested_end']<=28800135168 and x['original_stat']=={k:fixture[k] for k in ('dev','ino','size','mtime_sec','mtime_nsec','ctime_sec','ctime_nsec')}
  if mode=='sequential':assert x['min_requested_offset']==1048576 and x['max_requested_end']==1048576+count*1048576
  assert x['actual_f_getfl']&os.O_DIRECT and (x['actual_f_getfl']&os.O_ACCMODE)==os.O_RDONLY
  details=x['workers_detail'];assert len(details)==workers and {z['worker'] for z in details}==set(range(workers));assert sum(z['completed'] for z in details)==count and all(z['assignments']==z['completed']>=1 for z in details)
  deltas={k:r['counters_after']['ZFS_pool_iostats'][k]-v for k,v in r['counters_before']['ZFS_pool_iostats'].items()};assert all(v>=0 for v in deltas.values());r['whole_process_ZFS_deltas']=deltas
  r['whole_process_block_deltas']={k:[v-u for u,v in zip(r['counters_before'][k],r['counters_after'][k])] for k in ('block_leaf_nvme0n1','block_partition_nvme0n1p3')}
  r['whole_process_direct_DMU_to_timed_payload_ratio']=deltas['direct_read_bytes']/x['logical_bytes'];r['whole_process_leaf_read_bytes']=r['whole_process_block_deltas']['block_leaf_nvme0n1'][2]*512;r['whole_process_leaf_to_timed_payload_ratio']=r['whole_process_leaf_read_bytes']/x['logical_bytes']
  assert stat_id(SHARD)=={k:fixture[k] for k in ('dev','ino','size','mode','mtime_sec','mtime_nsec','ctime_sec','ctime_nsec')};assert ident(src)=={k:build['source'][k] for k in ('bytes','sha256')};assert ident(binary)=={k:build['binary'][k] for k in ('bytes','sha256')};assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==r['boot_id'];r.update(passed=True,complete=True)
 except BaseException as e:r['error']=type(e).__name__+': '+str(e);r['traceback']=traceback.format_exc()
 finally:r['active']=o.active is not None;save();print(json.dumps({k:r.get(k) for k in ['stage','mode','workers','repeat','passed','complete','active','error','whole_process_direct_DMU_to_timed_payload_ratio']}),flush=True)
 if not r['passed']:raise SystemExit(1)
