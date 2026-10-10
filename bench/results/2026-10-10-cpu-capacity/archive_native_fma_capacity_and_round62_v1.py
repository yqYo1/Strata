from pathlib import Path
from collections import defaultdict
import datetime,fcntl,hashlib,json,math,shutil,statistics,subprocess
B=Path(__file__).parent;R=B/'research-20261009';W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009');A=W/'bench/results/2026-10-10-parallel-round62';C=W/'bench/results/2026-10-10-cpu-capacity'
def ident(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def copy(p,d):assert not d.exists();d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d);assert ident(p)==ident(d)
def closed(r):
 assert r['passed'] and r['complete'] and not r['active']
 for cmd in r['commands']:
  assert cmd['exit_code']==0 and cmd['normal_exit'] and cmd['session_empty'] and cmd['observation_complete'] and not cmd['errors'] and not cmd['survivors'] and not cmd['cleanup']
  for o in cmd['owners']:
   p=Path('/proc')/str(o['pid'])/'stat'
   if p.exists():assert int(p.read_text().rsplit(')',1)[1].split()[19])!=o['start_ticks']
def compact(p,target,qualify=False):
 r=json.loads(p.read_text());rows=r.pop('output_rows',[]);r['original_receipt']=dict(path=str(p),**ident(p));r['normalized_archive']=True
 refs=[x for x in rows if x[0]=='REFERENCE_TOKEN'];metrics={}
 for stage in ('GU_SwiGLU','Down_actual_FF','full_chain_independent_FF'):
  selected=[x for x in refs if x[6]==stage]
  if selected:
   budget=(1e-3,1e-4) if stage=='full_chain_independent_FF' else (1e-4,1e-5)
   for x in selected:assert float(x[7])<=budget[0] and float(x[8])<=budget[1]
   metrics[stage]=dict(vectors=len(selected),rows=sum(int(x[13]) for x in selected),maxabs=max(float(x[7]) for x in selected),max_nrms=max(float(x[8]) for x in selected))
 r['verified_reference_summary']=metrics;r['all_reference_vectors_original_sha256']=hashlib.sha256(json.dumps(refs,separators=(',',':')).encode()).hexdigest()
 keep={'ROUND','WARMUP','ROUND_COUNTS','REFERENCE_COUNTS','REFERENCE_WORST','GATE','ROUTE','WORKING_SET','ACTIVE_LOGICAL_WORKING_SET','HOT_ID','PLACEMENT','TASK_PLAN','RESULT','META','ENV'}
 r['output_rows']=[x for x in rows if x[0] in keep];write(target,r);return r
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()=='2563819df746f78b4160ba1967e8f210efd9c23b';assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip();assert not C.exists() and not A.exists()
 matrix=B/'native-capacity-v2-matrix-plan-v1.json';plan=json.loads(matrix.read_text());assert plan['passed'] and plan['complete'] and not plan['active'] and len(plan['closed'])==24
 copy(matrix,C/matrix.name); groups=defaultdict(list);samples=[]
 for cell in plan['closed']:
  p=Path(cell['path']);assert ident(p)['sha256']==cell['sha256'];original=json.loads(p.read_text());closed(original);r=compact(p,C/'native'/(p.parent.name+'.json'))
  measured=[x for x in r['output_rows'] if x[0]=='ROUND' and x[1].isdecimal()];assert len(measured)==250
  for cohort in (0,1):
   for arm in ('GU','FFquant','Down','direct_complete','pool'):
    v=[x for x in measured if x[1]==str(cohort) and x[2]==arm];assert len(v)==25
    nt=cell['NT'];jobs=8 if cell['mode']=='hot' else 192;u=[float(x[12])*1000/jobs/nt for x in v];ms=[float(x[12]) for x in v]
    entry=dict(**{k:cell[k] for k in ('NT','mode','repeat')},cohort=cohort,arm=arm,measured_rounds=25,median_us_per_expert_token=statistics.median(u),median_ms=statistics.median(ms),sample_archive=str((C/'native'/(p.parent.name+'.json')).relative_to(C)))
    if arm!='FFquant':
     mac={'GU':3276800,'Down':1638400,'direct_complete':4915200,'pool':4915200}[arm];weight={'GU':1049600,'Down':921600,'direct_complete':1971200,'pool':1971200}[arm];entry.update(dense_equivalent_GFLOPs=2*mac/entry['median_us_per_expert_token']/1000,logical_packed_weight_GBps=weight/(entry['median_us_per_expert_token']*nt)/1000)
    samples.append(entry);groups[nt,cell['mode'],cohort,arm].append(entry)
 summary=[]
 for (nt,mode,cohort,arm),v in sorted(groups.items()):
  assert {x['repeat'] for x in v}=={1,2,3};values=[x['median_us_per_expert_token'] for x in v];e=dict(NT=nt,mode=mode,cohort=cohort,arm=arm,process_medians_us=values,median_us_per_expert_token=statistics.median(values),process_median_range_us=[min(values),max(values)])
  if arm!='FFquant':
   for key in ('dense_equivalent_GFLOPs','logical_packed_weight_GBps'):e[key]=statistics.median(x[key] for x in v)
  summary.append(e)
 write(C/'native-three-process-summary.json',dict(source_commit='dba6bea90192f907cbeecdb75c29aa0e7d1fa449',processes=24,measured_rows=6000,warmup_rows=720,convention='OneMAC=2 dense-equivalentFLOPs; packedquant instruction counts and physicalDRAM bytes are unmeasured.',samples=samples,cells=summary))
 for nt in (1,2,3,4):
  p=B/f'native-capacity-v2-qualify-nt{nt}-streaming-r1-controller2/record.json';closed(json.loads(p.read_text()));compact(p,C/'qualify'/f'nt{nt}-normalized.json',True)
 p=B/'native-capacity-v2-qualify-nt1-streaming-r1/record.json';old=json.loads(p.read_text());assert not old['passed'] and not old['active'] and old['commands'][0]['exit_code']==0;compact(p,C/'qualify'/'original-controller-finalmarker-failure.json',True)
 # One canonical ID/extent list preserves exact payload identity without duplicating it24times.
 canonical=json.loads((B/'native-capacity-v2-qualify-nt1-streaming-r1-controller2/record.json').read_text());write(C/'selected-payload-identity-rows.json',[x for x in canonical['output_rows'] if x[0] in ('ID','EXTENT')]);copy(B/'native-service-calibration-cohort-22-20-nt1-v2/manifest.json',C/'cohort-manifest.json');copy(B/'native-service-calibration-cohort-22-20-nt1-v2/cohort.tsv',C/'cohort.tsv')
 for name in ('run_native_capacity_v2.py','run_native_capacity_v2_controller2.py','run_native_capacity_matrix_v1.py','build_native_capacity_v2.py','cpu_fma_capacity_v1.cpp','run_cpu_fma_capacity_v1.py'):copy(B/name,C/name)
 copy(B/'native-capacity-v2-build/record.json',C/'native-build-record.json');copy(B/'cpu-fma-capacity-build-v1/record.json',C/'fma-build-record.json')
 fma=[]
 for repeat in (1,2,3):
  p=B/f'cpu-fma-capacity-v1-r{repeat}/record.json';r=json.loads(p.read_text());closed(r);copy(p,C/f'fma-r{repeat}-record.json')
  for threads in (1,6):
   values=[x['GFLOPs'] for x in r['output_rows'] if x.get('threads')==threads and x['sample']>=0];assert len(values)==7;fma.append(dict(repeat=repeat,physicalcores=threads,measured_GFLOPs=values,median_GFLOPs=statistics.median(values)))
 fmasummary=[]
 for c in (1,6):
  v=[x['median_GFLOPs'] for x in fma if x['physicalcores']==c];fmasummary.append(dict(physicalcores=c,median_GFLOPs=statistics.median(v),process_medians_GFLOPs=v,range_GFLOPs=[min(v),max(v)]))
 write(C/'fma-three-process-summary.json',dict(samples=fma,cells=fmasummary,scope='Register-only FP32 AVX2 FMA,12independentYMMaccumulators,64millioniterations,complete96float scalarFMAreferences and singleton physicalcoreaffinityeachrun. DifferentoperationfromIQ2_S/IQ4_NL.' ))
 lines=['# CPU attainable capacity and native expert service','','Ryzen5 5600X, unchanged powersave/EPPpower policy. No GPU or model inference. Three fresh processes per condition; no optimization adopted.','','| FP32 AVX2 register FMA | Median GFLOP/s | Range of process medians |','| --- | ---: | ---: |']
 for x in fmasummary:lines.append(f"| {x['physicalcores']} physical cores | {x['median_GFLOPs']:.2f} | {x['range_GFLOPs'][0]:.2f}–{x['range_GFLOPs'][1]:.2f} |")
 lines+=['','The loop contains exactly12 independent packed FP32 FMA instructions, no loop spill or memory operand. One warmup and seven measurements per core-count perprocess; full96float outputs compare to untimed scalarFMA references. This is attained arithmetic throughput under a register-only workload, not a quantized integer-dot ceiling.','','The native service fixture uses384 actual GU22/IQ2_S and Down20/IQ4_NL experts, deterministic synthetic activations, H2560/FF640. Fresh NT1–4 qualifiers preserve exact NT1 controls and predeclared maxabs/NRMS budgets; worst observed errors are below2.4e-7absolute GU and4.5e-8Down/fullchain. Every measured process repeats all references, guards, direct/pool parity, placement and full row counts.','','| NT | Active set | GU µs/expert-token | Down µs/expert-token | Direct complete µs/expert-token | Pool complete µs/expert-token |','| ---: | --- | ---: | ---: | ---: | ---: |']
 for nt in (1,2,3,4):
  for mode in ('hot','streaming'):
   v=[statistics.median([e['median_us_per_expert_token'] for e in samples if e['NT']==nt and e['mode']==mode and e['arm']==arm]) for arm in ('GU','Down','direct_complete','pool')];lines.append('| '+str(nt)+' | '+mode+' | '+' | '.join(f'{x:.3f}' for x in v)+' |')
 lines+=['','Table native values summarize the six process/cohort medians (3processes×2disjointcohorts). Structured results retain each separate process/cohort median and all6000 raw measurement rows plus720warmups. These are service intervals, not token throughput of the whole model. Hot uses8selectedexperts/cohort≈15.04MiB packedblobs; streaming uses192≈360.94MiB. Differentexpert subsets and jobcounts prevent a pure cache-causality inference; cache residency and physicalDRAM traffic are unmeasured.','','Direct runs use one pinned caller. Pool runs use five workers plus host, tasks0→18rowpartitions, batch1, synchronous phases with serial FFquant. This is no worker-count sweep. Dense-equivalent FLOPs normalize mathematical dot work; they cannot be divided by register FP32 capacity to claim a percent CPU utilization. NT4 emitted spills do not prevent it being faster pertoken in this fixture; spill dominance remains unmeasured.','','The first untimed process passed all numerical/placement checks and normal0, but its controller incorrectly expected the timed finalmarker; its original receipt remainsFAILED. Controller2 changed only stage-marker parsing, pinned the actual manifest cohort_sha256 and added traceback reporting; all four fresh qualifiers and24timingprocesses pass unchanged source/gates.','','Compact archives preserve original receipt hashes/statuses and each timing row, unique canonicalpayload identities, exact count/worst-error summaries, commands/ownership/policy/build/source/binarypins. Repeated successful reference-vector text is represented by verified count/worst summaries and a hash; no failed status is rewritten. Current four original qualifiers remain fixtures for the calibration controller. No full262144-position model boundary is qualified.']
 (C/'README.md').write_text('\n'.join(lines)+'\n')
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();prev=R/'report-registry-v84.json';assert ident(prev)['sha256']=='fd025ce071bd30fdbef0cb547f06ed31397541d922be29bb2897d09aca33b922';d=json.loads(prev.read_text());assert len(d['reports'])==335;new=[]
 for name,agent,digest,scope in [('round229-native-capacity-rate-denominators.txt','research_bottleneck_evidence_audit_v203','a971db770f861d7c37c44e45a78e034fbbc1f25f302e317bc8828e9f6120d29a','Actual native arithmetic/packed-byte denominators and closeduntimed qualifiers'),('round230-zfs-directio-storage-ceiling-contract.txt','research_native_iq4nl_esimd_k640_v202','89265f5dce55ac3d0840fdf917bd392bf6a1656d8c8a23dc1a5a1dfc35db21c1','OpenZFS2.4.1 directpath and storagecounter contract')]:
  p=R/name;assert ident(p)['sha256']==digest;copy(p,A/'research'/name);e=dict(path=str(p),**ident(p),agent='/root/'+agent,model='gpt-6-luna',scope=scope,status='completed-read-only',root_full_report_reviewed=True);new.append(e);d['reports'].append(e)
 d.update(registry_version=85,created_utc=now,research_completed=337,new_reports=new,previous_committed_registry=dict(path=str(prev),**ident(prev),storage_commit='2563819df746f78b4160ba1967e8f210efd9c23b'),scope='Native capacity and ZFSdirectpath review; actualCPU capacities complete',current_root_decisions=dict(CPU='24freshnative timingprocesses+4numericqualifiersPASS; registerFP32 FMA3freshprocessesPASS. No physicalDRAM/cycle or model claim.',storage='Kernel2.4.1, userspace2.2.2 lacks directproperty query; loadedDIO enabled1/strict0, primarycachemetadata/secondarynone/compressionoff/recordsize128KiB observed. O_DIRECT selection still needs pertrial counters.',GPU='Earlierhardware controls and no-overlap trace complete; actual32Kserviceledger/oneMKLcompleteinternalspan pending.',adopted=False),live_agent_snapshot=dict(time_utc=now,agents=[dict(agent=e['agent'],model='gpt-6-luna',status='completed') for e in new]+[dict(agent='/root/implement_prefill_phase_timer_validity',model='gpt-6.1-sol',status='running-source-only-storage-capacity')]))
 correction={'R229':'Exactdenominators accepted. Agent reviewed qualifiers only; root now supplies freshtimings. Hot/streamingdifferences alone do not prove RAM/instruction bottleneck; poolphasewallnotworkertime.', 'R230':'Sourceconditionaldirectpath contract accepted. Current userland2.2.2 cannot query directproperty throughitsCLI although loadedmodule2.4.1; primarycachemetadata/moduleDIO1 do not replace timed DMU/direct counter proof. SSDraw/NAND ceiling unmeasured.'};d['root_review_corrections_round62']=correction;reg=R/'report-registry-v85.json';assert not reg.exists();write(reg,d);copy(reg,A/reg.name);write(A/'root-review.json',dict(created_utc=now,reports=new,corrections=correction,registry=ident(reg),CPU_archive=str(C),native_measurement_count=24,FMA_fresh_processes=3));(A/'README.md').write_text('R229 gives exact native capacitydenominators and qualifier scope; R230 gives OpenZFS2.4.1 directI/O gating and actualcounter requirements. Root concurrently completed24freshnative timingprocesses and3freshregisterFMAprocesses, archived compact per-run evidence and all individual timings in ../2026-10-10-cpu-capacity. Quantized dense-equivalent rates and FP32arithmetic rates remain distinct; SSDraw/actual32K/full262K limits are not inferred.\n')
 copy(Path(__file__),C/Path(__file__).name)
 write(C/'archive-file-identities.json',{str(p.relative_to(C)):ident(p) for p in sorted(C.rglob('*')) if p.is_file()});write(A/'archive-file-identities.json',{str(p.relative_to(A)):ident(p) for p in sorted(A.rglob('*')) if p.is_file()})
 for args in [('diff','--check'),('add','bench/results/2026-10-10-parallel-round62','bench/results/2026-10-10-cpu-capacity'),('diff','--cached','--check'),('commit','-m','bench(cpu): measure native hot and streaming capacity with FP32 control'),('push',)]:subprocess.run(['git',*args],cwd=W,check=True)
 print(json.dumps(dict(registry=str(reg),**ident(reg),storage_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip(),native_rows=6000,FMA=fmasummary)))
