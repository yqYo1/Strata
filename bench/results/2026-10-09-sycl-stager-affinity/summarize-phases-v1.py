"""Assess prefill, cold decode and restored decode independently after all18 runs."""
from pathlib import Path
import datetime, fcntl, hashlib, itertools, json, math, statistics, sys
B=Path(__file__).parent
sys.path.insert(0,'/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05/sycl/tools')
from owned_gdb import process_identity
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
seq_path=B/'stager-affinity-v0141-decode-repeat-comparison-sequence-v1/record.json'
seq=json.loads(seq_path.read_text())
assert not seq['active'] and seq['passed'] and len(seq['steps'])==18
assert seq['controller_sha256']=='f6fd828a668d83d60713cf7ecb635c9c33512488dac8e60747895b13a6a53327'
assert seq['sequence_controller_sha256']=='0600cfad0060fb99c80a306dc6ac85df29d435321d4750a00f9dc77913347cc3'
assert seq['first_diagnostic_receipt_sha256']=='ff94770a606e18639b6e49c03c10722a74698b733fa32bb507091b9464a5aebc'
lock=(B/'owned-v0141-measurement.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert seq['boot_id']==boot
modes=['baseline','affinityoff','affinityon']; groups={mode:[] for mode in modes}; infos=[]
source_heads={'baseline':'23268953314d12588fd3f496414a46bd426a7306','affinityoff':'ba1494d9a0848f37066228cec08fb82022a2c307','affinityon':'ba1494d9a0848f37066228cec08fb82022a2c307'}
binaries={'baseline':'86972697ecb1903750d202f8be29a7a1da351eb3415678c3e3b6af790804d1c8','affinityoff':'7c014a1ddebb4878ca225e67c5508d4ae1caa951aa5ffdb884ded9e440951c55','affinityon':'7c014a1ddebb4878ca225e67c5508d4ae1caa951aa5ffdb884ded9e440951c55'}
for step in seq['steps']:
    p=Path(step['receipt']);assert sha(p)==step['receipt_sha256']
    r=json.loads(p.read_text());assert step['mode']==r['mode'] and step['block']==r['repetition']
    assert r['phase']=='clean' and not r['active'] and r['healthy'] and r['completed'] and r['math_gate_passed'] and r['phase_specific_decode_sequence_completed']
    assert r['exit_code']==0 and not r['exit_signal'] and not r['new_fault_messages'] and not any(r['cleanup'].values())
    assert r['boot_id']==boot and not r['adopted'] and not r['full_lifecycle_passed']
    assert r['commit']==source_heads[step['mode']] and r['binary_sha256']==binaries[step['mode']]
    for identity in [r['inferior'],r['debugger'],step['controller_identity']]:
        now=process_identity(identity['pid']);assert not now or now['start_ticks']!=identity['start_ticks']
    env=r['actual_target_environment'];assert env==r['environment']
    assert not any(k.startswith(('UR_LOG_','ZE_ENABLE_','ZEL_')) or k in ['UR_ENABLE_LAYERS','STRATA_TRACE','STRATA_PREFILL_COPY_ENGINE','STRATA_PREFILL_COPY_PHASE_RELEASE','STRATA_DUMP_FIRST_LOGITS','STRATA_PREFILL_DUMP_STATE'] for k in env)
    assert env.get('STRATA_STAGER_CPU_LIST')==('1,2,3' if step['mode']=='affinityon' else None)
    assert r['stager_affinity_startup_gate']['passed']
    assert r['first_stager_affinity_diagnostic_receipt_sha256']==seq['first_diagnostic_receipt_sha256']
    info_lines=[x for x in r['startup'] if x.startswith('INFO ')];assert len(info_lines)==1 and r['startup'][-1].split()[1]=='262144'
    info=dict(x.split('=',1) for x in info_lines[0].split()[1:]);infos.append(info)
    for key,value in {'context':'262144','kv':'int8','kv_resident':'32768','expert_slots':'128','expert_cache_mib':'325','spec':'4','pool_workers':'5','pcie_frac':'0.00','vram_free_mib':'1565','engine':'0.1.41'}.items(): assert info[key]==value,(step['mode'],step['block'],key,info[key])
    assert len(r['requests'])==14
    assert all(x['math_gate_passed'] and all(x['qualified_reference_output_comparison'].values()) for x in r['requests'])
    prime=r['requests'][0];warm=r['requests'][1]
    assert prime['phase_kind']=='prefill-prime' and prime['mtp_counts']==[41,66] and prime['measurement']['prompt_tokens']==32768 and prime['measurement']['generated_tokens']==64
    assert warm['warmup_decode'] and not warm['timing_eligible']
    samples=[x for x in r['requests'] if x['timing_eligible']]
    assert len(samples)==12 and samples==r['requests'][2:]
    assert all(x['phase_kind']=='decode-only' and not x['warmup_decode'] and x['resume_tokens']==[32831,32831] and x['mtp_counts']==[46,51] for x in samples)
    assert all(x['measurement']['generated_tokens']==64 and x['measurement']['prompt_tokens']==32832 for x in samples)
    ms=[x['measurement']['decode_ms'] for x in samples]
    assert all(math.isfinite(x) and x>0 for x in ms+[prime['measurement']['prompt_ms'],prime['measurement']['decode_ms']])
    groups[step['mode']].append({'block':step['block'],'position':step['position'],'receipt':str(p),'receipt_sha256':sha(p),'fresh_prefill_ms':prime['measurement']['prompt_ms'],'fresh_first_decode_ms':prime['measurement']['decode_ms'],'restored_mean_decode_ms':statistics.mean(ms),'restored_median_decode_ms':statistics.median(ms),'restored_decode_ms':ms,'within_process_sd_ms':statistics.stdev(ms),'warm_restored_decode_ms_excluded':warm['measurement']['decode_ms']})
assert all(x==infos[0] for x in infos)
for mode,ps in groups.items():
    assert sorted(x['block'] for x in ps)==list(range(1,7))
    assert all(sum(x['position']==p for x in ps)==2 for p in range(3))
phases={'prefill':('fresh_prefill_ms',32767),'fresh_first_decode':('fresh_first_decode_ms',64),'restored_decode':('restored_mean_decode_ms',64)}
# Protocol prompt count includes the final input token processed by decode.
# Preserve existing reported prefill convention32768/PPms separately from actual32767 batched prefix.
phases['prefill']=('fresh_prefill_ms',32768)
summaries={};comparisons={}; t=2.570581835636314
for phase,(key,tokens) in phases.items():
    summaries[phase]={}; comparisons[phase]={}
    for mode,ps in groups.items():
        vals=[x[key] for x in ps]
        summaries[phase][mode]={'independent_processes':6,'requests':72 if phase=='restored_decode' else 6,'tokens_per_request':tokens,'mean_ms':statistics.mean(vals),'tok_s':tokens*1000/statistics.mean(vals),'independent_process_sd_ms':statistics.stdev(vals),'minimum_process_ms':min(vals),'maximum_process_ms':max(vals),'process_values_ms':vals}
    for candidate,reference in [('affinityoff','baseline'),('affinityon','affinityoff'),('affinityon','baseline')]:
        cp={x['block']:x for x in groups[candidate]};rp={x['block']:x for x in groups[reference]}
        logs=[math.log(cp[k][key]/rp[k][key]) for k in range(1,7)]
        m=statistics.mean(logs);se=statistics.stdev(logs)/math.sqrt(6)
        ci=[100*(math.exp(-(m+t*se))-1),100*(math.exp(-(m-t*se))-1)]
        permutations=[abs(statistics.mean(s*x for s,x in zip(signs,logs))) for signs in itertools.product([-1,1],repeat=6)]
        comparisons[phase][candidate+'_vs_'+reference]={'paired_independent_blocks':6,'geometric_speed_change_percent':100*(math.exp(-m)-1),'speed_change_95_percent_ci':ci,'block_speed_changes_percent':[100*(math.exp(-x)-1) for x in logs],'paired_block_log_latency_ratios':logs,'paired_log_latency_standard_error':se,'student_t_degrees_of_freedom':5,'exact_sign_flip_two_sided_p':sum(x>=abs(m)-1e-14 for x in permutations)/64,'confidence_interval_includes_zero':ci[0]<=0<=ci[1],'interpretation':'Measured regression' if ci[1]<0 else 'Measured improvement' if ci[0]>0 else 'Direction unresolved; equivalence not established'}
out=B/'stager-affinity-v0141-phase-summary-v1.json';assert not out.exists()
result={'active':False,'passed':True,'adopted':False,'full_lifecycle_passed':False,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Three phases evaluated separately on B57010GiB/Ryzen5600X/128GB, Qwen3.8FlashNextIQ3_S. Six order-balanced independent process blocks per mode, one fresh32768 prime plus13 restored32831-state continuations; first restored warm request excluded,12 measured repeats. All IDs/LP/MTP match qualified869. No native-copy/release kernel or queue change.','sequence_receipt_sha256':sha(seq_path),'summarizer_sha256':sha(__file__),'actual_info':infos[0],'all18_geometry_equal':True,'groups':groups,'summary':summaries,'comparisons':comparisons,'phase_adoption_policy':'Prefill gains never compensate for decode regression; evaluate fresh and restored decode independently. Cross-zero CI does not establish equivalence. New candidate not physically262144 qualified.','limitations':['Six independent process blocks per mode;72 restored requests are not72 independent process samples.','Coding fixture32K and64-output continuation only; no general workload claim.','MTP acceptance41/66 fresh and46/51 restored are different output paths and are not pooled.','Reported prefill rate uses protocol prompt32768;32767 input tokens are batched by prefill and one input token enters decode.','Student t paired log-latency interval has5degrees of freedom; sign-flip check assumes exchangeable signs.','Physical variation and concurrent background host work cannot be eliminated by repetition.','No physical full256K qualification for this candidate and no production adoption.']}
out.write_text(json.dumps(result,indent=2)+'\n')
fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
print(json.dumps({'summary':summaries,'comparisons':comparisons,'output':str(out),'sha256':sha(out)},indent=2))
