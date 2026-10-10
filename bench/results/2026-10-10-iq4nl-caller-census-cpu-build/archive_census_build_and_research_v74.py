"""Root preserves closed census build and fully read R200-R203; no GPU run or cleanup."""
from pathlib import Path
import datetime, fcntl, hashlib, json, re, shutil, subprocess
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
R=B/'research-20261009'
M=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree')
C=M/'perf-sycl-iq4nl-caller-census-20261010'
W=M/'docs-sycl-storage-retention-20261009'
CP=C/'bench/results/2026-10-10-iq4nl-caller-census-cpu-build'
A=W/'bench/results/2026-10-10-parallel-round51'
def ident(p):
    p=Path(p)
    with p.open('rb') as f: return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def write(p,d): p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def copy(s,p):
    assert not p.exists(),str(p)
    p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(s,p)
    assert ident(s)==ident(p)
def git(w,*args): return subprocess.check_output(['git',*args],cwd=w,text=True).strip()
def commit(w,path,msg):
    for args in [('diff','--check'),('add',path),('diff','--cached','--check'),('commit','-m',msg),('push',)]:
        subprocess.run(['git',*args],cwd=w,check=True)
    return git(w,'rev-parse','HEAD')
with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for w in [C,W]:
        assert not git(w,'status','--porcelain'),str(w)
        for key in ['user.name','user.email']:
            assert subprocess.run(['git','config','--local','--get',key],cwd=w,capture_output=True).returncode==1
    assert git(C,'rev-parse','HEAD')=='484f38051c5cdc46313f232b9f746a931ab9e57b'
    assert git(W,'rev-parse','HEAD')=='313d3a9cc43728401f392db5b10bf4084d6fb35f'
    run=B/'iq4nl-caller-census-cpu-build-v1'
    d=json.loads((run/'record.json').read_text())
    assert d['passed'] and d['complete'] and not d['active']
    assert not d['GPU_executed'] and not d['model_opened'] and not d['adopted']
    for rel,pin in d['source_pins'].items(): assert ident(C/rel)==pin,rel
    for key in ['binary','engine_binary']: assert ident(d[key]['path'])=={k:d[key][k] for k in ['bytes','sha256']}
    assert all(x['exit_code']==0 and x['normal_exit'] and x['session_empty'] and x['direct_child_reaped'] and not x['errors'] and not x['cleanup'] for x in d['commands'])
    assert not CP.exists();CP.mkdir(parents=True)
    for name in ['record.json','configure.stdout','direct-imports.stdout']: copy(run/name,CP/name)
    copy(B/'build_iq4nl_caller_census_v1.py',CP/'build_iq4nl_caller_census_v1.py')
    build=Path(d['build'])
    copy(build/'compile_commands.json',CP/'compile_commands.json')
    copy(run/'target-commands.stdout',CP/'target-commands.stdout')
    raw=(run/'build.stdout').read_text()
    steps=re.findall(r'^\[(\d+)/(\d+)\] (.*)$',raw,re.M)
    assert len(steps)==126 and [int(x[0]) for x in steps]==list(range(1,127))
    (CP/'build-steps.txt').write_text('\n'.join(f'[{n}/{total}] {s}' for n,total,s in steps)+'\n')
    warnings={}
    for line in raw.splitlines():
        if ': warning:' in line:
            message=line.split(': warning:',1)[1].strip()
            row=warnings.setdefault(message,dict(count=0,first_diagnostic=line));row['count']+=1
    write(CP/'warning-summary.json',dict(original=ident(run/'build.stdout'),scope='Each compiler warning text and count; first source location preserved. No diagnosis of emitted native behavior or undefined behavior.',warnings=warnings))
    write(CP/'original-file-identities.json',{str(p):ident(p) for p in [run/'record.json',run/'build.stdout',run/'build.stderr',build/'compile_commands.json',run/'target-commands.stdout']})
    (CP/'REPORT.md').write_text("""# IQ4NL caller-census full CPU build

Root built fresh dependencies and the actual Strata engine from source
484f38051c5cdc46313f232b9f746a931ab9e57b with the caller-census diagnostic ON.
All126 Ninja steps completed; build elapsed363.665s. Every owned stage exited
normally, was reaped and left an empty observed session without forced cleanup.
No GPU command or model inference was executed by this build.

Exact source, compilers, commands, environment, limits, selected compilation
arguments and actual binary hashes are in the unchanged record. Complete
configured compilation commands and reachable target commands are retained.
The census macro is private to prefill.cpp. Private type20 Down dequant and
event-receipt fixture are also enabled; this is a diagnostic binary, not the
qualified default or an adopted performance result.

Earlier fourteen CPU-only ledger cases are in the sibling source-host
contract proof. Their synthetic262144 count does not qualify physical KV or
full model context. Runtime calls, async completion, actual>=32768 batched
positions, full262144 lifecycle and model speed remain unobserved here.

Repeated compiler warnings are represented by exact messages/counts
and first source locations, plus all126 build steps and original hashes.
The4.36MB raw build log remains outside Git; no file is deleted here.
""")
    copy(Path(__file__),CP/Path(__file__).name)
    write(CP/'archive-file-identities.json',{str(p.relative_to(CP)):ident(p) for p in sorted(CP.rglob('*')) if p.is_file()})
    ccommit=commit(C,str(CP.relative_to(C)),'docs(sycl): preserve closed caller-census engine build')
    prev=R/'report-registry-v73.json'
    assert ident(prev)['sha256']=='185d9d002c5eec59b01e04d46451a4a018098130e8854b95adcbec9efe26293a'
    registry=json.loads(prev.read_text());assert len(registry['reports'])==306
    assert not A.exists();A.mkdir(parents=True)
    entries=[
        (200,'round200-caller-census-aggregate-bounds-independent-audit.txt',7978,'53d190a9eb7c51b737415c2d92c4d75cb41981f9fc1f6d6073575cc10c0b1ff1','research_census_aggregate_bounds_v200','Frozen census bounds, aggregate limitations and completion scope'),
        (201,'round201-external-iq4nl-quantized-moe-gemm-eligibility.txt',17667,'63c3dec7de77db5f3b47baa7ecd4c94d88877f2167e52ec1c0ad0b8834de7d71','research_quantized_moe_gemm_eligibility_v201','External IQ4NL MoE GEMM format, K640 and numerical eligibility'),
        (202,'round202-native-iq4nl-esimd-k640-precision-design.txt',13694,'e37ae1d9914a9643915f8a4c2dcfb71b5739073a3542c6fa92a01122129bea6e','research_native_iq4nl_esimd_k640_v202','Native-layout direct packed XMX B design; root corrections required'),
        (203,'round203-prefill-decode-bottleneck-evidence-audit.txt',17272,'01ba06fdce9526bb282303de0523d7f7d135138531e3bea7a9f9dbfa0c679d15','research_bottleneck_evidence_audit_v203','Closed real-model bottleneck evidence and overlap limitations')]
    new=[]
    for number,name,size,digest,agent,scope in entries:
        p=R/name;assert ident(p)==dict(bytes=size,sha256=digest),name
        copy(p,A/'research'/name)
        e=dict(path=str(p),**ident(p),agent='/root/'+agent,model='gpt-6-luna',scope=scope,status='completed-read-only',root_full_report_reviewed=True)
        new.append(e);registry['reports'].append(e)
    corrections=[
        dict(report='R200',note='Aggregate counters lack perchunk expert identity and following GEMM ledger; caller return is not asynchronous GPU success. Root14hostcases and actual2360864B sizeof arrived after report. Full CPU SYCL build nowPASS; no model/physical262144 proof.'),
        dict(report='R201',note='External llm-scaler pinned0bf3789b9b1548b5d5078a8deaba214b59f12f04 remains non-drop-in. K640 per-row fallback, normalized ABI, weight/output rounding differ. No external local speed claim.'),
        dict(report='R202',note='Correct native payload:2560*640/32=51200blocks;51200*18=921600 compressedB, not3276800.3276800 is FP16 output bytes. For nativeqs[j], low nibble maps k=32*b+j and high k=32*b+16+j, not2*j/2*j+1. Correct before any implementation; preserve original.'),
        dict(report='R202',note='Frozen xmx kernel already reuses each packedB slab across up to32 M tiles/512 rows (MAXT4*NSG8*16) within each M chunk. A claimed R-fold saving from consecutive16-row tiles must account for existing reuse. K640 whole-slab reuse across M chunks is separate design, not measured. Current model fused/XMX route is inactive.'),
        dict(report='R202 final message',note='Final hyperlink had research-202609 typo; actual file is research-20261009 with verified digest. Original report filename and bytes are correct.'),
        dict(report='R203',note='Restored decode interval crossing0 does not apply to first decode after fresh prefill. Same closed repeat summary reports nativeon-vsbaseline fresh firstdecode geometric-7.9851%,95%[-12.8063,-2.8972] over6independentpairedprocesses; on-vsoff-5.8660%,95%[-10.1665,-1.3596]. A fresh-phase regression is established in that workload; do not state no decode regression globally.'),
        dict(report='R203',note='Existing firstdecode CPU-dispatch host counter baseline123.7767ms/window versus total165.3783 andGPUreach2.7633 is a concrete prioritization signal. Not exclusive CPU busy time; math/RAM/sync split missing. Input stage doublecounts PLE within enclosing stage, so stage increase is not an exclusive fraction.')]
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    registry.update(registry_version=74,created_utc=now,research_completed=310,new_reports=new,previous_committed_registry=dict(path=str(prev),**ident(prev),commit='313d3a9cc43728401f392db5b10bf4084d6fb35f'))
    registry['root_review_corrections_round51']=corrections
    registry['live_agent_snapshot']=dict(observed_utc=now,active_root=True,research_active=[],completed_fullread=['R200','R201','R202','R203'],implementer='completed; rootCPUcontract14cases and freshenginebuildPASS',next_assignment='Distinct fresh Luna scopes after this commit; main prioritizes actual prefilling/decode phase attribution.')
    registry['current_root_decisions'].update(
        census_source='484f38051c5cdc46313f232b9f746a931ab9e57b',census_build_proof_commit=ccommit,
        census_CPU_build=dict(path=str(run/'record.json'),**ident(run/'record.json'),passed=True,complete=True,active=False,GPU_executed=False),
        census_runtime='NOT RUN. Aggregate accounting only; actual model route, followingGEMM identity, async success and physical262144 unqualified.',
        current_bottleneck_evidence='Prefill nativecopy interventional gain~20.36%, not PCIe-only attribution. Freshdecode CPU-dispatch123.78ms/window vswall165.38 is priority; innerCPU math/RAM/sync unresolved. No exclusive currentkernel share.',
        native_copy_decode_policy='Freshfirstdecode-7.985% geometric paired regression95%[-12.806,-2.897]; restored-1.240%CIcrosses0. Keep phases separate; unadopted, do not offset decode loss with prefill gain.',
        XMX_direct_B='HOLD source-only native18B directpacked loader hypothesis; correctpayload/nibblemapping/existing512rowreuse before design. Actualcompiled/route/backend/wholemodel qualification absent.',
        next_gate='Prioritize current actual>=32768 batched prefill phase+MoE command attribution and first/restored decode pool GU/quant/down+sync separation before another candidate sweep.',
        root_execution='Root closed fresh126step census/parity/engine CPU build363.665s,14priorHOSTcases; read four returnedLunareports. No model or GPU work thisbuild.',
        model_performance_this_wave=False,actual_physical_262144_lifecycle=False,adopted=False)
    reg=R/'report-registry-v74.json';assert not reg.exists();write(reg,registry);copy(reg,A/reg.name)
    write(A/'root-review-corrections.json',dict(created_utc=now,corrections=corrections))
    write(A/'closed-build-pointer.json',dict(commit=ccommit,source=d['source_head'],report=str(CP/'REPORT.md'),receipt=ident(run/'record.json'),scope=d['scope']))
    (A/'REPORT.md').write_text("""# Recurring research round51 and measured bottleneck limits

Four returned Luna reports R200-R203 were read in full and preserved unchanged;
the catalog contains310 reviewed reports. Root concurrently completed the
fresh caller-census full engine CPU build. Source/host and build proofs
remain separate from actual model/GPU qualification.

R200 identifies aggregate-identity and async-success limits. R201 finds
no direct replacement for native IQ4NL Down. R202's direct packed-B hypothesis
is held: root corrects payload bytes, native low/high nibble positions
and existing512-row tile reuse before implementation. R203 separates
current real-model evidence from old logged phase times and synthetic helpers.
Its global decode conclusion is corrected: fresh first decode has a
measured regression, while restored-decode direction is unresolved.

Measured priorities are the prefill copy/staging/MoE schedule and the
decode CPU expert dispatch interval. Prefill native-copy changes coincide with
about20.36% faster later fresh32K prefill; this is not measured PCIe saturation
or a transfer-only fraction. Firstdecode CPU dispatch averages123.78ms/window,
with165.38ms total and2.76ms GPU-reach hostwait. CPU arithmetic/RAM/synchronization
remain unseparated. Timers overlap; their sums are not utilization or exclusive
wall shares. Old marker profiles do not establish current kernel percentages.

Next root work prioritizes attribution on actual>=32768 batched positions,
oneMKL command linkage and first/restored decode inner-pool separation.
Candidates remain unadopted; full physical262144 lifecycle does not transfer
from the qualified baseline. Source-only proposals and the synthetic
~1.237% rotating-address helper improvement are not model speed gains.
""")
    copy(Path(__file__),A/Path(__file__).name)
    write(A/'archive-file-identities.json',{str(p.relative_to(A)):ident(p) for p in sorted(A.rglob('*')) if p.is_file()})
    wcommit=commit(W,str(A.relative_to(W)),'docs(sycl): reconcile bottleneck evidence and repeated research')
    print(json.dumps(dict(build_proof_commit=ccommit,storage_commit=wcommit,registry=str(reg),registry_identity=ident(reg),reports=310)))
