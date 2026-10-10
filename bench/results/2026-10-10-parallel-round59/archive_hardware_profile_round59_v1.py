"""Root-owned compact profile and returned research archive, under the shared lock."""
from pathlib import Path
import datetime, fcntl, hashlib, json, shutil, subprocess
B = Path(__file__).parent
R = B / 'research-20261009'
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
C = W / 'bench/results/2026-10-10-hardware-concurrency'
A = W / 'bench/results/2026-10-10-parallel-round59'
P = B / 'hardware-concurrency-v1-profile'
def ident(p):
    with p.open('rb') as f:
        return dict(bytes=p.stat().st_size, sha256=hashlib.file_digest(f,'sha256').hexdigest())
def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
def copy(source, target):
    assert not target.exists(), target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert ident(source) == ident(target)
with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip() == '950776db6528bac943a867ab18c3a630665c2d54'
    assert subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip() == '?? bench/results/2026-10-10-hardware-concurrency/analyze_profile_v1.py'
    assert not A.exists()
    prev=R/'report-registry-v81.json'
    assert ident(prev)['sha256']=='4ad7cd5eb6860213acb97cdab6e338680ab067144d275e9fbd2401d9f5035f82'
    d=json.loads(prev.read_text()); assert len(d['reports'])==329
    entries=[
      ('round223-prefill-service-ledger-independent-admission.txt','research_bottleneck_evidence_audit_v203','Actual production returned-event ledger admission, boundedness and producer join','f155470482323725b9858c60f4e3ab10081e6d741f86ae144717259b596dbd6b'),
      ('round224-native-gu-emitted-code-and-zen3-capacity.txt','research_native_iq4nl_esimd_k640_v202','Pinned emitted AVX2 IQ2_S NT2–4 code and Zen3 capacity boundaries',None)]
    new=[]
    for name,agent,scope,digest in entries:
        source=R/name; identity=ident(source)
        if digest: assert identity['sha256']==digest
        assert not any(x.get('path',x.get('report'))==str(source) for x in d['reports'])
        copy(source,A/'research'/name)
        entry=dict(path=str(source),**identity,agent='/root/'+agent,model='gpt-6-luna',scope=scope,status='completed-read-only',root_full_report_reviewed=True)
        d['reports'].append(entry);new.append(entry)
    correction={
      'R223_unprofiled_admission':'Root read the actual complete ServiceLedger header. admit() invalidates an unprofiled lane before record() retention. Subsequent records increment dropped and keep only latest for completion; no pending timestamp queries exist in that first-admission case. The report claim that this case later queries retained events is unsupported.',
      'R223_query_failure_exit':'Timestamp query catches increment query_failures/invalidate and continue; they do not call fatal/_Exit. Completion/async/submission uncertainty can be fatal, a separate mechanism. No admission-related runtime defect is established by R223. Explicit fallback/reject remains an optional design suggestion, not a proven necessary fix.',
      'R223_coverage_cap':'Cumulative 512-token chunk ×48layer ×2role geometry can exceed16384 bucket cap at262144 positions. Telemetry is bounded and invalidated/suppressed; this is not an engine memory bug or full-context qualification.4096 chunks need6144 coverage buckets.',
      'R224_scope':'Direct emitted NT4 YMM stack spills are a credible cost to measure; no measured spill dominance or NT3/NT4 speed ordering. Existing calibration accepts NT1/2 and two-token buffers only, so it must be extended and correctness-qualified before NT3/4 timing.'}
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    d.setdefault('root_decision_history',[]).append(dict(at_utc=now,prior_current_root_decisions=d.get('current_root_decisions')))
    d.update(registry_version=82,created_utc=now,research_completed=len(d['reports']),new_reports=new,
       previous_committed_registry=dict(path=str(prev),**ident(prev),storage_commit='950776db6528bac943a867ab18c3a630665c2d54'),
       scope='Hardware empirical capacity and concurrency qualification; native grouped CPU and production internal-event gaps remain',
       live_agent_snapshot=dict(time_utc=now,method='Actual collaboration.list_agents after full return/review',agents=[
          dict(agent='/root/research_bottleneck_evidence_audit_v203',model='gpt-6-luna',round=223,status='completed'),
          dict(agent='/root/research_native_iq4nl_esimd_k640_v202',model='gpt-6-luna',round=224,status='completed'),
          dict(agent='/root/implement_prefill_phase_timer_validity',model='gpt-6.1-sol',status='completed-source-only')]),
       current_root_decisions=dict(hardware_capacity='3 fresh CPU processes and3 fresh GPU processes;7 samples/cell. Empirical rates, not absolute maxima.',
          copy_GEMM_concurrency='3 clean fresh-process repeats plus separate closed PTI qualification.144 target copies and3168 target GEMMs have zero device interval intersection in this probe with safe settings unchanged.',
          CPU_native='NT1 fallback is not grouped NT2–4 capacity; NT4 emitted spill warning requires hot/streaming actual-payload comparison.',
          production_ledger='495369cf committed CPU-contract v2 passed26cases. SYCL build/device/internal-kernel coverage and actual32K/full262K remain unqualified.',
          original_phase_controller='v2 remainsFAILED; separate source provenance audit admits phase-only evidence, not clean speed.',
          full_physical_context='No candidate inherits old baseline full262144 qualification.',adopted=False))
    d['root_review_corrections_round59']=correction
    registry=R/'report-registry-v82.json'; assert not registry.exists();write(registry,d);copy(registry,A/registry.name)
    write(A/'root-review.json',dict(created_utc=now,new_reports=new,completed_reports=331,registry=ident(registry),corrections=correction))
    copy(Path(__file__),A/Path(__file__).name)
    r=json.loads((P/'record.json').read_text()); s=json.loads((P/'profile-analysis.json').read_text())
    assert r['passed'] and r['complete'] and not r['active'] and s['passed']
    assert s['host_append_joins']==s['GPU_operations']==3330
    assert s['device_copy_GEMM_intersection_ns']==0 and s['closest_cross_role_gap_ns']>=1000
    for name in ['record.json','profile-analysis.json','compact-operation-intervals.json','benchmark.stdout','benchmark.stderr','kernel-cursor.stdout','kernel-cursor.stderr','kernel-interval.stdout','kernel-interval.stderr']:
        copy(P/name,C/'profile'/name)
    summary_file=P/'device-summary.1107780.txt'
    write(C/'profile/device-summary.json',dict(original_path=str(summary_file),**ident(summary_file),verbatim_text=summary_file.read_text()))
    pti=B/'pti-gpu-profiler-source'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=pti,text=True).strip()=='6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=pti,text=True)
    excerpts=[]
    for rel,lo,hi in [('tools/unitrace/src/chromelogger.h',269,288),('tools/unitrace/src/chromelogger.h',1024,1055),('tools/unitrace/src/levelzero/ze_collector.h',1216,1235)]:
        file=pti/rel;lines=file.read_text().splitlines();excerpts.append(dict(source=rel,source_identity=ident(file),first_line=lo,last_line=hi,lines=lines[lo-1:hi]))
    write(C/'profile/pti-source-meaning.json',dict(commit='6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033',excerpts=excerpts,
       conclusion='PTI labels queue group ordinal,index using tracked group properties; both flow directions are anchored at command_start. Device duration comes from command_end-command_start. Engine ordinal/index is not physical utilization.'))
    trace=next(P.glob('hardware-concur.*.json'))
    write(C/'profile/retention.json',dict(created_utc=now,raw_trace=dict(path=str(trace),**ident(trace)),budget_bytes=64<<20,
       owner='root hardware-concurrency-v1-profile',consumer='Next read-only independent audit of complete trace coverage, flow meaning and recorded engine mapping',
       review_point='After the next independent PTI capacity-evidence report is read and committed, retire raw trace if compact3330-operation evidence answers its question.',
       compact_replacement='compact-operation-intervals.json preserves every GPU interval and3330 joined host append calls; profile-analysis.json preserves exact counts, unions, intersection and limitations. Full16615+API timeline not committed.'))
    (C/'README.md').write_text('''# Safe runtime hardware concurrency controls

This probe uses two in-order queues on the same B570 device/context, with queue profiling disabled in the application. It copies independent next-slot host-USM buffers while running production-shaped FP16-input/FP32-output oneMKL GEMMs. Each copy-enabled batch moves four 8 MiB buffers. Small M80/M160 shapes run128 rotating-weight GEMMs; M8192 runs8. This is an independent capacity control, not Strata model or production ring performance.

## Closed clean repeats

A separate logged qualification and three fresh clean processes passed normal exit, output and fresh kernel-fault checks. All504 samples are preserved in the three process receipts and summary. Each cell has seven samples/process, with rotating mode order; seven rounds do not give exactly balanced position counts.

The following are medians of the three process medians, in milliseconds per batch.

| Role / M | Copy only | GEMM only | Forced serial | Concurrent submission |
| --- | ---: | ---: | ---: | ---: |
| GU /80 |5.263|5.146|10.366|10.323|
| GU /160 |5.250|5.201|10.438|10.376|
| GU /8192 |5.298|8.184|13.454|13.356|
| Down /80 |5.258|3.303|8.489|7.515|
| Down /160 |5.259|3.392|8.781|8.223|
| Down /8192 |5.270|5.176|10.439|10.352|

GU has little wall benefit. Small Down shapes have variable wall benefit, which alone does not prove device overlap. Final validation checks both copy buffers and the final surviving product; overwritten products are not checked. No optimization is adopted.

## Separate PTI qualification

The closed instrumented process used pinned unitrace with kernel and host-call logging. Source, binary, runtime, safe flags, owner lifetime, all output hashes and correctness/fault outcomes are in `profile/record.json`. Instrumented timings are excluded from the clean table.

All3330 GPU operations were reconciled to unique host append calls using flow IDs and timestamps. Source counts account for144 target8MiB copies,3168 GEMM kernels, and18 setup/validation operations. The full compact interval table preserves all3330 operations. Target copy/GEMM interval intersection is0ns; their closest cross-role gap is5084ns, above the conservative1us serialization guard. Both tracked device lanes are named `L0 Compute Engine<0,0>` by PTI. Pinned profiler source shows that this label uses queue group ordinal/index and recorded group properties, not a physical utilization counter. Both device flow directions anchor command_start, so D2H flow is not a completion timestamp.

This qualifies lack of device overlap in this specific independent probe with `UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1` and `EnableDirectSubmission=0`. It does not establish production engine coverage, identify the precise serialization cause, or justify changing either safety setting. Production oneMKL returned-event/internal-kernel coverage remains a separate qualification.

## Reproduction and retention

The controller stages are build, qualify, r1, r2, r3, profile; each stage writes a unique finite owned directory under the shared measurement lock. Historical receipts retain their original controller/source commits. The offline analyzer reproduces the full event join and intersection from the closed raw profile. The raw5.16MB timeline has a named next-audit consumer and review point in `profile/retention.json`; compact results are committed in Git.
''')
    (C/'HARDWARE_LIMITS.md').write_text('''# Measured hardware capacity and workload limits

Measured2026-10-10 on Ryzen5 5600X(6physical cores),128GB RAM and Arc B570. CPU and GPU capacity controls each used three fresh processes with seven samples/cell. Values below are medians of process medians; all individual samples and process ranges are in [the validated summary](../2026-10-10-parallel-round54/hardware/validated-three-process-summary.json). These are attained effective rates for the stated operations, not absolute hardware maxima. CPU power/governor policy and the previously qualified GPU safety settings were left unchanged.

| Component / operation | Measured rate | Scope |
| --- | ---: | --- |
| RAM read |36.87 GB/s|6physical cores;512MiB arrays exceed LLC|
| RAM non-temporal copy |37.58 GB/s|Logical read+write bytes|
| RAM cached copy |22.95 GB/s|Logical read+write bytes|
| RAM→GPU host USM |6.447 GB/s|256MiB, one-way payload, host-inclusive completion|
| GPU→RAM host USM |5.643 GB/s|256MiB, one-way payload|
| RAM→GPU pageable host |4.605 GB/s|256MiB; different memory path|
| VRAM copy |330.07 GB/s|256MiB, logical read+write bytes|
| GU GEMM /M8192 |52.69 TFLOP/s|Hot FP16 inputs, FP32 output;N1280,K2560|
| Down GEMM /M8192 |41.61 TFLOP/s|Hot FP16 inputs, FP32 output;N2560,K640|
| GU GEMM /M80 rotating8weights |13.16 TFLOP/s|Actual production-shaped kernel geometry, synthetic inputs|
| Down GEMM /M80 rotating8weights |10.08 TFLOP/s|Process medians10.04–14.21; variation retained|
| GU GEMM /M160 rotating8weights |25.78 TFLOP/s|Synthetic row grouping, not measured production row histogram|
| Down GEMM /M160 rotating8weights |20.03 TFLOP/s|Process medians19.19–22.55|

Different logical traffic conventions are explicit: RAM/VRAM copy counts both read and write; PCIe rates count transferred payload once. These are not directly comparable utilization ratios. Eight rotating weights do not prove cold caches. Large-shape rates cannot be applied to each small expert GEMM.

## What changes the architectural estimate

At32768 tokens,48layers,topK10,H2560 andFF640, routed GU/Down dense arithmetic is154.619TFLOP(4.718592GFLOP/token), excluding attention, shared experts, dequantization and other work. A1000token/s target therefore requires4.719TFLOP/s for these products, but attained expert shape and dependencies matter much more than the large GEMM peak.

An older matched32K route reported about190.2GB of logical expert copies. Holding that volume fixed, the attained host-USM rate6.447GB/s gives about29.5seconds of transfer. The1000token/s total budget is32.768seconds. This is a conditional estimate from a prior route and a separate capacity control, not the latest model's measured DMA service or a mathematical physical minimum. The current route's bytes, shapes and reuse must be reconciled before predicting speed.

The separate two-queue control found essentially serial GU wall time, and its PTI trace found zero intersection of144target copies and3168GEMM kernels with the safe runtime settings unchanged. Reordering submissions cannot be assumed to hide transfer at those settings. A scheduler change needs actual engine-overlap evidence; otherwise reducing transferred bytes, increasing useful work per weight load, or changing the data representation is required to recover that budget. Layer-major prefill already retains each loaded layer across token chunks, so an extra reuse claim must account for existing reuse.

RAM bandwidth alone does not characterize the native quantized CPU decoder. NT1 dispatches a different fallback from groupedNT2–4. The independent emitted-code audit found NT4 vector accumulator stack spills, but did not measure their cost. Grouped actual-payload hot/streaming service and worker-tail data remain to be qualified; no CPU instruction ceiling or decoder impossibility is claimed. PLE's foreground gather/wait markers also do not measure storage service: an actual row/page trace and matching read-only storage controls remain needed. Those missing capacities are explicit open work.

The existingDown phase-marker duration includes host waits and must not be equated to exclusive GPU execution. A default-off production returned-event ledger has passed host contracts, but still needs SYCL build, small device qualification, profiler coverage and a matched32K trace. Full physical262144-position validation is a separate candidate gate; a configured context length is not that validation.
''')
    write(A/'archive-file-identities.json',{str(p.relative_to(A)):ident(p) for p in sorted(A.rglob('*')) if p.is_file()})
    for args in [('diff','--check'),('add','bench/results/2026-10-10-parallel-round59','bench/results/2026-10-10-hardware-concurrency'),('diff','--cached','--check'),('commit','-m','docs(sycl): qualify hardware concurrency and preserve capacity limits'),('push',),('rev-parse','HEAD')]:
        subprocess.run(['git',*args],cwd=W,check=True)
    print(json.dumps(dict(registry=str(registry),**ident(registry),completed=331)))
