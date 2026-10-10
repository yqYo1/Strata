import datetime, fcntl, hashlib, json, math, os, shutil, statistics, subprocess
from pathlib import Path

B=Path(__file__).parent
R=B/'research-20261009'
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
SC=W/'bench/results/2026-10-10-storage-capacity'
A=W/'bench/results/2026-10-10-parallel-round64'
C=W/'bench/results/2026-10-10-hardware-concurrency'
HEAD='ded1e39ca579901e9264b70caa2f22909e308230'
def ident(p):
    p=Path(p);return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,d):
    Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def copy(p,q):
    q=Path(q);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);assert ident(p)==ident(q)
def no_owned_process(c):
    assert c['normal_exit'] and c['exit_code']==0 and c['session_empty']
    assert c['direct_child_reaped'] and not c['survivors'] and not c['errors']
    for o in c['owners']:
        p=Path('/proc')/str(o['pid'])/'stat'
        if p.exists():
            s=p.read_text();fields=s[s.rfind(')')+2:].split()
            assert int(fields[19])!=o['start_ticks'],('owned process still exists',o)

with open(B/'owned-v0141-measurement.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==HEAD
    status=subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip()
    # A previous archive-only attempt stopped at a mistaken controller-name assertion.
    # Allow only these generated, still byte-verified storage copies; no run status changed.
    assert not status or all(line.startswith('?? bench/results/2026-10-10-storage-capacity/') for line in status.splitlines())
    assert not A.exists() or not any(A.iterdir());A.mkdir(exist_ok=True)
    planp=B/'storage-capacity-v2-matrix-plan-v1.json';plan=json.loads(planp.read_text())
    assert plan['complete'] and plan['passed'] and not plan['active']
    assert len(plan['closed'])==len(plan['plan'])==15
    samples=[];mode_fnv={};pins={};boot=None
    for arm,entry in zip(plan['plan'],plan['closed']):
        assert all(arm[k]==entry[k] for k in ('mode','workers','repeat'))
        p=Path(entry['path']);assert ident(p)['sha256']==entry['sha256']
        r=json.loads(p.read_text());assert r['complete'] and r['passed'] and not r['active']
        assert r['stage']=='measure' and not r['gpu_work_submitted'] and not r['model_inference'] and not r['adopted']
        assert all(r[k]==arm[k] for k in ('mode','workers','repeat'))
        assert r['seed']==2026101001
        boot=boot or r['boot_id'];assert r['boot_id']==boot
        for k in ('source','binary'):
            assert ident(r[k]['path'])=={x:r[k][x] for x in ('bytes','sha256')}
            pins.setdefault(k,r[k]);assert pins[k]==r[k]
        assert len(r['commands'])==1;cmd=r['commands'][0];no_owned_process(cmd)
        assert len(r['output_rows'])==1;o=r['output_rows'][0]
        stdout=p.parent/'storage-cell.stdout';stderr=p.parent/'storage-cell.stderr'
        for log in (stdout,stderr):assert ident(log)==cmd['logs'][log.name]
        assert json.loads(stdout.read_text())==o and stderr.stat().st_size==0
        assert o['success'] and o['timing_claim'] and not o['qualification_only'] and o['stat_unchanged']
        assert o['requests']==o['completed_requests']==o['pread_syscalls']==sum(x['completed'] for x in o['workers_detail'])
        assert o['eintr_retries']==0 and o['workers']==arm['workers'] and o['final_buffer_checks']==arm['workers']
        assert o['logical_bytes']==o['requests']*o['block_bytes']
        assert math.isclose(o['logical_GBps'],o['logical_bytes']/o['wall_seconds_observed']/1e9,rel_tol=1e-12)
        assert math.isclose(o['IOPS'],o['requests']/o['wall_seconds_observed'],rel_tol=1e-12)
        assert all(x['assignments']==x['completed'] and x['assignments']>0 for x in o['workers_detail'])
        mode_fnv.setdefault(arm['mode'],o['planned_job_order_fnv']);assert mode_fnv[arm['mode']]==o['planned_job_order_fnv']
        if arm['mode']=='random':assert o['requests']==65536 and o['block_bytes']==4096
        else:assert o['requests']==27464 and o['block_bytes']==1048576 and o['sequential_begin']==1048576
        z=r['whole_process_ZFS_deltas'];ratio=z['direct_read_bytes']/o['logical_bytes']
        assert ratio==r['whole_process_direct_DMU_to_timed_payload_ratio']
        assert z['direct_read_count']>0 and z['direct_read_bytes']/z['direct_read_count']==131072
        dst=SC/'measure'/f"{arm['mode']}-w{arm['workers']}-r{arm['repeat']}-record.json";copy(p,dst)
        keep=('logical_bytes','requests','wall_seconds_observed','logical_GBps','IOPS','latency_median_ns','latency_p95_ns','latency_p99_ns','actual_peak_inflight_requests','final_buffer_checked_bytes','planned_job_order_fnv')
        samples.append(dict(**arm,**{k:o[k] for k in keep},receipt=str(dst.relative_to(SC)),receipt_identity=ident(p),whole_process_direct_DMU_to_timed_payload_ratio=ratio,whole_process_direct_bytes=z['direct_read_bytes'],whole_process_direct_count=z['direct_read_count'],whole_process_ARC_read_bytes=z['arc_read_bytes'],whole_process_leaf_read_bytes=r['whole_process_leaf_read_bytes'],whole_process_leaf_to_timed_payload_ratio=r['whole_process_leaf_to_timed_payload_ratio']))
    assert len({(x['mode'],x['workers'],x['repeat']) for x in samples})==15
    cells=[]
    for mode,workers in [('sequential',1),('sequential',16),('random',1),('random',16),('random',64)]:
        ss=[x for x in samples if x['mode']==mode and x['workers']==workers];assert len(ss)==3
        cell=dict(mode=mode,workers=workers,fresh_processes=3)
        for k in ('logical_GBps','IOPS','wall_seconds_observed','latency_median_ns','latency_p95_ns','latency_p99_ns','whole_process_direct_DMU_to_timed_payload_ratio','whole_process_leaf_to_timed_payload_ratio'):
            vv=[x[k] for x in ss];cell[k]=dict(median=statistics.median(vv),min=min(vv),max=max(vv),individual_process_values=vv)
        cells.append(cell)
    summary=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),closed_pass=True,processes=15,source=pins['source'],binary=pins['binary'],boot_id=boot,scope='Attained same-file application payload rates, not absolute SSD or NAND capacity, physical QD, PLE service or model speed. Three fresh processes per cell; no cache reset/property change; samples not assumed independent of filesystem history. Counter snapshots are whole-process pool/block-global, not timer/file attribution.',counter_event_mean_bytes=131072,cells=cells,samples=samples,planned_order_hashes=mode_fnv)
    write(SC/'three-process-summary.json',summary)
    for name in ('storage-capacity-v2-matrix-plan-v1.json','run_storage_capacity_matrix_v1.py'):
        copy(B/name,SC/name)
    assert ident(SC/'run_storage_capacity_v2.py')['sha256']==plan['controller_sha256']
    text=(SC/'README.md').read_text()
    text+='\n## Closed repeated capacities\n\nAll 15 fresh processes closed PASS with unchanged source, binary, shard, boot and properties. Repeat 2 reversed the cell order; all random cells use the same 65,536 replacement offsets and all sequential cells the same 27,464 planned 1 MiB offsets. No cache state was reset. Values are medians and ranges of three whole-process application rates. Full original receipts are preserved byte-identically in `measure/`; structured summary retains each measurement and latency quantile.\n\n| Read pattern | Workers | Payload GB/s median [range] | Logical IOPS median [range] |\n| --- | ---: | ---: | ---: |\n'
    for c in cells:
        g=c['logical_GBps'];i=c['IOPS'];text+=f"|{c['mode']} / {'1 MiB' if c['mode']=='sequential' else '4 KiB'}|{c['workers']}|{g['median']:.6f} [{g['min']:.6f}, {g['max']:.6f}]|{i['median']:.1f} [{i['min']:.1f}, {i['max']:.1f}]|\n"
    text+='\nThe 64-worker first repeat is retained, including its lower 0.037055 GB/s and longer tail; it is not discarded as an outlier. Increasing 16 to 64 workers does not establish a reliable gain. Worker high-water values are worker `pread` regions, not SSD queue depth.\n\nEvery cell reports exactly 131,072 bytes per counted direct-DMU read. Whole-process direct-DMU bytes / timed logical payload is 31.9956–31.9971 for 4 KiB random reads, and 0.999964–0.999977 for aligned sequential reads. These are pool-global counter observations over the whole supervised process; they are not timed per-file physical read amplification or NAND traffic. ARC and leaf-byte deltas remain separately recorded and unattributed. Pinned upstream 2.4.1 source suppresses predicted DIO data prefetch, but can still issue ARC indirect-metadata reads; it does not attribute this run\'s ARC bytes.\n\nThis supports testing record-coalesced PLE reads as a hypothesis. Sorting page jobs alone is not proof that each filesystem record is read once. Actual PLE row-cache hit rate, duplicate records, page/straddler shapes, reader return bytes, critical-path wait and matched 32K model correctness/performance must be measured before adopting a change. No source, runtime or production tuning was adopted by these controls.\n'
    (SC/'README.md').write_text(text)
    old=(C/'HARDWARE_LIMITS.md').read_text()
    old=old.replace('CPU and GPU capacity controls each used three fresh processes with seven samples/cell.', 'RAM and GPU controls and register-FMA controls each used three fresh processes with seven samples/cell. Native quantized CPU controls used 24 fresh processes (four group sizes, hot/streaming, three repeats), with 25 samples per arm/cohort. Same-file storage controls used 15 fresh processes (five patterns, three repeats).')
    old=old.replace('| RAM read |', '| CPU register FP32 FMA |0.818 TFLOP/s|6 physical cores; process medians 0.793–0.819; 12 independent AVX2 accumulators|\n| RAM read |')
    old=old.replace('Different logical traffic conventions', '| SSD sequential / 1 MiB |2.266 GB/s|16 workers; application payload; range 2.137–2.273|\n| SSD random / 4 KiB |0.05099 GB/s; 12,450 IOPS|16 workers; same-file ZFS path; range 0.05055–0.05127|\n| SSD random / 4 KiB |0.05307 GB/s; 12,958 IOPS|64 workers; range 0.03706–0.05313; not SSD queue depth|\n\nCPU operation details and all samples are in [CPU capacity](../2026-10-10-cpu-capacity/README.md); storage patterns, three repeats and whole-process counter boundaries are in [same-file storage capacity](../2026-10-10-storage-capacity/README.md).\n\nDifferent logical traffic conventions')
    begin=old.index('RAM bandwidth alone does not characterize')
    end=old.index('The existingDown phase-marker duration',begin)
    old=old[:begin]+'''Native quantized CPU capacity was measured with the actual selected IQ2_S/IQ4_NL payloads. In the streaming direct complete-chain fixture, NT1 costs 286.47 microseconds/expert-token versus 148.36 at NT4; the fixed five-worker-plus-host pool costs 90.98 versus 38.74. Grouping already improves attained per-token service substantially. This is not whole-model decoder latency or a physical-DRAM/cycle census. Hot/streaming use different expert subsets and job counts, and native quantized dense-equivalent FLOPs cannot be divided by register FP32 FMA capacity to infer CPU utilization. The NT4 emitted stack spills remain a source fact with unmeasured cost; NT4 is still the fastest tested per-token group. No decoder impossibility follows from logical packed bytes or RAM bandwidth alone.

PLE's foreground gather/wait markers do not measure storage service. The same-file control now shows about 2.27 GB/s for 1 MiB requests but only 0.051 GB/s for 4 KiB random requests at 16 workers; 64 workers offers a small median difference with a much wider range. All random cells show approximately 32 direct-DMU bytes per application byte, versus approximately one for aligned sequential requests. These pool-global whole-process counters support testing filesystem-record coalescing; they do not attribute NAND bytes or prove the actual PLE route's bottleneck. Reordering tiny jobs and increasing workers cannot be assumed to recover the contiguous-read rate. Actual PLE reader statistics and duplicate-record census remain necessary.

''' + old[end:]
    old=old.replace('has passed host contracts, but still needs SYCL build, small device qualification, profiler coverage and a matched32K trace.', 'has passed host contracts, isolated SYCL build and small device qualification, but still needs profiler coverage and a matched32K trace.')
    (C/'HARDWARE_LIMITS.md').write_text(old)
    now=datetime.datetime.now(datetime.timezone.utc).isoformat();prev=R/'report-registry-v86.json'
    assert ident(prev)['sha256']=='e4c2f35abcf01452a087989939f28e6bb99ce582bec2932a60d1ea610be1b489'
    d=json.loads(prev.read_text());assert len(d['reports'])==339;new=[]
    for name,sha,agent,scope in [
        ('round233-storage-capacity-source-and-admission-audit.txt','8bf3870ca9f304129330e29e48900445e778e2aa0faaafefad98cf4000bc9473','/root/research_bottleneck_evidence_audit_v203','Independent storage source/admission review; first full sample only'),
        ('round234-zfs-direct-read-extra-arc-prefetch-discriminator.txt','932dc9cd9d7b8a32e14fc5b71337a7056c11574d3de4434513069c3b8bb8dab8','/root/research_native_iq4nl_esimd_k640_v202','Pinned ZFS DIO data/metadata ARC classification and read-only discriminator')]:
        p=R/name;assert ident(p)['sha256']==sha;copy(p,A/name)
        e=dict(path=str(p),**ident(p),agent=agent,model='gpt-6-luna',scope=scope,status='completed-read-only',root_full_report_reviewed=True);new.append(e);d['reports'].append(e)
    corrections=dict(R233='Accepted source and admission limits. Root subsequently completed all 15 repeated cells; report reviewed only the first full W1 sample. Sequential 1MiB alignment avoids crossing an extra possible 128KiB record per request, not merely a 4KiB boundary. Emergency-write compiler warnings are preserved and do not change a normal-path result.',R234='Accepted pinned upstream distinction: DIO predictive data prefetch suppressed; indirect metadata demand/prefetch can use ARC. Pool-wide 4.30GB ARC delta remains unattributed. No claim that metadata or this PID caused all ARC traffic; no property, history, privilege or cache change.')
    d.update(registry_version=87,created_utc=now,research_completed=341,new_reports=new,previous_committed_registry=dict(path=str(prev),**ident(prev),storage_commit=HEAD),scope='Completed hardware capacity map including 15 same-file storage processes; actual 32K production critical path remains pending',live_agent_snapshot=dict(time_utc=now,agents=[dict(agent=e['agent'],model='gpt-6-luna',status='completed') for e in new]+[dict(agent='/root/implement_prefill_phase_timer_validity',model='gpt-6.1-sol',status='completed-source-only')]),current_root_decisions=dict(hardware='RAM/PCIe/VRAM/shape GPU/native CPU/registerFMA/storage repeated controls closed PASS. Attained operation-specific rates, not absolute maxima.',storage='15 full cells; random whole-process directDMU/payload ~32x, sequential ~1x. Actual PLE service and per-file/media attribution unmeasured.',production='No current model speed improvement, internal oneMKL whole-span or full262144-position candidate validation claimed.',adopted=False))
    d['root_review_corrections_round64']=corrections;reg=R/'report-registry-v87.json';assert not reg.exists();write(reg,d);copy(reg,A/reg.name)
    write(A/'root-review.json',dict(created_utc=now,original_reports=new,corrections=corrections,registry_identity=ident(reg),storage_summary_identity=ident(SC/'three-process-summary.json'),remaining=['actual matched32K PLE/PCIe/GEMM shape and critical-path ledger','complete internal oneMKL event span','full262144-position candidate validation'],adopted=False))
    (A/'README.md').write_text('R233 independently admits the bounded storage source and five qualifiers; R234 distinguishes DIO data from potential ARC metadata traffic in pinned upstream ZFS. Root completed all 15 repeated controls and committed individual original receipts and capacity limits. Whole-process global bytes remain distinct from timed payload and actual PLE/model service. No production optimization adopted.\n')
    copy(__file__,SC/Path(__file__).name)
    for args in [('diff','--check'),('add','bench/results/2026-10-10-storage-capacity','bench/results/2026-10-10-hardware-concurrency/HARDWARE_LIMITS.md','bench/results/2026-10-10-parallel-round64'),('diff','--cached','--check'),('commit','-m','bench: complete repeated storage capacity and hardware limit map'),('push',)]:
        subprocess.run(['git',*args],cwd=W,check=True)
    print(json.dumps(dict(closed_pass=True,registry=str(reg),registry_identity=ident(reg),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip(),capacities=[dict(mode=c['mode'],workers=c['workers'],GBps=c['logical_GBps']) for c in cells])))
