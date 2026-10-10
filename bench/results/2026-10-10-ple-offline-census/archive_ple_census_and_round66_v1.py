from pathlib import Path
import datetime, fcntl, hashlib, json, shutil, subprocess
B=Path(__file__).parent
R=B/'research-20261009'
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
A=W/'bench/results/2026-10-10-parallel-round66'
C=W/'bench/results/2026-10-10-ple-offline-census'
def ident(p):
    p=Path(p)
    return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def copy(p,q):shutil.copyfile(p,q);assert ident(p)==ident(q)
def closed(r):
    assert r['passed'] and r['complete'] and not r['active']
    for c in r['commands']:
        assert c['normal_exit'] and c['session_empty'] and c['direct_child_reaped'] and c['observation_complete']
        assert not c['errors'] and not c['survivors']
        for owner in c['owners']:
            p=Path('/proc')/str(owner['pid'])/'stat'
            if p.exists():assert int(p.read_text().rsplit(')',1)[1].split()[19])!=owner['start_ticks']
with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()=='7935f73c5b5b497e2c74937254457ef0d7322c38'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip()
    assert not A.exists() and not C.exists();A.mkdir();C.mkdir()
    source=R/'implementation-ple-offline-census-v1.txt';assert ident(source)['sha256']=='9be7725c9817ef76d5e65e81c228fd5d50895dc9ca3bc62d10e86f895f5a1f0b';copy(source,C/source.name)
    for name in ('ple_offline_census_v1.py','ple_ngram_oracle_v1.cpp','ple_offline_census_v2.py','ple_ngram_oracle_v2.cpp','prepare_ple_census_v2.py','ple-offline-census-v2-source-preparation.json','ple_hash_golden_root_v1.cpp','qualify_ple_census_v2.py','run_ple_census_v2.py'):
        copy(B/name,C/name)
    for dirname,name in [('ple-offline-census-source-object-build-v1','actual-ngram-source-object-build-record.json'),('ple-hash-golden-root-v1','cpp-external-golden-record.json'),('ple-offline-census-v2-owned','owned-census-record.json'),('ple-census-mutation-qualification-v1','matched-history-mutation-record.json')]:
        p=B/dirname/'record.json';r=json.loads(p.read_text());closed(r);copy(p,C/name)
    copy(B/'ple-offline-census-fixtures-v1/record.json',C/'derived-fixture-admission-record.json')
    owned=json.loads((C/'owned-census-record.json').read_text());assert len(owned['cases'])==2
    for case in owned['cases']:
        p=Path(case['census_path']);assert ident(p)==case['census_identity'];copy(p,C/(case['label']+'-census.json'))
        assert ident(Path(case['oracle_rows']['path']))=={k:case['oracle_rows'][k] for k in ('bytes','sha256')}
    actual=json.loads((C/'actual32K-census.json').read_text());full=json.loads((C/'context262144-census.json').read_text())
    assert actual['all_oracle_rows_compared']==524304 and full['all_oracle_rows_compared']==4194304
    assert actual['pp_tokens']==32768 and actual['chunk_count']==4 and full['pp_tokens']==262143 and full['chunk_count']==32
    text='''# Actual-input offline PLE request geometry

This is a CPU-only source/fixture census, not a storage/GPU benchmark or inference correctness result. Original unchanged `ngram_rows` was compiled with function/data sections and linked into the bounded oracle without GPU dependencies. Its external-origin golden cases pass all 304 rows; independent Python also passes the same golden vectors. Every generated row then matches the independent Python equation: 524,304 rows for actual A (32,768 PP tokens + one separate tail), and 4,194,304 rows for the full-context-shaped prefix (262,143 PP + one separate tail). The source full fixture actually has 262,145 IDs; root v2 corrects input admission while keeping prefix/output at most 262,144. Original Sol v1 sources/handoff remain unchanged.

Cold controls reset the hypothetical row cache independently for every 8,192-token chunk. They match first-page dedup and 4K/8K extension, preserving overlapping page jobs. They do not model completion-order-dependent eight-way row-cache replacement. Records are assumed to be 128KiB at whole-file offsets, not a discovered physical ZFS extent map. Values below are sums of separate chunks, never global unique counts.

| Shape | PP chunks | Row requests | Page jobs | Baseline logical page bytes | Conditional all-record bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
'''
    for label,result in [('Actual A /32768PP',actual),('Full-context-shaped /262143PP',full)]:
        t=result['totals_sum_of_independent_chunks'];text+=f"|{label}|{result['chunk_count']}|{t['row_requests']:,}|{t['first_page_dedup_jobs']:,}|{t['baseline_job_page_bytes_sum_with_overlap']:,}|{t['conditional_record_coalesced_bytes_clipped_at_shard_eof']:,}|\n"
    text+='''
In actual A, 79.2–81.1% of assumed records have only one expanded requested 4K page. Unconditionally reading all touched full records increases returned logical bytes 25.74–26.25 times per chunk. This rules out assuming that larger reads are automatically faster. It does not prove slower media service: the separate ZFS control counts full 128K DMU blocks for small requests, and the cold page census is not actual cached ReaderStats. Density-gated grouping, row caching and actual reader service need a matched fresh-state 32K comparison.

Qualification covers negative/null/EOS/current-EOS/token0/uint64-wrap history, all rows across prefixes8191/8192/8193/8208, independently derived page/record straddlers and EOF geometry, parser rejection and output overwrite refusal. The first 40-check qualifier's first/last mutation cases omitted its custom history, so those labels alone did not establish the location of the corrupt row. Root's separate matched-history control first passes all176 clean rows and then rejects exactly token0/head0 and token10/head15 after first/last in-range mutations. Original receipts remain unchanged; the correction is explicit. Numerical source/census algorithms and positive full row-stream results were unchanged.

Peak observed owned session RSS is142,610,432B, below the1GiB bound. No model shard payload is read and noGPU command is submitted. Aligned baseline last-page geometry can extend832B past admittedEOF; this is reported as geometry, not an actual successful read. Conditional record plans are clipped atEOF and are not an implementation of direct I/O. Full262144-position inference and a production implementation remain separate gates.

All exact source/fixture/binary/object hashes, commands, ownership, normal closures, expected failure statuses and structured per-chunk histograms are retained. Large derived row streams are reproducible temporary comparator inputs, not resume checkpoints; once the independent review completes they can be retired with a hash/replacement manifest. Original token fixtures and current numerical/RESTORE inputs stay.
'''
    (C/'README.md').write_text(text)
    estimates=C/'capacity-workload-estimates.json'
    h=json.loads((W/'bench/results/2026-10-10-parallel-round54/hardware/validated-three-process-summary.json').read_text())
    rates={tuple(c['cell']):c['median_of_process_medians'] for c in h['cells']};bw=rates['transfer','H2D_host_USM',268435456]
    gu=32768*48*10*2*1280*2560/1e12;down=32768*48*10*2*2560*640/1e12;cells=[]
    for rows,mode in [(80,'rotate8'),(160,'rotate8'),(640,'rotate8'),(8192,'hot')]:
        g=gu/rates['gemm','GU',rows,mode];d=down/rates['gemm','Down',rows,mode]
        cells.append(dict(all_GEMM_calls_assumed_shape_M=rows,cache_fixture=mode,GU_seconds=g,Down_seconds=d,total_GEMM_seconds=g+d,historical_logical_copy_seconds=190.2/bw,serialized_sum_seconds=190.2/bw+g+d,no_other_cost_serial_target_copy_budget_GB=(32.768-g-d)*bw))
    write(estimates,dict(scope='Counterfactual arithmetic, NOT measurements of actual model service or physical lower bounds. Every call is hypothetically assigned one shape, reuse/exactpath held fixed, historical190.2GB logical payload and separate H2D capacity reused; actual route shape histogram/bytes/dependencies/otherwork absent.',target_tokens_per_second=1000,target32768_seconds=32.768,GU_TFLOPs=gu,Down_TFLOPs=down,H2D_payload_GBps=bw,historical_logical_copies_GB=190.2,scenarios=cells))
    limits=W/'bench/results/2026-10-10-hardware-concurrency/HARDWARE_LIMITS.md'
    s=limits.read_text();s=s.replace('Layer-major prefill already retains each loaded layer across token chunks, so an extra reuse claim must account for existing reuse.', 'The source contains a layer-major route that retains each loaded layer across token chunks; the historical32K comparison explicitly used STRATA_PREFILL_LAYER_MAJOR=0. An estimate must state which route is selected and account for its actual reuse; existing source is not evidence that this alternative is adopted or that its separate full-context numerical failure is resolved.')
    s+='\nThe [actual-input CPU-only PLE census](../2026-10-10-ple-offline-census/README.md) independently verifies every row and shows sparse assumed-record occupancy for the admitted coding prompt. All-record reads would expand returned logical bytes about26times in cache-empty per8K controls, so unconditional coalescing is not selected. Actual cached reader service and physical record geometry remain pending. [Counterfactual workload estimates](../2026-10-10-ple-offline-census/capacity-workload-estimates.json) explicitly hold historical bytes and hypothetical per-call shapes fixed; they are not actual model timings or feasibility bounds.\n'
    limits.write_text(s)
    now=datetime.datetime.now(datetime.timezone.utc).isoformat();prev=R/'report-registry-v88.json';assert ident(prev)['sha256']=='1e0f37e1c09f93b5bc861d7e7ad433ff133b46d376fc45adce3527130450feea';registry=json.loads(prev.read_text());assert len(registry['reports'])==343;new=[]
    for name,sha,agent,scope in [
        ('round237-finite-onemkl-returned-event-kernel-span-qualification-plan.txt','fc287f8ff1504959b8ffb34b801de23e0969a043958e405bfe139193ccf4ed83','/root/research_bottleneck_evidence_audit_v203','Finite returned-event/internal oneMKL kernel-span qualification recipe'),
        ('round238-actual-prompt-offline-ple-census-contract.txt','e448757608f3b4233356a7177f843a0b325ef800b71a40dc9639791867de79ab','/root/research_native_iq4nl_esimd_k640_v202','Exact admitted prompt/history/chunk and offline geometry contract')]:
        p=R/name;assert ident(p)['sha256']==sha;copy(p,A/name);entry=dict(path=str(p),**ident(p),agent=agent,model='gpt-6-luna',scope=scope,status='completed-read-only',root_full_report_reviewed=True);new.append(entry);registry['reports'].append(entry)
    corrections=dict(R237='Finite marker/flow/clock/full-operation recipe accepted; source-specific span remains untested. Existing small canary used UR/LevelZero diagnostics, not PTI, so report statement about preserved PTI side outputs/no kernel entries has no collector evidence and is not a negative result. <=64token checks are debug qualification only, never performance comparison. Offline PLE row census does not supply MoE expert-routing T; actual clean32K route histogram remains needed for target shape coverage. Ledger cap/coverage bounds must be derived from exact drain/reset sites and actual route, not blindly multiplied chunk totals.',R238='ExactA insertion/history/pagejobs contract accepted. Root file-byte count proves full fixture has262145 IDs, not report claimed262144; root v2 accepts at most262145 original input but bounds computedprefix/output262144. Full-context-shaped census has262143PP+1 separate tail and no inference/GPU claim. Sourcehash vectors and all comparedrows independentlyqualified, runtimecache/physicalgeometry unmeasured.',implementation='Source-only Sol files preserved; root corrected fullfixturecount/provenance before running. Additional matched-history corruption qualification corrects two original roottest labels without rewriting original status or positive all-row results.',layer_major='Original historical32K controller explicitly selects layer_major0; existing alternative source reuse must not be mistaken for currentadoption. Separate originalfullcontext rejection remains unresolved.')
    registry.update(registry_version=89,created_utc=now,research_completed=345,new_reports=new,previous_committed_registry=dict(path=str(prev),**ident(prev),storage_commit='7935f73c5b5b497e2c74937254457ef0d7322c38'),scope='Full hardware capacity map plus independently verified actual-input offline PLE geometry',live_agent_snapshot=dict(time_utc=now,agents=[dict(agent=e['agent'],model='gpt-6-luna',status='completed') for e in new]+[dict(agent='/root/implement_prefill_phase_timer_validity',model='gpt-6.1-sol',status='completed-source-only')]),current_root_decisions=dict(hardware='Repeated attainedcapacities complete. Counterfactualshape/bytes estimates preserve conditional scope.',PLE='Actual A perchunkrecords ~80%onepage; no unconditionalrecordreads. All4.194M full-shaped rows match; coldcachecontrolnotactualReaderStats/physicalIO.',production='No current32Kspeed improvement/internaloneMKLspan/full262144candidate inference qualified.',adopted=False))
    registry['root_review_corrections_round66']=corrections;registry['implementation_handoffs_current_batch']=[dict(path=str(source),**ident(source),agent='/root/implement_prefill_phase_timer_validity',model='gpt-6.1-sol',status='completed-source-only',root_full_report_reviewed=True,root_CPU_qualification='closed PASS with explicit fullfixture/input and mutation-history corrections')]
    reg=R/'report-registry-v89.json';assert not reg.exists();write(reg,registry);copy(reg,A/reg.name)
    write(A/'root-review.json',dict(created_utc=now,original_reports=new,corrections=corrections,registry_identity=ident(reg),CPU_census_receipt=ident(C/'owned-census-record.json'),matched_history_correction=ident(C/'matched-history-mutation-record.json'),adopted=False))
    (A/'README.md').write_text('R237 supplies a finite internal-oneMKL-span recipe; root corrects an unsupported assertion about absent PTI output. R238 supplies exact admitted prompt/history/page geometry; root corrects fullfixture cardinality. Root concurrently compiles original CPU hashing source, qualifies external golden vectors, and independently compares all actual32K/full-shaped rows with bounded per-chunk geometry. No unconditionalrecordcoalescing, production optimization or model performance/fullcontext qualification claimed.\n')
    copy(__file__,C/Path(__file__).name)
    for args in [('diff','--check'),('add',str(A.relative_to(W)),str(C.relative_to(W)),str(limits.relative_to(W))),('diff','--cached','--check'),('commit','-m','bench: qualify actual-input offline PLE geometry and capacity estimates'),('push',)]:subprocess.run(['git',*args],cwd=W,check=True)
    print(json.dumps(dict(registry=str(reg),registry_identity=ident(reg),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip(),row_counts=[actual['all_oracle_rows_compared'],full['all_oracle_rows_compared']])))
