"""Root archives returned research and closed primitive GPU proof, under lock."""
from pathlib import Path
from collections import Counter
import datetime,fcntl,hashlib,json,re,shutil,subprocess
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
R=B/'research-20261009'
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
A=W/'bench/results/2026-10-10-parallel-round36'
def ident(p):
    p=Path(p)
    with p.open('rb') as stream:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def copy(source,target):
    assert not target.exists(),target;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);assert ident(source)==ident(target)

with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True)
    prev=R/'report-registry-v58.json';assert ident(prev)['sha256']=='91e875d1abf0023b04986f26283803b3ba4143de4d29cd20039aef70c985dce2'
    d=json.loads(prev.read_text());assert len(d['reports'])==243;new=[]
    entries=[
      (137,'prefill-gate-factor-production-interface','research_prefill_algorithms_blogs','Production factor interface and prefill-only integration scope','f68b53d17fc5bc9fcb13fc475fe6f76babe64d3cf9c701fffa4cc186300cc65c'),
      (138,'current-owner-controller-source-audit','research_dense_mirror_phase_ownership','Actual owner v1 source bounds/identity schema and durable observation','d5363b04b50f5d49579d79fea5a6175cddb70b5a0acbd826f2d501f84764337f'),
      (139,'public-pci-loader-driver-provenance','research_dense_mirror_phase_ownership','Standard public PCI loader versus vendor driver-extension resolver failure','c3ded5a7979cbfe426f2badaade0cb9d9dec07e381d8a87931c9d9cfb3c59236'),
      (140,'gdn-gate-exp-source-multiplicity','research_prefill_algorithms_blogs','Source reader multiplicity and required real recurrence performance discriminator','e112d8b4d86b50da327a3043f7045ad9d66e625ad61f1d9d8bed9d675832d054'),
      (141,'gdn-factor-real-recurrence-oracle','research_dense_mirror_phase_ownership','Concrete actual-state/conv/oracle and bounded full-prefix schedules','3769a760c923f4fa4e369a9af052ebe63b1286eb8b76d8ec137a01b41664d161'),
      (142,'moe-prefill-transfer-new-candidates','research_prefill_algorithms_blogs','Conditional nonfused MMQ GPU grouping candidate, compiled route unresolved','baa15f94e03bb43626bdb2fbeb227d672bde380632bac56ce0089bd01ed5dfc1'),
      (143,'gdn-factor-production-source-audit','research_dense_mirror_phase_ownership','Actual Sol factor copies, selector pairing, SYCL lifetime and default preservation','75d5fcef58e9a2e82cb04dcc0685d62a31ef6d03fe6294aecff3569bb1ec3fcb'),
      (144,'nonfused-mmq-gpu-grouping-interface','research_prefill_algorithms_blogs','Counts/off scratch publication and MMQ invariants; compiled stub correction pending','04c545f2eb9f05e99554660df846bedf315af5d0fc8de849f836154699d70de2'),
      (145,'real-gdn-factor-parity-source-audit','research_dense_mirror_phase_ownership','Actual root fixture input/guard/state carry and OFF-test gap','9f52c8c6fbf4b4cc939ceca9ba1356584d93856994ee4fa469f4f1605d206926'),
      (146,'compiled-sycl-moe-grouping-route-correction','research_prefill_algorithms_blogs','Withdraw inactive MMQ/fused premise; compiled FP16/GEMM grouping seam','a2366cedbb511b45b293d2ef82eff6bfb572f726f5b75127d7ac4b9575eb4cf4'),
      (147,'zen3-native-decode-new-factor-screen','research_dense_mirror_phase_ownership','External exact-format Zen3 decode factor screen; no new candidate','fc516864906703ff36c10d4575e60648d49b2c7f0dec3dbf32cfbe96c491bb2a')]
    for number,name,agent,scope,digest in entries:
        source=R/f'round{number}-{name}.txt';identity=ident(source);assert identity['sha256']==digest
        copy(source,A/'research'/source.name)
        entry=dict(path=str(source),**identity,agent='/root/'+agent,model='gpt-6-luna',scope=scope,status='completed-read-only',root_full_report_reviewed=True)
        d['reports'].append(entry);new.append(entry)
    runtime=B/'gdn-gate-factor-probe-runtime-v2';receipt=json.loads((runtime/'record.json').read_text())
    assert receipt['passed'] and receipt['complete'] and not receipt['active']
    assert ident(runtime/'record.json')['sha256']=='d101225b5206311e3a7c867d49a29844878187f94d5ba6ff555c0238fec44228'
    for name in ['record.json','probe.stdout','journal-cursor.stdout','journal-cursor.stderr','kernel-interval.stdout','kernel-interval.stderr']:
        copy(runtime/name,A/'evidence/gdn-primitive-v2'/name)
    copy(B/'run_gdn_gate_factor_probe_v2.py',A/'evidence/gdn-primitive-v2/run_gdn_gate_factor_probe_v2.py')
    copy(B/'gdn-owner-controller-host-checks-v2/record.json',A/'evidence/gdn-primitive-v2/host-owner-v2-record.json')
    trace=runtime/'probe.stderr';lines=trace.read_text().splitlines();probe=next(x for x in receipt['commands'] if x['label']=='probe')
    assert ident(trace)==probe['logs']['probe.stderr']
    counts=Counter(re.findall(r'SUCCESS \(ZE_RESULT_SUCCESS\) in (ze\w+)\(',trace.read_text()))
    error_lines=[line for line in lines if re.search(r'-> UR_RESULT_(?!SUCCESS)|ERROR \(ZE_RESULT_',line)]
    write(A/'evidence/gdn-primitive-v2/trace-summary.json',dict(original_path=str(trace),**ident(trace),lines=len(lines),successful_LevelZero_calls=dict(sorted(counts.items())),non_success_API_lines=len(error_lines),non_success_examples=error_lines[:16],first_lines=lines[:8],last_lines=lines[-14:],policy='Positive verbose trace compacted; original failure v1 retained unchanged in source commit609a270d. Original trace eligible for retirement after this archive commit and consumer closure.'))
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    d.update(registry_version=59,created_utc=now,research_completed=len(d['reports']),new_reports=new,previous_committed_registry=dict(path=str(prev),**ident(prev),storage_commit='ea3b6316771aace293cfc315e202f8f893896d53'))
    d['scope']='Recurring read-only research restored after completed-task gap; separate Sol real factor implementation and root actual parity/build work'
    d['live_agent_snapshot']=dict(time_utc=now,method='Reports returned and fully reviewed; completed boundary before next dispatch, verify actual live state on resume',agents=[dict(agent='/root/research_dense_mirror_phase_ownership',model='gpt-6-luna',round=147,status_at_dispatch='completed',scope='Distinct Zen3 native CPU decode external factor screen'),dict(agent='/root/research_prefill_algorithms_blogs',model='gpt-6-luna',round=146,status_at_dispatch='completed',scope='Actual compiled SYCL MoE route correction'),dict(agent='/root/implement_repeat_capture_reader_hardening',model='gpt-6.1-sol',status_at_dispatch='completed-source-only',handoff='implementation-gdn-prefill-gate-factor-v1.txt')])
    d['current_root_decisions']=dict(
       CPU_register_index='REJECT previous twelve real-weight process/cell medians; no reimplementation.',
       GDN_primitive='PASS actual B570 typed root0000:05:00.0:15cases/75stages/7970640bit comparisons;three8K fingerprints equal,normal0,empty ownedSID,newkernel journal interval clean.',
       GDN_primitive_receipt=dict(path=str(runtime/'record.json'),**ident(runtime/'record.json')),
       GDN_real_candidate='Private default-OFF prototype committed cbc77580050a327538be48b0ba7a759c14f7a49f. Separate Sol implementation, root parity source, independent static audit; actual compile/runtime/model/performance qualification pending at source boundary.',
       root_execution='Root is sole executor and tests serially under lock. Actual real-parity CPU build v1 closed PASS:119 steps,325.807s build,normal0/session empty. Corrected host fixture not yet rebuilt; no actual recurrence runtime claim.',
       private_API_contract='Typed explicit DecayFactor is allowed with buildON; env controls production prefill caller. Legacy public/decode APIs remain loggate.',
       R145_OFF_test='Must independently assert producer and recurrence factor refusal, because first expected throw skipped second call. Root fixed the independent assertions after pinned CPU build v1 closed; corrected source awaiting rebuild.',
       MoE_grouping='HOLD R142/R144 candidate: root found actual CMake links moe_fused_stub (group aborts;group_bytes0), not migrated moe_fused.dp.cpp. R146 confirms MMQ/fused stubs inactive. Only new standalone SYCL grouping of the active FP16/GEMM route may be considered, and remains unimplemented/unmeasured.',
       model_performance_this_wave=False,actual_physical_262144_lifecycle=False,adopted=False,
       next_gate='Compile corrected independent OFF-test fixture after successful owned CPU build v1; real GDN numerical and >=32768 timing then physical model262144 lifecycle before adoption.')
    d['additional_provenance_corrections'].update(R140_gate_count='48 unique gates/token;24576 source work-item expressions/token. Original report24576 gate scalars/token is incorrect,not emitted SFU count.',R141_quad_carry='Old quad --prefix fixture does carry recurrence state. Missing conv/full model coverage remains correct; no tolerance adoption.',R142_R144_compiled_route='Migrated fused source is not currently linked: actual SYCL CMake uses aborting stub. Candidate inactive/unqualified until compiled-route correction.')
    handoff=R/'implementation-gdn-prefill-gate-factor-v1.txt'
    d['implementation_handoffs_current_batch'].append(dict(path=str(handoff),**ident(handoff),model='gpt-6.1-sol',status='completed-source-only',root_full_report_reviewed=True,source_commit='cbc77580050a327538be48b0ba7a759c14f7a49f'))
    registry=R/'report-registry-v59.json';assert not registry.exists();write(registry,d);copy(registry,A/registry.name)
    write(A/'root-parallel-review-v59.json',dict(created_utc=now,new_reports=len(new),total_reports=len(d['reports']),registry_identity=ident(registry),decisions=d['current_root_decisions'],corrections=d['additional_provenance_corrections']))
    report=A/'REPORT.md';report.write_text(report.read_text()+'''\n\n## Renewed parallel work v59\n\nRoot confirmed a gap between completed tasks and follow-up assignments. Two Luna scopes were restarted with past reports/registry/prior decisions, plus a separate Sol real prefill implementation; root concurrently wrote actual producer/conv/recurrence parity source and launched a bounded full dependency CPU build. Returned reports R137–145 were read in full and originals retained.254 reports are now reviewed. New follow-ups R146 compiled-route correction and R147 Zen3 decode factors were dispatched while main continued build/qualification. Recorded running states are timestamped observations, not durable live claims.\n\nThe private expression primitive actually passed on B570:15cases,75stage records,7970640 exact comparisons,three matching8K repeat fingerprints,normal0 and complete clean kernel fault observation. Successful verbose trace is compacted with its original hash; first startup refusal remains unchanged. R139 resolves the startup refusal as standard-public-API versus vendor-extension resolver mismatch. No model speed or physical context pass is inferred. The real factor prototype is default-OFF and committed separately atcbc77580; actual recurrence/model qualification remains pending.\n\nR145 found a real OFF-test coverage gap (one expected producer throw skipped recurrence); it will be split after the pinned build closes. R140's unique-gate arithmetic and R141's old-quad carry claim are corrected separately. R142/R144's new MMQ grouping suggestion is held because root verified the actual SYCL build links an aborting fused stub instead of the migrated fused source. R146 withdraws the inactive premise and describes a new standalone grouping seam on compiled FP16/GEMM. R147 finds no qualifying new external Zen3 factor; it did not inspect the wrapper's external implementation body, so its source screen is partial.\n''')
    agents=W/'AGENTS.md'
    agents.write_text(agents.read_text()+'''\n\n### Continuing parallel optimization research\n\n- Inspect actual agent status at the start of resumed tuning; a historical running record is not evidence that an agent still runs.\n- Use multiple gpt-6-luna read-only investigators with distinct scopes. Once their reports return, read the full reports, preserve original files and separate corrections, and assign relevant follow-ups while the main agent continues implementation or qualification. Research is recurring, not one batch.\n- Share the past report directory, latest committed registry, prior decisions and immutable source with every follow-up. Verify compiled target/call-path eligibility before accepting an optimization from merely present source files.\n- Keep source implementation with the main agent or separate gpt-6.1-sol. The main agent manages all builds, tests and GPU/model runs serially; research agents never run them. Do not intervene in running research tasks.\n''')
    copy(Path(__file__),A/Path(__file__).name)
    index=A/'archive-file-identities.json';old=json.loads(index.read_text());new_index={str(p.relative_to(A)):ident(p) for p in sorted(A.rglob('*')) if p.is_file() and p!=index}
    for rel,identity in old.items():
        if rel!='REPORT.md':assert new_index[rel]==identity,rel
    write(index,new_index)
    for args in [('diff','--check'),('add','AGENTS.md','bench/results/2026-10-10-parallel-round36'),('diff','--cached','--check'),('commit','-m','docs(sycl): resume recurring parallel research and retain closed GPU proof'),('push',),('rev-parse','HEAD')]:subprocess.run(['git',*args],cwd=W,check=True)
    print(json.dumps(dict(registry=str(registry),**ident(registry),completed=d['research_completed'])))
