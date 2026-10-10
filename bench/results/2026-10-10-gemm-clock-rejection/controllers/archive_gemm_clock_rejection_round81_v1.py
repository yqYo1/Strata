"""Durable closed-run boundary; source/receipts, no workload or deletion."""
from pathlib import Path
import copy,datetime,fcntl,gzip,hashlib,json,shutil
B=Path(__file__).parent;R=B/'research-20261009'
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
A=W/'bench/results/2026-10-10-gemm-clock-rejection';AR=W/'bench/results/2026-10-10-parallel-round81'
def ident(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def write(p,j):p.write_text(json.dumps(j,indent=2)+'\n')
def cp(p,d):
 d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d);i=ident(p);j=ident(d);assert (i['bytes'],i['sha256'])==(j['bytes'],j['sha256']);inventory.append(dict(original=i,archive=str(d.relative_to(W))))
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not A.exists() and not AR.exists();A.mkdir(parents=True);AR.mkdir(parents=True);inventory=[]
 prior=R/'report-registry-v103.json';prev=json.loads(prior.read_text());assert len(prev['reports'])==382
 reports=[]
 for n in range(276,284):
  ps=list(R.glob(f'round{n}-*.txt'));assert len(ps)==1;p=ps[0];cp(p,AR/p.name)
  reports.append(dict(**ident(p),agent='/root/research_bottleneck_evidence_audit_v203' if n%2==0 else '/root/research_native_iq4nl_esimd_k640_v202',model='gpt-6-luna',status='completed-read-only',root_full_report_reviewed=True,scope=p.stem.split('-',1)[1]))
 handoffs=[]
 for name in ['implementation-mtp-prefill-owned-lease-v1.txt','implementation-observed-gemm-span-v1.txt']:
  p=R/name;cp(p,AR/name);handoffs.append(dict(**ident(p),agent='/root/implement_prefill_phase_timer_validity',model='gpt-6.1-sol',status='completed-source-only-not-compiled-or-tested',root_full_source_and_report_reviewed=True))
 for name in ['mtp-prefill-owned-lease-v1','qualify-observed-gemm-span-v1']:
  for p in sorted((B/name).rglob('*')):
   if p.is_file():cp(p,A/'source-only-handoffs'/name/p.relative_to(B/name))
 for name in ['run_prefill_gemm_only_contract_v1.py','build_prefill_gemm_only_tu_v1.py','link_prefill_gemm_only_v1.py','run_owned_prefill_gemm_only_code32k_v1.py','run_owned_fixed12k_chunk_major_code32k_v1.py','fixed12k-chunk-major-admission-ticket-v1.json',Path(__file__).name]:cp(B/name,A/'controllers'/name)
 for name in ['prefill-gemm-only-contract-root-v1','prefill-gemm-only-tu-root-v1','prefill-gemm-only-linked-root-v1']:
  p=B/name;r=json.loads((p/'record.json').read_text());assert r['passed'] and not r['active'];cp(p/'record.json',A/'qualification'/name/'record.json')
 name='owned-prefill-gemm-only-code32k-diagnostic-r1';P=B/name;r=json.loads((P/'record.json').read_text());assert not r['active'] and r['math_gate_passed'] and r['census_gate_passed'] and r['exit_code']==0 and not r['new_fault_messages'] and not any(r['cleanup'].values()) and not r['healthy']
 for rel in ['record.json','events.jsonl','protocol.stdout.raw','debugger/inferior-argv.json','probes/health.stdout','probes/health-environment.json','probes/kernel-before.stdout','probes/kernel-before.stderr','probes/kernel-after.stdout','probes/kernel-after.stderr']:cp(P/rel,A/'model'/rel)
 log=P/'debugger/inferior.stderr';lines=log.read_text().splitlines(keepends=True);context=''.join(l for l in lines if not l.startswith('{"kind":"prefill_route_'))
 (A/'model'/'diagnostic-context.stderr').write_text(context)
 validity=[json.loads(l.split(': ',1)[1]) for l in lines if l.startswith('strata prefill service validity: ')]
 assert len(validity)==1 and validity[0]['status']=='invalid' and validity[0]['reason']=='backward_event_timestamps'
 assert not any(l.startswith('strata prefill returned-event ledger: ') for l in lines)
 event_receipt=dict(scope='Derived result supplement; original failed receipt byte-identical and never rewritten',whole_run_timing_valid=False,numerical_gate_passed=True,normal_exit=True,new_XE_faults=[],validity=validity,original_record=ident(P/'record.json'),original_log=ident(log),retained_context=ident(A/'model/diagnostic-context.stderr'),returned_event_span_not_full_routine_qualified=True,same_flag_retry_permitted=False)
 write(A/'event-rejection.json',event_receipt)
 src=P/'census.jsonl';dest=A/'model/census.jsonl.gz'
 with src.open('rb') as fi,dest.open('wb') as fo,gzip.GzipFile(filename='',mode='wb',fileobj=fo,mtime=0) as z:shutil.copyfileobj(fi,z)
 with gzip.open(dest,'rb') as z:raw=z.read()
 assert len(raw)==ident(src)['bytes'] and hashlib.sha256(raw).hexdigest()==ident(src)['sha256'];inventory.append(dict(original=ident(src),archive=str(dest.relative_to(W)),decoded_verified=True,archive_identity=ident(dest)))
 review=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Root reviewed R276..283 plus closed main CPU/TU/link/model and two Sol source-only handoffs',adopted=False,performance_eligible=False,current_candidate_physical262144_passed=False,
  passed=['Exact extracted GEMM flag + real private header CPU13 cases, duplicate JSON key rejection and repeatedM7/calls2/rows14','Actual precise SYCL TU/link, binary81375b88f043f6b791bd2fee5df7b512e25b584ad5631e0b915d5ac354db237c, all cached/nonprefill objects unchanged','GEMM-only32768 input: all66live states/head/64output/LP/MTP exact; unchanged93505route calls and190.241GB; normalexit/noXE/ownersgone'],
  rejected=['GEMM returned-event timing:101 submit>start triples; correct invalid latch, no aggregate, laterdroppedrecords not valid subset; no retry or timestampguard relaxation'],
  correction_notes=['R276 says B580 once; actual hardware is B570.', 'R278 returned oneMKL event guarantees completion, not a full-routine profiling envelope. Existing model scope text must not be promoted into a service or FLOPS claim.', 'R275 preinit-bind/suspend ordering concerns CLI, not server. Sol design finds server persistent sp.init before bind/READY, optional verification and shared owner/alias gaps. No executable MTP lease was implemented.', 'R281 references old residual-reuse708837376B net12K estimator; it is not current ordinary owned-workspace delta. Current accounted12K delta is876871680B/836.25MiB; 16K1672.5MiB. R279 did not execute12K.', 'R283 says private caller tree was not found by its search; actual B/prefill-gemm-service-only-v1 exists, pinned by rootTU/link/model identities. Its analysis applies to Wsource, not wholebinaryproof.', 'R283 raw retired640MiB PLE plane is capped by next MoEsharedmax; net149.7MiB estimate is insufficient for16K512MiBguard. Actual nativebatchbranch admission unresolved; no alias implementation.'],
  next=['Conditional fixed12K ordinary chunk-major: accounted delta836.25MiB vs measured1565MiB baselinepoststartupfree; expected729MiBremaining. Controller requires actual12288 selection and>=512MiB free atREADY, alltiming/censusflagsOFF, exactwholemodelparity. Source/binary unchanged.', 'Only after numerical gates, separate clean repeatedprefill/decode comparisons and candidate physical262144 finaltoken/lifecycle qualification.', 'Observed-shape marker/PTI harness is source-only uncompiled; returned-clock contradictions remain rejected; future host-wallbatch denominator independent ofsubmit clock.', 'Current defaultdecode needs same-run role/type/NT and worker-service attribution; R282 task6hist cannotbejoinedto task0timing.'],
  live_agent_snapshot=[dict(agent='/root/research_bottleneck_evidence_audit_v203',model='gpt-6-luna',status='completed-read-only-round282'),dict(agent='/root/research_native_iq4nl_esimd_k640_v202',model='gpt-6-luna',status='completed-read-only-round283'),dict(agent='/root/implement_prefill_phase_timer_validity',model='gpt-6.1-sol',status='completed-source-only-observed-span-harness')])
 p=R/'root-round81-complete-evidence-review-v1.json';assert not p.exists();write(p,review);cp(p,AR/p.name)
 reg=copy.deepcopy(prev);reg.update(registry_version=104,created_utc=review['created_utc'],research_completed=390,reports=prev['reports']+reports,new_reports=reports,previous_committed_registry=dict(**ident(prior),storage_commit='ffd3b2e9'),scope='ActualGEMM-only numerical pass but invalid clocks; hardware/workspace alternative source budgets',current_root_decisions=review,live_agent_snapshot=review['live_agent_snapshot'],implementation_handoffs_current_batch=handoffs,root_review_corrections_round81=review['correction_notes'])
 p=R/'report-registry-v104.json';assert not p.exists();write(p,reg);cp(p,AR/p.name)
 write(A/'archive-inventory.json',dict(files=inventory,deleted=False))
 (A/'RESULTS_JA.md').write_text('''# GEMMだけの計測と次の容量候補\n\n新しいGEMM-only診断スイッチとM別のJSONを13CPUケースで確認し、実際のSYCL TUをproductionと同じprecise設定でcompile/linkした。コピーqueueやStagerの条件は元のまま。32K入力は64tokenを生成して正常終了し、head、全66live状態、IDs/logprobs/MTPがcanonical参照と一致した。censusも93505expert呼び出し・190240998400B転送で同一だった。\n\n計測自体は不採用。101個の時刻でsubmit>startが出たため、ledgerはinvalidとなり集計を出していない。後から棄却された140780件を除いた部分だけを有効sampleにしない。offset・clamp・toleranceで救済せず、同じ32K計測を繰り返さない。正常終了・新しいxe障害なし・所有プロセス消滅という結果とは区別する。oneMKLの返すeventは完了dependencyであり、全内部kernelの時間範囲を保証する仕様でもない。\n\n16Kは既存8K workspaceとの差1672.5MiBが、全startup後のfree1565MiBを超える。さらに512MiBの余裕を必要とする。現在のserverはpersistent prefillをbind/READY前に作るため、CLI位置のMTP解放callback移動だけでは安全なserver leaseにならない。Solはowner/alias・mandatory byte verify・cancel/復元・per-request退役の不足を明記した設計を返し、実装を作っていない。\n\nまず既存の固定12K、通常chunk-majorを確認する。accounted差836.25MiB、条件付き残り728.75MiBであり、実際の12288選択・fallbackなし・READY時512MiB以上をcontrollerが要求する。cache128/KV32768は保持し、イベント/census計測は無効。旧layer-major/lease/33K-contextの12K数学棄却とは別の構成なので、全数値・状態をあらためて比較する。この境界では未実行。clean speedや新candidateのphysical262144検証も未実施。\n\nハードウェア到達値・実仕事量・容量で優先順位を決め、測定の成立しなかった部分をボトルネック確定とは言わない。8つの読み取り専用Lunaレポート、2つのSol source-only handoff、元receiptと時刻矛盾は全てcommit用に保存。\n''')
 print(json.dumps(dict(archived=str(A),research_completed=390,files=len(inventory))),flush=True)
