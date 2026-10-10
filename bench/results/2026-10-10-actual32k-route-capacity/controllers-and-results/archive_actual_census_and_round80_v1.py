"""Root-only durable evidence boundary; no workload or deletion."""
from pathlib import Path
import copy,datetime,fcntl,gzip,hashlib,json,os,shutil
B=Path(__file__).parent;R=B/'research-20261009'
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
A=W/'bench/results/2026-10-10-actual32k-route-capacity'
AR=W/'bench/results/2026-10-10-parallel-round80'
def identity(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def copy_one(p,dst):
 p=Path(p);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
 i=identity(p);j=identity(dst);assert (i['bytes'],i['sha256'])==(j['bytes'],j['sha256'])
 inventory.append(dict(original=i,archive=str(dst.relative_to(W))))
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not A.exists() and not AR.exists();A.mkdir(parents=True);AR.mkdir(parents=True)
 inventory=[]
 prior=R/'report-registry-v102.json';assert identity(prior)['sha256']=='66fd91d0d8f35d99d8d41fd9a4c444d42ced257e31a51b224d2548de1a6b40b6'
 prev=json.loads(prior.read_text());assert len(prev['reports'])==374
 rs=[]
 for n,scope in [(268,'CPU diagnostic fault and output-cap contracts'),(269,'Modern versus legacy Sysman dispatch'),(270,'Conditional direct packed small-expert candidate'),(271,'Exact public .14 Sysman source and counter contract'),(272,'Actual32K format and route budget'),(273,'Chunk growth initial memory lifecycle'),(274,'Failed profiled-copy Stager completion stack/API contract'),(275,'Owned workspace retirement before MTP weight restore')]:
  paths=list(R.glob(f'round{n}-*.txt'));assert len(paths)==1
  p=paths[0];copy_one(p,AR/p.name)
  # These reports were returned and fully read before this archive boundary.
  ag='/root/research_bottleneck_evidence_audit_v203' if n in [268,270,272,274] else '/root/research_native_iq4nl_esimd_k640_v202'
  rs.append(dict(**identity(p),agent=ag,model='gpt-6-luna',scope=scope,status='completed-read-only',root_full_report_reviewed=True))
 handoff=R/'implementation-prefill-gemm-service-only-v1.txt';copy_one(handoff,AR/handoff.name)
 for p in sorted((B/'prefill-gemm-service-only-v1').rglob('*')):
  if p.is_file():copy_one(p,A/'gemm-only-source'/p.relative_to(B/'prefill-gemm-service-only-v1'))
 for name in ['prefill_census_admission_v3.cpp','run_prefill_census_admission_v3.py','prefill_census_admission_v4.cpp','run_prefill_census_admission_v4.py','prefill_census_admission_v5.cpp','run_prefill_census_admission_v5.py','link_prefill_route_census_v2.py','link_prefill_route_census_v3.py','link_prefill_route_census_v4.py','build_prefill_route_census_tu_v3.py','create_prefill_census_model_controller_v1.py','run_owned_prefill_route_census_v3_code32k_v1.py','run_owned_prefill_route_census_v3_off_code32k_parity_v1.py','run_owned_prefill_route_service_v3_code32k_v1.py','derive_actual_prefill_census_capacity_join_v1.py','run_health_after_service_stall_v1.py','actual-prefill-census-capacity-join-v1.json',Path(__file__).name]:
  copy_one(B/name,A/'controllers-and-results'/name)
 for p in sorted((B/'prefill-route-census-v3').rglob('*')):
  if p.is_file():copy_one(p,A/'census-source-v3'/p.relative_to(B/'prefill-route-census-v3'))
 for name in ['prefill-census-admission-root-v3','prefill-census-admission-root-v4','prefill-census-admission-root-v5','prefill-route-census-linked-root-v2','prefill-route-census-linked-root-v3','prefill-route-census-linked-root-v4','prefill-route-census-tu-root-v3','health-after-service-stall-root-v1']:
  p=B/name;r=json.loads((p/'record.json').read_text());assert not r['active']
  copy_one(p/'record.json',A/'receipts'/name/'record.json')
  # Preserve relevant failure errors. Successful build recipes/identities live in receipts.
  if name in ['prefill-census-admission-root-v3','prefill-route-census-linked-root-v2']:
   for q in p.glob('*.stderr'):copy_one(q,A/'receipts'/name/q.name)
  if name=='health-after-service-stall-root-v1':
   for q in p.rglob('*'):
    if q.is_file() and q.suffix=='.json' and q.name!='record.json':copy_one(q,A/'receipts'/name/q.relative_to(p))
 for name in ['owned-prefill-route-census-v3-code32k-diagnostic-r1','owned-prefill-route-census-v3-off-code32k-parity-r1','owned-prefill-route-service-v3-code32k-diagnostic-r1']:
  p=B/name;r=json.loads((p/'record.json').read_text());assert not r['active'] and not any(r['cleanup'][s] for s in ['inferior_survived','gdb_survived'])
  # Verify captured owners cannot still name a current process on this boot.
  for key in ['inferior','debugger']:
   own=r.get(key) or {};pid=own.get('pid');ticks=own.get('start_ticks')
   if pid and ticks:
    try:stat=Path(f'/proc/{pid}/stat').read_text();actual=int(stat[stat.rfind(')')+2:].split()[19])
    except FileNotFoundError:continue
    assert actual!=ticks,('owner still present',name,key,pid)
  for rel in ['record.json','events.jsonl','protocol.stdout.raw','debugger/inferior-argv.json','probes/health.stdout','probes/health-environment.json','probes/kernel-before.stdout','probes/kernel-before.stderr','probes/kernel-after.stdout','probes/kernel-after.stderr']:
   copy_one(p/rel,A/'models'/name/rel)
  if 'route-service' in name:
   for rel in ['debugger/inferior.stderr','debugger/failure.mi.txt']:copy_one(p/rel,A/'models'/name/rel)
  elif 'off-code32k' in name:copy_one(p/'debugger/inferior.stderr',A/'models'/name/'debugger/inferior.stderr')
  else:
   # Exact complete route histogram is a compact current model-join consumer.
   src=p/'census.jsonl';dest=A/'models'/name/'census.jsonl.gz'
   with src.open('rb') as fi,dest.open('wb') as fo,gzip.GzipFile(filename='',mode='wb',fileobj=fo,mtime=0) as z:shutil.copyfileobj(fi,z)
   with gzip.open(dest,'rb') as z:
    raw=z.read();assert len(raw)==identity(src)['bytes'] and hashlib.sha256(raw).hexdigest()==identity(src)['sha256']
   inventory.append(dict(original=identity(src),archive=str(dest.relative_to(W)),archive_identity=identity(dest),decoded_verified=True))
 decision=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Main reviewed current evidence; original reports and receipts retained without status edits',adopted=False,performance_eligible=False,current_binary_physical262144_qualified=False,
  current_source_hash='905f43162ae079b6785584da226d950cec14b13bd413ca51ab1a48f49de6ebaf',
  corrections=['R270, R272 and R274 transcribe the caller SHA with an extra a; authoritative source SHA is the current_source_hash recorded here and in actual TU/model receipts. Their original reports remain unchanged.',
   'R274 says exact .14 public source was unavailable; R271 already found and checked upstream/PPA .14 source. Source availability is established, installed library equivalence/debug symbols are still unqualified. R276 is researching the exact status-query source.',
   'R268 registry102 parent tree reference is historical; current storage parent is2a1eb686. CPU64/67 root receipts separately establish which exact header was tested.',
   'R272 and capacity join were written before defaultOFF model parity passed; current OFF result passed without emitted census. Neither ON nor OFF dump/diagnostic wall is a clean performance sample.',
   'R273 pre-init release alone omits restoration peak. R275 requires Prefill.reset after successful drained run and before restore. auto:16384 with no-prefill-borrow is coerced to2048; only fixed16384 is a candidate.',
   'CPU full-context-host is ledger coverage only; new linked candidate has not run the physical262144 final-token capacity gate.'],
  passed=['CPU v4 64 and v5 67 cases; v5 admits batched32767 from32768 input and rejects32766/262145',
   'Actual SYCL TU v3 and linked v4; cached dependencies and nonprefill archive members unchanged',
   'Fresh32768 ON route census and OFF parity: head, all66 live state parts,64 IDs/logprobs/MTP exact canonical; normal exit no newXE faults owners closed',
   'Actual192 layer-chunk rows/native quant metadata joined,93505 expert calls,92998 copies,190240998400 packed bytes',
   'Postfailed-profile integer GPU health passed without reset or newXE faults'],
  retained_failures=['CPU v3 Werror misleading indentation before tests','Link v2 root FSIZE8MiB too small for38MiB output; normal failure, fixed64MiB limit v3/v4 passed','Combined SERVICE_TIMING32K watchdogSIGABRT: main Stager readiness wait,3workers inside event status/UR/validation/driver stack; noXE fault and posthealthPASS. No valid timing/model result.'],
  next=['Qualify GEMM-only source exact env/JSON M distinction CPU and actual TU/link; copy queue stays unprofiled; no combined-mode retry or watchdog relaxation','One32768 GEMM-only diagnostic numerical+event/census joins with finite ownership gates','Compare attained/physical conditional capacities and M-conditioned actual service; do not label overlapping sum as exposed wait','Investigate fixed16384 external MTP lease with owned workspace retired before weight restore; test parity, physical262144 and independent prefill/decode samples before adoption'],
  live_assignments=[dict(agent='/root/research_bottleneck_evidence_audit_v203',model='gpt-6-luna',report='round276-exact-runtime-profiled-copy-query-source-boundary.txt',status='running-read-only'),dict(agent='/root/research_native_iq4nl_esimd_k640_v202',model='gpt-6-luna',report='round277-actual-route-scheduling-and-capacity-scenario-budget.txt',status='running-read-only'),dict(agent='/root/implement_prefill_phase_timer_validity',model='gpt-6.1-sol',report='implementation-mtp-prefill-owned-lease-v1.txt',status='running-source-only-no-build-or-test')])
 review=R/'root-round80-complete-evidence-review-v1.json';assert not review.exists();write(review,decision);copy_one(review,AR/review.name)
 reg=copy.deepcopy(prev);reg.update(registry_version=103,created_utc=decision['created_utc'],research_completed=382,reports=prev['reports']+rs,new_reports=rs,previous_committed_registry=dict(**identity(prior),storage_commit='2a1eb6864420738db7714587f26f517e1718879e'),scope='Round80 actual32K route join, defaultOFF parity, closed profile failure and posthealth; source-only next diagnostics',current_root_decisions=decision,live_agent_snapshot=decision['live_assignments'],implementation_handoffs_current_batch=[dict(**identity(handoff),agent='/root/implement_prefill_phase_timer_validity',model='gpt-6.1-sol',status='completed-source-only-not-compiled-or-tested',root_full_source_and_report_reviewed=True)],root_review_corrections_round80=decision['corrections'])
 rp=R/'report-registry-v103.json';assert not rp.exists();write(rp,reg);copy_one(rp,AR/rp.name)
 write(A/'archive-inventory.json',dict(scope='Byte-identical small evidence and verified lossless census; no artifacts removed by this script',files=inventory))
 (A/'RESULTS_JA.md').write_text('''# 実際の32Kルートと容量の照合\n\nB570 / 262144設定・KV常駐32768 / 固定8192チャンクで、入力32768（batched32767）を一回ずつON/OFF確認した。初回head、66個のlive状態、64出力ID・logprob・MTP結果が既存canonical参照と一致し、正常終了・新しいxe障害なし。出力保存と診断が入るため速度比較には使わない。現在のbinaryそのもののphysical262144検証は未実施。\n\n実ルートは4チャンク×48層、93505 expert呼び出し（GU+Down）、92998重みコピー、190240998400 B。約49.7%の呼び出しがM<=80だが、この集合の行数は約8.1%。平均Mや大行列ピークから現在のGEMM時間を補間しない。\n\n実測Host-USM H2D6.4468 GB/sに当てはめるとコピー29.509秒相当、pageable4.6052 GB/sでは41.31秒相当、外部リンクGen4x4の符号化後7.8769 GB/sでは24.152秒相当になる。これは達成済み単体帯域または物理リンク条件との照合であり、実モデルの露出待ち時間や絶対最大性能ではない。DQ単体fixtureのformat別呼び出し加重10.743秒も同様で、転送と重なり得るため足して直列時間としない。\n\nresidentかつM<=80の候補は168呼び出し・5732行だけで、全呼び出し0.18%、全行0.036%。単体DQ換算の削減余地は約21msで、優先候補から外す。チャンク拡大や層順序変更による重み再転送削減と実GEMM形状の改善を調査する。\n\nコピーとGEMMのイベントを同時に有効化した診断は、初層でwatchdog SIGABRTとなり無効。主スレッドはStager待ち、3準備スレッドはイベント状態問い合わせのUR/validation/driver内にいた。新xe障害やdumpはなく、所有プロセス終了後のGPU整数検査はPASS。ハードウェア障害や特定driverバグと断定しない。コピーqueueのprofilingを外すGEMMのみの診断をソース上で分離したが、この境界では未build/test。監視を緩めず、同じcombined modeを再試行しない。\n\n固定16384のowned workspaceを作る前にMTP expert/headをRAM参照から解放し、run終了後にPrefill.resetでworkspaceを退役してから復元する候補を別途実装調査中。auto:16384とno-borrowの併用は2048に縮退する。VRAMの同時live最大、キャンセル/エラー復元、同じアドレスとgraph再作成、prefill/decode個別比較、physical262144の最後の入力tokenを含む検証が採用条件。\n\n各原receipt・全サンプル・実行環境・compile/link・failed stackは本ディレクトリに保存。レポート268〜275の原文と転記訂正・次の並行調査は隣の2026-10-10-parallel-round80。元の結果を変更していない。\n''')
 print(json.dumps(dict(archive=str(A),registry_version=103,research_completed=len(reg['reports']),files=len(inventory),source_bytes=sum(x['original']['bytes'] for x in inventory))),flush=True)
