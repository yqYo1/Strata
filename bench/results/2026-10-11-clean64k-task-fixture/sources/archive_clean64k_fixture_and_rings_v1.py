import datetime,fcntl,hashlib,json,os,re,shutil
from pathlib import Path
B=Path(__file__).parent;R=B/'research-20261009';W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
A=W/'bench/results/2026-10-11-clean64k-task-fixture';P=W/'bench/results/2026-10-11-parallel-round86';OLD=W/'bench/results/2026-10-11-clean64k-and-xestrata-admission'
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def copy(p,q):
 p=Path(p);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);assert sha(p)==sha(q)
def absent(x):
 try:
  v=Path('/proc',str(x['pid']),'stat').read_text();assert int(v[v.rfind(')')+2:].split()[19])!=x['start_ticks'],'owned process still present'
 except FileNotFoundError:pass
with (B/'owned-v0141-measurement.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not A.exists()and not P.exists();A.mkdir();P.mkdir()
 for n in range(1,4):
  out=B/f'clean64k-task-fixture-cpu-root-r{n}';d=json.loads((out/'record.json').read_text());assert not d['active']
  assert d['passed']is(n==3)
  for c in d['commands']:
   assert c['session_empty']and c['observation_complete']and not c['survivors']and c['direct_child_reaped']
   for x in c['owners']:absent(x)
  for p in out.rglob('*'):
   if p.is_file()and '__pycache__'not in str(p):copy(p,A/'cpu'/out.name/p.relative_to(out))
 out=B/'clean64k-task-fixture-root-r3';m=json.loads((out/'manifest.json').read_text());p=Path(m['fixture']['path']);ids=list(map(int,p.read_text().split()))
 assert len(ids)==65536 and sha(p)==m['fixture']['sha256'];assert m['suffix']['encode_exact']and m['suffix']['decode_exact']and m['suffix']['original32k_tail_exact']
 copy(p,A/'coding-review-65536-tokens.txt');copy(out/'manifest.json',A/'manifest.json')
 for name in ['clean64k-task-fixture-v1','clean64k-task-fixture-v2']:
  for p in (B/name).iterdir():
   if p.is_file():copy(p,A/'sources'/name/p.name)
 for name in ['run_clean64k_fixture_cpu_v1.py','run_clean64k_fixture_cpu_v2.py','run_clean64k_fixture_cpu_v3.py']:
  copy(B/name,A/'sources'/name)
 copy(Path(__file__),A/'sources'/Path(__file__).name)
 copy(R/'implementation-clean64k-task-fixture-v1.txt',P/'implementation-clean64k-task-fixture-v1.txt')
 regpath=R/'report-registry-v108.json';reg=json.loads(regpath.read_text());new=[]
 for n in range(309,313):
  hits=list(R.glob(f'round{n}-*.txt'));assert len(hits)==1;p=hits[0];copy(p,P/p.name)
  new.append(dict(**pin(p),archive_path=str(P/p.name),model='gpt-6-luna',agent='/root/research_bottleneck_evidence_audit_v203'if n in [309,311]else '/root/research_native_iq4nl_esimd_k640_v202',status='completed-read-only',root_full_report_reviewed=True,scope=p.stem))
 correction=dict(R311_commit='b8cc2046 is D evidence commit, not Xe39bdadcc. Source919c/a844pins match immutableXe. PreviousR309 blanketHB assertion does not cover samechunk skip-release branch. Missingwait_issued can race plainused_of/optionalEvent.ev with issuer; actualskipoccurrence and GPUfaultcause unestablished.',R312_layers='Mainprefill has12QSA layers. Historical all13QSA fullcontext lifecycle includesdrafter and is a different scope. Do not multiply mainprefill payload by13. InspectedWNEWlayer/kernel linkinput equivalence is not independentlypinned:2.906652672GB remainssource-required conditional logicalpayload, notbusmeasurement.',fixture_cpu='Originalv1 FAILED on historicalsource262145vsbound262144. Newv2 pins exact262145 source count, constructs65536 only. V2usrbinpython FAILED lackingregex. Samev2 source withexistingpreviouslytestedvenv passes strictsuffixencode/decode/hash/count/domain; failedreceiptsunchanged. No renderedfullchat or independentqualitydata claim.',safety='No GPU/model test afterV8fault; no reset/rebind/dumpclear. Postfaulthealth admissibility/sourceauditR313pending; current13QSA physical262144 proof is oldbinary only.')
 reg.update(registry_version=109,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),research_completed=reg['research_completed']+len(new),previous_committed_registry=pin(regpath),new_reports=new)
 reg['reports']+=new;reg['root_review_corrections_round86']=correction
 reg['live_agent_snapshot']=[dict(agent='/root/research_bottleneck_evidence_audit_v203',model='gpt-6-luna',status='running-read-only-R313'),dict(agent='/root/research_native_iq4nl_esimd_k640_v202',model='gpt-6-luna',status='running-read-only-R314'),dict(agent='/root/implement_prefill_phase_timer_validity',model='gpt-6.1-sol',status='completed-source-only-fixture-v1; rootcorrectedv2CPUqualified')]
 reg['current_root_decisions']=dict(adopted=False,prefill_goal_tps=1000,actual_goal_input=65536,baseline_first64k_tps=403.53959196541444,baseline_decode3_ineligible=True,newfixture_cpu_qualified=True,newfixture=m['fixture'],GPU_work_stopped=True,full262144_currentbinary_qualified=False,next='R313postfaulthealth admission thenknownbaseline fresh64K newtaskfixture; no rawXe faultpath retry; matcheddecodelength3freshsamples afterfirstbaseline qualifies')
 write(R/'report-registry-v109.json',reg);copy(R/'report-registry-v109.json',P/'report-registry-v109.json')
 # Retire duplicated body, dummy CPU output and empty files, preserving results/hash manifests.
 retired=[]
 body=out/'body-prefix-65487-tokens.txt';expect=(' '.join(map(str,ids[:65487]))+'\n').encode();assert hashlib.sha256(expect).hexdigest()==m['body']['sha256']==sha(body)
 targets=[(body,'redundantbody reconstructs byte-exact from retainedfullfixture first65487IDs',str(A/'coding-review-65536-tokens.txt'))]
 for p in OLD.rglob('*'):
  if not p.is_file():continue
  rel=p.relative_to(OLD)
  dummy=p.name=='log-limit.stderr'and 'cpu' in rel.parts
  negative=p.parent.name=='xestrata-clean64k-protocol-contract-root-v3'and p.name!='record.json'and p.suffix=='.json'
  if p.stat().st_size==0 or dummy or negative:
   targets.append((p,'zero-byte/controlleddummy CPU log or one-use negative-admission input; source and original structuredresult/hashes retained',str(OLD/'RESULTS_JA.md')))
   if rel.parts[0]=='cpu':private=B/Path(*rel.parts[1:])
   elif rel.parts[0] in [x.name for x in B.glob('owned-xestrata39*')]+['owned-clean64k-v3-baseline-r1']:private=B/rel
   else:private=None
   if private is not None and private.exists():assert sha(private)==sha(p);targets.append((private,'private duplicate of retired CPUdummy/empty/negative input; structuredresult unchanged',str(OLD/'RESULTS_JA.md')))
 for p,why,replacement in targets:retired.append(dict(**pin(p),allocated_bytes=p.stat().st_blocks*512,reason=why,replacement=replacement))
 (A/'RESULTS_JA.md').write_text('実65536入力でprefill1000tok/sを評価する条件を継続する。現実装初回は403.5396tok/s、decode3IDstopは比較に不適格。新fixtureは歴史sourceの最初65487IDsと、既存32Kの検証済み49IDタスク/assistant末尾から構成。CPUの実hosttokenizer encode/decodeが49ID/textと完全一致し、実65536count/domain/ファイルhashを独立確認。新fixtureSHA5284d8fa53d1ef28f6ba916a629faf901c890bc0c669bf986ee04ed72aa2c3ff。既存corpusを共有しbodyはコード途中で切れるため、自然な新会話や独立した品質datasetとは扱わない。生成64IDは未確認。\n\nCPU1回目はsourceの262145IDsが想定上限262144を超えて失敗、GPUなし/出力なし。新v2は元sourcehashを維持してsourcecountをexact262145に固定。これは元資料の長さで、モデルへ渡すcontextはexact65536。2回目はusrbinPythonにregex依存がなく失敗。3回目は既存tokenizerテストvenvを使い同じv2builderでPASS。全所有process閉鎖、失敗は保持。\n\nR309/R311はrawXeのskip-releaseとissuerでsamechunkのmetadata/Event競合があり得ることを示す。crosschunkjoinはある。実routing skipの発生やCAT18原因は未確定。R310/R312は64Kの仕事量/容量を分け、mainprefill12QSAのprefix staging2.906652672GBをsourceconditional値として算出（未計測）。実MoE64Ktraffic/productionphaseは未測定。GPU障害後の追加モデル実行は停止中。R313/R314は異なるscopeで並行調査。\n\nstorageは原structuredreceiptを変更せず、空ログ、CPUdummylog、使い終えたnegativeinputs、fullfixtureからbyte-exact再構成できるbodyprefixをmanifestで退役。各実測値・失敗・driverfault/stackは保持。\n')
 (P/'REVIEW_JA.md').write_text('R309–R312を全文確認、元報告維持。R311のcommit番号はDの証拠commitでXeソース39bdadccではない点を別補正。R309の一般HB説明はskip branchのplainmetadata/event競合を除外しない。R312の12mainQSAとold13QSA全session/drafterのscopeを区別。logicalstaging2.907GBはproductionbus計測ではない。新64KfixtureのCPU資格はモデル/品質/GPU資格ではない。次はpostfaulthealthgateとcurrentDlargerchunkのring適用audit。\n')
 secrets=[v.encode()for k,v in os.environ.items()if re.search(r'TOKEN|SECRET|PASSWORD|API_KEY',k,re.I)and len(v)>=16]
 for root in [A,P]:
  for p in root.rglob('*'):
   if p.is_file():assert not any(v in p.read_bytes()for v in secrets),'credentialvalue in prospective archive'
 for e in retired:
  p=Path(e['path']);assert sha(p)==e['sha256'];p.unlink();e['removed']=True
 write(A/'retention-manifest.json',dict(complete=True,original_receipts_unchanged=True,removed=retired,removed_logical_bytes=sum(e['bytes']for e in retired),removed_file_allocation_bytes=sum(e['allocated_bytes']for e in retired),physical_space_gain_claimed=False))
 print(json.dumps(dict(passed=True,research_completed=reg['research_completed'],fixture_sha256=m['fixture']['sha256'],retired_files=len(retired),retired_logical_bytes=sum(e['bytes']for e in retired))))
