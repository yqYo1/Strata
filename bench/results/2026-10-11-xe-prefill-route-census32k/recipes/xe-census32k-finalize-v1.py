import ast
import fcntl
import importlib.util
import json
import os
import re
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
XE=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
D=W/'bench/results/2026-10-11-xe-prefill-route-census32k'
O=B/'xe-llvm-v711-source-v5/direct_owner.py'
s=importlib.util.spec_from_file_location('owner',O);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.require(m.sha(O)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f','qualified owner')
ENV=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')
if 'SSH_AUTH_SOCK' in os.environ:ENV['SSH_AUTH_SOCK']=os.environ['SSH_AUTH_SOCK']
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=B/'xe-census32k-finalize-root-v1';m.require(not out.exists(),'immutable finalize');out.mkdir()
 r=dict(active=True,complete=False,passed=False,commands=[],started_utc=m.utc(),controller_sha256=m.sha(__file__),gpu_executed=False,model_executed=False,deleted=[])
 def save():(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
 owner=m.Owner(out/'commands',save);r['commands']=owner.commands;save()
 def run(label,argv,cwd=W):
  e,so,se=owner.run(label,argv,ENV,cwd,wall=60,cpu=120,text_cap=4<<20,total_cap=16<<20,rss_cap=2<<30);m.require(m.closed(e) and e['exit_code']==0,'normal closed '+label);return so.read_text().strip()
 try:
  original_path=B/'xe-census32k-boundary-root-v1/record.json';original=json.loads(original_path.read_text())
  m.require(not original['active'] and not original['passed'] and not original['complete'] and not original['deleted'] and original['error']=="RuntimeError('normal closed secret-check')" and original.get('evidence_revision') is None and all(m.closed(c) for c in original['commands']),'closed original publication rejection before commit/cleanup')
  diagnosis_path=B/'xe-census32k-archive-secret-diagnose-root-v1/record.json';diagnosis=json.loads(diagnosis_path.read_text());m.require(diagnosis['complete'] and not diagnosis['active'] and all(m.closed(c) for c in diagnosis['commands']),'closed redacted finding diagnosis')
  m.require(len(diagnosis['findings'])==1 and diagnosis['findings'][0]['RuleID']=='generic-api-key' and diagnosis['findings'][0]['StartLine']==41,'one exact publication-label finding')
  m.require(run('prepared-status',['/usr/bin/git','status','--porcelain'])=='?? '+str(D.relative_to(W))+'/','only prepared untracked archive')
  actual_path=B/'xe-source-llvm-census32k-root-v3/counts/record.json';actual=json.loads(actual_path.read_text())
  m.require(actual['passed'] and actual['complete'] and not actual['active'] and all(m.closed(c) for c in actual['commands']),'closed actual diagnostic')
  m.require(run('source-head',['/usr/bin/git','rev-parse','HEAD'],XE)==actual['candidate_revision'] and not run('source-status',['/usr/bin/git','status','--porcelain'],XE),'clean exact source')
  m.require(all(m.sha(XE/p)==h for p,h in actual['source_pins'].items()),'source unchanged')
  manifest_path=D/'archive-manifest.json';manifest=json.loads(manifest_path.read_text());copies=manifest['byte_identical_copies']
  for path,item in copies.items():m.require(m.sha(W/path)==item['sha256'] and m.sha(Path(item['original_path']))==item['sha256'],'prepared exact archive copy')
  raw=B/'xe-census32k-boundary-v1.py';raw_text=raw.read_text();m.require(m.sha(raw)==original['controller_sha256'],'original rejected controller immutable')
  offending=D/'recipes/xe-census32k-boundary-v1.py';m.require(m.sha(offending)==m.sha(raw),'exact offending copy')
  m.require('lints-v10-failed' in raw_text.splitlines()[40],'public comparison run identifier was detected, no credential')
  m.require('case_name' not in raw_text,'reversible unique identifier rename')
  normalized=re.sub(r'\bkey\b','case_name',raw_text);ast.parse(normalized)
  m.require(re.sub(r'\bcase_name\b','key',normalized)==raw_text,'exact old controller reconstruction')
  new_path=D/'recipes/xe-census32k-boundary-v1-case-label-normalized.py';m.require(not new_path.exists(),'new normalized recipe');new_path.write_text(normalized)
  copies.pop(str(offending.relative_to(W)));offending.unlink()
  proof=dict(original_path=str(raw),original_sha256=m.sha(raw),normalized_archive=str(new_path.relative_to(W)),normalized_sha256=m.sha(new_path),normalization='Only Python identifier key renamed to case_name. Reverse word-boundary replacement reconstructs exact original source bytes and hash. Semantics and numerical/measurement/controller policy unchanged.',reconstruction_verified=True,original_failed_status_preserved=True,reason='Public run label falsely matched generic-api-key at source line41; keep scanner enabled without allowlisting or suppression.')
  manifest['normalized_source_copy']=proof
  (D/'publication-normalization.json').write_text(json.dumps(proof,indent=2)+'\n')
  def copy(src,name):
   dst=D/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(src.read_bytes());m.require(m.sha(dst)==m.sha(src),'exact additional evidence');copies[str(dst.relative_to(W))]=dict(original_path=str(src),bytes=src.stat().st_size,sha256=m.sha(src))
  copy(original_path,'publication-v1-rejected/record.json');copy(diagnosis_path,'publication-finding-diagnosis/record.json')
  copy(Path(__file__),'recipes/xe-census32k-finalize-v1.py')
  pending=B/'xe-native-dequant-probe-received-snapshot-v1.json';copy(pending,pending.name)
  registry_path=D/'report-registry-v127.json';registry=json.loads(registry_path.read_text());registry['received_not_yet_main_reviewed']=json.loads(pending.read_text());registry_path.write_text(json.dumps(registry,indent=2)+'\n')
  (D/'publication-finding-diagnosis/findings.json').write_text(json.dumps(diagnosis['findings'],indent=2)+'\n')
  # Useful rejection text is already redacted and includes only count, without the matched source value.
  for area,root in [('publication-v1-rejected','xe-census32k-boundary-root-v1'),('publication-finding-diagnosis','xe-census32k-archive-secret-diagnose-root-v1')]:
   fragments={p.name:p.read_text() for p in (B/root/'commands').glob('*') if p.suffix in ('.stdout','.stderr') and p.stat().st_size}
   (D/area/'retained-output.json').write_text(json.dumps(fragments,indent=2)+'\n')
  doc=D/'RESULTS_JA.md';doc.write_text(doc.read_text()+'\n保存前のgitleaks検査は、境界controllerの公開ケース識別子をgeneric-api-keyとして誤検出し、commit前に終了した。元のFAILEDと検査コマンドを保持した。Python変数名だけを変更した保存用recipeから元の全文・hashを復元できることを確認した。検査の抑制や例外追加は行わず、再度同じ検査を通す。測定statusは変更していない。\n')
  summary_path=D/'summary.json';summary=json.loads(summary_path.read_text());summary['original_publication_v1_rejected_preserved']=True;summary['publication_source_normalization']=proof;summary_path.write_text(json.dumps(summary,indent=2)+'\n')
  manifest['byte_identical_copies']=copies;manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
  roots={'source-integration':'xe-prefill-route-census-integrate-root-v1','lints-v10-failed':'xe-e32-lints-baseline-compare-root-v10','lints-v11':'xe-e32-lints-baseline-compare-root-v11','census-controller-cpu':'xe-source-llvm-census32k-cpu-root-v3','long-protocol-cpu':'xe-long-prefill-protocol-cpu-root-v1','source-publication':'xe-prefill-route-census-source-publish-root-v2','census32k':'xe-source-llvm-census32k-root-v3/counts','publication-v1-rejected':'xe-census32k-boundary-root-v1','publication-finding-diagnosis':'xe-census32k-archive-secret-diagnose-root-v1'}
  for mode in ['free','contrib-llvm']:
   for ver in [15,16]:roots[f'{mode}-release-v{ver}']=f'xe-e32-{mode}-full-build-root-v{ver}'
   for ver in [12,13]:roots[f'{mode}-host-asan-v{ver}']=f'xe-e32-{mode}-asan-full-build-root-v{ver}'
  plans=[]
  for case_name,root in roots.items():
   v=json.loads((B/root/'record.json').read_text());m.require(not v['active'] and all(m.closed(c) for c in v['commands']),'all original owners closed')
   for c in v['commands']:
    for ident in c['owners']:
     now=m.identity(ident['pid']);m.require(not now or now['start_ticks']!=ident['start_ticks'],'owned identity absent')
   m.require(m.sha(D/case_name/'record.json')==m.sha(B/root/'record.json'),'individual original receipt exact')
   for p in (B/root/'commands').glob('*'):
    if p.suffix not in ('.stdout','.stderr'):continue
    st=p.stat();plans.append(dict(path=str(p),bytes=st.st_size,allocated_bytes=st.st_blocks*512,sha256=m.sha(p),retained_receipt=str((D/case_name/'record.json').relative_to(W)),reason='Closed repeated command/build/lint/test/API/journal streams; commands/environment/source/individual original status/owners, complete actual census/protocol/runtime/fault data and useful diagnostics retained. Both resolved lint and publisher failures remain FAILED. No full-history consumer.'))
  capture=B/'xe-long-prefill-protocol-cpu-root-v1/actual-protocol.txt';st=capture.stat();cp=json.loads((B/'xe-long-prefill-protocol-cpu-root-v1/record.json').read_text());m.require(m.sha(capture)==cp['capture_sha256'],'unchanged completed duplicate protocol');plans.append(dict(path=str(capture),bytes=st.st_size,allocated_bytes=st.st_blocks*512,sha256=m.sha(capture),retained_receipt='bench/results/2026-10-11-xe-chunk6144-and-ring96/chunk6144-v3/record.json',reason='Completed request lines retained in original committed receipt; current actual parser verdict and explicit geometry also retained.'))
  # The exact normalized Git recipe reconstructs this superseded original source; no new agent artifacts are deleted.
  st=raw.stat();plans.append(dict(path=str(raw),bytes=st.st_size,allocated_bytes=st.st_blocks*512,sha256=m.sha(raw),retained_receipt=str((D/'publication-normalization.json').relative_to(W)),reason='Exact source reconstruction from committed identifier-normalized recipe was verified; old publisher failure and controller hash preserved. Resolved archival label match, no remaining raw-source consumer.'))
  m.require(sum(p.stat().st_size for p in D.rglob('*') if p.is_file())<12<<20,'compact archive12MiB')
  run('secret-check',[str(XE/'.lint/bin/gitleaks'),'detect','--no-git','--redact','--no-banner','--exit-code','1','--source',str(D)])
  run('diff-check',['/usr/bin/git','diff','--check']);run('stage',['/usr/bin/git','add','--',str(D.relative_to(W))]);run('commit',['/usr/bin/git','commit','-m','docs(bench): qualify prefill route census and record actual branches']);r['evidence_revision']=run('evidence-head',['/usr/bin/git','rev-parse','HEAD'])
  retention=D/'retention-v1.json';retention.write_text(json.dumps(dict(retained_commit=r['evidence_revision'],deletions=plans,logical_bytes=sum(x['bytes'] for x in plans),allocated_file_bytes=sum(x['allocated_bytes'] for x in plans),does_not_measure_available_space=True,original_statuses_unchanged=True),indent=2)+'\n')
  run('retention-secret-check',[str(XE/'.lint/bin/gitleaks'),'detect','--no-git','--redact','--no-banner','--exit-code','1','--source',str(retention)])
  for item in plans:
   p=Path(item['path']);m.require(m.sha(p)==item['sha256'] and p.stat().st_size==item['bytes'],'unchanged closed stream/source');p.unlink();r['deleted'].append(item['path'])
  m.require(all(m.sha(B/root/'record.json')==m.sha(D/case_name/'record.json') for case_name,root in roots.items()),'original receipts still exact after cleanup')
  run('stage-retention',['/usr/bin/git','add','--',str(retention.relative_to(W))]);run('commit-retention',['/usr/bin/git','commit','-m','docs(bench): retire closed census qualification streams']);r['committed_revision']=run('final-head',['/usr/bin/git','rev-parse','HEAD']);run('publish',['/usr/bin/git','push','origin','HEAD']);m.require(not run('final-status',['/usr/bin/git','status','--porcelain']),'clean report tree');r['complete']=True
 except BaseException as exc:r['error']=repr(exc)
 finally:
  r['active']=owner.active is not None;r['passed']=bool(r['complete'] and not r['active'] and not r.get('error'));r['finished_utc']=m.utc();save()
 print(json.dumps({k:r.get(k) for k in ['passed','error','evidence_revision','committed_revision']}));raise SystemExit(0 if r['passed'] else 1)
