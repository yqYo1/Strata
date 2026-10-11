import ast
import fcntl
import importlib.util
import json
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
C=B/'xe-long-prefill-protocol-source-v1'
O=B/'xe-llvm-v711-source-v5/direct_owner.py'
s=importlib.util.spec_from_file_location('owner',O);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.require(m.sha(O)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f','qualified owner')
ENV=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=B/'xe-long-prefill-protocol-cpu-root-v1';m.require(not out.exists(),'immutable CPU qualification');out.mkdir()
 r=dict(active=True,complete=False,passed=False,commands=[],started_utc=m.utc(),controller_sha256=m.sha(__file__),model_executed=False,gpu_executed=False,scope='Actual finite CPU protocol and optional trace fixtures, plus unchanged completed6144 request protocol; no numerical/GPU/throughput/full-context claim')
 def save():(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
 owner=m.Owner(out/'commands',save);r['commands']=owner.commands;save()
 try:
  review_path=B/'xe-long-prefill-protocol-main-review-v1.json';review=json.loads(review_path.read_text());m.require(review['main_reviewed'],'complete Main source review');r['main_review_sha256']=m.sha(review_path)
  for a in review['artifact_pins']:m.require(m.sha(Path(a['path']))==a['sha256'],'reviewed source unchanged')
  r['source_pins']={n:m.sha(C/n) for n in ['protocol.py','selfcheck.py']}
  for n in r['source_pins']:ast.parse((C/n).read_text(),filename=str(C/n))
  r['ast_parse_passed']=True
  e,so,se=owner.run('finite-fixtures',['/usr/bin/python3','-B',str(C/'selfcheck.py')],ENV,C,wall=20,cpu=20,text_cap=1<<20,total_cap=2<<20,rss_cap=2<<30)
  m.require(m.closed(e) and e['exit_code']==0,'normal closed finite fixtures');r['fixtures']=json.loads(so.read_text());m.require({k:r['fixtures'][k] for k in review['intended_cases']}==review['intended_cases'],'exact intended fixture counts')
  previous_path=B/'xe-source-llvm-chunk6144-diagnostic32k-root-v3/counts/record.json';previous=json.loads(previous_path.read_text());m.require(previous['passed'] and previous['complete'] and not previous['active'] and all(m.closed(c) for c in previous['commands']),'closed actual6144 request')
  r['actual_reference']=dict(path=str(previous_path),sha256=m.sha(previous_path),status_unchanged=True)
  capture=out/'actual-protocol.txt';capture.write_text('\n'.join(previous['request_protocol'])+'\n')
  expectations=out/'expectations.json';expected=dict(n_prompt=32768,batched_n=32767,chunk_limit=6144,max_generated=16,physical_context=262144,vocabulary=248320,output_requirement='positive',require_lp5=True);expectations.write_text(json.dumps(expected,indent=2)+'\n');r['actual_expectations']=expected
  e,so,se=owner.run('actual6144-capture',['/usr/bin/python3','-B',str(C/'selfcheck.py'),'--actual-protocol',str(capture),'--expectations',str(expectations)],ENV,C,wall=20,cpu=20,text_cap=1<<20,total_cap=2<<20,rss_cap=2<<30)
  m.require(m.closed(e) and e['exit_code']==0,'normal actual6144 protocol validation');r['actual_capture_validation']=json.loads(so.read_text())['actual']['protocol'];m.require(r['actual_capture_validation']['output_ids']==previous['validation']['output_ids'],'same actual captured IDs')
  r['capture_sha256']=m.sha(capture);r['source_stable']=all(m.sha(C/n)==h for n,h in r['source_pins'].items());m.require(r['source_stable'],'source stable');m.require(m.sha(previous_path)==r['actual_reference']['sha256'],'original actual receipt unchanged');r['complete']=True
 except BaseException as error:r['error']=repr(error)
 finally:
  r['active']=owner.active is not None;r['passed']=bool(r['complete'] and not r['active'] and not r.get('error'));r['finished_utc']=m.utc();save()
 print(json.dumps({k:r.get(k) for k in ['passed','error','fixtures']}));raise SystemExit(0 if r['passed'] else 1)
