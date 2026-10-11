import fcntl,importlib.util,json,time,re,shlex,difflib,hashlib
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
P=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/bench-xestrata-e32-pristine-20261011')
O=B/'xe-llvm-v711-source-v5/direct_owner.py';s=importlib.util.spec_from_file_location('o',O);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.require(m.sha(O)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f','qualified owner')
LLVM=W/'.tools/intel-llvm-v711/install';LINT=W/'.lint'
BASE=dict(PATH=str(LINT/'venv/bin')+':'+str(LINT/'bin')+':'+str(LINT/'cppcheck/usr/bin')+':/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',LD_LIBRARY_PATH=str(LINT/'cppcheck/usr/lib/x86_64-linux-gnu')+':'+str(LLVM/'lib')+':/opt/intel/oneapi/compiler/2026.1/lib:/opt/intel/oneapi/mkl/2026.1/lib')
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out=B/'xe-e32-lints-baseline-compare-root-v4';m.require(not out.exists(),'immutable output');out.mkdir()
 r=dict(active=True,complete=False,passed=False,commands=[],started_utc=m.utc(),controller_sha256=m.sha(__file__),owner_sha256=m.sha(O),gpu_executed=False,model_executed=False,scope='Required lints, original baseline compared by diagnostic source line/message; original v1 failure preserved; no warning suppression',total_wall_budget_seconds=4200,raw_log_budget_bytes=64<<20)
 def save():(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
 owner=m.Owner(out/'commands',save);r['commands']=owner.commands;begin=time.monotonic();save()
 def run(label,args,cwd=W,env=BASE,wall=60):
  m.require(time.monotonic()-begin<4200,'whole lint deadline');e,so,se=owner.run(label,args,env,cwd,wall=min(wall,4200-(time.monotonic()-begin)),cpu=int(wall)*4,text_cap=64<<20,total_cap=64<<20,rss_cap=16<<30)
  m.require(m.closed(e),'closed '+label);return e,so.read_text(),se.read_text()
 def key(row,root,line_map=None):
  path,line,column,severity,msg=row
  path=Path(path)
  if path.is_relative_to(root):rel=str(path.relative_to(root))
  elif path.is_relative_to(W):rel=str(path.relative_to(W))
  else:rel=str(path)
  ln=int(line)
  if line_map is not None and rel in line_map:ln=line_map[rel].get(ln)
  return (rel,ln,severity,msg)
 def diagnostics(text,root,line_map=None):
  return [key(v,root,line_map) for v in re.findall(r'^(.+?):(\d+):(\d+): (warning|error): (.+)$',text,re.M)]
 def check_misc(so,se):
  m.require('skipped (' not in so and 'possible secret' not in so,'all tools executed and no secret findings')
  for name in ['clang-tidy','cppcheck','codespell','reuse']:m.require('== '+name in so,'required tool '+name)
  spell=so.split('== codespell\n',1)[1].split('== reuse',1)[0]
  m.require(not spell.strip(),'no codespell findings')
  m.require(not so.split('== reuse\n',1)[1].strip(),'whole reuse clean')
  m.require(not any('error:' in x and '[unknownMacro]' not in x for x in se.splitlines()),'no unclassified tool errors')
 try:
  prior=json.loads((B/'xe-e32-lints-root-v1/record.json').read_text());m.require(not prior['active'] and prior['complete'],'closed original required lints')
  r['original_failed_lint_receipt_sha256']=m.sha(B/'xe-e32-lints-root-v1/record.json')
  for mode in ['free','contrib-llvm']:
   q=B/('xe-e32-'+mode+'-full-build-root-'+'v9'+'/record.json');v=json.loads(q.read_text());m.require(v['passed'] and not v['active'],'closed full build '+mode);r.setdefault('build_proofs',{})[mode]=m.sha(q)
  e,so,se=run('current-paths',['/usr/bin/git','diff','--name-only','HEAD']);m.require(e['exit_code']==0,'current paths');paths=so.splitlines()
  e,so,se=run('new-paths',['/usr/bin/git','ls-files','--others','--exclude-standard']);m.require(e['exit_code']==0,'new paths');paths=sorted(set(paths+so.splitlines()));r['source_pins']={p:m.sha(W/p) for p in paths};r['paths']=paths
  e,so,se=run('pristine-head',['/usr/bin/git','rev-parse','HEAD'],cwd=P);m.require(e['exit_code']==0 and so.strip()=='e32b8b0594095352365db0e140a68d5bf56d8b0f','exact pristine baseline')
  e,so,se=run('pristine-status',['/usr/bin/git','status','--porcelain','--untracked-files=no'],cwd=P);m.require(e['exit_code']==0 and not so.strip(),'pristine unmodified')
  baseline_paths=[p for p in prior['paths'] if (P/p).is_file()]
  current_free=['src/program/generate.cpp','tools/test_prefill_dense_gemm.cpp','src/kernels/parity_device.hpp']
  reusable_path=B/'xe-e32-lints-baseline-compare-root-v3/record.json'; reusable=json.loads(reusable_path.read_text())
  m.require(not reusable['active'] and not reusable['passed'] and all(m.closed(c) for c in reusable['commands']),'closed original v3 comparison failure')
  m.require(reusable['source_pins']['CMakeLists.txt']==m.sha(W/'CMakeLists.txt'),'CMake unchanged since executed v3 lint')
  m.require(all(m.sha(W/p)==h for p,h in reusable['source_pins'].items() if p not in {'src/program/generate.cpp','tools/test_prefill_dense_gemm.cpp'}),'only reviewed post-v3 source edits plus new parity helper')
  r['reused_v3_receipt_sha256']=m.sha(reusable_path)
  line_maps={}
  for p in set(paths)|set(baseline_paths):
   if not (P/p).is_file():line_maps[p]={};continue
   lm={};old=(P/p).read_text().splitlines();new=(W/p).read_text().splitlines()
   for a,z,k in difflib.SequenceMatcher(None,old,new,autojunk=False).get_matching_blocks():
    for i in range(k):lm[z+i+1]=a+i+1
   line_maps[p]=lm
  for mode,base_files,current_files in [('free',baseline_paths,current_free),('contrib-llvm',['src/prefill/gemm.cpp'],['src/prefill/gemm.cpp'])]:
   db=out/('baseline-db-'+mode);db.mkdir();source=W/'build'/mode/'compile_commands.json';rows=json.loads(source.read_text())
   def replace(a):
    for area in ['src','include','third_party','tools']:
     a=re.sub(re.escape(str(W)+'/'+area)+r'(?=/|$)',str(P)+'/'+area,a)
    return a
   for row in rows:
    row['file']=replace(row['file'])
    argv=row.get('arguments') or shlex.split(row['command'])
    row['arguments']=[replace(a) for a in argv];row.pop('command',None)
    m.require(all(not re.search(re.escape(str(W))+r'/(?:src|include|third_party|tools)(?=/|$)',a) for a in row['arguments']),'no current project headers/source mixed into baseline')
   (db/'compile_commands.json').write_text(json.dumps(rows,indent=2)+'\n');r.setdefault('compile_db_proofs',{})[mode]=dict(current_sha256=m.sha(source),baseline_adapted_sha256=m.sha(db/'compile_commands.json'),adaptation='Only source/include/third_party/tools prefixes replaced; actual installed compiler retained; no warning/check flags changed')
   if mode=='free':
    m.require(m.sha(source)==reusable['compile_db_proofs']['free']['current_sha256'],'identical real free compiler flags to executed baseline/current v3')
   if mode=='free':
    be=next(c for c in reusable['commands'] if c['label']=='baseline-free');m.require(m.closed(be) and be['exit_code']==1,'closed actual pristine baseline free v3')
    bso=(B/'xe-e32-lints-baseline-compare-root-v3/commands/baseline-free.stdout').read_text();bse=(B/'xe-e32-lints-baseline-compare-root-v3/commands/baseline-free.stderr').read_text()
    r['reused_baseline_free_logs']={s:m.sha(B/('xe-e32-lints-baseline-compare-root-v3/commands/baseline-free.'+s)) for s in ['stdout','stderr']}
   else:
    be,bso,bse=run('baseline-'+mode,['/bin/bash','tools/lint/run.sh',*base_files],cwd=P,env=dict(BASE,STRATA_LINT_BUILD=str(db)),wall=1200)
   ce,cso,cse=run('current-'+mode,['/bin/bash','tools/lint/run.sh',*current_files],env=dict(BASE,STRATA_LINT_BUILD=str(W/'build'/mode)),wall=1200)
   check_misc(bso,bse);check_misc(cso,cse)
   base=diagnostics(bso+'\n'+bse,P);current=diagnostics(cso+'\n'+cse,W,line_maps)
   def existing(k):
    if k in base:return True
    # Both checks existed on these exact classes. Compare avoidable padding, preserving full diagnostics;
    # field-name order naturally changes with required new members. No tool/check/message is suppressed.
    allowed={('include/strata/core/verify.hpp',60),('src/program/generate.cpp',203)}
    rx=r"Excessive padding in '(.+)' \((\d+) padding bytes, where (\d+) is optimal\)"
    if tuple(k[:2]) not in allowed or not k[3].endswith('[clang-analyzer-optin.performance.Padding]'):return False
    cur=re.match(rx,k[3])
    if not cur:return False
    for old in base:
     if old[:3]!=k[:3]:continue
     prev=re.match(rx,old[3])
     if prev and cur[1]==prev[1] and int(cur[2])-int(cur[3])<=int(prev[2])-int(prev[3]):
      r.setdefault('source_proven_existing_padding',[]).append(dict(path=k[0],baseline_line=k[1],baseline_message=old[3],current_message=k[3],baseline_avoidable=int(prev[2])-int(prev[3]),current_avoidable=int(cur[2])-int(cur[3]),scope=mode));return True
    return False
   new=[list(k) for k in current if not existing(k)];r.setdefault('comparisons',{})[mode]=dict(baseline_exit_code=be['exit_code'],current_exit_code=ce['exit_code'],baseline_diagnostics=[list(v) for v in base],current_diagnostics=[list(v) for v in current],new_findings=new)
   if mode=='free':
    original_map=json.loads((B/'xe-e32-lints-diagnostic-map-v1.json').read_text())['diagnostics']
    mapped=[(v['path'],v['unchanged_base_line'] if not v['path'].startswith('.tools/') else v['line'],v['severity'],v['message']) for v in original_map if v['unchanged_base_line'] or v['path'].startswith('.tools/')]
    prior_fixed=('src/program/generate.cpp',2287,'warning',"Value stored to 'helpers' during its initialization is never read [clang-analyzer-deadcode.DeadStores]")
    r['v1_same_line_finding_resolved_by_source_fix']=list(prior_fixed)
    m.require(not any(k[:3]==prior_fixed[:3] and k[3]==prior_fixed[3] for k in current),'new source helper dead store actually resolved')
    r['unchanged_v1_findings_not_in_baseline']=[list(k) for k in mapped if k!=prior_fixed and not existing(k)]
    m.require(not r['unchanged_v1_findings_not_in_baseline'],'previous findings actually seen in pristine baseline or quantitatively same prior padding check')
    def cmake_findings(text,root,lm=None):
     result=[]
     for ln,msg in re.findall(r'^CMakeLists.txt:(\d+)(?:,\d+)?: (.+)$',text,re.M):
      n=int(ln);result.append((lm.get(n) if lm is not None else n,msg))
     return result
    reused_current=(B/'xe-e32-lints-baseline-compare-root-v3/commands/current-free.stdout').read_text()
    bc=cmake_findings(bso,P);cc=cmake_findings(reused_current,W,line_maps['CMakeLists.txt']);r['cmake_baseline_findings']=bc;r['cmake_current_findings']=cc;m.require(all(v in bc for v in cc),'no new CMake findings')
   if mode=='free':
    reused_current=reusable['comparisons']['free']['current_diagnostics']
    carried=[tuple(v) for v in reused_current if v[0] in ['src/core/mtp.cpp','tools/test_mtp_completion.cpp']]
    m.require(all(existing(k) for k in carried),'unchanged earlier current MTP/test source no new findings')
    r['reused_unchanged_current_diagnostics']=len(carried)
   m.require(not new,'no new diagnostics '+mode)
   m.require((ce['exit_code']==0) or (be['exit_code']==1 and ce['exit_code']==1 and mode=='free'),'only source-proven existing script failure allowed')
   save()
  e,so,se=run('pristine-final-status',['/usr/bin/git','status','--porcelain','--untracked-files=no'],cwd=P);m.require(e['exit_code']==0 and not so.strip(),'pristine still unchanged')
  r['source_stable']=all(m.sha(W/p)==h for p,h in r['source_pins'].items());m.require(r['source_stable'],'source stable');r['complete']=True
 except BaseException as e:r['error']=repr(e)
 finally:r['active']=owner.active is not None;r['passed']=bool(r['complete'] and not r['active'] and not r.get('error'));r['finished_utc']=m.utc();save()
 print(json.dumps({k:r.get(k) for k in ['passed','error','unchanged_v1_findings_not_in_baseline']}));raise SystemExit(0 if r['passed'] else 1)
