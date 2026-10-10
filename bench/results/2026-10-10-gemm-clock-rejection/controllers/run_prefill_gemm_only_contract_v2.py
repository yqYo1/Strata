"""Main-owned real-header CPU flag/repeated-M qualification; no SYCL/GPU."""
from pathlib import Path
import ast,fcntl,hashlib,json,os,types
B=Path(__file__).parent;D=B/'prefill-gemm-service-only-v1';OUT=B/'prefill-gemm-only-contract-root-v2'
PARENT=B/'run_gdn_gate_factor_probe_v2.py'
HARNESS=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-prefill-service-qualification-20261010/sycl/tools/test_prefill_service_ledger.py')
def ident(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(bytes=p.stat().st_size,sha256=h)
def strict_pairs(pairs):
 d={}
 for k,v in pairs:
  if k in d:raise ValueError('duplicate JSON key '+k)
  d[k]=v
 return d
def parse(s,prefix):return [json.loads(l[len(prefix):],object_pairs_hook=strict_pairs) for l in s.splitlines() if l.startswith(prefix)]
def save_json(p,j):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(j,indent=2)+'\n');q.replace(p)
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert ident(PARENT)['sha256']=='7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f'
 caller=D/'prefill.cpp';header=D/'include/strata/prefill_service_ledger.hpp'
 assert ident(caller)['sha256']=='1cf7e911a555e8611c1368a4d18d4ba81c95027bb8de77f89ad65ac64b05fc3c'
 assert ident(header)['sha256']=='7d473f68dcfdb48660c50cff0e75c6191e5b3339ed95901a1aa4286f38bbff06'
 text=caller.read_text();start=text.index('    const char* const gemm_service_env =');end=text.index('    services.admit(cs);',start);flag=text[start:end]
 baseline=(B/'prefill-route-census-v3/prefill.cpp').read_text()
 old='    ExpertServiceLedger services{std::getenv("STRATA_PREFILL_SERVICE_TIMING") != nullptr, "expert_gemm"};\n'
 assert text[:start]+old+text[end:]==baseline, 'Only compute flag admission may change caller; all copy/queue conditions unchanged'
 tree=ast.parse(HARNESS.read_text());cpp=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CPP' for t in n.targets))
 cpp=cpp.replace('#include <string>','#include <string>\n#include <cstring>')
 cpp=cpp.replace(' const std::string mode = argv[1];',' (void)argc;\n const std::string mode = argv[1];')
 cpp=cpp.replace(' Ledger ledger(mode!="off","test");',flag+' auto& ledger=services;').replace(' Ledger ledger(', ' Ledger ledger(')
 cpp=cpp.replace('    ExpertServiceLedger services{','    Ledger services{')
 assert 'gemm_service_env' in cpp and 'Ledger ledger(mode' not in cpp
 assert not OUT.exists();OUT.mkdir(mode=0o700);src=OUT/'contract.cpp';src.write_text(cpp)
 m=types.ModuleType('root_gemm_only_cpu_owner');exec(compile(PARENT.read_text().split('\ndef parse_probe(',1)[0],str(PARENT),'exec'),m.__dict__);m.W=OUT;owner=m.Owner(OUT)
 allowed={'PATH','LANG','LANGUAGE','LC_ALL','LC_CTYPE','LD_LIBRARY_PATH','CPATH','CPLUS_INCLUDE_PATH','LIBRARY_PATH','TMPDIR','TMP','TEMP','CXX','OMP_NUM_THREADS','MKL_NUM_THREADS'}
 env={k:v for k,v in os.environ.items() if k in allowed};env['LC_ALL']='C'
 for k in ['STRATA_PREFILL_SERVICE_TIMING','STRATA_PREFILL_TRANSFER_TIMING','STRATA_PREFILL_GEMM_SERVICE_TIMING']:env.pop(k,None)
 cases=[('absent',{},'off','off'),('zero',{'STRATA_PREFILL_GEMM_SERVICE_TIMING':'0'},'off','off'),('one',{'STRATA_PREFILL_GEMM_SERVICE_TIMING':'1'},'normal','valid'),('empty',{'STRATA_PREFILL_GEMM_SERVICE_TIMING':''},'off','invalid'),('word',{'STRATA_PREFILL_GEMM_SERVICE_TIMING':'true'},'off','invalid'),('two',{'STRATA_PREFILL_GEMM_SERVICE_TIMING':'2'},'off','invalid'),('bad_old_full',{'STRATA_PREFILL_GEMM_SERVICE_TIMING':'yes','STRATA_PREFILL_SERVICE_TIMING':'1'},'off','invalid'),('old_full',{'STRATA_PREFILL_SERVICE_TIMING':'1'},'normal','valid'),('old_full_zero_presence',{'STRATA_PREFILL_SERVICE_TIMING':'0','STRATA_PREFILL_GEMM_SERVICE_TIMING':'0'},'normal','valid'),('transfer_only',{'STRATA_PREFILL_TRANSFER_TIMING':'1'},'off','off'),('equal',{'STRATA_PREFILL_GEMM_SERVICE_TIMING':'1'},'equal','valid'),('pipeline',{'STRATA_PREFILL_GEMM_SERVICE_TIMING':'1'},'pipeline','valid'),('full_chunk',{'STRATA_PREFILL_GEMM_SERVICE_TIMING':'1'},'full_chunk','valid')]
 r=dict(active=True,complete=False,passed=False,gpu_tested=False,source=ident(caller),header=ident(header),harness=ident(HARNESS),actual_flag_code=flag,compile_source=ident(src),controller=ident(__file__),parent_owner=ident(PARENT),all_copy_factory_timer_and_stager_source_unchanged=True,environment_base=env,commands=owner.commands,results=[],scope='Exact extracted caller flag plus real private header, strict unique JSON keys and repeated M rows; CPU stubs, not runtime/model proof')
 def save():save_json(OUT/'record.json',r)
 owner.persist=save
 try:
  cmd=['/usr/bin/c++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-I'+str(D/'include'),str(src),'-o',str(OUT/'contract')];r['effective_compile']=cmd
  e,so,se=owner.run('compile',cmd,env,wall=30,file_cap=8<<20);assert m.completed(e) and e['exit_code']==0, e
  for label,flags,mode,expected in cases:
   caseenv=dict(env);caseenv.update(flags)
   e,so,se=owner.run(label,[str(OUT/'contract'),mode],caseenv,wall=15,file_cap=8<<20);assert m.completed(e) and e['exit_code']==0,(label,e)
   stderr=se.read_text();stdout=so.read_text();receipts=parse(stderr,'strata prefill service validity: ');ledgers=parse(stderr,'strata prefill returned-event ledger: ')
   if expected=='off':assert not receipts and not ledgers and 'OPS 0' in stdout,(label,stderr,stdout)
   elif expected=='invalid':
    assert len(receipts)==1 and receipts[0]['status']=='invalid' and receipts[0]['reason']=='gemm_service_env_must_be_exact_0_or_1' and not ledgers and 'OPS 0' in stdout,(label,stderr,stdout)
   else:
    count=49152 if mode=='full_chunk' else 4 if mode=='normal' else 2
    assert len(receipts)==len(ledgers)==1 and receipts[0]['status']=='valid'
    q=receipts[0];g=ledgers[0]
    assert all(q[k]==count for k in ['attempts','submitted','retained','folded']) and q['query_attempts']==q['query_successes']==3*count
    assert q['expected_registered']==q['completed_role_pairs']==count//2 and q['expected_pending']==q['pending']==q['dropped']==q['query_failures']==0
    assert g['calls']==count and sum(s['calls'] for s in g['shapes'])==count and sum(c['calls'] for c in g['coverage'])==count
    for s in g['shapes']:assert s['M']>0 and s['rows']==s['M']*s['calls'],(label,s)
    if mode=='normal':assert len(g['shapes'])==2 and all(s['M']==7 and s['rows']==14 and s['calls']==2 for s in g['shapes'])
    if mode=='pipeline':assert g['returned_event_ms']==81/1e6
   r['results'].append(dict(case=label,flags=flags,expected=expected,passed=True,stdout_identity=ident(so),stderr_identity=ident(se),receipt=receipts,ledger=ledgers));save()
  r.update(complete=True,passed=True,binary=ident(OUT/'contract'))
 except BaseException as e:r['error']=type(e).__name__+': '+str(e)
 finally:
  r.update(active=owner.active is not None);r['passed']=r['passed'] and not r['active'] and all(m.completed(e) for e in owner.commands);save();print(json.dumps({k:r.get(k) for k in ['active','complete','passed','error']}),flush=True)
 if not r['passed']:raise SystemExit(1)
