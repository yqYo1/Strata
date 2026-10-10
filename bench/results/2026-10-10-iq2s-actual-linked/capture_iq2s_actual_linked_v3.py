from pathlib import Path
import datetime,fcntl,hashlib,json,os,resource,subprocess,time
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007');W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-iq2s-nt1-index-spread-20261010');D=B/'iq2s-actual-linked-capture-v3';receipt=B/'iq2s-actual-cohort-cpu-build-v1/record.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);q=json.loads(receipt.read_text());assert q['passed'] and q['complete'] and not q['active'] and not q['cleanup'] and not q['survivors'];binary=Path(q['binary']['path']);assert sha(binary)==q['binary']['sha256'];assert not D.exists();D.mkdir(mode=0o700);started=time.monotonic()
 r={'active':True,'complete':False,'passed':False,'scope':'Root CPU offline exact new linked symbol capture; no executable/model/GPU run','controller_sha256':sha(__file__),'build_receipt_sha256':sha(receipt),'binary':q['binary'],'source_commit':subprocess.check_output(['git','-C',str(W),'rev-parse','HEAD'],text=True).strip(),'commands':[],'cleanup':[],'survivors':[],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu_work_submitted':False,'model_opened':False,'performance_eligible':False,'adopted':False,'symbol_inventory':{}}
 def save():
  t=D/'record.json.tmp';t.write_text(json.dumps(r,indent=2)+'\n');t.replace(D/'record.json')
 def run(label,args):
  so=D/(label+'.txt');se=D/(label+'.stderr');cmd={'argv':args,'label':label};r['commands'].append(cmd);save()
  def limits():
   resource.setrlimit(resource.RLIMIT_AS,(512<<20,512<<20));resource.setrlimit(resource.RLIMIT_CPU,(30,31));resource.setrlimit(resource.RLIMIT_FSIZE,(2<<20,2<<20));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
  with so.open('wb') as o,se.open('wb') as e:
   result=subprocess.run(args,stdout=o,stderr=e,cwd=W,env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','LC_ALL':'C.UTF-8'},timeout=45,preexec_fn=limits)
  cmd.update(exit_code=result.returncode,stdout_sha256=sha(so),stdout_bytes=so.stat().st_size,stderr_sha256=sha(se));assert result.returncode==0,label
  assert time.monotonic()-started<120;assert sum(p.stat().st_size for p in D.iterdir() if p.is_file())<4<<20;save();return so.read_text()
 save()
 try:
  nm=run('nm-mangled',['/usr/bin/nm','--defined-only',str(binary)]);run('nm-demangled',['/usr/bin/nm','-C','--defined-only',str(binary)])
  keys={'custom-q8k':'q8k_quant_avx2','native-quant-act':'native_quant_act','direct-control':'direct_control','register-index':'index_candidate','trait-iq2s':'ggml_vec_dot_iq2_s_q8_K','reference-q8k':'quantize_row_q8_K_ref','trait-q8k-wrapper':'quantize_row_q8_K','cpu-avx2-gate':'cpu_avx2_ok'}
  symbols=[line.split(maxsplit=2)[2] for line in nm.splitlines() if len(line.split(maxsplit=2))==3 and line.split(maxsplit=2)[1] in ('t','T')]
  for label,needle in keys.items():
   found=[v for v in symbols if needle in v and not v.endswith('.cold') and not v.startswith('_ZZ')]
   if label in ('trait-q8k-wrapper','reference-q8k','trait-iq2s'):found=[v for v in found if v==needle]
   assert len(found)==1,(label,found);symbol=found[0]
   out=run(label,['/usr/bin/objdump','-d','--no-show-raw-insn','--disassemble='+symbol,str(binary)])
   assert '>:' in out and len(out.splitlines())>15,(label,'empty capture')
   r['symbol_inventory'][label]={'mangled':symbol,'path':str(D/(label+'.txt')),'sha256':sha(D/(label+'.txt')),'bytes':(D/(label+'.txt')).stat().st_size,'lines':len(out.splitlines())}
  assert sha(binary)==q['binary']['sha256'];assert sha(receipt)==r['build_receipt_sha256'];assert sha(__file__)==r['controller_sha256'];r.update(complete=True,passed=True,exit_pin_gate_passed=True)
 except BaseException as e:r['error']=repr(e)
 finally:r.update(active=False,elapsed_seconds=time.monotonic()-started,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
 print(json.dumps({k:r.get(k) for k in ['passed','error','elapsed_seconds','symbol_inventory']}))
 if not r['passed']:raise SystemExit(1)
