"""Inspect only current linked dot, never execute model/native binary."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,resource,subprocess
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-native-role-plan-v0141-20261010')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 binary=W/'build-native-service-calibration-v1/native_service_calibration'
 expected='3382d73a7b2db38727b3667192533e51f98165f842755331d292115ec24f18e1'
 assert sha(binary)==expected
 out=B/'native-iq2s-nt1-linked-isa-audit-v1';out.mkdir(mode=0o700)
 record=dict(passed=False,complete=False,active=True,source_binary_sha256=expected,controller_sha256=sha(__file__),tool='/usr/bin/objdump',tool_sha256=sha('/usr/bin/objdump'),commands=[],gpu_work=False,model_opened=False,binary_executed=False,performance_evidence=False)
 def limits():
  for kind,pair in ((resource.RLIMIT_AS,(512<<20,512<<20)),(resource.RLIMIT_CPU,(20,21)),(resource.RLIMIT_FSIZE,(2<<20,2<<20)),(resource.RLIMIT_CORE,(0,0)),(resource.RLIMIT_NOFILE,(128,128))):resource.setrlimit(kind,pair)
 try:
  for label,argv in [('version',['/usr/bin/objdump','--version']),('linked-dot',['/usr/bin/objdump','--no-show-raw-insn','-M','intel','-d','--disassemble=ggml_vec_dot_iq2_s_q8_K',str(binary)])]:
   so=out/(label+'.stdout');se=out/(label+'.stderr')
   with so.open('wb') as f,se.open('wb') as e:
    proc=subprocess.Popen(argv,env=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8'),stdout=f,stderr=e,preexec_fn=limits,start_new_session=True)
    try:code=proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
     os.killpg(proc.pid,9);proc.wait(timeout=5);raise
   record['commands'].append(dict(argv=argv,exit_code=code,stdout_sha256=sha(so),stderr_sha256=sha(se),stdout_bytes=so.stat().st_size));assert code==0
  asm=(out/'linked-dot.stdout').read_text();assert '<ggml_vec_dot_iq2_s_q8_K>:' in asm
  tokens={'AVX2_unsigned_signed_pair_products':'vpmaddubsw','AVX2_scaled_int32_pair_sum':'vpmaddwd','fused_YMM_accumulation':'vfmadd','float_int32_conversion':'vcvtdq2ps'}
  assert all(v in asm for v in tokens.values()) and 'ymm' in asm
  record.update(passed=True,complete=True,expected_instruction_patterns=tokens,pattern_counts={k:asm.count(v) for k,v in tokens.items()},scope='Linked current IQ2_S dot contains AVX2 integer dot and fused vector accumulation; not complete per-operation or performance proof',whole_function_sha256=sha(out/'linked-dot.stdout'))
 except BaseException as e:record['error']=type(e).__name__+': '+str(e)
 finally:
  if sha(binary)!=expected or sha(__file__)!=record['controller_sha256'] or sha('/usr/bin/objdump')!=record['tool_sha256']:record['passed']=False;record['pin_error']=True
  record['active']=False;record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
  (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record))
 if not record['passed']:raise SystemExit(1)
