#!/usr/bin/env python3
"""Main-only private R787 component build/first traced smoke; never product edits."""
from pathlib import Path
import argparse,datetime,difflib,fcntl,hashlib,importlib.util,json,re,shlex,time
A=Path(__file__).resolve().parent;P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
Q=A/'own-native16-attention-v1'
FAULT=re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
def require(x,s):
 if not x:raise RuntimeError(s)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(Path(p).read_text())
def main():
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['build','smoke']);parser.add_argument('--build-record',type=Path);parser.add_argument('--build-sha256');opts=parser.parse_args()
 mf=load(A/'own-native16-paired-gemm-bench-v1/manifest.json');ref=load(A/'own-native16-prefill-build-v2.json')
 helper=A/'own-native16-paired-gemm-bench-v1/main_controller.py';sp=importlib.util.spec_from_file_location('r803_probe_helpers',helper);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
 require(sha(helper)==mf['package_files']['main_controller.py']['sha256'],'reviewed R803 helper pin')
 ownpath=Path(mf['direct_owner']['path']);require(sha(ownpath)==mf['direct_owner']['sha256'],'Owner pin')
 sp=importlib.util.spec_from_file_location('r787_main_owner',ownpath);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 root=B/('main-r787-native16-attention-'+opts.mode+'-v1');target=A/('own-native16-attention-'+opts.mode+'-v1.json')
 require(not root.exists() and not target.exists(),'fresh roots only')
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);root.mkdir(mode=0o700)
  rec=dict(schema='strata-r787-private-attention-probe-v1',mode=opts.mode,passed=False,complete=False,active=False,boot_id=h.boot(),commands=[],GPU_executed=False,scope='component only; oldFP32 prefix unchanged; no quality/performance/full262144/adoption',private_root=str(root),source_pins=ref['source_pins'],references={},fault_boundaries=[],retention=dict(owner='Main',consumer='R818 component comparison and actual allocation-extent guard qualification',review_at='R818 component closure; by 2026-10-16T23:00:00Z',rawtext_cap=64<<20,rawtensor_cap=1<<20))
  def persist():
   rec['active']=owner.active is not None;(root/'record.json').write_text(json.dumps(rec,indent=2)+'\n')
  owner=m.Owner(root/'owned',persist);rec['commands']=owner.commands
  rec['controller_sha256']=sha(__file__)
  rec['input_pins']={str(p):sha(p) for p in [Q/'files/src/kernels/sycl/prompt_attention.cpp',Q/'files/tests/sycl/prompt_native16.cpp',Q/'files/include/strata/kernels/qsa_prompt_attn.hpp',A/'own-native16-paired-gemm-bench-v1/manifest.json',helper,A/'own-native16-prefill-build-v2.json']}
  cursor=None;workflow=False;deadline=time.monotonic()+600
  def pins():
   require(h.boot()==rec['boot_id']==ref['boot_id'],'boot unchanged')
   require(sha(__file__)==rec['controller_sha256'],'controller changed')
   for path,value in rec['input_pins'].items():require(sha(path)==value,'input changed '+path)
   for name,value in ref['source_pins'].items():require(sha(P/name)==value,'qualified source changed '+name)
   for path,pin in mf['external_pins'].items():h.pinned(path,pin)
   require(sha(ownpath)==mf['direct_owner']['sha256'],'Owner changed')
   for path,value in rec.get('private_pins',{}).items():require(sha(path)==value,'private input changed '+path)
  def run(label,args,env,building=False,native=False):
   pins();wall=min(300 if building else 180 if native else 20,deadline-time.monotonic());require(wall>0,'whole deadline')
   entry,out,err=owner.run(label,list(map(str,args)),env,root,wall=wall,cpu=1200 if building else 180,rss_cap=(32 if building else 8)<<30,text_cap=64<<20,total_cap=512<<20,file_cap=64<<20)
   pins();require(m.closed(entry) and entry.get('exit_code')==0,label+' owned command failed');return out,err
  hostenv=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1')
  def dumpguard():
   d=Path('/sys/bus/pci/devices/0000:05:00.0');require((d/'driver').resolve().name=='xe' and not (d/'devcoredump').exists() and not list(Path('/sys/class/devcoredump').glob('devcd*')),'xe/dump guard')
  def journal(label):
   nonlocal cursor
   argv=['/usr/bin/journalctl','-k','-b','--no-pager','-o','json','--show-cursor']+(['--after-cursor',cursor] if cursor else ['-n','0'])
   out,err=run(label,argv,hostenv);text=out.read_text();tail=re.search(r'^-- cursor: (\S+)\s*$',text,re.M)
   entries=[json.loads(s) for s in text.splitlines() if s and not s.startswith('-- cursor: ') and s!='-- No entries --'];require(tail is not None or(cursor is not None and not entries),'journal cursor')
   faults=[v for v in entries if FAULT.search(str(v.get('MESSAGE','')))];rec['fault_boundaries'].append(dict(after=cursor,end=tail[1] if tail else cursor,entries=len(entries),faults=faults,filter=FAULT.pattern));cursor=tail[1] if tail else cursor;persist();require(not faults,'fresh GPU fault');dumpguard()
  try:
   pins();dumpguard()
   # Latest hardware job is a closed CPU-side numerical rejection, never used as a correctness PASS.
   prior=A/'own-native16-onednn-check-v3.json';old=load(prior);require(old['complete'] and not old['active'] and old['ownership_closed'] and old['boot_id']==rec['boot_id'] and not old['fresh_faults'] and not old['driver_dump_exists'],'prior failed job closure')
   for c in old['commands']:
    require(m.closed(c),'prior ownership closure')
    for identity in c['owners']:
     now=m.identity(identity['pid']);require(now is None or now['start_ticks']!=identity['start_ticks'],'prior PID/start alive')
   rec['references'][prior.name]=dict(sha256=sha(prior),original_status='FAIL numerical corner; unchanged; unrelated component, no GPU fault')
   for name in ['own-native16-prefill-build-v2.json','own-native16-gemm-correctness-v2.json']:
    r=load(A/name);require(r['passed'] and r['complete'] and not r['active'],'qualification reference');rec['references'][name]=dict(sha256=sha(A/name),status='PASS unchanged')
    for command in r['commands']:
     require(m.closed(command) and command.get('exit_code')==0,'qualification ownership')
     for identity in command['owners']:
      now=m.identity(identity['pid']);require(now is None or now['start_ticks']!=identity['start_ticks'],'qualification identity alive')
   h.no_model_owners(mf['known_model_files']);journal('journal-before')
   env=dict(mf['build_environment'] if opts.mode=='build' else mf['measurement_environment']);(root/'cache').mkdir();(root/'tmp').mkdir();env.update(XDG_CACHE_HOME=str(root/'cache'),TMPDIR=str(root/'tmp'))
   if opts.mode=='build':
    source=(Q/'files/src/kernels/sycl/prompt_attention.cpp').read_text();require(source.startswith((P/'src/kernels/sycl/prompt_attention.cpp').read_text()),'old FP32 prefix unchanged')
    start=source.index('void prompt_bf16_registered(');end=source.index('void launch_prompt_bf16(',start)
    source=source[:start]+'void prompt_bf16_registered(sycl::queue &queue, Span span) {\n  ::strata::sycl_backend::validate_device_span(span.pointer, span.bytes, queue);\n}\n'+source[end:]
    fixture=(Q/'files/tests/sycl/prompt_native16.cpp').read_text()
    anchor='template <class T> struct Buffer {';require(fixture.count(anchor)==1,'fixture anchor')
    fixture=fixture.replace(anchor,'void private_checked_copy(void *dst, const void *src, size_t bytes) {\n  try { runtime->wait(runtime->compute().memcpy(dst,src,bytes)); }\n  catch (...) {\n    try { runtime->wait(); }\n    catch (...) { std::cerr << "private attention copy drain failed\\n"; std::_Exit(86); }\n    throw;\n  }\n}\ntemplate <class T> struct Buffer {')
    for original,replacement in [('runtime->wait(runtime->compute().memcpy(data(), values.data(), allocation.size()));','private_checked_copy(data(),values.data(),allocation.size());'),('runtime->wait(runtime->compute().memcpy(values.data(), data(), allocation.size()));','private_checked_copy(values.data(),data(),allocation.size());')]:
     require(fixture.count(original)==1,'fixture copy anchor');fixture=fixture.replace(original,replacement)
    inc=root/'include/strata/kernels';inc.mkdir(parents=True);header=inc/'qsa_prompt_attn.hpp';header.write_bytes((Q/'files/include/strata/kernels/qsa_prompt_attn.hpp').read_bytes())
    core=root/'prompt_attention.cpp';test=root/'prompt_native16.cpp';core.write_text(source);test.write_text(fixture)
    rec['derivation']=dict(original_core_sha256=sha(Q/'files/src/kernels/sycl/prompt_attention.cpp'),original_fixture_sha256=sha(Q/'files/tests/sycl/prompt_native16.cpp'),changes=['registered actual Device allocation extents/context instead of endpoint-only membership; private qualification narrowed to Device allocations','checked drain before copy host vector can unwind; mathematical oracle and kernel unchanged'],source_diff=''.join(difflib.unified_diff((Q/'files/src/kernels/sycl/prompt_attention.cpp').read_text().splitlines(True),source.splitlines(True))),fixture_diff=''.join(difflib.unified_diff((Q/'files/tests/sycl/prompt_native16.cpp').read_text().splitlines(True),fixture.splitlines(True))))
    rec['private_pins']={str(p):sha(p) for p in [core,test,header]}
    argv=shlex.split(mf['original_v2_compile_entry']['command']);argv=argv[:argv.index('-o')];argv=[a for a in argv if not a.startswith('-DSTRATA_VERSION=')]
    argv[1:1]=['-I'+str(root/'include'),'-I'+str(P/'src/kernels/sycl')]
    objects=[]
    for label,src in [('core',core),('fixture',test)]:
     obj=root/(label+'.o');run('compile-'+label,argv+['-c',src,'-o',obj],env,building=True);objects.append(obj);rec['private_pins'][str(obj)]=sha(obj)
    exe=root/'sycl_prompt_native16_test';run('link',[mf['compiler'],'-O3','-fsycl','-fsycl-id-queries-range=size_t','-fsycl-allow-device-image-dependencies',*objects,*mf['linked_file_realpaths'],'-Wl,-rpath,/opt/intel/oneapi/mkl/2026.1/lib','-lm','-ldl','-lpthread','-o',exe],env,building=True)
    rec.update(binary=str(exe),binary_sha256=sha(exe));rec['private_pins'][str(exe)]=sha(exe)
   else:
    require(opts.build_record and sha(opts.build_record)==opts.build_sha256,'explicit build receipt SHA');built=load(opts.build_record);require(built['passed'] and built['complete'] and not built['active'] and built['mode']=='build','closed passing build')
    for command in built['commands']:
     require(m.closed(command) and command.get('exit_code')==0,'private build closure')
     for identity in command['owners']:
      now=m.identity(identity['pid']);require(now is None or now['start_ticks']!=identity['start_ticks'],'build identity alive')
    rec['private_pins']=built['private_pins'];exe=Path(built['binary']);require(sha(exe)==built['binary_sha256'],'binary pin');rec.update(binary=str(exe),binary_sha256=sha(exe),build_receipt=dict(path=str(opts.build_record),sha256=opts.build_sha256))
    env.update(mf['first_smoke_trace_environment']);rec['GPU_executed']=True;dumpfile=root/'selected-bf16.bin';out,err=run('attention-smoke',[exe,'--smoke','--dump',dumpfile],env,native=True)
    text=out.read_text();require('fixture=PASS model_quality=UNRUN whole_prefill=UNRUN physical262144=UNRUN faults=EXTERNAL_REQUIRED' in text,'fixture result');require('urEnqueueKernelLaunch' in err.read_text(),'flushed UR trace');require(dumpfile.is_file() and 0<dumpfile.stat().st_size<=1<<20,'bounded tensor');rec['stdout_original']=text;rec['capture']=dict(path=str(dumpfile),bytes=dumpfile.stat().st_size,sha256=sha(dumpfile));rec['stderr_sha256']=sha(err)
   journal('journal-after');pins();workflow=True
  except BaseException as exc:
   rec['error']=repr(exc)
  finally:
   if cursor and owner.active is None and not workflow:
    try:journal('journal-final-after-failure')
    except BaseException as exc:rec.setdefault('closure_errors',[]).append(repr(exc))
   try:pins();dumpguard()
   except BaseException as exc:rec.setdefault('closure_errors',[]).append(repr(exc))
   rec.update(active=owner.active is not None,finished_utc=m.utc(),boot_unchanged=h.boot()==rec['boot_id'],driver_dump_exists=Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump').exists(),fresh_faults=[v for w in rec['fault_boundaries'] for v in w['faults']])
   rec['ownership_closed']=not rec['active'] and all(m.closed(c) for c in owner.commands);rec['complete']=rec['ownership_closed'] and rec['boot_unchanged'] and not rec.get('closure_errors');rec['passed']=rec['complete'] and workflow and not rec.get('error') and not rec['fresh_faults'] and not rec['driver_dump_exists'];persist()
   if rec['complete']:
    with target.open('x') as f:json.dump(rec,f,indent=2);f.write('\n')
  print(json.dumps(dict(passed=rec['passed'],complete=rec['complete'],active=rec['active'],receipt=str(target) if target.exists() else None)),flush=True)
  return 0 if rec['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
