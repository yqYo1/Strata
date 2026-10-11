import argparse,fcntl,importlib.util,json,shlex,time
from pathlib import Path
EXPECTED_API_V2_PINS={'src/kernels/xe/iq_kernels.cpp': '75f2ac1dfdb337c424bba19e62c34159a1bde547a776b9b5d6cf7975dfc29d50', 'include/strata/kernels/iq_kernels.hpp': '61513ccb71831d1065bd25829541c508adf3ae58e1e3df6d3bc0c9681cb54f6d', 'src/prefill/prefill.cpp': '1b30e3de22b30f4eb2160756c9fdde623abe87a2360fe029f0767be33aa9d664'}
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
LLVM=W/'.tools/intel-llvm-v711/install';O=B/'xe-llvm-v711-source-v5/direct_owner.py'
s=importlib.util.spec_from_file_location('o',O);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.require(m.sha(O)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f','qualified owner')
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['free','contrib-llvm'],required=True);a=p.parse_args()
ENV=dict(PATH=str(LLVM/'bin')+':/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',LD_LIBRARY_PATH=str(LLVM/'lib'))
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=B/('xe-native-dequant-group-component-build-root-v2-'+a.mode);m.require(not out.exists(),'immutable build');out.mkdir()
 r=dict(active=True,complete=False,passed=False,commands=[],started_utc=m.utc(),controller_sha256=m.sha(__file__),gpu_executed=False,model_executed=False,mode=a.mode,source_base='d73288cba04f728dadfd69e9797729c1a10af3ab',api_version=2,compiler_runtime='source intel LLVM v7.1.1',scope='Real production library closure; exact target compile/device-link settings; no extracted decoders or Runtime stubs',raw_log_budget_bytes=64<<20)
 def save():(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
 owner=m.Owner(out/'commands',save);r['commands']=owner.commands;save()
 def run(label,args,wall=300):
  e,so,se=owner.run(label,args,ENV,W/'build'/a.mode,wall=wall,cpu=wall*4,text_cap=64<<20,total_cap=64<<20,rss_cap=40<<30)
  m.require(m.closed(e) and e['exit_code']==0,'normal closed '+label);return so.read_text()
 try:
  proof=B/('xe-e32-'+a.mode+'-full-build-root-v18/record.json');v=json.loads(proof.read_text());m.require(v['passed'] and v['complete'] and not v['active'] and all(m.closed(c) for c in v['commands']),'closed release build')
  m.require(all(v['source_pins'].get(f)==h for f,h in EXPECTED_API_V2_PINS.items()),'fresh exact constant-selector APIv2 release pins')
  m.require(all(m.sha(W/f)==h for f,h in v['source_pins'].items()),'matching production source');r['release_proof_sha256']=m.sha(proof);r['source_pins']=v['source_pins']
  src=B/'xe-native-dequant-group-component-source-v2/native_dequant_group_component.cpp';m.require(m.sha(src)=='f9168793027bfd3aa62fd8c8ff607b98e9b25d9ece54fe31d54d1945439b89a7','reviewed selector source');r['probe_source']=dict(path=str(src),sha256=m.sha(src))
  build=W/'build'/a.mode;db=build/'compile_commands.json';rows=json.loads(db.read_text());row=next(x for x in rows if x['file'].endswith('/src/kernels/bf16_gemv_parity.cpp'))
  args=shlex.split(row['command']);m.require(args[0]==str(LLVM/'bin/clang++') and args[-2]=='-c' and '-fsycl' in args and '-Werror' in args,'actual source LLVM compile flags')
  obj=out/'probe.o';args[-1]=str(src);args[args.index('-o')+1]=str(obj);args.insert(1,'-I'+str(W/'third_party/main/ggml'))
  r['compile_db_sha256']=m.sha(db);r['compile_arguments']=args;run('compile-probe',args)
  recipes=run('actual-target-device-link-recipe',['/usr/bin/ninja','-C',str(build),'-t','commands','bf16_gemv_parity'],60).splitlines();link=shlex.split(recipes[-1]);m.require(link[:2]==[':','&&'] and link[-2:]==['&&',':'],'exact simple CMake linker recipe');link=link[2:-2]
  m.require(link[0]==str(LLVM/'bin/clang++') and '-fsycl' in link and 'libstrata_kernels.a' in link and 'libstrata_core.a' in link,'actual production archive closure')
  at=link.index('CMakeFiles/bf16_gemv_parity.dir/src/kernels/bf16_gemv_parity.cpp.o');link[at]=str(obj);binary=out/'native-dequant-group-component';link[link.index('-o')+1]=str(binary)
  r['archive_pins']={x:m.sha(build/x) for x in link if x.endswith('.a')};r['link_arguments']=link;run('link-real-production-closure',link)
  r['binary']=dict(path=str(binary),sha256=m.sha(binary),bytes=binary.stat().st_size)
  e,so,se=owner.run('invalid-cli-before-runtime',[str(binary),'--invalid'],ENV,W,wall=15,cpu=15,text_cap=1<<20,total_cap=64<<20,rss_cap=2<<30);m.require(m.closed(e) and e['exit_code']==2 and se.read_text()=='Use --smoke-host, --full-host, --smoke-tags or --full-tags\n','bad CLI before Runtime')
  r['source_stable']=all(m.sha(W/f)==h for f,h in r['source_pins'].items()) and m.sha(src)==r['probe_source']['sha256'];m.require(r['source_stable'],'source stable');m.require(all(m.sha(build/x)==h for x,h in r['archive_pins'].items()),'archives stable');r['complete']=True
 except BaseException as exc:r['error']=repr(exc)
 finally:r['active']=owner.active is not None;r['passed']=bool(r['complete'] and not r['active'] and not r.get('error'));r['finished_utc']=m.utc();save()
 print(json.dumps({k:r.get(k) for k in ['mode','passed','error','binary']}));raise SystemExit(0 if r['passed'] else 1)
