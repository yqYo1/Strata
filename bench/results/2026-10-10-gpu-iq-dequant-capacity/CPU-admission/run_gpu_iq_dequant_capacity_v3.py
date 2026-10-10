"""Root-owned isolated actual-pack control. Every stage is serialized."""
from pathlib import Path
import datetime,fcntl,hashlib,json,math,os,re,subprocess,sys,types
B=Path(__file__).parent
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
P=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-prefill-service-events-20261010')
F=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-native-service-capacity-20261010')
G=Path('/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned/ggml')
C=W/'bench/results/2026-10-10-gpu-iq-dequant-capacity/source'
BASE=B/'gpu-iq-dequant-capacity-root-v2'
PACK=Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s')
PRIMARY=Path('/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf')
PARENT=B/'run_gdn_gate_factor_probe_v2.py'
REF=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-native-role-plan-v0141-20261010/build-native-service-host-v1')
STAGE=sys.argv[1];assert STAGE in ['build','parser','host-check','host-check2','qualify','timing1','timing2','timing3']
def ident(p):
 p=Path(p)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(bytes=p.stat().st_size,sha256=h)
def closed(stage):
 p=BASE/stage/'record.json';r=json.loads(p.read_text())
 assert r['passed'] and r['complete'] and not r['active']
 return p,r
def save_json(p,r):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2)+'\n');t.replace(p)
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert ident(PARENT)['sha256']=='7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f'
 assert ident(C/'gpu_iq_dequant_capacity_v1.cpp')['sha256']=='de0e86cb4c74df03b51517dce96d223ddc4149f0bb71a97c7da05dd12f9242a7'
 assert ident(C/'gpu_iq_dequant_capacity_v1_CMakeLists.txt')['sha256']=='60069c4875f6b1a7ec608e7f583dcabdf5f05e164002d0cf7b31cff8c2e8c278'
 m=types.ModuleType('root_gpu_iq_owner');s=PARENT.read_text().split('\ndef parse_probe(',1)[0]
 assert s.count('(16 << 30, 16 << 30)')==1 and s.count('(120, 121)')==1
 s=s.replace('(16 << 30, 16 << 30)','(128 << 30, 128 << 30)').replace('(120, 121)','(600, 601)')
 exec(compile(s,str(PARENT),'exec'),m.__dict__)
 BASE.mkdir(exist_ok=True);out=BASE/STAGE;assert not out.exists();out.mkdir(mode=0o700);m.W=out
 owner=m.Owner(out);clean=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8')
 r=dict(active=True,complete=False,passed=False,stage=STAGE,started_utc=m.utc(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),controller=ident(__file__),parent_owner=ident(PARENT),owner_adaptations=['AS128GiB for unchanged NativeRolePlan virtual54.8GB mapping','CPU600/601s'],limits=dict(AS_each=128<<30,RSS_session_sampled=2<<30,CPU_each=[600,601],wall_each=600,text_each=64<<20),commands=owner.commands,adopted=False,model_executed=False,scope='Isolated actual-pack generic GPU IQ dequant; host submission/completion wall, not exclusive device service or current production route',source=ident(C/'gpu_iq_dequant_capacity_v1.cpp'))
 def save():save_json(out/'record.json',r)
 owner.persist=save
 def run(label,args,env=clean,wall=600,expected=0,cap=64<<20):
  e,so,se=owner.run(label,args,env,wall=wall,text_cap=cap,file_cap=cap);save()
  assert m.completed(e) and e['exit_code']==expected,(label,e)
  return so,se
 try:
  br=json.loads((B/'prefill-service-qualification-cpu-build-v1/record.json').read_text());assert br['passed'] and not br['active']
  env=dict(br['environment']);r['build_environment']=env
  libs=[REF/'ggml/src/libggml-cpu.a',REF/'ggml/src/libggml-base.a']
  r['cached_CPU_reference_files']={str(p):ident(p) for p in libs}
  r['reference_build_receipt']=ident(B/'native-service-host-cpu-build-v1/record.json')
  r['reference_compile_commands']=ident(REF/'compile_commands.json')
  binary=BASE/'build/gpu_iq_dequant_capacity_v1'
  if STAGE=='build':
   project=BASE/'project';project.mkdir();recipe=(C/'gpu_iq_dequant_capacity_v1_CMakeLists.txt').read_text();assert recipe.count('-device intel_gpu_bmg_g21')==1;recipe=recipe.replace('-device intel_gpu_bmg_g21','-device bmg-g21');(project/'CMakeLists.txt').write_text(recipe);r['root_recipe_correction']={'reason':'SYCL architecture enum name is not an OCLOC device token; original failed normally before GPU execution','original':ident(C/'gpu_iq_dequant_capacity_v1_CMakeLists.txt'),'corrected':ident(project/'CMakeLists.txt'),'effective_OCLOC_device':'bmg-g21','primary_reference':'https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2026-0/ahead-of-time-compilation.html','source_compiler_policy_string':'Original harness names SYCL architecture enum intel_gpu_bmg_g21; actual compiler command uses OCLOC bmg-g21'}
   sources=[F/'src/artifact/native_role_plan.cpp',F/'src/kernels/cpu/expert_layout.cpp',F/'src/kernels/cpu/native_expert.cpp']
   r['host_source_pins']={str(p):ident(p) for p in sources};r['GPU_source_pin']=ident(P/'sycl/src/kernels/cuda/iq_kernels.dp.cpp')
   objects=[]
   for p in sources:
    obj=out/(p.stem+'.o');objects.append(obj)
    cmd=['/opt/intel/oneapi/compiler/2026.1/bin/icpx','-std=c++20','-O2','-fp-model=precise','-ffunction-sections','-fdata-sections','-DSTRATA_NATIVE_EXPERTS=1','-I'+str(F/'include'),'-I'+str(G/'include'),'-I'+str(G/'src'),'-I'+str(G/'src/ggml-cpu'),'-c',str(p),'-o',str(obj)]
    run('compile-host-'+p.stem,cmd,env)
   refs=';'.join(str(p) for p in objects+libs)
   run('configure',['/usr/bin/cmake','-S',str(project),'-B',str(BASE/'build'),'-DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icpx','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DHARNESS_SOURCE='+str(C/'gpu_iq_dequant_capacity_v1.cpp'),'-DHOST_REFERENCE_FILES='+refs],env)
   run('build',['/usr/bin/cmake','--build',str(BASE/'build'),'--parallel','1'],env)
   r['host_object_pins']={str(p):ident(p) for p in objects};r['binary']=dict(path=str(binary),**ident(binary));r['compile_commands']=json.loads((BASE/'build/compile_commands.json').read_text())
   so,se=run('binary-needed',['/usr/bin/readelf','-d',str(binary)],env,30);r['binary_needed']=so.read_text()
  else:
   bp,build=closed('build');r['build_receipt']=ident(bp);r['binary']=build['binary'];assert ident(binary)=={k:build['binary'][k] for k in ['bytes','sha256']}
   if STAGE=='parser':
    args=[str(binary),str(PACK),str(PRIMARY),'qualify','32','1','0']
    cases=[('missing',args[:-1]),('excess',args+['extra'])]
    for field,value in [(3,'unknown'),(4,'0'),(4,'1'),(4,'15'),(4,'17'),(4,'33'),(5,'-1'),(5,'18446744073709551616'),(5,'01'),(5,'1x'),(6,'16'),(1,'x'*4097)]:
     a=list(args);a[field]=value;cases.append(('bad-'+str(field)+'-'+str(len(cases)),a))
    r['parser_results']=[]
    for label,a in cases:
     so,se=run(label,a,env,30,1,8<<20);j=json.loads(se.read_text());assert not j['success'] and j['no_timing_rows_valid'] and not so.read_bytes();r['parser_results'].append(dict(label=label,result=j))
    for label,name,value in [('env-width','ONEAPI_DEVICE_SELECTOR','x'*4097),('preload-present','LD_PRELOAD','')]:
     e=dict(env);e[name]=value;so,se=run(label,args,e,30,1,8<<20);j=json.loads(se.read_text());assert not j['success'] and j['no_timing_rows_valid'] and not so.read_bytes();r['parser_results'].append(dict(label=label,result=j))
   elif STAGE in ['host-check','host-check2']:
    text=(C/'gpu_iq_dequant_capacity_v1.cpp').read_text()
    blocks=[text[text.index('void need('):text.index('uint64_t decimal(')],text[text.index('uint16_t rne('):text.index('[[noreturn]] void fatal(')],text[text.index('void guards('):text.index('std::vector<uint8_t> readback(')],text[text.index('void compare('):text.index('void launch(')]]
    header='#include "strata/kernels/cpu/native_expert.hpp"\n#include "ggml.h"\n#include <array>\n#include <bit>\n#include <cmath>\n#include <cstdint>\n#include <cstring>\n#include <iostream>\n#include <stdexcept>\n#include <string>\n#include <vector>\nconstexpr size_t H=2560,FF=640,GU=2*FF*H,D=H*FF,G=64;\nstruct Source{int layer,expert;};\n'
    body='''
int main(){host_edges();uint32_t seed=17;size_t values=0,rejected=0;
for(size_t i=0;i<100000;++i){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;
if(((seed>>23)&255)==255&&(seed&0x7fffff))continue;
float f=std::bit_cast<float>(seed);ggml_fp16_t h;ggml_fp32_to_fp16_row(&f,&h,1);
need(uint16_t(h)==rne(f),"random independent/host mismatch i="+std::to_string(i)+" fp32bits="+std::to_string(seed)+" GGMLbits="+std::to_string(uint16_t(h))+" RNEbits="+std::to_string(rne(f)));++values;}
for(size_t n: {GU,D}){
std::vector<uint8_t> a(2*n+2*G,0xa5);std::vector<uint16_t> ref(GU+D,0);size_t off=n==GU?0:GU;
std::memset(a.data()+G,0,2*n);guards(a,2*n);compare(a,ref,off,n,Source{47,511},"host");
for(size_t index:{size_t(0),n-1}){a[G+2*index]=1;try{compare(a,ref,off,n,Source{47,511},"host");throw std::logic_error("uncaught bit change");}catch(const std::runtime_error& e){need(std::string(e.what()).find("generic_vs_independent")!=std::string::npos,"wrong mismatch gate");++rejected;}a[G+2*index]=0;}
for(size_t index:{size_t(0),G-1,G+2*n,G+2*n+G-1}){a[index]=0;try{guards(a,2*n);throw std::logic_error("uncaught guard change");}catch(const std::runtime_error& e){need(std::string(e.what()).find("canary")!=std::string::npos,"wrong guard gate");++rejected;}a[index]=0xa5;}}
std::cout<<"{\\"passed\\":true,\\"literal_edges\\":24,\\"random_half_values\\":"<<values<<",\\"injected_changes_rejected\\":"<<rejected<<"}\\n";
}
'''
    src=out/'exact_extracted_host_contract.cpp';src.write_text(header+'\n'.join(blocks)+body);r['extracted_source']=ident(src);exe=out/'host-contract'
    run('compile-host-contract',['/opt/intel/oneapi/compiler/2026.1/bin/icpx','-std=c++20','-O2','-fp-model=precise','-I'+str(F/'include'),'-I'+str(G/'include'),str(src),str(libs[1]),'-ldl','-pthread','-o',str(exe)],env)
    so,se=run('host-contract',[str(exe)],env,120);r['host_result']=json.loads(so.read_text());assert r['host_result']['passed'] and r['host_result']['injected_changes_rejected']==12
   else:
    for prev in ['parser','host-check']:
     pp,pv=closed(prev);r[prev+'_receipt']=ident(pp)
    if STAGE.startswith('timing'):
     qp,q=closed('qualify');assert q['binary']==r['binary'];r['qualification_receipt']=ident(qp)
    env.update(UR_ADAPTERS_FORCE_LOAD='/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero_v2.so.0',UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',ONEAPI_DEVICE_SELECTOR='level_zero:gpu',SYCL_CACHE_PERSISTENT='0',EnableDirectSubmission='0',NEOReadDebugKeys='1')
    if STAGE=='qualify':
     helper=P/'sycl/tools/recover-xe.sh';r['diagnostic_helper']=ident(helper);h=types.ModuleType('root_iq_diagnostic_env');exec(compile(helper.read_text().split("<<'PY'\n",1)[1].split('\nPY',1)[0],str(helper),'exec'),h.__dict__);env=h.diagnostic_environment(env)
    r['effective_environment']=env
    assert not Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump').exists()
    census=json.loads((W/'bench/results/2026-10-10-current-native-pack-census/receipt.json').read_text());r['current_census_receipt']=ident(W/'bench/results/2026-10-10-current-native-pack-census/receipt.json')
    assert ident(PACK/'native_experts.txt')['sha256']=='d9ac2dfa3ee63c55c9c6a6db26f72da0cec0ee41733f007e5cbbbba71617aa5d'
    with PRIMARY.open('rb') as f:prefix=f.read(11024384)
    assert hashlib.sha256(prefix).hexdigest()=='d6932b90e1b001f706ee5c8a1aebbab3a8a6ae35892df2fb09af459add1140ef'
    so,se=run('journal-cursor',['/usr/bin/journalctl','-k','-n','0','--show-cursor','--no-pager'],clean,10,0,8<<20);match=re.search(r'^-- cursor: (.+)$',so.read_text(),re.M);assert match;r['kernel_cursor']=match[1]
    args=[str(binary),str(PACK),str(PRIMARY),'qualify' if STAGE=='qualify' else 'timing','32','1','0']
    e,so,se=owner.run('probe',args,env,wall=600,text_cap=64<<20,file_cap=64<<20);save()
    try:r['probe_result']=json.loads(so.read_text())
    except (ValueError,FileNotFoundError):r['probe_failure_text']=se.read_text()[-262144:]
    js,je=run('journal-interval',['/usr/bin/journalctl','-k','--after-cursor='+r['kernel_cursor'],'--no-pager','-o','short-monotonic'],clean,10,0,8<<20)
    r['kernel_filter']=r'\bxe\b|i915|GPU HANG|devcoredump';r['visible_kernel_GPU_entries']=[s for s in js.read_text().splitlines() if re.search(r['kernel_filter'],s,re.I)];r['devcoredump_after']=Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump').exists()
    assert m.completed(e) and e['exit_code']==0 and not r['visible_kernel_GPU_entries'] and not r['devcoredump_after'],('GPU execution/closure/fault gate',e)
    j=r['probe_result'];assert j['success'] and j['generic_vs_independent_all_bits'] and j['all_allocated_source_and_output_guards_checked'] and len(j['pairs'])==7
    assert j['H']==2560 and j['FF']==640 and j['NE']==512 and j['DQ']==2 and 'B570' in j['device']
    assert [(p['gu_type'],p['down_type']) for p in j['pairs']]==[(18,20),(18,42),(21,20),(21,42),(22,20),(22,42),(23,20)]
    for p in j['pairs']:
     n=1 if STAGE=='qualify' else 32;assert p['selected_distinct_experts']==n and len(set(z['expert'] for z in p['selected']))==n
     assert p['values_compared_all_selected']==n*4915200 and p['device_packed_payload_bytes_fully_checked']==n*p['packed_bytes_each']
     assert p['wrapper_launches_total']==(2 if n==1 else 2+18*n)
     if n>1:
      assert len(p['individual_samples'])==7 and p['values_compared_timed_end_slots']==7*2*4915200
      for row in p['individual_samples']:
       assert 0<row['host_submit_completion_wall_ns'] and 0<=row['host_submit_ns']<=row['host_submit_completion_wall_ns']
       assert row['logical_packed_input_bytes']==32*p['packed_bytes_each'] and row['logical_FP16_output_bytes']==32*9830400
    if STAGE=='qualify':assert j['device_edges_qualified'] and not j['timing_rows_complete'] and j['samples']==0
    else:assert j['timing_rows_complete'] and not j['device_edges_qualified'] and j['samples']==7
  r.update(passed=True,complete=True)
 except BaseException as e:r['error']=type(e).__name__+': '+str(e)
 finally:
  r.update(active=owner.active is not None,finished_utc=m.utc());r['passed']=r['passed'] and not r['active'] and all(m.completed(e) for e in owner.commands);save();print(json.dumps({k:r.get(k) for k in ['stage','active','complete','passed','error']}),flush=True)
 if not r['passed']:sys.exit(1)
