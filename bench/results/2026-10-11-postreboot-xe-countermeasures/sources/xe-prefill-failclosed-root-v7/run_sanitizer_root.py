import collections,fcntl,importlib.util,json,re,shutil
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007');W=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-b570-prefill-publication-20261011');HERE=Path(__file__).resolve().parent
p=B/'xestrata-clean-64k-comparison-v4/direct_owner.py';s=importlib.util.spec_from_file_location('owner',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.require(m.sha(p)=='8b40cc573b9abd069f8386deb6af65f0a55f7992bcda2a6a62b51a5c9a47b686','owner pin')
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out=B/'xe-prefill-sanitizer-root-v1';m.require(not out.exists(),'new output');out.mkdir(mode=0o700)
 r=dict(active=True,complete=False,passed=False,gpu_executed=False,model_executed=False,full_SYCL_sanitizer_suite_verified=False,free_SYCL_build_verified=False,started_utc=m.utc(),controller_sha256=m.sha(__file__))
 def save():(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
 own=m.Owner(out/'commands',save);r['commands']=own.commands;env=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
 def run(label,args,cwd=out,wall=60):
  e,so,se=own.run(label,args,env,cwd,wall=wall,text_cap=8<<20,total_cap=64<<20,rss_cap=6<<30,cpu=wall)
  m.require(m.closed(e) and e['exit_code']==0,'normal '+label);m.require(not re.search(r'ERROR: AddressSanitizer|runtime error:|LeakSanitizer: detected',so.read_text()+se.read_text()),'no sanitizer finding '+label);return so
 save()
 try:
  cpu=B/'xe-prefill-failclosed-cpu-root-v7/record.json';j=json.loads(cpu.read_text());m.require(j['passed'] and not j['active'] and j['complete'],'production CPU gate')
  lint=B/'xe-prefill-lint-root-v3/record.json';q=json.loads(lint.read_text());m.require(q['passed'] and not q['active'],'lint gate')
  def findings(path):return collections.Counter(re.sub(r'^.*?:\d+:\d+: warning: ','',l) for l in path.read_text().splitlines() if ': warning:' in l)
  old=findings(lint.parent/'commands/baseline-tidy.stdout');new=findings(lint.parent/'commands/lint.stdout');m.require(old==new,'no new static-analysis findings')
  r['lint_qualification']=dict(path=str(lint),sha256=m.sha(lint),new_findings=0,existing_warning_count=sum(old.values()),existing_warning_messages=list(old.elements()))
  for f,sha in j['source_sha256'].items():m.require(m.sha(W/f)==sha,'source pin')
  r['source_sha256']=j['source_sha256'];r['cpu_qualification_sha256']=m.sha(cpu)
  source=out/'source';source.mkdir();build=out/'build'
  for name in ['stager-exact.inc','stager-guard-exact.inc','hostvec-exact.inc']:shutil.copyfile(cpu.parent/name,source/name)
  shutil.copyfile(HERE/'qualify_stager.cpp',source/'qualify_stager.cpp')
  (source/'expect_terminal.py').write_text('import subprocess,sys\nr=subprocess.run([sys.argv[1],"--cleanup-failure"],capture_output=True,text=True,timeout=10)\nassert r.returncode==74 and r.stdout=="BEFORE_UNWIND\\n" and "GPU completion unconfirmed" in r.stderr and "RELEASED" not in r.stdout\nassert "ERROR: AddressSanitizer" not in r.stderr and "runtime error:" not in r.stderr\nprint("expected fail-stop: exit 74 before buffer-release destructor")\n')
  cm='cmake_minimum_required(VERSION 3.24)\nproject(XePrefillHostSafety LANGUAGES CXX)\nset(CMAKE_CXX_STANDARD 20)\nenable_testing()\nset(STRATA_SANITIZE "address,undefined" CACHE STRING "Host sanitizers")\nadd_compile_options(-Wall -Wextra -fsanitize=${STRATA_SANITIZE} -fno-omit-frame-pointer)\nadd_link_options(-fsanitize=${STRATA_SANITIZE})\n'
  cm+=f'include_directories("{W}/include" "{source}")\nadd_executable(publication "{W}/tools/test_prefill_publication.cpp")\nadd_executable(stager qualify_stager.cpp)\ntarget_link_libraries(publication PRIVATE pthread)\ntarget_link_libraries(stager PRIVATE pthread)\nadd_test(NAME production_publication COMMAND publication)\nadd_test(NAME actual_stager_lifecycle COMMAND stager)\nadd_test(NAME terminal_before_release COMMAND /usr/bin/python3 "{source}/expect_terminal.py" $<TARGET_FILE:stager>)\nset_tests_properties(production_publication actual_stager_lifecycle terminal_before_release PROPERTIES TIMEOUT 15)\n'
  (source/'CMakeLists.txt').write_text(cm)
  run('configure-host-safety',['/usr/bin/cmake','-S',str(source),'-B',str(build),'-G','Ninja','-DCMAKE_CXX_COMPILER=/usr/bin/g++','-DCMAKE_BUILD_TYPE=Debug','-DSTRATA_SANITIZE=address,undefined','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'])
  run('build-host-safety',['/usr/bin/cmake','--build',str(build),'--parallel','2'],wall=60)
  tests=run('ctest-host-safety',['/usr/bin/ctest','--test-dir',str(build),'--output-on-failure','--timeout','15'],wall=60).read_text();m.require('100% tests passed, 0 tests failed out of 3' in tests,'all three host safety tests');r['ctest_host_safety']=tests
  # The published fork omits tests/core and tests/platform. Build/test its available CPU foundation separately.
  bootstrap=out/'enable_ctest.cmake';bootstrap.write_text('enable_testing()\n')
  available=out/'project-host-asan'
  run('configure-available-project-host',['/usr/bin/cmake','-S',str(W),'-B',str(available),'-G','Ninja','-DCMAKE_CXX_COMPILER=/usr/bin/g++','-DCMAKE_BUILD_TYPE=Debug','-DSTRATA_ENABLE_XE=OFF','-DSTRATA_NATIVE_EXPERTS=OFF','-DSTRATA_BUILD_TESTS=OFF','-DSTRATA_LICENSE=free','-DSTRATA_SANITIZE=address,undefined','-DSTRATA_CUDA_ARCHS=','-DSTRATA_HIP_ARCHS=','-DCMAKE_PROJECT_INCLUDE='+str(bootstrap),'-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],wall=60)
  run('build-available-project-host',['/usr/bin/cmake','--build',str(available),'--target','platform_memory_test','ple_reader_test','--parallel','2'],wall=120)
  result=run('ctest-available-project-host',['/usr/bin/ctest','--test-dir',str(available),'-R','^(platform_memory_test|ple_reader_selftest)$','--output-on-failure','--timeout','15'],wall=45).read_text();m.require('100% tests passed, 0 tests failed out of 2' in result,'available host project tests');r['ctest_available_project_host']=result
  r['scope']='3 ASan/UBSan host tests using exact production publication and extracted actual Stager/HostVec/guard with CPU GPU-API stand-ins; 2 available project CPU-foundation CTests. No SYCL device or full engine sanitized execution.'
  r['unverified']=['Free SYCL mode: no intel/llvm DPC++ 7+ compiler installed; ordinary Clang is not a SYCL replacement.','Full contrib-icpx sanitizer CTest suite: published source omits tests/core and tests/platform; standalone host safety and available host CTests do not replace it.','Smaller/no-XMX/iGPU configurations: not executed; target machine B570 only.']
  r['standalone_compile_commands_sha256']=m.sha(build/'compile_commands.json');r['available_project_compile_commands_sha256']=m.sha(available/'compile_commands.json')
  for f,sha in j['source_sha256'].items():m.require(m.sha(W/f)==sha,'stable source')
  r.update(complete=True,passed=True)
 except BaseException as exc:r['error']=repr(exc)
 finally:r['active']=own.active is not None;r['finished_utc']=m.utc();save()
 print(json.dumps(dict(record=str(out/'record.json'),passed=r['passed'],error=r.get('error'))))
 raise SystemExit(0 if r['passed'] else 1)
