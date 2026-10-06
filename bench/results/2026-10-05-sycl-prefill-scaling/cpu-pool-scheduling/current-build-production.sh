#!/bin/bash
set -eo pipefail
task_work=/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05
task_probe=/home/yayoi/.local/state/strata-sycl/cpu-pool-scheduling-probe/production-current-20261006
source /opt/intel/oneapi/setvars.sh > "$task_probe/production-env.log" 2>&1
export MKLROOT=/opt/intel/oneapi/mkl/2026.1
export LIBRARY_PATH=/opt/intel/oneapi/mkl/2026.1/lib
cmake -S "$task_work/sycl" -B "$task_work/build-sycl-upstream-jit" -DSTRATA_NATIVE_POOL_TASK_FACTOR=0 > "$task_probe/production-configure-default.log" 2>&1
ninja -C "$task_work/build-sycl-upstream-jit" -j 2 strata > "$task_probe/production-build-default.log" 2>&1
python3 - <<'PY'
import hashlib
from pathlib import Path
p=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05/build-sycl-upstream-jit/strata')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='1e37f0a910828046d9ad4701f44428e98a78f0056841a9d6e36376ca0d03347e','Default executable changed'
PY
cp "$task_work/build-sycl-upstream-jit/libstrata_kernels_cpu.a" "$task_probe/production-constant3.a"
/opt/intel/oneapi/compiler/2026.1/bin/icpx -fsycl "$task_probe/probe.o" "$task_probe/production-constant3.a" "$task_work/build-sycl-upstream-jit/ggml/src/libggml-cpu.a" "$task_work/build-sycl-upstream-jit/ggml/src/libggml-base.a" -lpthread -ldl -o "$task_probe/pool-production-reference"
cmake -S "$task_work/sycl" -B "$task_work/build-sycl-upstream-jit" -DSTRATA_NATIVE_POOL_TASK_FACTOR=9 > "$task_probe/production-configure-tasks9.log" 2>&1
ninja -C "$task_work/build-sycl-upstream-jit" -j 2 strata > "$task_probe/production-build-tasks9.log" 2>&1
cp "$task_work/build-sycl-upstream-jit/strata" "$task_probe/strata-upstream-arc-native-tasks9-jit"
cp "$task_work/build-sycl-upstream-jit/libstrata_kernels_cpu.a" "$task_probe/production-constant9.a"
cp "$task_work/build-sycl-upstream-jit/native-pool-task-factor.cpp" "$task_probe/production-constant9.cpp"
/opt/intel/oneapi/compiler/2026.1/bin/icpx -fsycl "$task_probe/probe.o" "$task_probe/production-constant9.a" "$task_work/build-sycl-upstream-jit/ggml/src/libggml-cpu.a" "$task_work/build-sycl-upstream-jit/ggml/src/libggml-base.a" -lpthread -ldl -o "$task_probe/pool-production-tasks9"
cmake -S "$task_work/sycl" -B "$task_work/build-sycl-upstream-jit" -DSTRATA_NATIVE_POOL_TASK_FACTOR=0 > "$task_probe/production-configure-restored.log" 2>&1
ninja -C "$task_work/build-sycl-upstream-jit" -j 2 strata > "$task_probe/production-build-restored.log" 2>&1
python3 - <<'PY'
import hashlib,json,os
from pathlib import Path
work=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05');p=Path('/home/yayoi/.local/state/strata-sycl/cpu-pool-scheduling-probe/production-current-20261006')
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
default=sha(work/'build-sycl-upstream-jit/strata')
assert default=='1e37f0a910828046d9ad4701f44428e98a78f0056841a9d6e36376ca0d03347e'
original=(work/'src/kernels/cpu/pool.cpp').read_text();generated=(p/'production-constant9.cpp').read_text()
needle='        mtasks_ = 3 * threads;\n        mrows_ = (int64_t) nb * FF;'
assert original.count(needle)==1 and generated==original.replace(needle,needle.replace('3 * threads','9 * threads'))
record={'scope':'Build only; engine GPU execution unverified','build_passed':True,'gpu_verified':False,'source_revision':'80b3920 plus SYCL safety and task-factor changes identified by source hashes','cmake_sha256':sha(work/'sycl/CMakeLists.txt'),'original_pool_sha256':sha(work/'src/kernels/cpu/pool.cpp'),'generated_pool_sha256':sha(p/'production-constant9.cpp'),'only_source_change':'Native ExpertPool row task multiplier 3 -> 9','default_executable_restored_sha256':default,'candidate_executable_sha256':sha(p/'strata-upstream-arc-native-tasks9-jit'),'candidate_archive_sha256':sha(p/'production-constant9.a'),'reference_archive_sha256':sha(p/'production-constant3.a'),'reference_binary_sha256':sha(p/'pool-production-reference'),'source_digests':{str(f):sha(f) for f in [work/'sycl/CMakeLists.txt'] + sorted((work/'sycl/include').rglob('*.hpp')) + sorted((work/'sycl/src').rglob('*.cpp'))},'harness_binary_sha256':sha(p/'pool-production-tasks9'),'harness_object_sha256':sha(p/'probe.o'),'script_sha256':sha(p/'build-production.sh'),'environment':{k:v for k,v in os.environ.items() if k.startswith(('ONEAPI_','SYCL_','OMP_','KMP_')) or k in ['MKLROOT','LIBRARY_PATH','LD_LIBRARY_PATH']}}
(p/'production-build.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='environment'},indent=2))
PY
