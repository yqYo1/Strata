"""Offline accounting of the executed private compact layout; no GPU access."""
from pathlib import Path
import datetime, hashlib, json, os, subprocess

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
private = root / 'build-sycl-event-ack-registered-copy-v3-20261008/source'
src = private / 'sycl/src/prefill/prefill.cpp'
text = src.read_text()
out = base / 'compact-chunk-v01402-host-accounting'
out.mkdir(mode=0o700)

def function(prefix, source=text):
    start = source.index(prefix)
    brace = source.index('{', start)
    depth = 0
    for i in range(brace, len(source)):
        depth += (source[i] == '{') - (source[i] == '}')
        if depth == 0:
            return source[start:i+1]
    raise AssertionError(prefix)

# Keep the actual take/count arithmetic, removing the unexecuted GPU path.
alloc_start = text.index('struct Alloc {')
alloc_end = text.index('        if (base != nullptr)', alloc_start)
alloc = text[alloc_start:alloc_end].replace(' T *take(size_t n, bool &ok) try {', ' T *take(size_t n, bool &ok) {')
alloc += '        std::abort();\n    }\n};\n'
regions = '\n'.join(function(prefix) for prefix in [
    'template <typename V>\nvoid compact_take', 'void take_compact_hc(',
    'void take_compact_base(', 'uint64_t hc_set_bytes(', 'uint64_t ple_set_bytes(',
    'uint64_t gdn_set_bytes(', 'uint64_t qsa_set_bytes(',
    'MoeBufs moe_bufs(', 'uint64_t moe_set_bytes('])
body = function('uint64_t Prefill::bytes_needed_impl(')
attn_src = private / 'sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp'
attn = function('uint64_t qsa_decode_attn_scratch_floats(', attn_src.read_text())
qsa_header = private / 'include/strata/kernels/qsa.hpp'
assert 'constexpr int HD = 256;' in attn_src.read_text()
assert 'constexpr int CHUNK = 64;' in attn_src.read_text()

program = r'''
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "strata/kernels/qsa.hpp"
namespace strata::kernels {
constexpr int NG_HC_DIM = 10240, HD = 256, CHUNK = 64;
struct KvHostPools {};
ATTN
}
namespace core {
struct ModelGeometry { int64_t n_head=24,n_head_kv=2,head_dim=256,idx_q_heads=4,idx_key_dim=128,n_expert=512; };
struct QsaState { int64_t max_cells=32768; };
struct SessionState { QsaState qsa_states[1]; int qsa_primary() const { return 0; } };
}
constexpr int64_t N=2560,HC=4,D=N*HC,LR=320,K=10,NE=512,C=10240,ZV=6144,HV=48;
constexpr int64_t GEMM_SCRATCH=32ll<<20;
constexpr size_t GEMM_WS=0,MMQ_TAIL=4096;
constexpr int DQ=2,MMQ_GROUP=16;
bool gr_unfused() { return false; }
bool prompt_f16() { return true; }
bool bf16x2(bool) { return false; }
bool bf16x2_hc(bool) { return false; }
bool compact_prefill() { return true; }
bool compact_hc_prefill() { return true; }
bool fused_layout(size_t,bool) { return false; }
int64_t stream_all_min() { return 1024; }
int64_t prompt_attn_batch(size_t T) { return std::min<int64_t>(32,T); }
int ring_slots(size_t) { return 8; }
int64_t MAXBLOB() { return 1363148800/512; }
namespace mmq { size_t q8_bytes(int64_t,int64_t) { return 0; } }
namespace fused { size_t group_bytes(int64_t,int) { return 0; } size_t act_bytes(int64_t,int64_t) { return 0; } }
struct MmqPlan { bool any=false,fallback=true; size_t gu_max=0,d_max=0; };
const MmqPlan& mmq_plan() { static MmqPlan p;return p; }
struct MoeBufs { size_t gu,h,xq,hq; };
struct Prefill {
    struct Impl {
        bool f16_io=true;
        float *xn=nullptr,*lo=nullptr,*gated=nullptr,*emb=nullptr,*mixed=nullptr,*bo=nullptr,*R=nullptr,*grs=nullptr,*inj=nullptr;
        uint16_t *xn16=nullptr,*lo16=nullptr,*xn16_lo=nullptr,*lo16_lo=nullptr,*mixed_bf=nullptr,*mixed_h=nullptr,*mixed_bf_lo=nullptr;
        int32_t* steps_dev=nullptr;
    };
    static uint64_t bytes_needed_impl(const core::ModelGeometry&,const core::SessionState&,int64_t,bool);
};
ALLOC
void take_stage(Alloc&,const core::SessionState&,const strata::kernels::QsaShapes&,strata::kernels::KvHostPools&,bool&) {}
REGIONS
BODY
int main() {
    core::ModelGeometry g;core::SessionState ss;
    for(int64_t T:{8192,12288,16384}) {
        uint64_t accounted=Prefill::bytes_needed_impl(g,ss,T,false)-(8u<<20);
        uint64_t estimated_owned=Prefill::bytes_needed_impl(g,ss,T,true);
        auto s=strata::kernels::qsa_real_shapes();
        auto cap=strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells,s);
        uint64_t region=std::max({hc_set_bytes(T,true),ple_set_bytes(T,true),gdn_set_bytes(T,true),qsa_set_bytes(T,cap,8194,256,32,s,true),moe_set_bytes(T,512,false,true)});
        std::printf("%lld %llu %llu %llu\n",(long long)T,(unsigned long long)accounted,(unsigned long long)estimated_owned,(unsigned long long)region);
    }
}
'''.replace('ATTN', attn).replace('ALLOC', alloc).replace('REGIONS', regions).replace('BODY', body)
cpp = out / 'count.cpp'
cpp.write_text(program)
env = dict(os.environ)
env.pop('LD_PRELOAD', None)
record = {'active':True, 'passed':False, 'gpu_access':False,
          'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scope':'Actual sizing helper/count arithmetic extracted from the executed private source. Fixed executed compact-hc/FP16/MMQ-off/ring8 geometry and resident KV; no allocator/runtime capacity guarantee. 2MiB owned-page granules are the engine estimator, not a measurement of SYCL physical granularity.',
          'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [src,attn_src,qsa_header,Path(__file__)]}}
try:
    build=subprocess.run(['/usr/bin/c++','-std=c++20','-O2','-I'+str(private/'include'),str(cpp),'-o',str(out/'count')],env=env,capture_output=True,timeout=30)
    (out/'build.stdout').write_bytes(build.stdout);(out/'build.stderr').write_bytes(build.stderr)
    assert build.returncode==0,build.stderr.decode()
    run=subprocess.run([str(out/'count')],env=env,capture_output=True,timeout=10)
    (out/'run.stdout').write_bytes(run.stdout);(out/'run.stderr').write_bytes(run.stderr);assert run.returncode==0
    rows=[dict(zip(['chunk','accounted_bytes','owned_estimator_bytes','shared_region_bytes'],map(int,line.split()))) for line in run.stdout.decode().splitlines()]
    assert rows[0]['accounted_bytes']==1861857536
    assert rows[0]['shared_region_bytes']==1166147584
    for row in rows:
        row['new_all_gpu_residual_bytes']=(32767-row['chunk'])*40960
        row['owned_estimator_and_residual_bytes']=row['owned_estimator_bytes']+row['new_all_gpu_residual_bytes']
        row['estimated_increment_over_8192_bytes']=row['owned_estimator_and_residual_bytes']-(rows[0]['owned_estimator_bytes']+(32767-8192)*40960)
    record.update(passed=True,rows=rows,observed_8192_accounting_matches=True)
except BaseException as error:
    record['error']=repr(error)
    raise
finally:
    record['active']=False
    (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
