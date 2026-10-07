"""CPU-only production-helper comparison of a typed SYCL vector-load candidate."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import time

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
source = root/'sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp'
out = base/'qsa-vector-load-cpu'
out.mkdir(mode=0o700)
original = source.read_text()
candidate = original
old = '''    const sycl::uint4 raw = *reinterpret_cast<const sycl::uint4 *>(base);
    const sycl::half2 *h2 = reinterpret_cast<const sycl::half2 *>(&raw);'''
new = '''    sycl::vec<uint16_t, 8> bits;
    bits.load(0, base);
    const sycl::half8 halves = bits.as<sycl::half8>();
    const sycl::half2 h2[] = {halves.template swizzle<0, 1>(), halves.template swizzle<2, 3>(),
                             halves.template swizzle<4, 5>(), halves.template swizzle<6, 7>()};'''
assert candidate.count(old) == 1
candidate = candidate.replace(old, new)
old = '''    const sycl::uint2 raw = *reinterpret_cast<const sycl::uint2 *>(codes);
    const int8_t* c = reinterpret_cast<const int8_t*>(&raw);'''
new = '''    sycl::vec<int8_t, 8> c;
    c.load(0, codes);'''
assert candidate.count(old) == 1
candidate = candidate.replace(old, new)
(out/'qsa_decode_attn.original.dp.cpp').write_text(original)
(out/'qsa_decode_attn.candidate.dp.cpp').write_text(candidate)

def helpers(text):
    a = text.index('__dpct_inline__ void load8_f16(')
    b = text.index('__dpct_inline__ void load8_q4(', a)
    return text[a:b].replace('__dpct_inline__', 'inline')

fixture = '''#include <sycl/sycl.hpp>
#include "strata/kernels/qsa_decode_attn.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdint>
using strata::kernels::QsaAttnPools;
constexpr int HD = 256, KV_Q8_GROUP = 64;
'''
fixture += '\nnamespace original {\n'+helpers(original)+'\n}\n'
fixture += '\nnamespace candidate {\n'+helpers(candidate)+'\n}\n'
fixture += r'''
float ieee_half(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    const uint32_t exponent = (h >> 10) & 31, mantissa = h & 1023;
    if (exponent == 31) return std::bit_cast<float>(sign | 0x7f800000 | (mantissa << 13));
    if (exponent) return std::bit_cast<float>(sign | ((exponent + 112) << 23) | (mantissa << 13));
    if (!mantissa) return std::bit_cast<float>(sign);
    unsigned bit = 0;
    for (unsigned m = mantissa; m > 1; m >>= 1) ++bit;
    return std::bit_cast<float>(sign | ((103 + bit) << 23) | ((mantissa - (1u << bit)) << (23 - bit)));
}
bool exact_or_nan(float value, float reference) {
    return std::isnan(reference) ? std::isnan(value)
           : std::bit_cast<uint32_t>(value) == std::bit_cast<uint32_t>(reference);
}
int main() {
    alignas(64) std::array<uint16_t, 3 * HD> k16{}, v16{};
    alignas(64) std::array<int8_t, 3 * HD> k8{}, v8{};
    std::array<uint16_t, 3 * HD / KV_Q8_GROUP> ks{}, vs{};
    QsaAttnPools pools;
    pools.k_pool=k16.data(); pools.v_pool=v16.data();
    pools.k_q=k8.data(); pools.v_q=v8.data(); pools.k_scale=ks.data(); pools.v_scale=vs.data();
    uint64_t mismatches=0, comparisons=0, finite_bits=0, nan_classes=0;
    const int8_t codes[8]={-128, -127, -1, 0, 1, 2, 126, 127};
    for (unsigned pattern=0; pattern<65536; ++pattern) {
        const int row=pattern%3, d0=((pattern/3)%32)*8;
        for (int j=0; j<8; ++j) {
            k16[row*HD+d0+j]=uint16_t(pattern + 8191*j);
            v16[row*HD+d0+j]=uint16_t(pattern + 8191*j) ^ 0x8000;
            k8[row*HD+d0+j]=codes[(j+pattern)%8];
            v8[row*HD+d0+j]=codes[(7-j+pattern)%8];
        }
        ks[row*(HD/KV_Q8_GROUP)+d0/KV_Q8_GROUP]=uint16_t(pattern);
        vs[row*(HD/KV_Q8_GROUP)+d0/KV_Q8_GROUP]=uint16_t(pattern)^0x8000;
        for (bool value : {false,true}) {
            float old16[8], new16[8], old8[8], new8[8];
            original::load8_f16(pools,value,row,d0,old16);
            candidate::load8_f16(pools,value,row,d0,new16);
            original::load8_q8(pools,value,row,d0,old8);
            candidate::load8_q8(pools,value,row,d0,new8);
            for (int j=0; j<8; ++j) {
                const float reference16=ieee_half((value?v16:k16)[row*HD+d0+j]);
                const float reference8=float((value?v8:k8)[row*HD+d0+j]) *
                    ieee_half((value?vs:ks)[row*(HD/KV_Q8_GROUP)+d0/KV_Q8_GROUP]);
                mismatches += !exact_or_nan(old16[j],reference16);
                mismatches += !exact_or_nan(new16[j],reference16);
                mismatches += !exact_or_nan(old8[j],reference8);
                mismatches += !exact_or_nan(new8[j],reference8);
                mismatches += !exact_or_nan(old16[j],new16[j]);
                mismatches += !exact_or_nan(old8[j],new8[j]);
                comparisons += 6;
                finite_bits += !std::isnan(reference16) + !std::isnan(reference8);
                nan_classes += std::isnan(reference16) + std::isnan(reference8);
            }
        }
    }
    std::printf("{\"patterns\":65536,\"comparisons\":%llu,\"non_nan_bit_checks\":%llu,"
                "\"nan_class_checks\":%llu,\"mismatches\":%llu}\n",
                (unsigned long long)comparisons,(unsigned long long)finite_bits,
                (unsigned long long)nan_classes,(unsigned long long)mismatches);
    return mismatches ? 1 : 0;
}
'''
(out/'fixture.cpp').write_text(fixture)
record = {'scope': 'CPU-only exact-helper comparison against an independent IEEE binary16 reference; no queue, device, GPU submission, capacity or hang-cause proof',
          'passed': False, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'original_source_sha256': hashlib.sha256(original.encode()).hexdigest(),
          'candidate_source_sha256': hashlib.sha256(candidate.encode()).hexdigest()}
env=dict(os.environ,PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin')
env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
argv=['/opt/intel/oneapi/compiler/2026.1/bin/icpx','-fsycl','-std=c++20','-O3',
      '-fp-model=precise','-fstrict-aliasing','-I'+str(root/'include'),str(out/'fixture.cpp'),
      '-o',str(out/'fixture')]
record['compile_argv']=argv
start=time.monotonic()
try:
    with (out/'compile.stdout').open('wb') as stdout,(out/'compile.stderr').open('wb') as stderr:
        compiled=subprocess.run(argv,env=env,stdout=stdout,stderr=stderr,timeout=120)
    record['compile_exit_code']=compiled.returncode;assert compiled.returncode==0
    record['fixture_sha256']=hashlib.sha256((out/'fixture.cpp').read_bytes()).hexdigest()
    record['binary_sha256']=hashlib.sha256((out/'fixture').read_bytes()).hexdigest()
    # Trace any accidental enumeration, but construct no SYCL reference-semantic objects.
    run_env=dict(env,LD_LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.1/lib:/opt/intel/oneapi/umf/1.1/lib',
                 UR_ENABLE_LAYERS='UR_LAYER_TRACING',UR_LOG_TRACING='level:info;flush:info;output:stderr')
    run=subprocess.run([str(out/'fixture')],env=run_env,capture_output=True,text=True,timeout=15)
    (out/'fixture.stdout').write_text(run.stdout);(out/'fixture.stderr').write_text(run.stderr)
    record['exit_code']=run.returncode;record['result']=json.loads(run.stdout)
    assert run.returncode==0 and record['result']['mismatches']==0
    assert '---> ur' not in run.stderr, 'Unexpected runtime call in the CPU-only fixture'
    assert source.read_text()==original, 'Production source changed during the CPU comparison'
    record['production_source_unchanged']=True
    record['passed']=True
except BaseException as error:
    record['error']=repr(error)
    raise
finally:
    record['elapsed_seconds']=time.monotonic()-start
    record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
