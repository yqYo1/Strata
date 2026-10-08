from pathlib import Path
import json,hashlib
b=Path(__file__).parent;r=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05');p=r/'sycl/src/prefill/kernels.dp.cpp';s=p.read_text()
a=s.index('__dpct_inline__ void moe_combine4_kernel(');z=s.index('// ---------------------------------------------------------------- QSA helpers',a);body=s[a:z]
for old,new in [
 ('const float *__restrict__ w, const sycl::float4 *__restrict__ shared,','const float *__restrict__ w, const float *__restrict__ shared,'),
 ('const float *__restrict__ sg, sycl::float4 *__restrict__ bo, int64_t T)', 'const float *__restrict__ sg, float *__restrict__ bo, int64_t T)'),
 ('v[k] = *reinterpret_cast<const sycl::float4 *>(\n            Dm + (int64_t)slot[t * 10 + k] * N + d);', 'const float* p = Dm + (int64_t)slot[t * 10 + k] * N + d;\n        v[k] = sycl::float4(p[0], p[1], p[2], p[3]);'),
 ('const sycl::float4 sh = shared[i];\n    bo[i] = sycl::float4(s.x() + sh.x() * g, s.y() + sh.y() * g,\n                         s.z() + sh.z() * g, s.w() + sh.w() * g);', 'const float* sh = shared + 4 * i;\n    bo[4 * i] = s.x() + sh[0] * g;\n    bo[4 * i + 1] = s.y() + sh[1] * g;\n    bo[4 * i + 2] = s.z() + sh[2] * g;\n    bo[4 * i + 3] = s.w() + sh[3] * g;')]:
 assert body.count(old)==1,old;body=body.replace(old,new)
s=s[:a]+body+s[z:]
a=s.index('void moe_combine(const float* Dm');z=s.index('\nvoid ',a+5);wrapper=s[a:z]
old='sycl::ext::oneapi::experimental::use_root_sync';assert wrapper.count(old)==1;wrapper=wrapper.replace(old,'')
old='''moe_combine4_kernel(
                            Dm, slot, w,
                            reinterpret_cast<const sycl::float4 *>(shared), sg,
                            reinterpret_cast<sycl::float4 *>(bo), T);'''
assert wrapper.count(old)==1;wrapper=wrapper.replace(old,'moe_combine4_kernel(Dm, slot, w, shared, sg, bo, T);')
s=s[:a]+wrapper+s[z:];p.write_text(s)
record={'scope':'New upstream four-component MoE combine accesses float storage through float pointers; no sycl::float4 object reinterpretation. Each element retains ten ordered FMAs and the same gated shared addition. Independent work groups need no root synchronization; launch range, local size, subgroup, kernel identity and queue order unchanged','additional_ordinary_launches':1,'cumulative_ordinary_launch_changes':53,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(b/'upstream-e8ca-refresh-20261007/float4-storage-and-launch-review.json').write_text(json.dumps(record,indent=2)+'\n')
