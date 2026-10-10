from pathlib import Path
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import types

B = Path(__file__).parent
OLD = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-prefill-service-events-20261010')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/fix-sycl-layer-major-prefix-boundary-20261010')
OUT = B / 'layer-major-prefix-host-regression-v1'
PARENT = B / 'run_gdn_gate_factor_probe_v2.py'


def ident(p):
    p = Path(p)
    return dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())


with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not OUT.exists()
    oldp = OLD / 'sycl/src/prefill/prefill.cpp'
    newp = W / 'sycl/src/prefill/prefill.cpp'
    assert ident(oldp)['sha256'] == 'f6cbc4acb9f84b19480a25892a2e6864896a3e3ab0f68ace3c3f51a788591f69'
    assert ident(newp)['sha256'] == 'd6597bedd4959836d55714f5aa0df9302347f8fed8b4ef7f1b8c58313a051c0e'
    assert ident(PARENT)['sha256'] == '7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f'
    mod = types.ModuleType('prefix_regression_owner')
    prefix = PARENT.read_text().split('\ndef parse_probe(', 1)[0]
    exec(compile(prefix, str(PARENT), 'exec'), mod.__dict__)
    OUT.mkdir(mode=0o700)
    mod.W = OUT
    owner = mod.Owner(OUT)
    record = dict(active=True, complete=False, passed=False, commands=owner.commands,
                  started_utc=mod.utc(), controller=ident(__file__), parent_owner=ident(PARENT),
                  source_old=dict(path=str(oldp), **ident(oldp)), source_fixed=dict(path=str(newp), **ident(newp)),
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  scope='CPU execution of exact extracted outer-prefix and inner-chunk calculations with stub cache geometry, plus independent span-bounds oracle and actual new guard. SYCL translation-unit compile only; no GPU, model inference or numerical/full-context qualification.',
                  gpu_executed=False, model_executed=False, adopted_optimization=False)

    def save():
        tmp = OUT / 'record.json.tmp'
        tmp.write_text(json.dumps(record, indent=2) + '\n')
        tmp.replace(OUT / 'record.json')

    owner.persist = save
    clean = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8')

    def run(label, argv, env=clean, wall=120):
        entry, so, se = owner.run(label, argv, env, wall=wall, text_cap=1 << 20, file_cap=2 << 20)
        save()
        assert mod.completed(entry) and entry['exit_code'] == 0, (label, entry)
        return so, se

    try:
        a, b = oldp.read_text(), newp.read_text()

        def outer(s):
            part = s.split('bool Prefill::run_layer_major(', 1)[1]
            start = part.index('    const char* gpu_env =')
            end = part.index('    const char* inplace_env =', start)
            return part[start:end]

        def inner(s):
            start = s.index('    const bool all_cached = m.cache != nullptr')
            end = s.index('    auto ple_gather =', start)
            return s[start:end]

        old_outer, fixed_outer = outer(a), outer(b)
        assert inner(a) == inner(b)
        chunk = inner(b)
        gs = b.index('        if (m.transfer_context && c0 < m.residual_gpu_tokens')
        ge = b.index('        last_chunk_len = T;', gs)
        guard = b[gs:ge]
        snippets = dict(old_outer=old_outer, fixed_outer=fixed_outer, unchanged_inner=chunk, new_guard=guard)
        (OUT / 'extracted-snippets.json').write_text(json.dumps(snippets, indent=2) + '\n')
        source = '''#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
struct Cache { int64_t count; int64_t slots() const { return count; } };
struct Geometry { int64_t n_layers=48, n_expert=512; };
struct Impl { Cache* cache; Geometry* g; int64_t T; void* transfer_context=nullptr; int64_t residual_gpu_tokens=0; };
'''
        for label, snippet in (('old', old_outer), ('fixed', fixed_outer)):
            source += f'''int64_t outer_{label}(Impl& m, const Cache& layer_cache, int64_t n) {{
auto* impl_=&m; const auto& g=*m.g; (void)impl_; (void)g; (void)layer_cache;
{snippet}
return gpu_tokens;
}}
'''
        source += f'''std::vector<int64_t> chunk_lengths(Impl& m, int64_t n) {{
{chunk}
std::vector<int64_t> r; for(int64_t c0=0;c0<n;c0+=chunk_len(c0)) {{
const auto length=chunk_len(c0); assert(length>0 && length<=m.T && length<=n-c0); r.push_back(length);
}} return r;
}}
bool actual_guard(Impl& m, int64_t c0, int64_t T, std::string& err) {{
{guard}
return true;
}}
'''
        source += r'''
bool spans_valid(int64_t n,int64_t G,const std::vector<int64_t>& chunks) {
 if(G<0||G>n) return false;
 int64_t row=0;
 for(auto count:chunks) {
  if(row+count<=G) { if(row<0 || count>G-row) return false; }
  else { const auto offset=row-G; if(offset<0 || offset>n-G || count>n-G-offset) return false; }
  row+=count;
 }
 return row==n;
}
int main() {
 Geometry g; Cache layer{512}; uint64_t cases=0,old_bad=0,guard_rejections=0;
 const int64_t ns[]={1,255,256,257,511,512,513,8191,8192,8193,16385,32768,262143,262144};
 const int64_t ts[]={128,256,257,512,8192,32768};
 const int64_t slots[]={0,511,512,24575,24576,24577};
 for(auto n:ns) for(auto T:ts) for(auto sl:slots) {
  Cache original{sl}; Impl m{&original,&g,T};
  const int64_t requests[]={-1,0,1,255,256,257,8191,8192,8193,8448,16384,16640,32768,n-1,n,n+1};
  for(auto want:requests) {
   const auto value=std::to_string(want); assert(!setenv("STRATA_PREFILL_LAYER_MAJOR_R_GPU",value.c_str(),1));
   m.cache=&original; auto oldG=outer_old(m,layer,n); auto G=outer_fixed(m,layer,n);
   m.cache=&layer; const auto chunks=chunk_lengths(m,n);
   assert(spans_valid(n,G,chunks));
   m.transfer_context=&m; m.residual_gpu_tokens=G; int64_t c0=0;
   for(auto length:chunks) { std::string error; assert(actual_guard(m,c0,length,error)); assert(error.empty()); c0+=length; }
   if(!spans_valid(n,oldG,chunks)) {
    ++old_bad; m.residual_gpu_tokens=oldG; c0=0; bool rejected=false;
    for(auto length:chunks) { std::string error; if(!actual_guard(m,c0,length,error)) { assert(error=="prefill layer-major: GPU residual prefix splits a chunk"); rejected=true; } c0+=length; }
    assert(rejected); ++guard_rejections;
   }
   ++cases;
  }
 }
 Cache full{24576}; Impl m{&full,&g,8192};
 assert(!setenv("STRATA_PREFILL_LAYER_MAJOR_R_GPU","8192",1));
 const auto oldG=outer_old(m,layer,32768), newG=outer_fixed(m,layer,32768);
 m.cache=&layer; const auto chunks=chunk_lengths(m,32768);
 const bool unset=std::getenv("STRATA_PREFILL_FIRST")==nullptr;
 if(unset) { assert(oldG==256 && newG==8192); assert(!spans_valid(32768,oldG,chunks)); assert(spans_valid(32768,newG,chunks)); }
 m.transfer_context=&m; m.residual_gpu_tokens=1; std::string err; assert(!actual_guard(m,0,8192,err));
 m.transfer_context=nullptr; err.clear(); assert(actual_guard(m,0,8192,err) && err.empty());
 std::cout<<"{\"passed\":true,\"cases\":"<<cases<<",\"original_invalid_spans\":"<<old_bad
 <<",\"guard_rejected_original_invalid_spans\":"<<guard_rejections<<",\"fixed_invalid_spans\":0,\"named_unset_original_G\":"<<oldG
 <<",\"named_unset_fixed_G\":"<<newG<<",\"named_unset_original_first_host_offset\":"<<-oldG<<"}\n";
}
'''
        cp = OUT / 'exact_extracted_prefix_regression.cpp'
        cp.write_text(source)
        record['extracted_source'] = dict(path=str(cp), **ident(cp))
        record['result_rows'] = []
        for kind, flags in (('release', ['-O2']), ('ubsan', ['-O1', '-fsanitize=undefined', '-fno-sanitize-recover=all'])):
            binary = OUT / ('prefix-regression-' + kind)
            run('build-' + kind, ['/usr/bin/g++', '-std=c++20', '-Wall', '-Wextra', '-Werror', *flags, str(cp), '-o', str(binary)])
            for label, value in (('unset', None), ('zero', '0'), ('short256', '256'), ('one', '1'), ('full8192', '8192'), ('negative', '-1')):
                env = dict(clean)
                if value is not None:
                    env['STRATA_PREFILL_FIRST'] = value
                so, se = run(kind + '-' + label, [str(binary)], env)
                row = json.loads(so.read_text())
                assert row['passed'] and row['cases'] == 14 * 6 * 6 * 16
                assert row['fixed_invalid_spans'] == 0 and row['original_invalid_spans'] == row['guard_rejected_original_invalid_spans']
                assert not se.read_text()
                if label == 'unset':
                    assert row['original_invalid_spans'] > 0 and row['named_unset_original_G'] == 256 and row['named_unset_fixed_G'] == 8192
                record['result_rows'].append(dict(build=kind, first=value, binary=ident(binary), **row))
                save()
        buildp = B / 'prefill-service-qualification-cpu-build-v1/record.json'
        build = json.loads(buildp.read_text())
        assert build['passed'] and build['complete'] and not build['active']
        qualroot = build['root']
        cmd = [arg.replace(qualroot, str(W)) for arg in build['changed_compile']['prefill.cpp.o']]
        cmd[cmd.index('-o')+1] = str(OUT / 'prefill.cpp.o')
        assert cmd[-1] == str(newp)
        record['SYCL_compile'] = dict(prior_recipe_receipt=dict(path=str(buildp), **ident(buildp)), argv=cmd, environment=build['environment'])
        save()
        run('compile-fixed-SYCL-prefill', cmd, build['environment'], wall=120)
        record['SYCL_object'] = dict(path=str(OUT / 'prefill.cpp.o'), **ident(OUT / 'prefill.cpp.o'))
        assert ident(oldp) == {k: record['source_old'][k] for k in ('bytes','sha256')}
        assert ident(newp) == {k: record['source_fixed'][k] for k in ('bytes','sha256')}
        record.update(passed=True, complete=True)
    except BaseException as error:
        record['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        record.update(active=owner.active is not None, finished_utc=mod.utc())
        record['passed'] = record['passed'] and not record['active'] and all(mod.completed(c) for c in owner.commands)
        save()
        print(json.dumps({k: record.get(k) for k in ('active','complete','passed','error','result_rows','SYCL_object')}), flush=True)
    if not record['passed']:
        sys.exit(1)
