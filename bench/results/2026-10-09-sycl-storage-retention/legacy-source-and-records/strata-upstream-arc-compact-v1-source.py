import difflib,hashlib,json,shutil,subprocess
from pathlib import Path
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05');p=root/'sycl/src/prefill/prefill.cpp';s=p.read_text()
s=s.replace('bool borrowed = false, compact = false, compact_hc = false;','bool borrowed = false, compact = false;').replace('bool compact_hc_prefill();\n','')
a=s.index('void take_compact_hc(');b=s.index('uint64_t gdn_set_bytes',a)
s=s[:a]+'''void take_compact_base(Alloc& a, Prefill::Impl* m, size_t T, bool& ok) {
    float* image = a.take<float>(T * N, ok);
    if (m) m->emb = m->mixed = m->bo = image;
    auto take = [&](auto field, size_t n) { compact_take(a, m, field, n, ok); };
    take(&Prefill::Impl::R, T * D);
    if (gr_unfused()) take(&Prefill::Impl::xn, T * D);
    else if (m) m->xn = nullptr;
    take(&Prefill::Impl::grs, T * HC);
    take(&Prefill::Impl::xn16, T * D);
    take(&Prefill::Impl::lo, T * LR);
    take(&Prefill::Impl::lo16, T * LR);
    take(&Prefill::Impl::gated, T * D);
    take(&Prefill::Impl::inj, T * HC);
    take(&Prefill::Impl::mixed_bf, T * N);
    take(&Prefill::Impl::mixed_h, T * N);
    if (bf16x2_hc()) {
        take(&Prefill::Impl::xn16_lo, T * D);
        take(&Prefill::Impl::lo16_lo, T * LR);
    }
    if (bf16x2()) take(&Prefill::Impl::mixed_bf_lo, T * N);
    take(&Prefill::Impl::steps_dev, T * strata::kernels::kStepCount);
}

'''+s[b:]
s=s.replace('return value && (std::atoi(value) == 1 || std::atoi(value) == 2);','return value && std::atoi(value) == 1;')
a=s.index('bool compact_hc_prefill() {');b=s.index('// #136 P3:',a);s=s[:a]+s[b:]
s=s.replace('    m.compact_hc = compact_hc_prefill();\n','').replace('take_compact_base(o, &m, T, ok, m.compact_hc);','take_compact_base(o, &m, T, ok);')
s=s.replace('std::max({hc_set_bytes(T, m.compact_hc), gdn_set_bytes(T, m.compact),','std::max({gdn_set_bytes(T, m.compact),')
a=s.index('        if (m.compact_hc) {');b=s.index('        Alloc a;',a);s=s[:a]+s[b:]
s=s.replace('m.compact_hc ? "compact-hc" : m.compact ? "compact" : "original"','m.compact ? "compact" : "original"')
s=s.replace('    const bool compact_hc = compact_hc_prefill();\n','').replace('take_compact_base(o, nullptr, T, ok, compact_hc);','take_compact_base(o, nullptr, T, ok);').replace('std::max({hc_set_bytes(T, compact_hc), gdn_set_bytes(T, compact),','std::max({gdn_set_bytes(T, compact),')
assert 'compact_hc' not in s and 'take_compact_hc' not in s
out=Path('/tmp/strata-upstream-arc-compact-v1-source');out.mkdir(exist_ok=True)
(out/'prefill.cpp').write_text(s)
old=subprocess.check_output(['git','show','931ae49:sycl/src/prefill/prefill.cpp'],cwd=root,text=True)
(out/'source.patch').write_text(''.join(difflib.unified_diff(old.splitlines(keepends=True),s.splitlines(keepends=True),fromfile='a/sycl/src/prefill/prefill.cpp',tofile='b/sycl/src/prefill/prefill.cpp')))
obj=root/'build-sycl-upstream-jit/CMakeFiles/strata_prefill.dir/src/prefill/prefill.cpp.o'
shutil.copy2(obj,out/'prefill.cpp.o')
meta={'base_revision':'931ae49','source_sha256':hashlib.sha256(s.encode()).hexdigest(),'compiled_object_sha256':hashlib.sha256(obj.read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256(Path('/tmp/strata-upstream-arc-compact-diag-jit').read_bytes()).hexdigest(),'roundtrip_object_check_pending':True}
(out/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta),flush=True)
