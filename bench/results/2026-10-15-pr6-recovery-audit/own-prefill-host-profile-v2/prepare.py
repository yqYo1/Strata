# Authored for Strata; repository LICENSE (MIT) applies.
"""Materialize review-only source snapshots and diff; never edits active sources."""
from pathlib import Path
import hashlib, json, difflib
root = Path(__file__).resolve().parents[4]
out = Path(__file__).resolve().parent
paths = ['include/strata/sycl/cpu_wait_profile.hpp', 'src/prefill/prefill.cpp', 'src/prefill/layer_major.inc', 'src/prefill/sycl/gemm.cpp']
# After first materialization, regenerate exclusively from the pinned snapshot.
prior = json.loads((out / 'manifest.json').read_text()) if (out / 'manifest.json').exists() else None
base = {p: ((out / 'base' / p) if prior else (root / p)).read_text() for p in paths}
if prior:
    for entry in prior['files']:
        if hashlib.sha256(base[entry['path']].encode()).hexdigest() != entry['base_sha256']:
            raise RuntimeError('base snapshot hash mismatch: ' + entry['path'])
def once(s, a, b):
    if s.count(a) != 1: raise RuntimeError('ambiguous source anchor: ' + a[:80])
    return s.replace(a, b, 1)
p = paths[0]; s = base[p]
s = once(s, '#include <exception>', '#include <exception>\n#include <cmath>')
s = once(s, 'enum Kind {', 'enum HostKind { source_copy, host_group, gemm_submit, host_kinds };\nenum Kind {')
s = once(s, 'struct State {', '''inline bool duration(double x) noexcept { return std::isfinite(x) && x >= 0; }
struct State {''')
s = once(s, '    unsigned depth = 0;', '''    double host_cpu[host_kinds]{}, host_wall[host_kinds]{};
    double host_wait_cpu[host_kinds]{}, host_wait_wall[host_kinds]{};
    unsigned long long host_count[host_kinds]{}, host_throws[host_kinds]{};
    unsigned host_depth = 0;
    unsigned depth = 0;''')
s = once(s, '!begin_.valid || !end.valid || end.cpu < begin_.cpu || end.wall < begin_.wall', '!begin_.valid || !end.valid || !duration(end.cpu - begin_.cpu) || !duration(end.wall - begin_.wall)')
s = once(s, 'class Scope {', '''// Regions must not overlap or begin inside an explicit wait. Invalid nesting
// poisons the enclosing scope instead of reporting a double-counted partition.
class Host {
    State* s_ = active;
    HostKind kind_;
    Stamp begin_;
    double wc_ = 0, ww_ = 0;
    int exceptions_ = 0;
    bool open_ = false, measured_ = false;
public:
    explicit Host(HostKind k) noexcept : kind_(k) {
        if (!s_) return;
        open_ = true;
        ++s_->host_count[kind_];
        if (s_->host_depth++ || s_->depth) { s_->invalid = true; return; }
        measured_ = true;
        for (int i = 0; i < kinds; ++i) { wc_ += s_->cpu[i]; ww_ += s_->wall[i]; }
        exceptions_ = std::uncaught_exceptions();
        begin_ = stamp();
    }
    void finish() noexcept {
        if (!open_) return;
        open_ = false;
        --s_->host_depth;
        if (!measured_) return;
        const auto end = stamp();
        if (std::uncaught_exceptions() > exceptions_) ++s_->host_throws[kind_];
        double wc = 0, ww = 0;
        for (int i = 0; i < kinds; ++i) { wc += s_->cpu[i]; ww += s_->wall[i]; }
        const double cpu = end.cpu - begin_.cpu, wall = end.wall - begin_.wall;
        wc -= wc_; ww -= ww_;
        if (!begin_.valid || !end.valid || s_->depth || s_->host_depth ||
            !duration(cpu) || !duration(wall) || !duration(wc) || !duration(ww) ||
            !duration(cpu - wc) || !duration(wall - ww)) { s_->invalid = true; return; }
        s_->host_cpu[kind_] += cpu; s_->host_wall[kind_] += wall;
        s_->host_wait_cpu[kind_] += wc; s_->host_wait_wall[kind_] += ww;
    }
    ~Host() noexcept { finish(); }
    Host(const Host&) = delete;
    Host& operator=(const Host&) = delete;
};
class Scope {''')
s = s.replace('version=1', 'version=2')
s = once(s, '        const bool available =', '''        double hc = 0, hw = 0;
        bool host_valid = !state_.host_depth;
        for (int k = 0; k < host_kinds; ++k) {
            const double c = state_.host_cpu[k] - state_.host_wait_cpu[k];
            const double w = state_.host_wall[k] - state_.host_wait_wall[k];
            host_valid = host_valid && duration(state_.host_cpu[k]) && duration(state_.host_wall[k]) &&
                duration(state_.host_wait_cpu[k]) && duration(state_.host_wait_wall[k]) && duration(c) && duration(w);
            hc += c; hw += w;
        }
        const bool available =''')
s = once(s, '!state_.depth && cpu >= wc && wall >= ww;', '''!state_.depth && host_valid && duration(cpu) && duration(wall) &&
            duration(wc) && duration(ww) && duration(cpu - wc - hc) && duration(wall - ww - hw);''')
anchor = '    }\n    Scope(const Scope&) = delete;'
addition = '''        // One bounded extra line; no per-call output or dynamic allocation.
        std::fprintf(stderr, "strata prefill cpu host: version=2 available=%d open_regions=%u "
            "unclassified_cpu_s=%.9f unclassified_wall_s=%.9f",
            int(available), state_.host_depth, available ? cpu - wc - hc : -1., available ? wall - ww - hw : -1.);
        const char* names[host_kinds] = {"source_copy", "host_group", "gemm_submit"};
        for (int k = 0; k < host_kinds; ++k)
            std::fprintf(stderr, " %s_count=%llu %s_throws=%llu %s_cpu_s=%.9f %s_wall_s=%.9f "
                "%s_outside_wait_cpu_s=%.9f %s_outside_wait_wall_s=%.9f",
                names[k], state_.host_count[k], names[k], state_.host_throws[k],
                names[k], available ? state_.host_cpu[k] : -1., names[k], available ? state_.host_wall[k] : -1.,
                names[k], available ? state_.host_cpu[k] - state_.host_wait_cpu[k] : -1.,
                names[k], available ? state_.host_wall[k] - state_.host_wait_wall[k] : -1.);
        std::fputc('\\n', stderr);
'''
s = once(s, anchor, addition + anchor)
proposed = {p:s}
p = paths[1]; s = base[p]
s = once(s, '                        std::fill(m.cnt.begin(), m.cnt.end(), 0);', '''#if defined(STRATA_ENABLE_SYCL)
                        sycl_backend::cpu_wait_profile::Host host_group_profile(sycl_backend::cpu_wait_profile::host_group);
#endif
                        std::fill(m.cnt.begin(), m.cnt.end(), 0);''')
s = once(s, '                        n_order = order.size();', '''                        n_order = order.size();
#if defined(STRATA_ENABLE_SYCL)
                        host_group_profile.finish();
#endif''')
proposed[p] = s
p = paths[2]; s = base[p]
s = once(s, '        for (int32_t e = 0; e < NE; ++e) {\n            if ((e % 32)', '''        sycl_backend::cpu_wait_profile::Host source_copy_profile(sycl_backend::cpu_wait_profile::source_copy);
        for (int32_t e = 0; e < NE; ++e) {
            if ((e % 32)''')
s = once(s, '        if (profile) profile->add(LmProfile::source, load_start);', '''        source_copy_profile.finish();
        if (profile) profile->add(LmProfile::source, load_start);''')
proposed[p] = s
p = paths[3]; s = base[p]
s = once(s, '#include "strata/sycl/launch.hpp"', '#include "strata/sycl/launch.hpp"\n#include "strata/sycl/cpu_wait_profile.hpp"')
s = once(s, '  auto event = oneapi::mkl::blas::row_major::gemm(\n      queue_for(stream),', '''  auto &gemm_queue = queue_for(stream);
  cpu_wait_profile::Host gemm_submit_profile(cpu_wait_profile::gemm_submit);
  auto event = oneapi::mkl::blas::row_major::gemm(
      gemm_queue,''')
s = once(s, '  finish(stream, event);', '  gemm_submit_profile.finish();\n  finish(stream, event);')
proposed[p] = s
patch = ''
files = []
for p in paths:
    for directory, content in [('base', base[p]), ('proposed', proposed[p])]:
        dest = out / directory / p; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_text(content)
    patch += ''.join(difflib.unified_diff(base[p].splitlines(True), proposed[p].splitlines(True), fromfile='a/'+p, tofile='b/'+p))
    files.append({'path':p, 'base_sha256':hashlib.sha256(base[p].encode()).hexdigest(), 'proposed_sha256':hashlib.sha256(proposed[p].encode()).hexdigest()})
(out/'source.diff').write_text(patch)
(out/'manifest.json').write_text(json.dumps({'base_commit':'9b83fbf8e05115fa70b136c083896e7841608ba1', 'base_identity':'parent-supplied commit; exact touched source snapshots SHA256-pinned; no Git commands performed', 'provenance':'Independently authored from active repository MIT source and own cpu_wait_profile.hpp only; no XeStrata or archived implementation consulted.', 'status':'source-only; untested', 'files':files, 'patch_sha256':hashlib.sha256(patch.encode()).hexdigest()}, indent=2)+'\n')
print(hashlib.sha256(patch.encode()).hexdigest())
