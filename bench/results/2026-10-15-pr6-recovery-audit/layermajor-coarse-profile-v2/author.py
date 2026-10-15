from pathlib import Path
import subprocess, difflib, hashlib, json
root=Path(__file__).resolve().parent
base='9177ad4fb7cc1587255db7235be474aca3eb4e2a'
paths=['src/prefill/prefill.cpp','src/prefill/layer_major.inc','src/program/generate.cpp']
old={p:subprocess.check_output(['git','show',base+':'+p]).decode() for p in paths}
new=dict(old)
def edit(path,a,b,n=1):
    assert new[path].count(a)==n,(path,a,new[path].count(a))
    new[path]=new[path].replace(a,b)
p=paths[0]; l=paths[1]; gen=paths[2]
edit(p,'#include <cuda_runtime.h>','#include <cuda_runtime.h>\n#include "strata/prefill/layermajor_coarse_profile.hpp"')
edit(p,'    detail::LayerMajorShapeCensus* layer_major_census = nullptr;','    detail::LayerMajorShapeCensus* layer_major_census = nullptr;\n    detail::LayerMajorCoarseProfile* layer_major_profile = nullptr;')
edit(l,'    detail::LayerMajorShapeBorrow census_borrow(m.layer_major_census, census.get());','''    detail::LayerMajorShapeBorrow census_borrow(m.layer_major_census, census.get());
    const char* profile_switch = std::getenv("STRATA_PREFILL_LAYER_MAJOR_PROFILE");
    const bool profile_requested = profile_switch && std::strcmp(profile_switch, "1") == 0;
    std::unique_ptr<detail::LayerMajorCoarseProfile> profile;
    if (profile_requested) {
        uint64_t need = detail::layer_major_add(host_need, audit ? 4ull << 20 : 0);
        need = detail::layer_major_add(need, phase_audit ? detail::Layer26PhaseAudit::host_reservation : 0);
        need = detail::layer_major_add(need, census ? sizeof(detail::LayerMajorShapeCensus) : 0);
        need = detail::layer_major_add(need, 1ull << 20); // fixed rows and twelve small compat event owners
        if (!core::conversation_memory_admit(core::conversation_available_memory(), need, host_headroom)) {
            err = "prefill layer-major profile: diagnostic storage exceeds RAM headroom"; return false;
        }
        profile.reset(new detail::LayerMajorCoarseProfile);
    }
    detail::LayerMajorCoarseBorrow profile_borrow(m.layer_major_profile, profile.get());
    using LmProfile = detail::LayerMajorCoarseProfile;''')
edit(l,'    auto drain = [&]() noexcept {\n        compute_drain', '    auto drain = [&]() noexcept {\n        const auto profile_drain_start = profile ? LmProfile::Clock::now() : LmProfile::Clock::time_point{};\n        compute_drain')
edit(l,'        return compute_drain == cudaSuccess && copy_drain == cudaSuccess;', '        if (profile) profile->add(LmProfile::wrapper_drain, profile_drain_start);\n        return compute_drain == cudaSuccess && copy_drain == cudaSuccess;')
edit(l,'    if (!drain()) { err = "prefill layer-major: entry queue drain failed"; return false; }','    if (!drain()) { err = "prefill layer-major: entry queue drain failed"; return false; }\n    if (profile) profile->open();')
edit(l,'    for (; layer < g.n_layers; ++layer) {','    for (; layer < g.n_layers; ++layer) {\n        if (profile) profile->layer = int(layer);')
edit(l,'        for (int32_t e = 0; e < NE; ++e) {\n            if (!layer_cache.fill_slot', '''        if (profile) profile->add(LmProfile::source, load_start);
        const auto upload_start = profile ? LmProfile::Clock::now() : LmProfile::Clock::time_point{};
        if (profile) profile->begin(LmProfile::upload, m.cs);
        for (int32_t e = 0; e < NE; ++e) {
            if (!layer_cache.fill_slot''')
edit(l,'        if (!drain()) { err = "prefill layer-major: expert upload drain failed"; return false; }','''        if (profile) {
            profile->end(LmProfile::upload, m.cs);
            profile->add(LmProfile::upload_submit, upload_start);
            profile->rows[layer].upload_bytes += uint64_t(NE) * bytes;
        }
        const auto upload_drain_start = profile ? LmProfile::Clock::now() : LmProfile::Clock::time_point{};
        if (!drain()) { err = "prefill layer-major: expert upload drain failed"; return false; }
        if (profile) {
            profile->add(LmProfile::upload_drain, upload_drain_start);
            profile->retired(LmProfile::upload); // existing checked upload drain above
        }''')
edit(l,'        if (activate_native_dense_layer) {','''        const auto activation_start = profile ? LmProfile::Clock::now() : LmProfile::Clock::time_point{};
        if (activate_native_dense_layer) {''')
edit(l,'        stage_lb_ = layer; stage_le_ = layer + 1;','''        if (profile && activate_native_dense_layer) profile->add(LmProfile::activation, activation_start);
        stage_lb_ = layer; stage_le_ = layer + 1;''')
edit(l,'            const auto span = plan.span(row, count);','''            const auto store_start = profile ? LmProfile::Clock::now() : LmProfile::Clock::time_point{};
            const auto span = plan.span(row, count);
            if (profile) {
                profile->rows[layer].store_bytes[int(span.device)] += span.bytes;
                profile->begin(LmProfile::store, m.cs, span.device);
            }''')
edit(l,'            if (status != cudaSuccess || !drain()) {\n                error = "prefill layer-major: residual store copy/drain failed"; return false;\n            }','''            if (profile) profile->end(LmProfile::store, m.cs);
            const auto store_drain_start = profile ? LmProfile::Clock::now() : LmProfile::Clock::time_point{};
            if (status != cudaSuccess || !drain()) {
                error = "prefill layer-major: residual store copy/drain failed"; return false;
            }
            if (profile) {
                profile->add(span.device ? LmProfile::store_drain_device : LmProfile::store_drain_host, store_drain_start);
                profile->add(span.device ? LmProfile::store_wall_device : LmProfile::store_wall_host, store_start);
                profile->retired(LmProfile::store); // existing checked residual-store drain above
            }''')
edit(l,'    lease.retire();\n    if (census) census->complete();','''    lease.retire();
    if (profile) {
        profile->wrapper_ms = ms_since(started);
        profile->complete = true;
        profile->close(); // checked lease has retired; status checked before emit
        profile->emit();
    }
    if (census) census->complete();''')
edit(p,'                const auto span = m.layer_major_residual.span(c0, T);','''                const auto profile_input_start = m.layer_major_profile ? detail::LayerMajorCoarseProfile::Clock::now()
                    : detail::LayerMajorCoarseProfile::Clock::time_point{};
                const auto span = m.layer_major_residual.span(c0, T);
                if (m.layer_major_profile) {
                    m.layer_major_profile->rows[LB].input_bytes[int(span.device)] += span.bytes;
                    m.layer_major_profile->begin(detail::LayerMajorCoarseProfile::input, m.cs, span.device);
                }''')
edit(p,'                if (m.layer_major_audit && !m.layer_major_audit->input(LB, c0, T, m.R, span.device,','''                if (m.layer_major_profile) {
                    m.layer_major_profile->end(detail::LayerMajorCoarseProfile::input, m.cs);
                    m.layer_major_profile->add(span.device ? detail::LayerMajorCoarseProfile::input_submit_device : detail::LayerMajorCoarseProfile::input_submit_host, profile_input_start);
                }
                if (m.layer_major_audit && !m.layer_major_audit->input(LB, c0, T, m.R, span.device,''')
edit(p,'                    // ======================= QSA =======================','''                    // ======================= QSA =======================
                    if (m.layer_major_profile) m.layer_major_profile->begin(detail::LayerMajorCoarseProfile::qsa, m.cs);''')
edit(p,'                    ++qsa_index;\n                } else {','''                    ++qsa_index;
                    if (m.layer_major_profile) m.layer_major_profile->end(detail::LayerMajorCoarseProfile::qsa, m.cs);
                } else {''')
edit(p,'                    // ======================= MoE =======================','''                    // ======================= MoE =======================
                    if (m.layer_major_profile) m.layer_major_profile->begin(detail::LayerMajorCoarseProfile::moe, m.cs);''')
edit(p,'                    // debug: STRATA_DBG_NAN=1 reports','''                    if (m.layer_major_profile) m.layer_major_profile->end(detail::LayerMajorCoarseProfile::moe, m.cs);
                    // debug: STRATA_DBG_NAN=1 reports''')
edit(p,'            const auto toc2 = Clock::now();\n            if (on_stage_chunk','''            const auto toc2 = Clock::now();
            if (m.layer_major_profile) {
                m.layer_major_profile->add(detail::LayerMajorCoarseProfile::chunk_drain, toc);
                // Existing checked callback drain above covers all marks.
                m.layer_major_profile->chunk_retired();
            }
            if (on_stage_chunk''')
edit(p,'                    pt.mark(kPfGdn, cs);','''                    if (m.layer_major_profile) m.layer_major_profile->begin(detail::LayerMajorCoarseProfile::gdn, m.cs);
                    pt.mark(kPfGdn, cs);''')
edit(p,'                    if (!native_proj(m.gemm, wo, m.y_h, m.bo, T, v.name("ssm_out.weight"), err)) return false;','''                    if (!native_proj(m.gemm, wo, m.y_h, m.bo, T, v.name("ssm_out.weight"), err)) return false;
                    if (m.layer_major_profile) m.layer_major_profile->end(detail::LayerMajorCoarseProfile::gdn, m.cs);''')
exec((root/'native_dense_edits.py').read_text())
header='include/strata/prefill/layermajor_coarse_profile.hpp'
new[header]=(root/'layermajor_coarse_profile.hpp').read_text()
diff=''
for path in [*paths,header]:
    diff+=''.join(difflib.unified_diff(old.get(path,'').splitlines(True),new[path].splitlines(True),fromfile='a/'+path if path in old else '/dev/null',tofile='b/'+path))
(root/'source.diff').write_text(diff)
(root/'pins.json').write_text(json.dumps({'base_commit':base,'base_source_sha256':{p:hashlib.sha256(s.encode()).hexdigest() for p,s in old.items()},'candidate_source_sha256':{p:hashlib.sha256(s.encode()).hexdigest() for p,s in new.items()},'source_diff_sha256':hashlib.sha256(diff.encode()).hexdigest(),'status':'SOURCE_ONLY_UNAPPLIED_UNCOMPILED_UNTESTED'},indent=2)+'\n')
print(len(diff),hashlib.sha256(diff.encode()).hexdigest())
# Preservation receipts describe textual identity, not execution validation.
pins_path=root/'pins.json'
pins=json.loads(pins_path.read_text())
def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def trace(s):
    a=s.index('    const bool trace_dense=')
    b=s.index('    auto drain=',a)
    return s[a:b]
def outside(s):
    a=s.index('bool run_prompt_with_native_dense_phase(')
    b=s.index('// Windows\' WDDM driver model:',a)
    return s[:a],s[b:]
pins['preserved_generator_trace_sha256']={'base':sha(trace(old[gen])),'candidate':sha(trace(new[gen]))}
pins['preserved_generator_outside_native_phase_sha256']={str(i):{'base':sha(a),'candidate':sha(b)} for i,(a,b) in enumerate(zip(outside(old[gen]),outside(new[gen])))}
pins['compat_audit_sha256']={s:hashlib.sha256(subprocess.check_output(['git','show',base+':'+s])).hexdigest() for s in ['src/sycl/compat.cpp','src/sycl/runtime.cpp','include/strata/sycl/compat/cuda_runtime.h','include/strata/prefill/draft_uploads.hpp']}
pins_path.write_text(json.dumps(pins,indent=2)+'\n')
