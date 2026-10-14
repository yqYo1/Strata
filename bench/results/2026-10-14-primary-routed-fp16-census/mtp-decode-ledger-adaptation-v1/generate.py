#!/usr/bin/env python3
"""Source-pinned artifact fabrication only. Never writes or imports product source."""
from pathlib import Path
import difflib
import hashlib
import json

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
base={}
proposed={}
anchors=[]
def edit(path,before,after):
    if path not in base:
        base[path]=(ROOT/path).read_text()
        proposed[path]=base[path]
    if proposed[path].count(before)!=1:
        raise RuntimeError(f"nonunique source anchor: {path}: {before[:100]!r}")
    proposed[path]=proposed[path].replace(before,after,1)
    anchors.append({'path':path,'before':before,'after':after})
def new(path,artifact):
    if (ROOT/path).exists(): raise RuntimeError(f"new target exists; review/rebase: {path}")
    base[path]=''; proposed[path]=(OUT/artifact).read_text()

new('include/strata/kernels/cpu/service_budget.hpp','adapted_service_budget.hpp')
new('include/strata/core/decode_route_ledger.hpp','adapted_decode_route_ledger.hpp')
new('include/strata/core/decode_service_budget.hpp','adapted_decode_service_budget.hpp')
edit('include/strata/kernels/cpu/pool.hpp','#pragma once','#pragma once\n#include "strata/kernels/cpu/service_budget.hpp"')
edit('include/strata/kernels/cpu/pool.hpp','    int workers() const { return n_; }',
'''    // Serialized issuer only. Bind before publishing native jobs; clear after
    // run_split_multi_native's existing final parked boundary, never from a worker.
    void service_budget(NativeServiceBudget* value) { service_budget_=value; }
    int workers() const { return n_; }''')
edit('include/strata/kernels/cpu/pool.hpp','    int n_ = 0;',
     '    NativeServiceBudget* service_budget_=nullptr;\n    int n_ = 0;')
edit('include/strata/core/expert_source.hpp','struct ExpertDispatch {\n',
     'class DecodeRouteLedger;\nstruct ExpertDispatch {\n    DecodeRouteLedger* decode_route_ledger=nullptr; // joined request owner, ordinary synchronous MTP only\n')
edit('src/core/expert_source.cpp','#include "strata/core/expert_source.hpp"',
     '#include "strata/core/expert_source.hpp"\n#include "strata/core/decode_service_budget.hpp"')
edit('src/core/expert_source.cpp','#include <memory>','#include <memory>\n#include <new>')
edit('src/core/expert_source.cpp','namespace strata::core {\n',
     'namespace strata::core {\n'+(OUT/'scope_impl.cpp.in').read_text()+'\n')
edit('src/core/expert_source.cpp',
'''                                float* out, const uint8_t* hit_mask) {
    using namespace strata::kernels::cpu;
    if (d.failed) return;''',
'''                                float* out, const uint8_t* hit_mask) {
    using namespace strata::kernels::cpu;
    auto* const decode_observer=d.decode_route_ledger;
    if (decode_observer) decode_observer->attempt_dispatch(d.layers,n_tok,k);
    if (d.failed) return;''')
edit('src/core/expert_source.cpp',
'''    if (native) d.pool->run_split_multi_native(lay.fmt[(size_t) d.layers], d.jobs_multi.data(), njobs);
    else d.pool->run_split_multi(d.jobs_multi.data(), njobs);''',
'''    if (native) {
        if (decode_observer && decode_observer->active) {
            decode_observer->service.actual_t=(int)n_tok;
            decode_observer->pool_bound=true;
            d.pool->service_budget(&decode_observer->service);
        }
        d.pool->run_split_multi_native(lay.fmt[(size_t) d.layers], d.jobs_multi.data(), njobs);
        // Native batches have completed and workers parked before this returns.
        if (decode_observer && decode_observer->active) {
            d.pool->service_budget(nullptr);
            decode_observer->pool_bound=false;
        }
    } else d.pool->run_split_multi(d.jobs_multi.data(), njobs);''')
edit('src/core/expert_source.cpp','    pt("ran");\n    auto ms =',
'''    pt("ran");
    if (decode_observer) {
        if (decode_route_ledger_dispatch_eligible(native,n_tok,d.remote_count,d.peer!=nullptr) &&
            d.layers>=0 && (size_t)d.layers<lay.fmt.size())
            decode_observer->record(d.layers,n_tok,ids,kind,n,d.jobs_multi.data(),njobs,d.jobs_multi.size(),
                                    lay.fmt[(size_t)d.layers],d.plan ? d.plan->pcie_mode : 0);
        else decode_observer->invalid=true;
    }
    auto ms =''')

edit('src/kernels/cpu/pool.cpp',
'''            // plan v0.3 P6: native layers, 5 = gate/up rows, 6 = down rows
            const int per = mode_ == 5 ? FF : H;
            const int64_t g0 = mrows_ * (int64_t) i / mtasks_, g1 = mrows_ * (int64_t) (i + 1) / mtasks_;''',
'''            // plan v0.3 P6: native layers, 5 = gate/up rows, 6 = down rows
            const int per = mode_ == 5 ? FF : H;
            const int64_t g0 = mrows_ * (int64_t) i / mtasks_, g1 = mrows_ * (int64_t) (i + 1) / mtasks_;
            if (service_budget_) service_budget_->task(id,mode_-5,g1-g0);''')
for before,after in [
('''                if (mode_ == 5 && nfmt_->gu_type == 42) {
                    // a native Q2_0 pack''',
'''                if (mode_ == 5 && nfmt_->gu_type == 42) {
                    ServiceClockScope service(service_budget_ ? service_budget_->counter(id,3,mjobs_[e].nt) : nullptr);
                    // a native Q2_0 pack'''),
('''                    native_gu_rows(*nfmt_, mjobs_[e].blob, mjobs_[e].nact, mjobs_[e].nt, ff, r0, r1);''',
'''                    ServiceClockScope service(service_budget_ ? service_budget_->counter(id,0,mjobs_[e].nt) : nullptr);
                    native_gu_rows(*nfmt_, mjobs_[e].blob, mjobs_[e].nact, mjobs_[e].nt, ff, r0, r1);'''),
('''                } else if (nfmt_->d_type == 42) {
                    // Q2_0 down''',
'''                } else if (nfmt_->d_type == 42) {
                    ServiceClockScope service(service_budget_ ? service_budget_->counter(id,5,mjobs_[e].nt) : nullptr);
                    // Q2_0 down'''),
    ('''                    native_down_rows(*nfmt_, mjobs_[e].blob, hq, mjobs_[e].nt, mjobs_[e].out, r0, r1);''',
'''                    ServiceClockScope service(service_budget_ ? service_budget_->counter(id,2,mjobs_[e].nt) : nullptr);
                    native_down_rows(*nfmt_, mjobs_[e].blob, hq, mjobs_[e].nt, mjobs_[e].out, r0, r1);''')]: edit('src/kernels/cpu/pool.cpp',before,after)
edit('src/kernels/cpu/pool.cpp',
     '// Q2_0 down (most IQ layers): the AVX-512 kernel, ggml-cpu has only a scalar one on x86',
     '// Q2_0 down: q2_rows_any selects AVX-512 or AVX2 on the current CPU.')
edit('src/kernels/cpu/pool.cpp',
'''        done_.fetch_add(1, std::memory_order_release);
    }
}

void ExpertPool::run_phase''',
'''        if (service_budget_ && mode_>=5) service_budget_->task_done(id,mode_-5);
        done_.fetch_add(1, std::memory_order_release);
    }
}

void ExpertPool::run_phase''')
edit('src/kernels/cpu/pool.cpp','''void ExpertPool::run_phase(int mode, int n_tasks) {
    wait_parked("before a phase");''',
'''void ExpertPool::run_phase(int mode, int n_tasks) {
    if (service_budget_ && (mode==5 || mode==6)) service_budget_->phase_begin(mode-5,n_tasks,mrows_);
    wait_parked("before a phase");''')
edit('src/kernels/cpu/pool.cpp','''    wait_done(n_tasks);
    wait_parked("after a phase");''',
'''    wait_done(n_tasks);
    wait_parked("after a phase");
    if (service_budget_ && (mode==5 || mode==6)) service_budget_->phase_end(mode-5);''')
edit('src/kernels/cpu/pool.cpp',
'''                if (f.d_type == 42) act_quant_any(split_multi_[(size_t) e].ff[t], FF, split_multi_[(size_t) e].a2[t]);
                else native_quant_h(f, split_multi_[(size_t) e].ff[t], split_multi_[(size_t) e].hq[t]);''',
'''                if (f.d_type == 42) {
                    ServiceClockScope service(service_budget_ ? service_budget_->counter(-1,4,mjobs_[e].nt) : nullptr);
                    act_quant_any(split_multi_[(size_t) e].ff[t], FF, split_multi_[(size_t) e].a2[t]);
                } else {
                    ServiceClockScope service(service_budget_ ? service_budget_->counter(-1,1,mjobs_[e].nt) : nullptr);
                    native_quant_h(f, split_multi_[(size_t) e].ff[t], split_multi_[(size_t) e].hq[t]);
                }''')

edit('src/program/generate.cpp','#include "strata/core/expert_source.hpp"',
     '#include "strata/core/expert_source.hpp"\n#include "strata/core/decode_service_budget.hpp"')
edit('src/program/generate.cpp','''            int64_t dec_windows = 0, dec_T = 0;
            const int64_t decode_hits0''',
'''            int64_t dec_windows = 0, dec_T = 0;
            const strata::core::DecodeLedgerSources decode_sources{
                {&ver.ms_wait,&ver.ms_pool,&ver.ms_host,&ver.ms_commit,&drive.d.ms_plan,&drive.d.ms_actq,
                 &drive.d.ms_jobs,&drive.d.ms_run,&pool.ms_multi_gu,&pool.ms_multi_q,&pool.ms_multi_down,
                 &dt_run,&dt_commit,&dt_draft},
                {&ver.windows,&drive.d.multi_misses,&drive.d.multi_entries,&drive.d.cache_hits,&drive.d.pcie_experts,&pool.multi_bytes}};
            // This root ordinary serve loop has neither batch admission nor a lookup-chain mode.
            strata::core::DecodeServiceBudgetScope decode_service(drive.d,g.n_layers,K,
                strata::core::decode_route_ledger_eligible(!geni && native_pack && !multi_gpu && !o.no_pool &&
                    o.spec_oracle.empty() && o.spec_corrupt==0,0,n_stages,!o.mtp.empty(),o.suffix_draft,0,S,o.spec_split),
                ids.data(),n,max_new,o.max_context,decode_sources);
            const int64_t decode_hits0''')
edit('src/program/generate.cpp','''                if (p + T > o.max_context) break;''',
'''                if (p + T > o.max_context) { decode_service.context_limit(); break; }''')
edit('src/program/generate.cpp','''                const Clock::time_point tw0 = Clock::now();
                if (!ver.run(T, window.data(), p, win_pool_fn''',
'''                const Clock::time_point tw0 = Clock::now();
                decode_service.begin_window(T);
                if (!ver.run(T, window.data(), p, win_pool_fn''')
edit('src/program/generate.cpp','''                int a = 0;
                while (a < T - 1 && window[(size_t) a + 1] == outv[(size_t) a]) ++a;
                if (from_sfx)''',
'''                decode_service.verified(T);
                int a = 0;
                while (a < T - 1 && window[(size_t) a + 1] == outv[(size_t) a]) ++a;
                if (from_sfx)''')
edit('src/program/generate.cpp','''                // the window's first a + 1 tokens are in the session now (the last output is not: it is next x)''',
'''                decode_service.committed(a,a+1);
                // the window's first a + 1 tokens are in the session now (the last output is not: it is next x)''')
edit('src/program/generate.cpp','''                bool eos = false;
                for (int i = 0; i <= a && produced_n < max_new''',
'''                bool eos = false;
                const int64_t decode_emit_start=produced_n;
                for (int i = 0; i <= a && produced_n < max_new''')
edit('src/program/generate.cpp','''                std::fflush(stdout);
                ++rounds;
                const Clock::time_point tw2''',
'''                std::fflush(stdout);
                decode_service.emitted(produced_n-decode_emit_start,eos || produced_n>=max_new);
                ++rounds;
                const Clock::time_point tw2''')
edit('src/program/generate.cpp','''            if (dec_timing && dec_windows > 0) {''',
'''            decode_service.wait_commit_succeeded();
            if (dec_timing && dec_windows > 0) {''')
edit('src/program/generate.cpp','''            if (drive.routing != nullptr) std::fflush(drive.routing);   // the routing trace survives a crash and is watchable mid-session''',
'''            decode_service.finish(finish,produced_n,draft_offered,draft_accepted,decode_ms);
            if (drive.routing != nullptr) std::fflush(drive.routing);   // the routing trace survives a crash and is watchable mid-session''')

digest=lambda s:hashlib.sha256(s.encode()).hexdigest()
patch=''.join(''.join(difflib.unified_diff(base[p].splitlines(True),proposed[p].splitlines(True),
    fromfile='a/'+p if base[p] else '/dev/null',tofile='b/'+p)) for p in sorted(base))
(OUT/'unified.patch').write_text(patch)
identity={'status':'artifact-only; unapplied/unbuilt/unrun/unadopted','branch':'feature/sycl',
          'source_revision_note':'No current Git inspection; byte hashes are authoritative.',
          'origins':['8e85169a4002e08671e09d84139390e110f2cdc5','1cd442981faf137e4370fffdc9d73ab91cce8c7a'],
          'sources':[{'path':p,'base_sha256':digest(base[p]) if base[p] else None,'proposed_sha256':digest(proposed[p]),
                      'base_bytes':len(base[p].encode()),'proposed_bytes':len(proposed[p].encode())} for p in sorted(base)],
          'patch_sha256':digest(patch),'gate':'both STRATA_DECODE_ROUTE_LEDGER=1 and STRATA_DECODE_SERVICE_CPU=1 exactly',
          'scope':'Linux/SYCL ordinary single-device native serve MTP/spec4/splitOFF/suffix0; bounded64layers128workers',
          'state_cap':262144,'close_output_cap':262144,'fixed_combined_cap':524288}
(OUT/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
(OUT/'transform-anchors.json').write_text(json.dumps(anchors,indent=2)+'\n')
print(json.dumps({'patch_sha256':identity['patch_sha256'],'product_files_written':0,'sources':len(base)}))
