#!/usr/bin/env python3
"""Regenerate proposal diff/pins only; never modifies product or executes anything."""
from pathlib import Path
import difflib, hashlib, json
root=Path('/home/yayoi/orca/workspaces/Strata/sycl')
out=Path(__file__).resolve().parent
changes={}
def edit(path, old,new):
    current=changes.get(path,(root/path).read_text())
    assert current.count(old)==1, (path,old[:120],current.count(old))
    changes[path]=current.replace(old,new,1)
p='include/strata/core/expert_source.hpp'
edit(p,'class DecodeRouteLedger;','class DecodeRouteLedger;\nstruct CpuGuReuseObserver;')
edit(p,'    DecodeRouteLedger* decode_route_ledger=nullptr;', '    CpuGuReuseObserver* cpu_gu_reuse=nullptr; // issuer-only, borrowed request owner\n    DecodeRouteLedger* decode_route_ledger=nullptr;')
p='include/strata/core/decode_service_budget.hpp'
edit(p,'#include "strata/core/decode_route_ledger.hpp"','#include "strata/core/decode_route_ledger.hpp"\n#include "strata/core/cpu_gu_reuse_observer.hpp"')
edit(p,'    std::unique_ptr<DecodeRouteLedger> state_;','    std::unique_ptr<DecodeRouteLedger> state_;\n    std::unique_ptr<CpuGuReuseObserver> gu_reuse_;')
p='src/core/expert_source.cpp'
edit(p,'    d.decode_route_ledger=state_.get(); // after prompt; one synchronous ordinary issuer','''    d.decode_route_ledger=state_.get(); // after prompt; one synchronous ordinary issuer
    const char* gu_env=std::getenv("STRATA_CPU_GU_REUSE_OBSERVER");
    if (gu_env && !std::strcmp(gu_env,"1") && layers==48 && d.n_expert==512 && !d.cpu_gu_reuse) {
        int types[48];
        for (int l=0;l<48;++l) types[l]=layout.fmt[(size_t)l].gu_type;
        gu_reuse_.reset(new(std::nothrow) CpuGuReuseObserver(types));
        if (gu_reuse_) {
            for (int l=0;l<48;++l) {
                const auto& f=layout.fmt[(size_t)l];
                const size_t block=f.gu_type==18?98:f.gu_type==22?82:f.gu_type==21?110:0;
                if (f.n_embd!=2560 || f.n_ff!=640 ||
                    (block && (f.gu_row!=10*block || f.up_off!=640*f.gu_row || f.down_off!=2*640*f.gu_row)))
                    gu_reuse_->invalid=true;
            }
            d.cpu_gu_reuse=gu_reuse_.get();
        }
    }''')
edit(p,'    std::unique_ptr<char[]> bytes(new(std::nothrow) char[DecodeLedgerOutput::kCap]);','''    if (gu_reuse_) {
        if (dispatch_.cpu_gu_reuse!=gu_reuse_.get()) gu_reuse_->invalid=true;
        else dispatch_.cpu_gu_reuse=nullptr;
        std::unique_ptr<char[]> gu_bytes(new(std::nothrow) char[CpuGuReuseObserver::OutputCap]);
        if (gu_bytes) {
            const bool normal=status && (!std::strcmp(status,"length") || !std::strcmp(status,"stop"));
            bool valid=state_->complete(normal,emitted,offered,accepted) && !state_->invalid && !state_->overflow;
            for (unsigned nt=0;nt<4;++nt) {
                uint64_t total=0;
                for (unsigned f=0;f<4;++f) if (!gu_reuse_->add(total,gu_reuse_->format_nt[f][nt])) valid=false;
                if (total!=state_->job_nt[nt]) valid=false;
            }
            const size_t used=gu_reuse_->receipt(gu_bytes.get(),sequence_,state_->cpu_jobs,state_->cpu_rows,valid);
            std::fwrite(gu_bytes.get(),1,used,stderr);
        } else std::fprintf(stderr,"strata cpu_gu_reuse_v1 request=%llu complete=0 reason=output_allocation\\n",(unsigned long long)sequence_);
    }
    std::unique_ptr<char[]> bytes(new(std::nothrow) char[DecodeLedgerOutput::kCap]);''')
edit(p,'    auto* const decode_observer=d.decode_route_ledger;','''    auto* const decode_observer=d.decode_route_ledger;
    auto* const gu_observer=d.cpu_gu_reuse;
    int16_t gu_job_expert[kMaxWindowEntries]; // initialized only at successful CPU job creation
    uint64_t gu_completed_rows=0;''')
edit(p,'                nj.blob = b;\n                nj.nt = 0;','''                nj.blob = b;
                nj.nt = 0;
                // e is the exact source identity used by blob(d.layers,e), not a route slot.
                if (gu_observer) gu_job_expert[(size_t)jo]=(int16_t)e;''')
edit(p,'    pt("run", njobs);\n    if (native) {','''    pt("run", njobs);
    if (gu_observer) {
        if (!native || !decode_observer || !decode_observer->active || njobs<0 || njobs>kMaxWindowEntries)
            gu_observer->invalid=true;
        else for (int j=0;j<njobs;++j) {
            const auto& job=d.jobs_multi[(size_t)j];
            if (!job.blob) gu_observer->invalid=true;
            gu_observer->touch((int)d.layers,gu_job_expert[j],job.nt);
            if(job.nt>=1 && job.nt<=4) gu_completed_rows+=(uint64_t)job.nt;
        }
    }
    if (native) {''')
edit(p,'        // Native batches have completed and workers parked before this returns.','''        // Native batches have completed and workers parked before this returns.
        if (gu_observer) gu_observer->completed((uint64_t)njobs,gu_completed_rows);''')
parts=[]
for path,new in changes.items():
    parts.extend(difflib.unified_diff((root/path).read_text().splitlines(True),new.splitlines(True),fromfile='a/'+path,tofile='b/'+path))
path='include/strata/core/cpu_gu_reuse_observer.hpp'
parts.extend(difflib.unified_diff([], (out/'cpu_gu_reuse_observer.hpp').read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+path))
(out/'source.diff').write_text(''.join(parts))
refs=['AGENTS.md','docs/ARTIFACT_RETENTION.md','.claude/skills/strata-upstream-sync/SKILL.md','include/strata/core/decode_route_ledger.hpp','include/strata/kernels/cpu/native_expert.hpp','include/strata/kernels/cpu/expert_layout.hpp','src/kernels/cpu/native_expert.cpp','CMakeLists.txt','cmake/sycl_backend.cmake','src/program/generate.cpp']
refs+=list(changes)
base='bench/results/2026-10-15-pr6-recovery-audit/'
refs += [base+x for x in ['research-index.json','research-lossless-cpu-predecode-round595-v1.txt','research-cpu-predecode-tradeoff-round598-v1.txt','cpu-predecode-probe-v1/README.ja.md','cpu-predecode-probe-v1/pins.json']]
(out/'pins.json').write_text(json.dumps({'status':'proposal_only_uncompiled_unrun_unapplied','inputs_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sorted(set(refs))},'proposed_product_paths':list(changes)+[path],'owner':'Main; apply/rebase/build/mock/model runs serialized by Main only'},indent=2)+'\n')
