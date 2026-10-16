// Unexecuted actual-loader/actual-owner fixture. Main owns all compile/GPU work.
#include <cuda_runtime.h>
#include "strata/core/native_dense.hpp"
#include "strata/core/weights.hpp"
#include "strata/core/layout.hpp"
#include "strata/kernels/f16_bits.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#ifdef STRATA_CANONICAL_ALLOC_FAULT
static const char* fault_mode=nullptr;
static int armed=0;
static cudaError_t canonical_fixture_alloc(void** p,size_t n) {
    if(!armed || --armed) return cudaMalloc(p,n);
    static int fake;
    *p=std::strstr(fault_mode,"nonnull")?&fake:nullptr;
    if(std::strstr(fault_mode,"success")) return cudaSuccess;
    if(std::strstr(fault_mode,"oom")) return cudaErrorMemoryAllocation;
    return cudaErrorUnknown;
}
#define cudaMalloc canonical_fixture_alloc
#endif
#include "strata/core/canonical_dense_bank.hpp"
#ifdef STRATA_CANONICAL_ALLOC_FAULT
#undef cudaMalloc
#endif
using namespace strata::core;
static void need(bool b,const std::string& why){if(!b){std::fprintf(stderr,"FAIL %s\n",why.c_str());std::abort();}}
static void gpu(cudaError_t status){need(status==cudaSuccess,"checked runtime call");}
static void payload(const WeightRef& ref,const uint8_t* expected) {
    std::vector<uint8_t> chunk(1<<20);
    for(uint64_t at=0;at<ref.bytes;){size_t n=size_t(std::min<uint64_t>(chunk.size(),ref.bytes-at));
        gpu(cudaMemcpy(chunk.data(),static_cast<const uint8_t*>(ref.data)+at,n,cudaMemcpyDeviceToHost));
        need(!std::memcmp(chunk.data(),expected+at,n),"actual canonical device bytes differ");at+=n;}
}
static void metadata(WeightRef a,WeightRef b) {
#define SAME(f) need(a.f==b.f,"metadata " #f)
    SAME(bytes);SAME(ne0);SAME(ne1);SAME(elements);SAME(kind);SAME(code_bits);SAME(code_bias);SAME(group_elems);
    SAME(codebook_iq4nl);SAME(has_offset);SAME(codes_bytes);SAME(scales_bytes);SAME(offset_bytes);SAME(src_off);SAME(src_bytes);
    SAME(file_id);SAME(scales_fp16);SAME(act_kind);SAME(native_data);SAME(native_q8_1);SAME(native_type);SAME(resident);
#undef SAME
}
// Independent source-file materialization for every captured conversion. This
// reread is fixture-only; the component never reloads/reopens runtime payloads.
static void source_reference(const std::string& pack,const EngineSnapshot::Row& row,const uint8_t* expected) {
    const auto& r=row.original;const char* files[]={"dense.bin","embd.bin","experts.bin","extra.bin"};
    need(r.file_id>=0 && r.file_id<4,"source file id");
    std::ifstream file(pack+"/"+files[r.file_id],std::ios::binary);need(bool(file),"source reopen");
    file.seekg(std::streamoff(r.src_off));std::vector<uint8_t> input(size_t(r.src_bytes)),out(size_t(r.bytes));
    if(!input.empty())file.read(reinterpret_cast<char*>(input.data()),std::streamsize(input.size()));
    need(bool(file),"source exact read");
    if(r.codes_bytes) {
        need(r.codes_bytes<=input.size(),"source code span");std::memcpy(out.data(),input.data(),size_t(r.codes_bytes));
        const auto in_scales=r.scales_fp16?r.scales_bytes/2:r.scales_bytes;
        need(r.codes_bytes+in_scales+r.offset_bytes==r.src_bytes,"independent source planes");
        if(r.scales_fp16) for(uint64_t i=0;i<r.scales_bytes/4;++i){uint16_t h;std::memcpy(&h,input.data()+r.codes_bytes+i*2,2);
            float x=strata::kernels::f32_from_f16(h);std::memcpy(out.data()+r.codes_bytes+i*4,&x,4);}
        else if(r.scales_bytes)std::memcpy(out.data()+r.codes_bytes,input.data()+r.codes_bytes,size_t(r.scales_bytes));
        if(r.offset_bytes)std::memcpy(out.data()+r.codes_bytes+r.scales_bytes,input.data()+r.codes_bytes+in_scales,size_t(r.offset_bytes));
    } else if(r.src_bytes==r.bytes) out=input; // includes raw16 index kinds4/5
    else {
        need(r.src_bytes==uint64_t(r.elements)*4 && r.bytes==uint64_t(r.elements)*2,"promoted source widths");
        for(int64_t i=0;i<r.elements;++i){uint16_t h;
            if(r.kind==WeightKind::Bf16InF32){uint32_t x;std::memcpy(&x,input.data()+i*4,4);h=uint16_t(x>>16);}
            else {need(r.kind==WeightKind::F16InF32,"promoted source kind");float x;std::memcpy(&x,input.data()+i*4,4);h=strata::kernels::f16_from_f32(x);}
            std::memcpy(out.data()+i*2,&h,2);}
    }
    need(out.empty() || !std::memcmp(out.data(),expected,out.size()),"independent converted source differs");
}
int main(int argc,char** argv) {
#ifdef STRATA_CANONICAL_ALLOC_FAULT
    need(argc>=4,"MODE PACK GGUF...");fault_mode=argv[1];--argc;++argv;
#endif
    need(argc>=3,"PACK GGUF...");std::string err,pack=argv[1];std::vector<std::string> shards(argv+2,argv+argc);
    std::set<std::string> skip;need(NativeDense::served_names(shards,true,skip,err),err);
    need(NativeDense::keep_unquantized_ple_key(pack,skip,err),err);skip.insert("token_embd.weight");skip.insert("output.weight");
    uint64_t bytes=0;need(WeightTable::pool_bytes(pack,bytes,err,&skip),err);void* arena=nullptr;gpu(cudaMalloc(&arena,bytes));
    cudaStream_t queue=nullptr;gpu(cudaStreamCreate(&queue));
    {
        WeightTable wt;EngineSnapshot captured;
        need(!wt.load(pack,arena,bytes,err,&skip,&captured,0),"zero snapshot budget accepted");
        need(wt.load(pack,arena,bytes,err,&skip,&captured),err);
        NativeDense native;need(native.load(shards,wt,err,true,true),err);
        // Foreign/interior owner rejected before ownership is consumed.
        void* alias=static_cast<uint8_t*>(arena)+256;
        {CanonicalDenseBank wrong;need(!wrong.adopt(captured,wt,alias,native,err),"interior owner accepted");}
        {auto it=captured.rows.begin();const auto saved=it->second.original.native_data;
            it->second.original.native_data=reinterpret_cast<void*>(uintptr_t(1));
            {CanonicalDenseBank wrong;need(!wrong.adopt(captured,wt,arena,native,err),"changed source/native overlap accepted");}
            it->second.original.native_data=saved;}
        {auto it=captured.rows.begin();const auto at=it->second.offset;need(it->second.original.bytes>0,"payload fixture row");
            captured.bytes[size_t(at)]^=1;
            {CanonicalDenseBank wrong;need(!wrong.adopt(captured,wt,arena,native,err),"changed captured payload accepted");}
            captured.bytes[size_t(at)]^=1;}
        for(const auto& [name,row]:captured.rows){source_reference(pack,row,captured.bytes.data()+row.offset);payload(*wt.find(name),captured.bytes.data()+row.offset);}
        CanonicalDenseBank bank;need(bank.adopt(captured,wt,arena,native,err),err);
        // This fixture's target census is the pinned current IQ3_S pack.
        need(bank.arena_bytes()==1538035200ULL && bank.bank_bytes()==32284672ULL && bank.persistent_bytes()==82821120ULL,"target canonical inventory/capacity differs");
        const auto original_arena=arena;
#ifdef STRATA_CANONICAL_ALLOC_FAULT
        const bool restoring=std::strstr(fault_mode,"restore")!=nullptr;
        if(!restoring){armed=std::strstr(fault_mode,"second")?2:1;const bool ok=bank.prepare(queue,err);
            if(!std::strstr(fault_mode,"oom-null")){std::fprintf(stderr,"FAIL unexpected ordinary prepare return %d\n",int(ok));std::exit(91);}
            need(!ok && bank.state()==CanonicalDenseBank::State::Full && arena==original_arena,"entry OOM changed owner/state");}
#endif
        need(bank.prepare(queue,err) && native.prepare_layer_bank(err),err);
        need(arena==original_arena && native.residency()==NativeDense::Residency::Full,"prepare withdrew owners");
        need(native.enter_layer_bank({true},err),err);bank.enter();need(arena==nullptr,"exact canonical owner not withdrawn");
        strata::kernels::PleWeights ple;need(bank.bind_ple(ple,err),err);
        for(int layer=0;layer<48;++layer){need(native.activate_layer(layer,queue,{true},err),err);need(bank.activate(layer,queue,native.layer_view(),err),err);
            LayerView current(wt,layer,&bank.view());
            for(const auto& [name,row]:bank.snapshot().rows){const bool persistent=strata::core::detail::canonical_persistent(name);
                if(persistent || strata::core::detail::canonical_layer(name)==layer){const auto it=bank.view().find(name);need(it!=bank.view().end(),"active or persistent row absent");
                    metadata(it->second,row.original);if(!persistent || layer==0)payload(it->second,bank.snapshot().bytes.data()+row.offset);
                }else need(!wt.find(name)->data && !bank.view().count(name),"inactive row remained readable");}
            const int lookahead=std::min(layer+1,47);LayerView next(wt,lookahead,&bank.view());
            need(current.get("hc_ffn_norm.weight")->data && next.get("hc_attn_norm.weight")->data,"HC lookahead pool absent");
            if(layer==1){need(ple.value_bf16 && ple.norm_key && ple.norm_query && ple.norm_conv && ple.conv1d_f16,"PLE persistent bindings absent");}
        }
        gpu(cudaStreamSynchronize(queue));need(native.restore_full(queue,{true},err),err);
#ifdef STRATA_CANONICAL_ALLOC_FAULT
        if(restoring){armed=1;const bool ok=bank.restore_full(queue,err);
            if(!std::strstr(fault_mode,"oom-null")){std::fprintf(stderr,"FAIL unexpected ordinary restore return %d\n",int(ok));std::exit(91);}
            need(!ok && bank.state()==CanonicalDenseBank::State::Bank && !arena,"restore OOM published owner");}
#endif
        need(bank.restore_full(queue,err),err);need(bank.bind_ple(ple,err),err);bank.finish_restore();
        for(const auto& [name,row]:bank.snapshot().rows){metadata(*wt.find(name),row.original);payload(*wt.find(name),bank.snapshot().bytes.data()+row.offset);}
        need(arena && bank.state()==CanonicalDenseBank::State::Full,"full canonical restoration absent");
        std::printf("COMPONENT_ONLY canonical=%llu bank=%llu persistent=%llu rows=%zu; no model/runtime-quality claim\n",
            (unsigned long long)bank.arena_bytes(),(unsigned long long)bank.bank_bytes(),(unsigned long long)bank.persistent_bytes(),bank.snapshot().rows.size());
    }
    gpu(cudaStreamSynchronize(queue));gpu(cudaStreamDestroy(queue));gpu(cudaFree(arena));
}
