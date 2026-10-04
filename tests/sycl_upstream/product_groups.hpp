#pragma once
// Connected original MoE pipeline, checked against pinned CPU dequantization,
// independent Q8_1 bytes, separate/fused stages and changed-state graph replay.
void grouped_test(int gu,int down,int embd,int ff,cudaStream_t s,cudaStream_t other,size_t& cases,size_t& graphs,bool capture){
    current_case="group gu="+std::to_string(gu)+" down="+std::to_string(down)+" H="+std::to_string(embd)+" FF="+std::to_string(ff);
    check(k::native_expert_supported(gu,down,embd,ff),"supported original type/dimension pair refused");
    const auto L=k::native_expert_layout(gu,down,embd,ff);
    Weights g0(gu,ff,embd),u0(gu,ff,embd),d0(down,embd,ff),g1(gu,ff,embd),u1(gu,ff,embd),d1(down,embd,ff);
    const std::array<const Weights*,2> gate{&g0,&g1},up{&u0,&u1},dw{&d0,&d1};
    check(L.gu_row==g0.stride && L.d_row==d0.stride && L.up_off==g0.raw.size() && L.down_off==g0.raw.size()+u0.raw.size() && L.bytes==g0.raw.size()+u0.raw.size()+d0.raw.size(),"original blob layout disagrees with pinned GGML");
    auto blob=[&](int i){std::vector<uint8_t> b;for(const auto* w:{gate[i],up[i],dw[i]})b.insert(b.end(),w->raw.begin(),w->raw.end());return b;};
    const auto blob0=blob(0),blob1=blob(1);Buffer db0(L.bytes),db1(L.bytes);db0.put(blob0,s);db1.put(blob1,s);
    constexpr int cap_groups=7,cap_entries=13,tokens=9,output_rows=17;
    const std::array<int,4> owner{0,1,1,0};
    std::vector<int32_t> starts{2,3,3,8,10,10,10,10},tok{0,0,8,0,4,2,4,6,1,3,0,0,0},dst{0,0,10,1,12,5,0,15,6,8,0,0,0};
    auto pointers=[&](bool flip){std::vector<unsigned long long> p(cap_groups);for(int g=0;g<cap_groups;++g){const int who=(g<4?owner[g]:0)^int(flip);p[g]=reinterpret_cast<uintptr_t>(who?db1.p:db0.p);}return p;};
    auto x=input(tokens,embd);auto cpuq=quantize(x);
    Buffer dx(x.size()*4),dq(cpuq.size()*36),dp(cap_groups*8),ds((cap_groups+1)*4),dn(4),dt(cap_entries*4),dd(cap_entries*4);
    const size_t scratch_bytes=k::native_expert_scratch_bytes(cap_entries,ff),fa=(size_t(cap_entries)*ff*4+255)&~size_t(255),hq_bytes=size_t(cap_entries)*ff/32*36;
    Buffer scratch(scratch_bytes+64),out((size_t(output_rows)*embd+16)*4);
    dx.put(x,s);k::quantize_q8_1_rows(dx.as<float>(),tokens,embd,dq.p,s);auto qb=dq.get<uint8_t>(s);quant_check(qb,cpuq,x);
    dp.put(pointers(false),s);ds.put(starts,s);dt.put(tok,s);dd.put(dst,s);dn.put(std::vector<int32_t>{4},s);
    std::vector<float> initial(out.bytes/4,-1234567.f),baseline_out;std::vector<uint8_t> baseline_scratch;
    auto initialize=[&]{ck(cudaMemsetAsync(scratch.p,0,scratch.bytes,s));ck(cudaMemsetAsync(static_cast<uint8_t*>(scratch.p)+scratch_bytes,0xbd,64,s));out.put(initial,s);};
    auto float_at=[](const std::vector<uint8_t>& b,size_t at){float v;std::memcpy(&v,b.data()+at,4);return v;};
    auto verify=[&](int ng,bool flip,bool v1,bool compare_baseline){
        const auto got=out.get<float>(s);const auto sh=scratch.get<uint8_t>(s);std::vector<Q81> hq(hq_bytes/36);std::memcpy(hq.data(),sh.data()+3*fa,hq_bytes);
        std::set<int> written;
        for(int group=0;group<ng;++group)for(int e=starts[group];e<starts[group+1];++e){
            const int who=owner[group]^int(flip);written.insert(dst[e]);
            for(int r=0;r<ff;++r){const auto [gv,gl]=gate[who]->dot(r,cpuq.data()+size_t(tok[e])*embd/32);const auto [uv,ul]=up[who]->dot(r,cpuq.data()+size_t(tok[e])*embd/32);
                const float actual_g=float_at(sh,(size_t(e)*ff+r)*4),actual_u=float_at(sh,fa+(size_t(e)*ff+r)*4);
                value_check(actual_g,gv,gl,"group gate CPU reference");value_check(actual_u,uv,ul,"group up CPU reference");
                if(v1){const double h=double(actual_g)/(1+std::exp(-double(actual_g)))*actual_u;value_check(float_at(sh,2*fa+(size_t(e)*ff+r)*4),h,std::abs(h),"group SwiGLU independent exponential");}
                if(compare_baseline){check(std::bit_cast<uint32_t>(actual_g)==std::bit_cast<uint32_t>(float_at(baseline_scratch,(size_t(e)*ff+r)*4)),"group gate decode-once order");check(std::bit_cast<uint32_t>(actual_u)==std::bit_cast<uint32_t>(float_at(baseline_scratch,fa+(size_t(e)*ff+r)*4)),"group up decode-once order");bitwise+=2;}
            }
            for(int r=0;r<embd;++r){const auto [v,l1]=dw[who]->dot(r,hq.data()+size_t(e)*ff/32);value_check(got[size_t(dst[e])*embd+r],v,l1,"group down CPU reference");}
            if(compare_baseline){const size_t begin=3*fa+size_t(e)*ff/32*36;check(std::memcmp(sh.data()+begin,baseline_scratch.data()+begin,size_t(ff)/32*36)==0,"fused SwiGLU changed Q8_1 bytes");bytes_checked+=size_t(ff)/32*36;}
        }
        if(v1){std::vector<float> h(size_t(cap_entries)*ff);std::memcpy(h.data(),sh.data()+2*fa,h.size()*4);auto expected=quantize(h);check(std::memcmp(hq.data(),expected.data(),hq_bytes)==0,"separate intermediate Q8_1 independent CPU tree");bytes_checked+=hq_bytes;}
        if(!v1)for(int e=0;e<cap_entries;++e)if(e<starts[0] || e>=starts[ng])for(size_t i=size_t(e)*ff/32*36;i<size_t(e+1)*ff/32*36;++i){check(sh[3*fa+i]==0,"fused stage modified entries outside this call");++guards;}
        for(size_t i=0;i<got.size();++i)if(i>=size_t(output_rows)*embd || !written.count(int(i/embd))){check(got[i]==-1234567.f,"group scatter modified unselected row or guard");++guards;}
        for(size_t i=scratch_bytes;i<sh.size();++i){check(sh[i]==0xbd,"group scratch guard");++guards;}
        if(compare_baseline)equal_bits(got,baseline_out,"group stride/old/fused output changed bits");
        if(baseline_out.empty()){baseline_out=got;baseline_scratch=sh;}
    };
    for(bool old:{true,false})for(bool v1:{true,false}){
        k::iq_set_old_kernels(old);k::native_grouped_set_v1(v1);initialize();
        k::native_expert_grouped(L,dp.as<unsigned long long>(),ds.as<int32_t>(),dn.as<int32_t>(),dd.as<int32_t>(),dt.as<int32_t>(),cap_groups,cap_entries,dq.p,scratch.p,out.as<float>(),s,v1?0:1);
        verify(4,false,v1,!baseline_out.empty());++cases;
    }
    k::iq_set_old_kernels(false);k::native_grouped_set_v1(false);dn.put(std::vector<int32_t>{0},s);initialize();
    k::native_expert_grouped(L,dp.as<unsigned long long>(),ds.as<int32_t>(),dn.as<int32_t>(),dd.as<int32_t>(),dt.as<int32_t>(),cap_groups,cap_entries,dq.p,scratch.p,out.as<float>(),s,2);
    verify(0,false,false,false);++cases;
    if(capture){
        dn.put(std::vector<int32_t>{4},s);initialize();ck(cudaStreamSynchronize(s));
        cudaGraph_t graph{};cudaGraphExec_t exec{};ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));
        k::native_expert_grouped(L,dp.as<unsigned long long>(),ds.as<int32_t>(),dn.as<int32_t>(),dd.as<int32_t>(),dt.as<int32_t>(),cap_groups,cap_entries,dq.p,scratch.p,out.as<float>(),s,1);
        ck(cudaStreamEndCapture(s,&graph));size_t count=0;ck(cudaGraphGetNodes(graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);ck(cudaGraphGetNodes(graph,nodes.data(),&count));int kernels=0;
        for(auto node:nodes){cudaGraphNodeType type;ck(cudaGraphNodeGetType(node,&type));kernels+=type==cudaGraphNodeTypeKernel;}check(kernels==3,"fused original group capture must contain three native kernels");
        ck(cudaGraphInstantiate(&exec,graph,0ull));ck(cudaGraphDestroy(graph));check(out.get<float>(other)==initial,"group graph executed during instantiation");
        // Captured launch choices persist after changing the original flags.
        k::iq_set_old_kernels(true);k::native_grouped_set_v1(true);
        for(int ng:{0,2,4}){const bool flip=ng!=0;x=input(tokens,embd);cpuq=quantize(x);dx.put(x,s);k::quantize_q8_1_rows(dx.as<float>(),tokens,embd,dq.p,s);dp.put(pointers(flip),s);dn.put(std::vector<int32_t>{ng},s);initialize();ck(cudaGraphLaunch(exec,s));verify(ng,flip,false,false);++graphs;}
        ck(cudaGraphExecDestroy(exec));k::iq_set_old_kernels(false);k::native_grouped_set_v1(false);
        // A warmed whole original pipeline remains asynchronous behind a
        // held callback; another native stream still completes GPU work.
        Buffer flag(4096);
        struct Gate{std::promise<void> release;std::shared_future<void> ready=release.get_future().share();bool open=false;void finish(){if(!open){release.set_value();open=true;}}~Gate(){finish();}} gate_hold;
        ck(backend::submit(s,[ready=gate_hold.ready](sycl::queue& q){return q.submit([&](sycl::handler& h){h.host_task([ready]{ready.wait();});});}));
        auto submit=std::async(std::launch::async,[&]{k::native_expert_grouped(L,dp.as<unsigned long long>(),ds.as<int32_t>(),dn.as<int32_t>(),dd.as<int32_t>(),dt.as<int32_t>(),cap_groups,cap_entries,dq.p,scratch.p,out.as<float>(),s,1);});
        const bool returned=submit.wait_for(2s)==std::future_status::ready;if(!returned)gate_hold.finish();submit.get();check(returned,"whole group host submission blocked");
        ck(cudaMemsetAsync(flag.p,0xa7,flag.bytes,other));ck(cudaStreamSynchronize(other));check(cudaStreamQuery(s)==cudaErrorNotReady,"group host call synchronized held pipeline");
        gate_hold.finish();verify(4,true,false,false);++cases;
    }
    check(db0.get<uint8_t>(s)==blob0 && db1.get<uint8_t>(s)==blob1 && dt.get<int32_t>(s)==tok && dd.get<int32_t>(s)==dst && ds.get<int32_t>(s)==starts,"group modified source blob or index arrays");
}
