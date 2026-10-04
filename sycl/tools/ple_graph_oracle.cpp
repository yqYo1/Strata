// PLE fixture outputs from ggml's CPU graph, using dequantized artifact weights.
// The nodes follow llama.cpp qwen4exp.cpp::build_ple at the pinned ggml revision.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) try {
    if (argc != 3) throw std::runtime_error("usage: ple_graph_oracle ple_in.bin ple_out.bin");
    FILE* in = std::fopen(argv[1], "rb");
    if (!in) throw std::runtime_error("cannot open input");
    int32_t dims[5]; float eps;
    if (std::fread(dims, 4, 5, in) != 5 || std::fread(&eps, 4, 1, in) != 1)
        throw std::runtime_error("input header truncated");
    const int64_t N=dims[0], HC=dims[1], T=dims[2], K=dims[3], D=dims[4], H=N*HC, hist=(K-1)*D;
    if (N != 2560 || HC != 4 || T < 1 || T > 64 || K != 4 || D != 3 || eps != 1e-6f)
        throw std::runtime_error("unexpected input geometry");
    ggml_init_params params{256u*1024u*1024u, nullptr, false};
    ggml_context* ctx = ggml_init(params);
    if (!ctx) throw std::runtime_error("ggml context allocation failed");
    auto read = [&](int64_t n0, int64_t n1) {
        auto* t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n0, n1);
        if (std::fread(t->data, 4, n0*n1, in) != (size_t)(n0*n1))
            throw std::runtime_error("input tensor truncated");
        return t;
    };
    auto* emb=read(N,T); auto* hidden=read(H,T);
    auto* wk=read(N,H); auto* wv=read(N,N);
    auto* nk=read(N,HC); auto* nq=read(N,HC); auto* nc=read(N,HC);
    auto* wc=read(K,H); auto* state=read(hist,H);
    if (std::fgetc(in) != EOF) throw std::runtime_error("input has trailing data");
    std::fclose(in);
    auto norm = [&](ggml_tensor* x, ggml_tensor* w) {
        return ggml_mul(ctx, ggml_rms_norm(ctx, ggml_reshape_3d(ctx,x,N,HC,T),eps),w);
    };
    auto* key=norm(ggml_mul_mat(ctx,wk,emb),nk);
    auto* value=ggml_mul_mat(ctx,wv,emb);
    auto* query=norm(hidden,nq);
    auto* s=ggml_scale(ctx,ggml_sum_rows(ctx,ggml_mul(ctx,key,query)),1.0f/std::sqrt(float(N)));
    auto* mag=ggml_sqrt(ctx,ggml_clamp(ctx,ggml_abs(ctx,s),1e-6f,INFINITY));
    auto* gate=ggml_sigmoid(ctx,ggml_mul(ctx,ggml_sgn(ctx,s),mag));
    auto* v3=ggml_repeat_4d(ctx,ggml_reshape_3d(ctx,value,N,1,T),N,HC,T,1);
    auto* gated=ggml_mul(ctx,v3,gate);
    auto* normalized=ggml_reshape_2d(ctx,norm(ggml_reshape_2d(ctx,gated,H,T),nc),H,T);
    auto* transposed=ggml_cont(ctx,ggml_transpose(ctx,normalized));
    auto* padded=ggml_concat(ctx,state,transposed,0);
    ggml_tensor* conv=nullptr;
    for (int64_t k=0;k<K;++k) {
        const int64_t start=hist-(K-1-k)*D;
        auto* shifted=ggml_cont(ctx,ggml_transpose(ctx,
            ggml_view_2d(ctx,padded,T,H,padded->nb[1],start*padded->nb[0])));
        auto* weight=ggml_reshape_1d(ctx,ggml_cont(ctx,
            ggml_view_2d(ctx,wc,1,H,wc->nb[1],k*wc->nb[0])),H);
        auto* term=ggml_mul(ctx,shifted,weight);
        conv=conv ? ggml_add(ctx,conv,term) : term;
    }
    conv=ggml_reshape_3d(ctx,ggml_cont(ctx,ggml_silu(ctx,conv)),N,HC,T);
    auto* result=ggml_add(ctx,ggml_reshape_3d(ctx,hidden,N,HC,T),ggml_add(ctx,gated,conv));
    auto* graph=ggml_new_graph(ctx);
    ggml_build_forward_expand(graph,result);
    if (ggml_graph_compute_with_ctx(ctx,graph,6) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("ggml graph compute failed");
    FILE* out=std::fopen(argv[2],"wb");
    if (!out) throw std::runtime_error("cannot open output");
    const int32_t header[]={int32_t(N),int32_t(H),int32_t(T)};
    if (std::fwrite(header,4,3,out) != 3) throw std::runtime_error("output header write failed");
    for (auto* t : {key,value,gate,gated,normalized,conv,result}) {
        if (!ggml_is_contiguous(t)) throw std::runtime_error("oracle output is not contiguous");
        const size_t count=ggml_nelements(t);
        for (size_t i=0;i<count;++i) if (!std::isfinite(static_cast<float*>(t->data)[i]))
            throw std::runtime_error("nonfinite oracle output");
        if (std::fwrite(t->data,4,count,out) != count) throw std::runtime_error("output tensor write failed");
    }
    if (std::fclose(out)) throw std::runtime_error("output close failed");
    ggml_free(ctx);
    std::puts("PASS ggml CPU PLE graph fixture");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr,"ple_graph_oracle: %s\n",e.what());
    return 1;
}
