# SOURCE-ONLY packaging: reads source bytes, writes ONLY this proposal directory.
from pathlib import Path
import hashlib, difflib, json
ROOT=Path('/home/yayoi/orca/workspaces/Strata/sycl')
OUT=ROOT/'bench/results/2026-10-15-pr6-recovery-audit/prefill-teacherforced-nll-v1'
changes={}
def edit(name, fn):
    old=(ROOT/name).read_text(); changes[name]=(old,fn(old))
def replace(s,a,b):
    if s.count(a)!=1: raise ValueError('nonunique anchor: '+a[:90])
    return s.replace(a,b)
writer=r'''#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace strata::program::prefill_nll {
// Native endian records: int32 gold, float32 NLL; position is record index.
// Tokens are stored exactly in the same file, including the first input token.
// Whole raw file <= 16 + 4*262144 + 8*262143 = 3145736 bytes;
// metadata <= 4096 bytes. Completion is the exclusively published .meta link.
class Writer {
public:
    Writer() = default;
    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;
    ~Writer() { if (data_) std::fclose(data_); if (meta_) std::fclose(meta_); }
    bool begin(const std::string& path, const std::vector<int64_t>& tokens,
               int64_t vocab, int head_type, std::string& err) {
        static_assert(sizeof(float)==4 && sizeof(int32_t)==4);
        if (started_ || failed_ || path.empty() || tokens.size()<2 || tokens.size()>262144 ||
            vocab<=0 || vocab>std::numeric_limits<int32_t>::max())
            return fail(err,"invalid dimensions or destination");
        // Validate ALL IDs before narrowing, file creation, upload, or head execution.
        for (int64_t id:tokens) if (id<0 || id>=vocab) return fail(err,"input token outside native vocab");
        tokens_.reserve(tokens.size());
        for (int64_t id:tokens) tokens_.push_back(int32_t(id));
        n_=int64_t(tokens.size())-1;
        host_.resize(size_t(n_));
        final_=path+".meta"; pending_=final_+".partial";
        std::error_code ec;
        if (std::filesystem::exists(final_,ec) || ec) return fail(err,"metadata exists or cannot be checked");
        data_=std::fopen(path.c_str(),"wbx");
        if (!data_) return fail(err,"exclusive raw creation failed");
        meta_=std::fopen(pending_.c_str(),"wbx");
        if (!meta_) return fail(err,"exclusive partial metadata creation failed");
        const int32_t header[4]={0x314c4c4e,int32_t(tokens.size()),int32_t(vocab),0};
        if (std::fwrite(header,sizeof(header),1,data_)!=1 ||
            std::fwrite(tokens_.data(),4,tokens_.size(),data_)!=tokens_.size() ||
            std::fprintf(meta_,"strata-prefill-nll-v1\nformat=native-endian-int32-header4-tokens-gold-float32-nll\n"
                "scope=initial-full-batched-prefill\nhead=native\nhead_ggml_type=%d\nvocab=%lld\n"
                "input_tokens=%zu\nexpected_rows=%lld\nresidual_callback=fp32-final-layer\n"
                "storage_dtype_placement=external-recipe-required\ninput_identity=exact-int32-tokens-in-raw\n",
                head_type,(long long)vocab,tokens.size(),(long long)n_)<0)
            return fail(err,"header write failed");
        started_=true; return true;
    }
    bool chunk(int64_t p,int64_t t,std::string& err) {
        if (!started_ || failed_ || seen_!=written_ || p!=seen_ || t<=0 || t>n_-seen_)
            return fail(err,"noncanonical or incomplete callback range");
        seen_+=t; return true;
    }
    const int32_t* gold() const { return tokens_.data()+1; }
    float* host() { return host_.data(); }
    int64_t rows() const { return n_; }
    bool append(int64_t p,int64_t t,std::string& err) {
        if (!started_ || failed_ || p!=written_ || t<=0 || t!=seen_-written_)
            return fail(err,"invalid readback coverage");
        // Inspect entire chunk BEFORE writing any of it.
        for (int64_t i=p;i<p+t;++i) if (!std::isfinite(host_[size_t(i)]) || host_[size_t(i)]<0)
            return fail(err,"nonfinite or negative NLL");
        for (int64_t i=p;i<p+t;++i) {
            const int32_t id=tokens_[size_t(i+1)]; const float nll=host_[size_t(i)];
            if (std::fwrite(&id,4,1,data_)!=1 || std::fwrite(&nll,4,1,data_)!=1)
                return fail(err,"record write failed");
            ++written_;
        }
        return true;
    }
    // Only after complete prompt/MTP restore AND checked GPU owner retirement.
    bool finish(std::string& err) {
        if (!started_ || failed_ || seen_!=n_ || written_!=n_)
            return fail(err,"incomplete coverage");
        const int32_t count=int32_t(written_);
        if (std::fseek(data_,12,SEEK_SET) || std::fwrite(&count,4,1,data_)!=1)
            return fail(err,"count commit failed");
        const int flush=std::fflush(data_), close=std::fclose(data_); data_=nullptr;
        if (flush || close) return fail(err,"raw flush/close failed");
        const uint64_t bytes=16+4*tokens_.size()+8*uint64_t(written_);
        if (bytes+4096>(4ull<<20) || std::fprintf(meta_,"written_rows=%lld\ncovered_rows=%lld\n"
            "raw_bytes=%llu\ncomplete=true\n",(long long)written_,(long long)seen_,(unsigned long long)bytes)<0)
            return fail(err,"completion write/budget failed");
        const long meta_bytes=std::ftell(meta_);
        const int fm=std::fflush(meta_), cm=std::fclose(meta_); meta_=nullptr;
        if (meta_bytes<0 || meta_bytes>4096 || fm || cm) return fail(err,"metadata size/flush/close failed");
        std::error_code ec;
        // no-clobber publication; rename() can replace a destination and is not used.
        std::filesystem::create_hard_link(pending_,final_,ec);
        if (ec) return fail(err,"exclusive completion publication failed");
        started_=false; return true;
    }
private:
    bool fail(std::string& err,const char* why) { failed_=true; err=std::string("prefill NLL: ")+why; return false; }
    std::FILE *data_=nullptr,*meta_=nullptr;
    std::string final_,pending_;
    std::vector<int32_t> tokens_; std::vector<float> host_;
    int64_t n_=0,seen_=0,written_=0; bool started_=false,failed_=false;
};
} // namespace strata::program::prefill_nll
'''
changes['include/strata/program/prefill_nll.hpp']=('',writer)
kernel=r'''// Diagnostic only: stable FP32 logsumexp, never log(float(probability)).
// Explicit queue and enqueue-only. Invalid target/nonfinite logits yield NaN;
// callers must reject that chunk rather than publish complete quality data.
void row_target_nll(const float *logits, int rows, int vocab, const int32_t *ids,
                    float *nll, void *stream) {
  require(stream != nullptr && rows >= 0 && vocab > 0);
  const auto elements = count(rows, vocab), outputs = count(rows);
  validate_spans({{nll, outputs * 4}},
                 {{logits, elements * 4}, {ids, outputs * 4}});
  auto &q = queue_for(stream); // enforces in-order
  // Require USM in this queue's context; span checks above handle alignment,
  // overflow and all write/read overlap. Allocation extents remain caller-owned.
  for (const void *p : {static_cast<const void *>(logits), static_cast<const void *>(ids),
                        static_cast<const void *>(nll)})
    require(sycl::get_pointer_type(p, q.get_context()) != sycl::usm::alloc::unknown);
  if (!rows) return;
  q.parallel_for(sycl::nd_range<1>(size_t(rows)*256,256), [=](sycl::nd_item<1> it) {
    const int row=int(it.get_group_linear_id()), tid=int(it.get_local_linear_id());
    const int id=ids[row];
    if (id<0 || id>=vocab) { // uniform workgroup branch; never read logits[id]
      if (!tid) nll[row]=sycl::nan(0u);
      return;
    }
    const float *l=logits+size_t(row)*size_t(vocab);
    float m=-INFINITY; int bad=0;
    for (int64_t i=tid;i<vocab;i+=256) {
      const float v=l[i]; bad |= !sycl::isfinite(v); m=sycl::fmax(m,v);
    }
    bad=sycl::reduce_over_group(it.get_group(),bad,sycl::bit_or<int>());
    m=sycl::reduce_over_group(it.get_group(),m,sycl::maximum<float>());
    if (bad) { if (!tid) nll[row]=sycl::nan(0u); return; }
    float sum=0;
    for (int64_t i=tid;i<vocab;i+=256) sum+=sycl::exp(l[i]-m);
    sum=sycl::reduce_over_group(it.get_group(),sum,sycl::plus<float>());
    // Parentheses preserve small NLL near very large common logit offsets.
    if (!tid) nll[row]=(m-l[id])+sycl::log(sum);
  });
}
'''
edit('src/kernels/sycl/sequence.cpp',lambda s:replace(s,'void embedding_gather_dev(',kernel+'void embedding_gather_dev('))
edit('include/strata/kernels/verify_kernels.hpp',lambda s:replace(s,'void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream);','void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream);\n#ifdef STRATA_ENABLE_SYCL\n/// Stable FP32 NLL. Explicit in-order queue; enqueue-only. Invalid IDs/nonfinite logits => NaN.\nvoid row_target_nll(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* nll, void* stream);\n#endif'))
def generate(s):
 s=replace(s,'#include "strata/program/prefill_logits.hpp"','#include "strata/program/prefill_logits.hpp"\n#include "strata/program/prefill_nll.hpp"\n#include "strata/sycl/checked_cleanup.hpp"')
 anchor='    // Normally off. Validate the small position list BEFORE any GPU preflight.'
 pre=r'''    const char* prefill_nll_path = std::getenv("STRATA_PREFILL_NLL");
    const bool collect_prefill_nll = prefill_nll_path != nullptr;
    if (collect_prefill_nll) {
        const bool remote = std::any_of(o.expert_cache_remote.begin(), o.expert_cache_remote.end(),
                                        [](auto n) { return n > 0; });
        if (!*prefill_nll_path || std::getenv("STRATA_PREFILL_LOGITS") ||
            std::getenv("STRATA_PREFILL_LOGITS_POSITIONS") || !have_tokens ||
            o.tokens.size() < 2 || o.tokens.size() > 262144 || o.prefill_chunk <= 0 ||
            o.serve || !o.layer_split.empty() || o.peer_device >= 0 || remote ||
            o.gpu_stages || o.graph_only || o.gpu_only_full || o.native_preset.empty() ||
            (o.prefill_until > 0 && uint64_t(o.prefill_until) < o.tokens.size()-1)) {
            std::fprintf(stderr, "strata generate: STRATA_PREFILL_NLL needs a nonempty exclusive output path, "
                "--native, --prefill, 2..262144 fresh input tokens and full batched scope; "
                "standalone single GPU only, no selected logits/peer/stage/remote/short prefill-until\n");
            return 2;
        }
    }
'''
 s=replace(s,anchor,pre+anchor)
 anchor='    strata::program::prefill_logits::Writer prefill_logits;'
 setup=r'''    strata::program::prefill_nll::Writer prefill_nll;
    // Declared AFTER writer: checked drain/free precede host writer destruction.
    struct PrefillNLLOwner {
        int32_t* gold = nullptr; float* nll = nullptr; cudaStream_t stream{};
        bool active = false;
        bool close() noexcept {
            if (!active) return true;
            if (cudaStreamSynchronize(stream) != cudaSuccess)
                strata::sycl_backend::cleanup_fail_stop("prefill NLL queue drain failed");
            if (gold && cudaFree(gold) != cudaSuccess)
                strata::sycl_backend::cleanup_fail_stop("prefill NLL gold release failed");
            gold = nullptr;
            if (nll && cudaFree(nll) != cudaSuccess)
                strata::sycl_backend::cleanup_fail_stop("prefill NLL output release failed");
            nll = nullptr; active = false; return true;
        }
        ~PrefillNLLOwner() { close(); }
    } prefill_nll_owner;
    if (collect_prefill_nll) {
        try {
            if (!native_head.loaded() || !prefill_nll.begin(prefill_nll_path, o.tokens, n_vocab,
                                                           native_head.type(), err)) {
                if (err.empty()) err = "prefill NLL: native head required";
                std::fprintf(stderr, "strata generate: %s\n", err.c_str()); return 1;
            }
            prefill_nll_owner.stream = main_stream;
            prefill_nll_owner.active = true; // includes partially successful allocations/enqueues
            const size_t bytes = size_t(n_batched) * 4;
            if (cudaMalloc((void**)&prefill_nll_owner.gold, bytes) != cudaSuccess ||
                cudaMalloc((void**)&prefill_nll_owner.nll, bytes) != cudaSuccess ||
                cudaMemcpyAsync(prefill_nll_owner.gold, prefill_nll.gold(), bytes,
                                cudaMemcpyHostToDevice, main_stream) != cudaSuccess) {
                std::fprintf(stderr, "strata generate: prefill NLL allocation/upload failed\n"); return 1;
            }
        } catch (const std::exception& exc) {
            std::fprintf(stderr, "strata generate: prefill NLL open: %s\n", exc.what()); return 1;
        }
    }
'''
 s=replace(s,anchor,setup+anchor)
 anchor='        if (collect_prefill_logits) {\n            // Compose with the already installed MTP callback.'
 cb=r'''        if (collect_prefill_nll) {
            auto previous = std::move(prefill.on_chunk);
            prefill.on_chunk = [&, previous = std::move(previous)]
                (const float* R_rows, int64_t T, int64_t p0, std::string& e) -> bool {
                try {
                    if (!R_rows || !prefill_nll.chunk(p0, T, e)) {
                        if (e.empty()) e = "prefill NLL: null callback residual";
                        return false;
                    }
                    auto fatal = []() noexcept {
                        strata::sycl_backend::cleanup_fail_stop("prefill NLL callback drain failed");
                    };
                    strata::prefill::detail::DraftUploadGuard lease(
                        [&]() noexcept { return cudaStreamSynchronize(main_stream) == cudaSuccess; }, fatal);
                    lease.arm();
                    for (int64_t t = 0; t < T; ++t) {
                        // Same GR read -> NativeHead path as selected-prefill logits.
                        // R is read only; the local BlockBuffers copy redirects its row.
                        strata::core::BlockBuffers bb = ss.block;
                        bb.R = const_cast<float*>(R_rows + size_t(t) * size_t(g.hc * g.n_embd));
                        if (!strata::core::lm_head_mix(wt, g, bb, main_cs, e) ||
                            !native_head.run(bb.mixed, d_logits, main_cs, e)) return false;
                        strata::kernels::row_target_nll(d_logits, 1, int(n_vocab),
                            prefill_nll_owner.gold + p0 + t, prefill_nll_owner.nll + p0 + t, main_cs);
                    }
                    // One bounded readback and checked wait per chunk, never per row.
                    if (cudaMemcpyAsync(prefill_nll.host() + p0, prefill_nll_owner.nll + p0,
                                        size_t(T) * 4, cudaMemcpyDeviceToHost, main_stream) != cudaSuccess) {
                        e = "prefill NLL: chunk readback enqueue failed"; return false;
                    }
                    if (cudaStreamSynchronize(main_stream) != cudaSuccess) fatal();
                    lease.retire();
                    if (!prefill_nll.append(p0, T, e)) return false;
                    return !previous || previous(R_rows, T, p0, e);
                } catch (const std::exception& exc) {
                    e = std::string("prefill NLL: callback: ") + exc.what(); return false;
                }
            };
        }
'''
 s=replace(s,anchor,cb+anchor)
 anchor='        if (collect_prefill_logits && !prefill_logits.finish(err)) {'
 s=replace(s,anchor,'''        // run_prompt_with_mtp_experts has restored MTP experts; retire diagnostic
        // device owners before making any complete artifact visible.
        if (collect_prefill_nll) {
            prefill_nll_owner.close();
            if (!prefill_nll.finish(err)) {
                std::fprintf(stderr, "strata generate: %s\\n", err.c_str()); return 1;
            }
        }
'''+anchor)
 return s
edit('src/program/generate.cpp',generate)
# Patch and SHA pins are reconstructed from exact UTF-8 source, no Git invocation.
patch=''; pins=[]
for name,(old,new) in changes.items():
 patch+=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+name if old else '/dev/null',tofile='b/'+name))
 pins.append({'path':name,'base_sha256':hashlib.sha256(old.encode()).hexdigest() if old else None,'proposed_sha256':hashlib.sha256(new.encode()).hexdigest(),'base_bytes':len(old.encode()),'proposed_bytes':len(new.encode())})
(OUT/'source.diff').write_text(patch)
(OUT/'source-pins.json').write_text(json.dumps({'status':'SOURCE ONLY / NOT BUILT / NOT TESTED / NO GPU','base':'exact current worktree bytes; no Git inspection permitted','files':pins,'patch_sha256':hashlib.sha256(patch.encode()).hexdigest()},indent=2)+'\n')
