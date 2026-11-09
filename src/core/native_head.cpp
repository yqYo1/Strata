#include "strata/core/native_head.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>
#include <climits>
#include <cstdio>
#include <cstring>
#include <exception>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#ifdef STRATA_ENABLE_SYCL
#include "strata/sycl/checked_cleanup.hpp"
#include "../kernels/sycl/q6_dpas_component.hpp"
#endif

namespace strata::core {

NativeHead::~NativeHead() {
#ifdef STRATA_ENABLE_SYCL
    if(head_prepared_ || head_detached_)
        strata::sycl_backend::cleanup_fail_stop("native head teardown while prompt head scope active");
    if(head_release_used_ || decode_dpas_ || primary_packed_) {
        // The caller's reader lease has already drained main/verifier/MTP queues.
        if(scratch_ && cudaFree(scratch_)!=cudaSuccess)strata::sycl_backend::cleanup_fail_stop("native scratch free");
        if(weights_ && cudaFree(weights_)!=cudaSuccess)strata::sycl_backend::cleanup_fail_stop("native restored head free");
        return;
    }
#endif
    if (scratch_) cudaFree(scratch_);
    if (weights_) cudaFree(weights_);
}

bool NativeHead::load(const std::vector<std::string>& shards, int64_t n_in, int64_t n_out, std::string& err) {
    if (head_prepared_ || head_detached_) {err="native head load forbidden during head release";return false;}
    if (loaded()) { err = "native head is already loaded"; return false; }
    if (n_in <= 0 || n_out <= 0 || n_in > INT_MAX || n_out > INT_MAX || n_in % 256) {
        err = "native head requires positive int32 dimensions and whole 256-value rows";
        return false;
    }
    try {
#ifdef STRATA_ENABLE_SYCL
        const char* dpas_value=std::getenv("STRATA_SYCL_DECODE_Q6_DPAS_HEAD");
        if(dpas_value && std::strcmp(dpas_value,"0") && std::strcmp(dpas_value,"1") && std::strcmp(dpas_value,"2")) {
            err="native head: STRATA_SYCL_DECODE_Q6_DPAS_HEAD must be 0 or 1 or 2";return false;
        }
        const bool dpas=dpas_value && std::strcmp(dpas_value,"1")==0;
        const bool packed=dpas_value && std::strcmp(dpas_value,"2")==0;
        const char* release_heads=std::getenv("STRATA_PREFILL_RELEASE_HEADS");
        if((dpas || packed) && (!release_heads || std::strcmp(release_heads,"1"))) {
            err="native head: DPAS requires STRATA_PREFILL_RELEASE_HEADS=1";return false;
        }
#endif
        // The architecture is the metadata shard's; output.weight comes from whichever shard holds it (shard 2
        // of Unsloth's UD-Q4_K_XL, whose shard 1 holds no tensor).  GgufModel refuses a duplicate across shards.
        const strata::GgufModel model(shards);
        err = strata::check_architecture(model.meta());
        if (!err.empty()) return false;
        size_t at = 0;
        const strata::TensorInfo* tensor = model.find("output.weight", &at);
        const strata::GgufFile& gguf = model.shard(at);
        if (!tensor || !strata::kernels::native_mmvq_supported((int) tensor->type) || tensor->shape.size() != 2 ||
            tensor->shape[0] != (uint64_t) n_in || tensor->shape[1] != (uint64_t) n_out) {
            err = "native head: expected a natively supported output.weight with the canonical head dimensions";
            return false;
        }
        const uint64_t bytes = strata::kernels::native_mmvq_weight_bytes((int) tensor->type, (int) n_in, (int) n_out);
        const uint64_t payload = gguf.file_size() - gguf.data_start();
        if (tensor->offset > payload || bytes > payload - tensor->offset) {
            err = "native head: truncated output.weight payload";
            return false;
        }
#ifdef STRATA_ENABLE_SYCL
        if(dpas || packed) {
            // Only this head/C4 cell won the repeated component comparison.
            // Reject unsupported explicit requests before any device allocation.
            if(tensor->type!=14 || n_in!=2560 || n_out!=248320 ||
               !strata::kernels::native_mmvq_multi_exact()) {
                err="native head: DPAS requires exact Q8_1 and Q6_K 2560x248320";return false;
            }
        }
        std::vector<uint16_t> packed_host;
        if(packed) {
            const size_t packed_bytes=strata::kernels::component::q6_dpas_packed_bytes(int(n_in),int(n_out));
            if(packed_bytes!=bytes){err="native head: sole packed owner must preserve exact weight bytes";return false;}
            packed_host.resize(packed_bytes/2);
            strata::kernels::component::q6_dpas_pack(
                {static_cast<const uint8_t*>(gguf.tensor_data(*tensor)),size_t(bytes)},packed_host,int(n_in),int(n_out));
        }
#endif
        void* weights = nullptr;
        void* scratch = nullptr;
        cudaError_t status = cudaMalloc(&weights, bytes);
        if (status == cudaSuccess)
            status = cudaMalloc(&scratch, strata::kernels::native_q8_1_bytes((int) n_in, 1));
        if (status == cudaSuccess) {
            const void* upload=gguf.tensor_data(*tensor);
#ifdef STRATA_ENABLE_SYCL
            if(packed)upload=packed_host.data();
#endif
            status = cudaMemcpy(weights, upload, bytes, cudaMemcpyHostToDevice);
#ifdef STRATA_ENABLE_SYCL
            if((dpas || packed) && status!=cudaSuccess)
                strata::sycl_backend::cleanup_fail_stop("native head initial H2D completion unknown");
#endif
        }
        if (status != cudaSuccess) {
#ifdef STRATA_ENABLE_SYCL
            if(dpas || packed) {
                if(scratch && cudaFree(scratch)!=cudaSuccess)
                    strata::sycl_backend::cleanup_fail_stop("native scratch failed-allocation cleanup");
                if(weights && cudaFree(weights)!=cudaSuccess)
                    strata::sycl_backend::cleanup_fail_stop("native head failed-allocation cleanup");
                err=std::string("native head allocation: ")+cudaGetErrorString(status);return false;
            }
#endif
            if (scratch) cudaFree(scratch);
            if (weights) cudaFree(weights);
            err = std::string("native head upload: ") + cudaGetErrorString(status);
            return false;
        }
        weights_ = weights;
        scratch_ = scratch;
        bytes_ = bytes;
        n_in_ = (int) n_in;
        n_out_ = (int) n_out;
        type_ = (int) tensor->type;
#ifdef STRATA_ENABLE_SYCL
        decode_dpas_=dpas;
        primary_packed_=packed;
        if(dpas) std::fprintf(stderr,"strata native head DPAS: ready=1 columns=4 layout=canonical canonical_bytes=%llu extra_device_weight_bytes=0 extra_host_weight_bytes=0 default_adopted=0\n",
            (unsigned long long)bytes_);
        if(packed)std::fprintf(stderr,"strata native head packed: ready=1 columns=1..4 weight_bytes=%llu canonical_device_weight_bytes=0 extra_device_weight_bytes=0 permanent_host_weight_bytes=0 default_adopted=0\n",
            (unsigned long long)bytes_);
#endif
        return true;
    } catch (const std::exception& error) {
        err = std::string("native head: ") + error.what();
        return false;
    }
}

const void* NativeHead::weights() const {
    if(primary_packed_ && weights_)throw std::logic_error("native head: packed owner has no row-major GPU weights");
    return weights_;
}

void NativeHead::run_packed_columns(const void* q8,float* logits,int columns,void* stream) const {
#ifndef STRATA_ENABLE_SYCL
    (void)q8;(void)logits;(void)columns;(void)stream;
    throw std::invalid_argument("native packed head requires SYCL");
#else
    if(!primary_packed_ || head_detached_ || !weights_ || type_!=14 || n_in_!=2560 || n_out_!=248320 ||
       !strata::kernels::native_mmvq_multi_exact())
        throw std::invalid_argument("native packed head owner/format");
    strata::kernels::component::q6_dpas_columns(static_cast<const uint16_t*>(weights_),q8,logits,n_in_,n_out_,columns,stream);
#endif
}

void NativeHead::run_decode_c4(const void* q8,float* logits,void* stream) const {
#ifndef STRATA_ENABLE_SYCL
    (void)q8;(void)logits;(void)stream;
    throw std::invalid_argument("native DPAS decode head requires SYCL");
#else
    if(!decode_dpas_ready() || type_!=14 || n_in_!=2560 || n_out_!=248320 ||
       !strata::kernels::native_mmvq_multi_exact())
        throw std::invalid_argument("native DPAS decode head owner/format");
    strata::kernels::component::q6_dpas_c4_canonical(weights_,
        q8,logits,n_in_,n_out_,4,stream);
#endif
}

bool NativeHead::run(const float* mixed, float* logits, void* stream, std::string& err) const {
    if (head_detached_) {err="native head is detached for prompt";return false;}
    if (!loaded() || !mixed || !logits || !stream) {
        err = "native head requires loaded weights, device buffers and an explicit stream";
        return false;
    }
    try {
        if (type_ == 13) {
            strata::kernels::native_q5_k_f32(weights_, mixed, scratch_, logits, n_in_, n_out_, 1, stream);
        } else {
            strata::kernels::native_quantize_q8_1(mixed, scratch_, n_in_, 1, stream);
            if(primary_packed_)run_packed_columns(scratch_,logits,1,stream);
            else strata::kernels::native_mmvq(type_, weights_, scratch_, logits, n_in_, n_out_, 1, stream);
        }
    } catch (const std::exception& error) {
        err = std::string("native head launch: ") + error.what();
        return false;
    }
    const cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
        err = std::string("native head launch: ") + cudaGetErrorString(status);
        return false;
    }
    return true;
}

bool NativeHead::prepare_prefill_weights(std::string& err) try {
#ifndef STRATA_ENABLE_SYCL
    err="head release requires SYCL";return false;
#else
    if(head_prepared_ || head_detached_ || !loaded() || !bytes_ || bytes_>(1ull<<30) ||
       bytes_>uint64_t(std::numeric_limits<size_t>::max()) ||
       bytes_!=strata::kernels::native_mmvq_weight_bytes(type_,n_in_,n_out_) ||
       (decode_dpas_ && !decode_dpas_ready())) {
        err="native head release requires ready exact weights within 1 GiB snapshot budget";return false;
    }
    try {head_snapshot_.resize(size_t(bytes_));}
    catch(const std::exception& e){err=std::string("native head RAM snapshot: ")+e.what();return false;}
    head_prepared_=true;return true;
#endif
} catch(const std::exception& e) {err=std::string("native head prepare: ")+e.what();return false;}
void NativeHead::cancel_prefill_weights() noexcept {
#ifdef STRATA_ENABLE_SYCL
    if(!head_prepared_ || head_detached_)strata::sycl_backend::cleanup_fail_stop("invalid native head prepare cancellation");
    head_prepared_=false;std::vector<uint8_t>().swap(head_snapshot_);
#endif
}
void NativeHead::detach_prefill_weights() noexcept {
#ifdef STRATA_ENABLE_SYCL
    using strata::sycl_backend::cleanup_fail_stop;
    if(!head_prepared_ || head_detached_ || !weights_ || head_snapshot_.size()!=bytes_)cleanup_fail_stop("native head detach state");
    // Every head reader was retired by the common request boundary before here.
    if(cudaMemcpy(head_snapshot_.data(),weights_,size_t(bytes_),cudaMemcpyDeviceToHost)!=cudaSuccess)
        cleanup_fail_stop("native head D2H snapshot completion unknown");
    head_detached_=true;head_release_used_=true;
    if(cudaFree(weights_)!=cudaSuccess)cleanup_fail_stop("native head detach free");
    weights_=nullptr;
    std::fprintf(stderr,"strata prefill heads: target detached bytes=%llu source=owned-RAM D2H=%llu H2D-on-restore=%llu\n",
        (unsigned long long)bytes_,(unsigned long long)bytes_,(unsigned long long)bytes_);
#endif
}
void NativeHead::restore_prefill_weights() noexcept {
#ifdef STRATA_ENABLE_SYCL
    using strata::sycl_backend::cleanup_fail_stop;
    if(!head_prepared_ || !head_detached_ || weights_ ||
       head_snapshot_.size()!=bytes_)cleanup_fail_stop("native head restore state");
    if(cudaMalloc(&weights_,size_t(bytes_))!=cudaSuccess)cleanup_fail_stop("native head restore allocation");
    if(cudaMemcpy(weights_,head_snapshot_.data(),size_t(bytes_),cudaMemcpyHostToDevice)!=cudaSuccess)
        cleanup_fail_stop("native head restore H2D completion unknown");
    // Full-forward remains blocked until both target and draft owners are ready.
#endif
}
void NativeHead::resume_prefill_weights() noexcept {
#ifdef STRATA_ENABLE_SYCL
    if(!head_prepared_ || !head_detached_ || !weights_)
        strata::sycl_backend::cleanup_fail_stop("native head resume state");
    head_detached_=false;head_prepared_=false;
    std::vector<uint8_t>().swap(head_snapshot_);
#endif
}

// ================================ plan v0.3 P6: THE NATIVE EMBEDDING ================================

namespace {
const NativeEmbed* g_embed = nullptr;
}
void set_native_embed(const NativeEmbed* e) { g_embed = e; }
const NativeEmbed* native_embed() { return g_embed; }

NativeEmbed::~NativeEmbed() {
    if (host_) cudaFreeHost(host_);
    else if (dev_) cudaFree(const_cast<void*>(dev_));   // the VRAM fallback below
}

bool NativeEmbed::load(const std::vector<std::string>& shards, int64_t n_embd, int64_t n_vocab, std::string& err) {
    try {
        const strata::GgufModel model(shards);
        // --embd-gguf's one-tensor file (tools/embd_bf16_pack.py) says "strata-embd": only its tensor is checked
        const strata::MetaValue* arch = model.meta().get("general.architecture");
        err = arch != nullptr && arch->s == "strata-embd" ? std::string() : strata::check_architecture(model.meta());
        if (!err.empty()) { err = "native embedding: " + err; return false; }
        size_t at = 0;
        const strata::TensorInfo* t = model.find("token_embd.weight", &at);
        const strata::GgufFile& gguf = model.shard(at);
        if (!t || t->shape.size() != 2 || t->shape[0] != (uint64_t) n_embd || t->shape[1] != (uint64_t) n_vocab ||
            !strata::kernels::embed_type_supported((int) t->type) || n_embd % 256) {
            err = "native embedding: token_embd.weight is absent, of another shape, or of a type without a GPU "
                  "dequantizer";
            return false;
        }
        row_ = strata::kernels::iq_row_bytes((int) t->type, n_embd);
        bytes_ = (uint64_t) row_ * (uint64_t) n_vocab;
        // the table is copied out of the mapping below: a truncated shard must be an error, not a read past EOF
        if (!model.in_bounds(*t, at) || strata::tensor_payload_bytes(*t) != bytes_) {
            err = "native embedding: token_embd.weight's payload is truncated or not " + std::to_string(bytes_) +
                  " B (" + gguf.path() + ")";
            bytes_ = 0;
            return false;
        }
        if (cudaHostAlloc(&host_, bytes_, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
            // Under WSL2 the driver's pinned/mapped host budget (~1 GiB) can be spent by the GPU contexts
            // themselves (three cards). The table is only gathered from, so keep it in the current device's VRAM
            // instead: it costs its size there and reads faster than over PCIe.
            cudaGetLastError();
            host_ = nullptr;
            void* d = nullptr;
            if (cudaMalloc(&d, bytes_) != cudaSuccess ||
                cudaMemcpy(d, gguf.tensor_data(*t), bytes_, cudaMemcpyHostToDevice) != cudaSuccess) {
                if (d) cudaFree(d);
                cudaGetLastError();
                err = "native embedding: cannot pin " + std::to_string(bytes_ >> 20) + " MiB, nor place it in VRAM";
                return false;
            }
            std::fprintf(stderr, "strata: native embedding: cannot pin %llu MiB, kept in VRAM instead\n",
                         (unsigned long long) (bytes_ >> 20));
            dev_ = d;
        } else {
            std::memcpy(host_, gguf.tensor_data(*t), bytes_);
            void* d = nullptr;
#if defined(STRATA_USE_HIP)
            // #325: the Windows HIP stack can refuse the device alias of a mapped allocation (and, when it gives
            // one, it is the host address itself - unified addressing; kernels read it correctly there, a
            // device-to-device copy into it does not land: tests/hip/mapped_alias). The table is only gathered from,
            // so without an alias it goes into VRAM like the unpinnable case above, instead of failing the start.
            if (cudaHostGetDevicePointer(&d, host_, 0) != cudaSuccess || d == nullptr) {
                cudaGetLastError();
                d = nullptr;
                if (cudaMalloc(&d, bytes_) != cudaSuccess ||
                    cudaMemcpy(d, gguf.tensor_data(*t), bytes_, cudaMemcpyHostToDevice) != cudaSuccess) {
                    if (d) cudaFree(d);
                    cudaGetLastError();
                    err = "native embedding: no device alias for the mapped table, and no VRAM to copy it into";
                    return false;
                }
                cudaFreeHost(host_);   // the destructor frees dev_ when host_ is null
                host_ = nullptr;
                std::fprintf(stderr, "strata: native embedding: no device alias for the mapped table, kept in VRAM\n");
            }
#else
            if (cudaHostGetDevicePointer(&d, host_, 0) != cudaSuccess) {
                err = "native embedding: no device alias for the mapped table";
                return false;
            }
#endif
            dev_ = d;
        }
        type_ = (int) t->type;
        n_embd_ = n_embd;
        n_vocab_ = n_vocab;
        return true;
    } catch (const std::exception& e) {
        err = std::string("native embedding: ") + e.what();
        return false;
    }
}

void NativeEmbed::gather_dev(const int32_t* tokens, int64_t n_tok, float* out, void* stream) const {
    strata::kernels::iq_embed_rows(type_, dev_, row_, tokens, n_tok, n_embd_, out, stream);
}

void NativeEmbed::gather_one(int64_t token, float* out, void* stream) const {
    strata::kernels::iq_dequant_f32(type_, (const uint8_t*) dev_ + (size_t) token * row_, n_embd_, out, stream);
}

}  // namespace strata::core
