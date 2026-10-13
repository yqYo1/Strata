// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/platform/memory.hpp"

#include "strata/core/gpu.hpp"
#include <climits>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <cstring>
#include <exception>
#include <stdexcept>

namespace strata::core {

NativeHead::~NativeHead() {
    // Terminate before member destruction can retire a pending DMA source.
    if (upload_.uncertain()) std::terminate();
    upload_.clear([](void* pointer) { strata::gpu::free(pointer); });
}

bool NativeHead::load(const std::string& path, int64_t n_in, int64_t n_out, std::string& err) {
    if (upload_.uncertain()) { err = "native head upload owner is poisoned"; return false; }
    if (loaded()) { err = "native head is already loaded"; return false; }
    if (n_in <= 0 || n_out <= 0 || n_in > INT_MAX || n_out > INT_MAX || n_in % 256) {
        err = "native head requires positive int32 dimensions and whole 256-value rows";
        return false;
    }
    try {
        std::shared_ptr<const void> source_owner;
        const void* source = nullptr;
        uint64_t bytes = bytes_;
        int source_type = source_type_;
        if (source_snapshot_) {
            if (path != source_path_ || n_in != n_in_ || n_out != n_out_) {
                err = "native head immutable source dimensions/path changed"; return false;
            }
            source_owner = source_snapshot_;
            source = source_snapshot_->data();
        } else {
            // Retain the mapping in the staged owner before any upload can submit.
            auto model = std::make_shared<strata::GgufModel>(strata::GgufModel::open(path));
            err = strata::check_architecture(model->meta());
            if (!err.empty()) return false;
            size_t at = 0;
            const strata::TensorInfo* tensor = model->find("output.weight", &at);
            if (!tensor || !strata::kernels::native_mmvq_supported((int) tensor->type) ||
                tensor->shape.size() != 2 || tensor->shape[0] != (uint64_t) n_in ||
                tensor->shape[1] != (uint64_t) n_out) {
                err = "native head: expected a natively supported output.weight with the canonical head dimensions";
                return false;
            }
            const strata::GgufFile& gguf = model->shard(at);
            source_type = (int) tensor->type;
            bytes = strata::kernels::native_mmvq_weight_bytes(source_type, (int) n_in, (int) n_out);
            if (!bytes || bytes > std::numeric_limits<size_t>::max() ||
                !model->in_bounds(*tensor, at) || strata::tensor_payload_bytes(*tensor) != bytes) {
                err = "native head: truncated or inconsistent output.weight payload"; return false;
            }
            source = gguf.tensor_data(*tensor);
            strata::platform::advise_willneed(source, bytes);
            source_path_ = path;
            source_type_ = source_type; bytes_ = bytes;
            n_in_ = (int) n_in; n_out_ = (int) n_out;
            const char* lease = std::getenv("STRATA_PHASE_HEAD_LEASE");
            if (lease && std::strcmp(lease, "1") == 0) {
                // Copy exactly the validated raw GGUF payload BEFORE its first H2D.
                source_snapshot_ = HeadUploadOwner::snapshot(source, (size_t) bytes);
                source_owner = source_snapshot_; source = source_snapshot_->data();
            } else {
                source_owner = model; // default: no additional CPU payload storage
            }
        }
        int type = source_type;
        const bool status = upload_.stage(source_owner, source, (size_t) bytes,
            strata::kernels::native_q8_1_bytes((int) n_in, 1),
            [](void** pointer, size_t size) { return strata::gpu::alloc_device(pointer, size); },
            [](void* pointer) { strata::gpu::free(pointer); },
            [](void* dest, const void* src, size_t size) { return strata::gpu::copy(dest, src, size); },
            [&](void* weights) {
                if (type == 14 && strata::kernels::native_q6_k_rows_ok((int) n_in)) {
                    sycl::queue& q = strata::core::Runtime::get().compute();
                    strata::kernels::native_q6_k_to_rows(weights, (int) n_in, (int) n_out, &q);
                    if (source_snapshot_) strata::core::Runtime::get().finish(q); else q.wait();
                    type = strata::kernels::kNativeQ6KRows;
                } else if (type == 8 && strata::kernels::native_q8_0_rows_ok((int) n_in)) {
                    sycl::queue& q = strata::core::Runtime::get().compute();
                    strata::kernels::native_q8_0_to_rows(weights, (int) n_in, (int) n_out, &q);
                    if (source_snapshot_) strata::core::Runtime::get().finish(q); else q.wait();
                    type = strata::kernels::kNativeQ8Rows;
                }
                return true;
            });
        if (!status) {
            err = std::string("native head upload: ") + strata::gpu::last_error();
        } else if (leased_ && type != converted_type_) {
            upload_.clear([](void* pointer) { strata::gpu::free(pointer); });
            err = "native head reload conversion changed"; return false;
        } else {
            converted_type_ = type; leased_ = false;
            weights_ = upload_.weights(); scratch_ = upload_.scratch(); type_ = type;
            return true;
        }
    } catch (const std::exception& error) {
        err = std::string("native head: ") + error.what();
    } catch (...) {
        err = "native head: unknown upload/conversion exception";
    }
    if (upload_.uncertain()) {
        std::ostringstream retained;
        retained << "; uncertain completion; retained weights=" << upload_.weights()
                 << " bytes=" << upload_.bytes() << " scratch=" << upload_.scratch()
                 << " scratch_bytes=" << upload_.scratch_bytes() << " source=" << upload_.source();
        err += retained.str();
    }
    return false;
}

uint64_t NativeHead::scratch_bytes() const {
    return n_in_ > 0 ? strata::kernels::native_q8_1_bytes(n_in_, 1) : 0;
}

bool NativeHead::release_for_prefill(DrainedReaders proof, std::string& err) {
    if (upload_.uncertain() || !source_snapshot_ || !proof.device_complete || !proof.graphs_retired || !loaded() || leased_) {
        err = "native head release requires loaded owner and explicit drained/retired readers"; return false;
    }
    upload_.clear([](void* pointer) { strata::gpu::free(pointer); });
    scratch_ = nullptr; weights_ = nullptr;
    leased_ = true;
    return true;
}

bool NativeHead::reload_after_prefill(std::string& err) {
    if (upload_.uncertain() || !leased_ || loaded() || !source_snapshot_) {
        err = "native head is not safely leased"; return false;
    }
    return load(source_path_, n_in_, n_out_, err);
}

bool NativeHead::run(const float* mixed, float* logits, void* stream, std::string& err) const {
    if (!loaded() || !mixed || !logits || !stream) {
        err = "native head requires loaded weights, device buffers and an explicit stream";
        return false;
    }
    try {
        if (type_ == 13) {
            strata::kernels::native_q5_k_f32(weights_, mixed, scratch_, logits, n_in_, n_out_, 1, stream);
        } else {
            strata::kernels::native_quantize_q8_1(mixed, scratch_, n_in_, 1, stream);
            strata::kernels::native_mmvq(type_, weights_, scratch_, logits, n_in_, n_out_, 1, stream);
        }
    } catch (const std::exception& error) {
        err = std::string("native head launch: ") + error.what();
        return false;
    }
    return true;
}

// ================================ plan v0.3 P6: THE NATIVE EMBEDDING ================================

namespace {
const NativeEmbed* g_embed = nullptr;
}
void set_native_embed(const NativeEmbed* e) { g_embed = e; }
const NativeEmbed* native_embed() { return g_embed; }

NativeEmbed::~NativeEmbed() {
    for (void* c : copies_)
        if (c) strata::gpu::free(c);
    if (host_) strata::gpu::free(host_);
}

const void* NativeEmbed::table_for(void* stream) const {
    const Runtime* table_rt = Runtime::owner(host_);
    const Runtime* run_rt = stream ? Runtime::owner_of_stream(stream) : &Runtime::get();
    if (!table_rt || !run_rt || table_rt == run_rt) return dev_;
    const int dev = run_rt->ordinal();
    std::lock_guard<std::mutex> lock(copies_mutex_);
    if (copies_.size() <= (size_t) dev) copies_.resize((size_t) dev + 1, nullptr);
    if (!copies_[(size_t) dev]) {
        const OnDevice on(dev);
        void* c = nullptr;
        if (!strata::gpu::alloc_host(&c, bytes_))
            throw std::runtime_error("native embedding: cannot pin a copy of " + std::to_string(bytes_ >> 20) +
                                     " MiB for GPU " + std::to_string(dev));
        std::memcpy(c, host_, bytes_);
        copies_[(size_t) dev] = c;
    }
    return copies_[(size_t) dev];
}

bool NativeEmbed::load(const std::string& path, int64_t n_embd, int64_t n_vocab, std::string& err) {
    try {
        // token_embd.weight from whichever shard of `path`'s model holds it (upstream 02cfe36)
        const strata::GgufModel model = strata::GgufModel::open(path);
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
        strata::platform::advise_willneed(gguf.tensor_data(*t), bytes_);   // copied out of the mapping below
        if (!strata::gpu::alloc_host(&host_, bytes_)) {
            host_ = nullptr;
            err = "native embedding: cannot pin " + std::to_string(bytes_ >> 20) + " MiB";
            return false;
        }
        std::memcpy(host_, gguf.tensor_data(*t), bytes_);
        void* d = nullptr;
        if (!strata::gpu::device_pointer(&d, host_)) {
            err = "native embedding: no device alias for the mapped table";
            return false;
        }
        dev_ = d;
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
    strata::kernels::iq_embed_rows(type_, table_for(stream), row_, tokens, n_tok, n_embd_, out, stream);
}

void NativeEmbed::gather_one(int64_t token, float* out, void* stream) const {
    strata::kernels::iq_dequant_f32(type_, (const uint8_t*) table_for(stream) + (size_t) token * row_, n_embd_, out,
                                    stream);
}

}  // namespace strata::core
