#include "strata/core/native_dense.hpp"
#include "strata/core/native_dense_bank_policy.hpp"
#include "strata/core/weights.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <exception>
#include <limits>
#include <memory>
#include <set>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace strata::core {
namespace {
[[noreturn]] void bank_stop(const char* operation) noexcept;
uint64_t packed_stride(uint64_t bytes) {
    if (bytes > UINT64_MAX - 255) throw std::length_error("native dense bank alignment overflow");
    return (bytes + 255) & ~uint64_t(255);
}
bool eligible(const strata::TensorInfo& tensor, bool include_ple_key) {
    const auto& name = tensor.name;
    if (name.rfind("blk.", 0) != 0) return false;
    // Match the native PLE kernel: Q2_0, IQ3_XXS, IQ4_XS and Q8_0 (UD-Q4_K_XL). Other keys retain the packed BF16
    // fallback.
    if (name == "blk.1.ple_key.weight")
        return include_ple_key && (tensor.type == 42 || tensor.type == 18 || tensor.type == 23 || tensor.type == 8);
    static const char* suffixes[] = {".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight",
        ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_output.weight",
        ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight", ".ffn_down_shexp.weight"};
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
struct DeviceFree {
    bool checked = false;
    void operator()(void* p) const {
        if (p && cudaFree(p) != cudaSuccess && checked) bank_stop("startup pending owner free");
    }
};
using DevicePtr = std::unique_ptr<void, DeviceFree>;
struct Pending {
    WeightRef* ref;
    int type;
    uint64_t bytes;
    DevicePtr data;
    NativeDense::PackedSource source;
};
}

bool NativeDense::served_names(const std::vector<std::string>& shards, bool include_ple_key,
                               std::set<std::string>& out, std::string& err) {
    try {
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            for (const auto& tensor : gguf.tensors())
                if (eligible(tensor, include_ple_key) && strata::kernels::native_mmvq_supported(tensor.type) &&
                    tensor.shape.size() == 2)
                    out.insert(tensor.name);
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}

bool NativeDense::keep_unquantized_ple_key(const std::string& pack_dir, std::set<std::string>& skip,
                                           std::string& err) {
    const std::string key = "blk.1.ple_key.weight";
    if (!skip.count(key)) return true;
    int code_bits = -1;
    if (!WeightTable::index_code_bits(pack_dir, key, code_bits, err)) return false;
    if (code_bits == 0) skip.erase(key);
    return true;
}

NativeDense::~NativeDense() {
    if (residency_ != Residency::Full || bank_) std::abort();
    const bool checked = !sources_.empty();
    if (scratch_ && cudaFree(scratch_) != cudaSuccess && checked) bank_stop("full scratch destruction");
    for (void* p : weights_)
        if (cudaFree(p) != cudaSuccess && checked) bank_stop("full owner destruction");
}

bool NativeDense::load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
                       bool include_ple_key, bool capture_layer_bank, uint64_t layer_bank_ram_limit) {
    if (scratch_ || !weights_.empty()) { err = "native dense: already loaded"; return false; }
    if (shards.empty()) { err = "native dense: at least one GGUF shard is required"; return false; }
    try {
        std::vector<Pending> pending;
        std::set<std::string> seen;
        int max_in = 0;
        uint64_t total = 0;
        uint64_t split_count = 0, split_tensors = 0;
        std::set<uint64_t> split_numbers;
        bool have_architecture = false;
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            const auto* count = gguf.get("split.count");
            const auto* number = gguf.get("split.no");
            const auto* tensors = gguf.get("split.tensors.count");
            if (gguf.get("general.architecture")) {
                err = strata::check_architecture(gguf);
                if (!err.empty()) return false;
                have_architecture = true;
                if (count && number && tensors && number->u == 0 && count->u > 1) {
                    split_count = count->u;
                    split_tensors = tensors->u;
                }
            } else if (!have_architecture || !split_count || !count || !number || !tensors ||
                       count->u != split_count || number->u == 0 || number->u >= split_count ||
                       tensors->u != split_tensors) {
                err = "native dense: additional shard must match the architecture-validated first shard's split metadata";
                return false;
            }
            if (number && !split_numbers.insert(number->u).second) {
                err = "native dense: duplicate split shard number"; return false;
            }
            std::vector<uint64_t> offsets;
            for (const auto& tensor : gguf.tensors()) offsets.push_back(tensor.offset);
            std::sort(offsets.begin(), offsets.end());
            if (std::adjacent_find(offsets.begin(), offsets.end()) != offsets.end()) {
                err = "native dense: tensor payload offsets overlap"; return false;
            }
            // Validate every directory span, including tensors we do not upload:
            // an ignored tensor must not overlap the native matrix that follows it.
            const uint64_t payload = gguf.file_size() - gguf.data_start();
            for (const auto& tensor : gguf.tensors()) {
                int block_elements = 0, block_bytes = 0;
                uint64_t elements = 1;
                if (tensor.shape.empty() || !strata::block_geometry(tensor.type, block_elements, block_bytes) ||
                    tensor.shape[0] % (uint64_t) block_elements != 0) {
                    err = "native dense: invalid block geometry " + tensor.name; return false;
                }
                for (uint64_t dimension : tensor.shape) {
                    if (!dimension || elements > (std::numeric_limits<uint64_t>::max)() / dimension) {
                        err = "native dense: invalid tensor extent " + tensor.name; return false;
                    }
                    elements *= dimension;
                }
                const uint64_t blocks = elements / (uint64_t) block_elements;
                if (blocks > (std::numeric_limits<uint64_t>::max)() / (uint64_t) block_bytes) {
                    err = "native dense: tensor byte count overflow " + tensor.name; return false;
                }
                const uint64_t bytes = blocks * (uint64_t) block_bytes;
                if (tensor.offset > payload || bytes > payload - tensor.offset) {
                    err = "native dense: truncated payload " + tensor.name; return false;
                }
                const auto next = std::upper_bound(offsets.begin(), offsets.end(), tensor.offset);
                if (next != offsets.end() && bytes > *next - tensor.offset) {
                    err = "native dense: overlapping payload " + tensor.name; return false;
                }
            }
            for (const auto& tensor : gguf.tensors()) {
                if (!eligible(tensor, include_ple_key)) continue;
                if (!seen.insert(tensor.name).second) {
                    err = "native dense: duplicate tensor " + tensor.name; return false;
                }
                auto found = table.table_.find(tensor.name);
                if (found == table.table_.end()) {
                    err = "native dense: tensor absent from canonical table: " + tensor.name; return false;
                }
                auto& ref = found->second;
                if (ref.native_data) { err = "native dense: override already attached"; return false; }
                if (!strata::kernels::native_mmvq_supported(tensor.type)) continue;
                // #326: the pack keeps an unquantized (--compat-bf16) key, which the PLE reads from the arena
                if (tensor.name == "blk.1.ple_key.weight" && !ref.quantized()) continue;
                if (!ref.quantized() || tensor.shape.size() != 2 ||
                    ref.ne0 <= 0 || ref.ne0 > INT_MAX || ref.ne1 <= 0 || ref.ne1 > INT_MAX ||
                    tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1) {
                    err = "native dense: incompatible matrix " + tensor.name; return false;
                }
                const auto bytes = strata::kernels::native_mmvq_weight_bytes(
                    tensor.type, (int) ref.ne0, (int) ref.ne1);
                if (capture_layer_bank && (bytes > layer_bank_ram_limit || total > layer_bank_ram_limit - bytes)) {
                    err = "native dense: startup snapshot RAM budget exceeded"; return false;
                }
                PackedSource source;
                if (capture_layer_bank) {
                    const auto dot = tensor.name.find('.', 4);
                    const auto digits = tensor.name.substr(4, dot - 4);
                    if (dot == std::string::npos || digits.empty() ||
                        digits.find_first_not_of("0123456789") != std::string::npos) {
                        err = "native dense: invalid layer name " + tensor.name; return false;
                    }
                    source.layer = std::stoi(digits);
                    source.name = tensor.name; source.shard = path;
                    source.gguf_offset = tensor.offset; source.data_start = gguf.data_start();
                    source.file_bytes = gguf.file_size(); source.original = ref;
                    source.packed.resize(bytes);
                    std::memcpy(source.packed.data(), gguf.tensor_data(tensor), bytes);
                    source.original.native_type = tensor.type;
                }
                void* allocation = nullptr;
                auto status = cudaMalloc(&allocation, bytes);
                DevicePtr data(allocation, DeviceFree{capture_layer_bank});
                if (status == cudaSuccess)
                    status = cudaMemcpy(data.get(), capture_layer_bank ? source.packed.data() : gguf.tensor_data(tensor), bytes, cudaMemcpyHostToDevice);
                if (status != cudaSuccess) {
                    if (capture_layer_bank && (allocation || status != cudaErrorMemoryAllocation))
                        bank_stop("startup allocation/upload uncertainty");
                    err = "native dense upload " + tensor.name + ": " + cudaGetErrorString(status); return false;
                }
                max_in = (std::max)(max_in, (int) ref.ne0);
                total += bytes;
                pending.push_back(Pending{&ref, (int) tensor.type, bytes, std::move(data), std::move(source)});
            }
        }
        if (pending.empty()) { err = "native dense: no supported GDN/QSA matrices in supplied shards"; return false; }
        void* allocation = nullptr;
        const auto status = cudaMalloc(&allocation, strata::kernels::native_q8_1_bytes(max_in));
        DevicePtr scratch(allocation, DeviceFree{capture_layer_bank});
        if (status != cudaSuccess) {
            if (capture_layer_bank && (allocation || status != cudaErrorMemoryAllocation))
                bank_stop("startup scratch allocation uncertainty");
            err = std::string("native dense scratch: ") + cudaGetErrorString(status); return false; }
        // All checks and allocations finish before publishing any reference.
        weights_.reserve(pending.size());
        std::vector<PackedSource> snapshots;
        if (capture_layer_bank) snapshots.reserve(pending.size());
        std::map<int, uint64_t> layer_bytes;
        for (const auto& item : pending) if (capture_layer_bank) {
            if (packed_stride(item.bytes) > UINT64_MAX - layer_bytes[item.source.layer]) {
                err = "native dense: layer byte overflow"; return false;
            }
            layer_bytes[item.source.layer] += packed_stride(item.bytes);
        }
        uint64_t maximum = 0;
        for (const auto& item : layer_bytes) maximum = (std::max)(maximum, item.second);
        for (auto& item : pending) {
            item.ref->native_data = item.data.get();
            item.ref->native_type = item.type;
            item.ref->native_q8_1 = scratch.get();
            weights_.push_back(item.data.release());
            if (capture_layer_bank) {
                item.source.original = *item.ref;
                snapshots.push_back(std::move(item.source));
            }
        }
        scratch_ = scratch.release();
        bytes_ = total;
        sources_ = std::move(snapshots); table_ = capture_layer_bank ? &table : nullptr; bank_bytes_ = maximum;
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}

namespace {
[[noreturn]] void bank_stop(const char* operation) noexcept {
    // abort deliberately bypasses every destructor: a failed drain/free/copy
    // cannot establish which owner the runtime still references.
    std::fprintf(stderr, "native dense layer bank: fail-stop in %s\n", operation);
    std::abort();
}
void checked_bank(cudaError_t status, const char* operation) noexcept {
    if (status != cudaSuccess) bank_stop(operation);
}
bool same_ref(const WeightRef& a, const WeightRef& b) {
#define FIELD(f) if (a.f != b.f) return false
    FIELD(data); FIELD(bytes); FIELD(ne0); FIELD(ne1); FIELD(elements); FIELD(kind);
    FIELD(code_bits); FIELD(code_bias); FIELD(group_elems); FIELD(codebook_iq4nl);
    FIELD(has_offset); FIELD(codes_bytes); FIELD(scales_bytes); FIELD(offset_bytes);
    FIELD(src_off); FIELD(src_bytes); FIELD(file_id); FIELD(scales_fp16); FIELD(act_kind);
    FIELD(native_data); FIELD(native_q8_1); FIELD(native_type); FIELD(resident);
#undef FIELD
    return true;
}
WeightRef withdrawn_ref(WeightRef value) {
    // The normal table is explicitly unavailable, even if canonical bytes
    // happen to be resident. Preserve ALL shape/type/provenance fields.
    value.data = nullptr; value.native_data = nullptr; value.native_q8_1 = nullptr;
    value.resident = false;
    return value;
}
}

bool NativeDense::validate_table(bool withdrawn, std::string& err) const {
    if (!table_ || sources_.empty() || sources_.size() != weights_.size()) {
        err = "native dense: startup layer-bank snapshots unavailable"; return false;
    }
    for (size_t i = 0; i < sources_.size(); ++i) {
        const auto& s = sources_[i];
        const auto* ref = table_->find(s.name);
        const auto expected = withdrawn ? withdrawn_ref(s.original) : s.original;
        if (!ref || !same_ref(*ref, expected) ||
            (!withdrawn && weights_[i] != s.original.native_data) ||
            s.original.native_q8_1 != scratch_ ||
            s.packed.size() != strata::kernels::native_mmvq_weight_bytes(
                s.original.native_type, (int)s.original.ne0, (int)s.original.ne1)) {
            err = "native dense: owner/metadata changed: " + s.name; return false;
        }
    }
    return true;
}

bool NativeDense::enter_layer_bank(ReaderRetirement retirement, std::string& err) {
    if (residency_ != Residency::Full || !retirement.all_readers_retired) {
        err = "native dense: enter requires Full state and retired readers"; return false;
    }
    if (!validate_table(false, err)) return false;
    // Allocate before withdrawing/freeing: allocation failure leaves every
    // original owner and published reference alive. One max-layer allocation.
    void* allocation = nullptr;
    const auto status = cudaMalloc(&allocation, bank_bytes_);
    if (status != cudaSuccess) {
        // No copy has been submitted. An unexpected non-null failed allocation
        // is runtime uncertainty, so do not optimistically free/reuse it.
        if (detail::classify_bank_allocation(status, cudaSuccess, cudaErrorMemoryAllocation, allocation) != detail::BankAllocation::retryable_oom)
            bank_stop("uncertain bank allocation failure");
        err = "native dense: bank allocation failed"; return false;
    }
    if (!allocation) bank_stop("successful bank allocation returned null");
    bank_ = allocation;
    for (const auto& s : sources_) table_->table_.at(s.name) = withdrawn_ref(s.original);
    residency_ = Residency::Bank;
    // Caller, not this component, proved graph/callback retirement. A global
    // device wait cannot replace that proof (a graph could be replayed later).
    for (auto& p : weights_) {
        checked_bank(cudaFree(p), "release original packed owner");
        p = nullptr;
    }
    return true;
}

bool NativeDense::activate_layer(int layer, void* stream, ReaderRetirement retirement, std::string& err) {
    if (residency_ != Residency::Bank || !retirement.all_readers_retired || !stream) {
        err = "native dense: activate requires Bank state, owned stream and retired view readers"; return false;
    }
    if (!validate_table(true, err)) bank_stop("withdrawn table identity");
    std::map<std::string, WeightRef> next;
    uint64_t offset = 0;
    // All fallible host allocations/validation precede retiring the old view.
    try {
        for (const auto& s : sources_) if (s.layer == layer) {
            if (s.packed.size() > bank_bytes_ - offset) bank_stop("bank extent");
            auto ref = s.original;
            ref.native_data = static_cast<uint8_t*>(bank_) + offset;
            next.emplace(s.name, ref);
            offset += packed_stride(s.packed.size());
        }
    } catch (const std::exception& e) { err = e.what(); return false; }
    if (next.empty()) { err = "native dense: unknown native layer"; return false; }
    auto queue = static_cast<cudaStream_t>(stream);
    checked_bank(cudaStreamSynchronize(queue), "retire previous layer readers");
    layer_view_.clear();
    offset = 0;
    for (const auto& s : sources_) if (s.layer == layer) {
        checked_bank(cudaMemcpyAsync(static_cast<uint8_t*>(bank_) + offset, s.packed.data(),
                     s.packed.size(), cudaMemcpyHostToDevice, queue), "layer H2D submit");
        offset += packed_stride(s.packed.size());
    }
    checked_bank(cudaStreamSynchronize(queue), "layer H2D completion");
    layer_view_.swap(next); // complete copies before any pointer publication
    return true;
}

bool NativeDense::restore_full(void* stream, ReaderRetirement retirement, std::string& err) {
    if (residency_ != Residency::Bank || !retirement.all_readers_retired || !stream) {
        err = "native dense: restore requires Bank state, owned stream and retired view readers"; return false;
    }
    if (!validate_table(true, err)) bank_stop("restore table identity");
    std::vector<void*> next;
    try { next.resize(sources_.size(), nullptr); }
    catch (const std::exception& e) { err = e.what(); return false; }
    auto queue = static_cast<cudaStream_t>(stream);
    checked_bank(cudaStreamSynchronize(queue), "restore retire bank readers");
    // Allocate full owners before changing bank/view. On OOM no table is
    // published and the prior bank remains valid/retryable.
    for (size_t i = 0; i < next.size(); ++i) {
        auto status = cudaMalloc(&next[i], sources_[i].packed.size());
        if (status != cudaSuccess) {
            if (next[i] || status != cudaErrorMemoryAllocation) bank_stop("uncertain restore allocation failure");
            for (auto p : next) if (p) checked_bank(cudaFree(p), "restore allocation rollback");
            err = "native dense: full restore allocation failed; bank retained"; return false;
        }
    }
    for (size_t i = 0; i < next.size(); ++i)
        checked_bank(cudaMemcpyAsync(next[i], sources_[i].packed.data(), sources_[i].packed.size(),
                     cudaMemcpyHostToDevice, queue), "restore H2D submit");
    checked_bank(cudaStreamSynchronize(queue), "restore H2D completion");
    if (!validate_table(true, err)) bank_stop("restore prepublication identity");
    // No reader may run during this publication. Assignment of WeightRef and
    // vector swap are non-allocating. The canonical pointer/metadata are exact.
    for (size_t i = 0; i < next.size(); ++i) {
        auto ref = sources_[i].original;
        ref.native_data = next[i];
        table_->table_.at(sources_[i].name) = ref;
        sources_[i].original = ref;
    }
    weights_.swap(next);
    layer_view_.clear();
    checked_bank(cudaFree(bank_), "restore bank free");
    bank_ = nullptr;
    residency_ = Residency::Full;
    // Caller must recapture all graphs: restored addresses may have changed.
    return true;
}

} // namespace strata::core
