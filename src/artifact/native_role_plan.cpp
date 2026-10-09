#include "strata/artifact/native_role_plan.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace strata {
namespace {
uint64_t multiply(uint64_t a, uint64_t b) {
    if (b && a > (std::numeric_limits<uint64_t>::max)() / b)
        throw std::runtime_error("native role geometry overflows");
    return a * b;
}
uint64_t add(uint64_t a, uint64_t b) {
    if (a > (std::numeric_limits<uint64_t>::max)() - b)
        throw std::runtime_error("native role offset overflows");
    return a + b;
}
uint64_t row_bytes(int type, int64_t columns) {
    int elements = 0, bytes = 0;
    if (type < 0 || columns <= 0 || !block_geometry(static_cast<uint32_t>(type), elements, bytes) ||
        static_cast<uint64_t>(columns) % static_cast<uint64_t>(elements))
        throw std::runtime_error("invalid native role block geometry");
    return multiply(static_cast<uint64_t>(columns) / static_cast<uint64_t>(elements), bytes);
}
std::string role_path(const std::string& primary, const kernels::cpu::ExpertLayout& layout, size_t index) {
    if (layout.gguf_file.empty() || layout.gguf_file[index].empty()) return primary;
    const size_t cut = primary.find_last_of("/\\");
    return (cut == std::string::npos ? std::string() : primary.substr(0, cut + 1)) + layout.gguf_file[index];
}
}  // namespace

struct NativeRolePlan::Impl {
    struct Role { Extent extent; const uint8_t* data = nullptr; };
    int64_t layers = 0, experts = 0;
    std::map<std::string, std::unique_ptr<GgufFile>> files;
    std::vector<Role> roles;
    std::vector<uint64_t> blobs;
};

NativeRolePlan::NativeRolePlan() = default;
NativeRolePlan::~NativeRolePlan() = default;
void NativeRolePlan::close() noexcept { impl_.reset(); }
bool NativeRolePlan::is_open() const noexcept { return impl_ != nullptr; }
size_t NativeRolePlan::file_count() const noexcept { return impl_ ? impl_->files.size() : 0; }
const NativeRolePlan::Extent* NativeRolePlan::extent(int64_t layer, int role) const noexcept {
    if (!impl_ || layer < 0 || layer >= impl_->layers || role < 0 || role >= 3) return nullptr;
    return &impl_->roles[static_cast<size_t>(layer) * 3 + static_cast<size_t>(role)].extent;
}
uint64_t NativeRolePlan::blob_bytes(int64_t layer) const noexcept {
    return impl_ && layer >= 0 && layer < impl_->layers ? impl_->blobs[static_cast<size_t>(layer)] : 0;
}

bool NativeRolePlan::open(const std::string& primary, const kernels::cpu::ExpertLayout& layout,
                          std::string& error) {
    close();
    error.clear();
    try {
        if (!layout.native || layout.n_layers <= 0 || layout.n_expert <= 0)
            throw std::runtime_error("native role plan requires positive native layer/expert counts");
        const uint64_t count = multiply(static_cast<uint64_t>(layout.n_layers), 3);
        if (count > (std::numeric_limits<size_t>::max)() ||
            layout.fmt.size() != static_cast<uint64_t>(layout.n_layers) ||
            layout.bytes.size() != static_cast<uint64_t>(layout.n_layers) ||
            layout.gguf_off.size() != count || (!layout.gguf_file.empty() && layout.gguf_file.size() != count))
            throw std::runtime_error("native role layout vector lengths do not match layer count");
        auto next = std::make_unique<Impl>();
        next->layers = layout.n_layers;
        next->experts = layout.n_expert;
        next->roles.resize(static_cast<size_t>(count));
        next->blobs = layout.bytes;
        static const char* names[3] = {"gate", "up", "down"};
        for (int64_t layer = 0; layer < layout.n_layers; ++layer) {
            const auto& format = layout.fmt[static_cast<size_t>(layer)];
            if (format.n_embd <= 0 || format.n_ff <= 0)
                throw std::runtime_error("native role dimensions must be positive");
            const uint64_t gu_row = row_bytes(format.gu_type, format.n_embd);
            const uint64_t down_row = row_bytes(format.d_type, format.n_ff);
            const uint64_t gu = multiply(gu_row, static_cast<uint64_t>(format.n_ff));
            const uint64_t down = multiply(down_row, static_cast<uint64_t>(format.n_embd));
            const uint64_t down_offset = multiply(gu, 2);
            const uint64_t blob = add(down_offset, down);
            if (gu_row != format.gu_row || down_row != format.d_row || gu != format.up_off ||
                down_offset != format.down_off || blob != format.bytes ||
                blob != layout.bytes[static_cast<size_t>(layer)] || blob > (std::numeric_limits<size_t>::max)())
                throw std::runtime_error("native role row/blob geometry disagrees with layout");
            const uint64_t strides[3] = {gu, gu, down};
            for (int role = 0; role < 3; ++role) {
                const size_t index = static_cast<size_t>(layer) * 3 + static_cast<size_t>(role);
                const std::string path = role_path(primary, layout, index);
                auto& file = next->files[path];
                if (!file) file = std::make_unique<GgufFile>(path);
                const std::string name = "blk." + std::to_string(layer) + ".ffn_" + names[role] + "_exps.weight";
                const TensorInfo* tensor = file->find(name);
                const uint32_t type = static_cast<uint32_t>(role < 2 ? format.gu_type : format.d_type);
                const uint64_t columns = static_cast<uint64_t>(role < 2 ? format.n_embd : format.n_ff);
                const uint64_t rows = static_cast<uint64_t>(role < 2 ? format.n_ff : format.n_embd);
                const uint64_t total = multiply(strides[role], static_cast<uint64_t>(layout.n_expert));
                std::string why;
                if (!tensor) why = "is not in it";
                else if (tensor->type != type) why = "has a different weight type";
                else if (tensor->shape != std::vector<uint64_t>{columns, rows, static_cast<uint64_t>(layout.n_expert)})
                    why = "has different input/row/expert dimensions";
                else if (tensor_payload_bytes(*tensor) != total) why = "has a different payload size";
                else {
                    const uint64_t payload = file->file_size() - file->data_start();
                    // Check bounds before addition or tensor_data pointer construction.
                    if (tensor->offset > payload || total > payload - tensor->offset)
                        why = "runs past the end of the file (a truncated shard?)";
                    else if (add(file->data_start(), tensor->offset) != layout.gguf_off[index])
                        why = "starts at a different absolute byte offset";
                }
                if (!why.empty()) {
                    error = "the pack's native_experts.txt does not match the model: " + name + " in " + path + " " +
                            why + " - repack with tools/iq_pack.py from this model's shards";
                    return false;
                }
                auto& view = next->roles[index];
                view.extent = {path, name, layout.gguf_off[index], strides[role], total};
                view.data = file->tensor_data(*tensor);
            }
        }
        impl_ = std::move(next);
        return true;
    } catch (const std::exception& exception) {
        error = std::string("native experts from the GGUF: ") + exception.what();
        return false;
    }
}

bool NativeRolePlan::copy_blob(int64_t layer, int64_t expert, uint8_t* destination,
                               size_t capacity, std::string& error) const {
    error.clear();
    if (!impl_ || layer < 0 || layer >= impl_->layers || expert < 0 || expert >= impl_->experts) {
        error = "native role plan is closed or layer/expert is out of range";
        return false;
    }
    if (!destination || capacity < impl_->blobs[static_cast<size_t>(layer)]) {
        error = "native role destination is null or too small";
        return false;
    }
    size_t offset = 0;
    for (int role = 0; role < 3; ++role) {
        const auto& view = impl_->roles[static_cast<size_t>(layer) * 3 + static_cast<size_t>(role)];
        const size_t stride = static_cast<size_t>(view.extent.bytes_per_expert);
        // open validated stride * experts and the entire mapped tensor extent.
        std::memcpy(destination + offset, view.data + static_cast<uint64_t>(expert) * stride, stride);
        offset += stride;
    }
    return true;
}
}  // namespace strata
