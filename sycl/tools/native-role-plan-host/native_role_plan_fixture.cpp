#include "strata/artifact/native_role_plan.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;
using Layout = strata::kernels::cpu::ExpertLayout;
using Plan = strata::NativeRolePlan;
namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class T> void put(std::vector<uint8_t>& bytes, T value) {
    const auto* data = reinterpret_cast<const uint8_t*>(&value);
    bytes.insert(bytes.end(), data, data + sizeof(value));
}
void text(std::vector<uint8_t>& bytes, const std::string& value) {
    put<uint64_t>(bytes, value.size());
    bytes.insert(bytes.end(), value.begin(), value.end());
}
struct Tensor {
    int layer, role;
    uint32_t type;
    std::vector<uint64_t> dimensions;
    uint64_t relative = 0, absolute = 0;
    std::vector<uint8_t> bytes;
};
// Expected bytes depend only on logical coordinates, not the reader or offsets.
uint8_t value(int layer, int role, int expert, size_t index) {
    return static_cast<uint8_t>(13 + layer * 47 + role * 61 + expert * 19 + index * 7);
}
Tensor tensor(int layer, int role) {
    Tensor result{layer, role, role < 2 ? 1u : 0u,
                  role < 2 ? std::vector<uint64_t>{4, 2, 3} : std::vector<uint64_t>{2, 4, 3}, 0, 0, {}};
    const size_t stride = role < 2 ? 16 : 32;
    for (int expert = 0; expert < 3; ++expert)
        for (size_t byte = 0; byte < stride; ++byte) result.bytes.push_back(value(layer, role, expert, byte));
    return result;
}
std::string name(const Tensor& t) {
    const char* roles[3] = {"gate", "up", "down"};
    return "blk." + std::to_string(t.layer) + ".ffn_" + roles[t.role] + "_exps.weight";
}
// Independent small GGUF v3 writer. Reverse physical ordering and add padding
// so expert * full_blob_bytes cannot accidentally serve as a role stride.
void write(const fs::path& path, std::vector<Tensor>& tensors) {
    check(std::endian::native == std::endian::little, "fixture requires little endian");
    std::reverse(tensors.begin(), tensors.end());
    uint64_t cursor = 0;
    for (auto& t : tensors) {
        t.relative = cursor;
        cursor = ((cursor + t.bytes.size() + 63) / 32) * 32;
    }
    std::vector<uint8_t> bytes;
    put<uint32_t>(bytes, 0x46554747);
    put<uint32_t>(bytes, 3);
    put<uint64_t>(bytes, tensors.size());
    put<uint64_t>(bytes, 0);
    for (const auto& t : tensors) {
        text(bytes, name(t));
        put<uint32_t>(bytes, t.dimensions.size());
        for (auto d : t.dimensions) put<uint64_t>(bytes, d);
        put<uint32_t>(bytes, t.type);
        put<uint64_t>(bytes, t.relative);
    }
    const uint64_t data_start = ((bytes.size() + 31) / 32) * 32;
    bytes.resize(static_cast<size_t>(data_start + cursor), 0xa7);
    for (auto& t : tensors) {
        t.absolute = data_start + t.relative;
        std::copy(t.bytes.begin(), t.bytes.end(), bytes.begin() + static_cast<size_t>(t.absolute));
    }
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    check(stream.good(), "fixture write failed");
}
struct Fixture { fs::path primary; Layout layout; std::vector<fs::path> paths; };
Fixture prepare(const fs::path& directory, int mode) {
    fs::create_directory(directory);
    Fixture f;
    f.primary = directory / "primary.gguf";
    f.layout.native = true; f.layout.n_layers = 2; f.layout.n_expert = 3;
    f.layout.bytes = {64, 64}; f.layout.gguf_off.resize(6);
    strata::kernels::cpu::NativeFmt format;
    format.gu_type = 1; format.d_type = 0; format.n_embd = 4; format.n_ff = 2;
    format.gu_row = 8; format.d_row = 8; format.up_off = 16; format.down_off = 32; format.bytes = 64;
    f.layout.fmt = {format, format};
    if (mode != 0) f.layout.gguf_file.resize(6);
    const int shards = mode == 0 ? 1 : mode == 1 ? 2 : 3;
    std::vector<std::vector<Tensor>> groups(static_cast<size_t>(shards));
    for (int layer = 0; layer < 2; ++layer)
        for (int role = 0; role < 3; ++role) {
            const int shard = mode == 0 ? 0 : mode == 1 ? layer : role;
            groups[static_cast<size_t>(shard)].push_back(tensor(layer, role));
            if (mode != 0) f.layout.gguf_file[static_cast<size_t>(layer * 3 + role)] =
                shard == 0 ? "" : "shard" + std::to_string(shard) + ".gguf";
        }
    for (int shard = 0; shard < shards; ++shard) {
        fs::path path = shard == 0 ? f.primary : directory / ("shard" + std::to_string(shard) + ".gguf");
        write(path, groups[static_cast<size_t>(shard)]);
        f.paths.push_back(path);
        for (const auto& t : groups[static_cast<size_t>(shard)])
            f.layout.gguf_off[static_cast<size_t>(t.layer * 3 + t.role)] = t.absolute;
    }
    return f;
}
void verify_copy(Plan& plan) {
    std::string error;
    for (int layer = 0; layer < 2; ++layer)
        for (int expert = 0; expert < 3; ++expert) {
            std::vector<uint8_t> output(80, 0xcd), expected;
            for (int role = 0; role < 3; ++role)
                for (size_t index = 0; index < (role < 2 ? 16u : 32u); ++index)
                    expected.push_back(value(layer, role, expert, index));
            check(plan.copy_blob(layer, expert, output.data() + 8, 64, error), "valid fixture copy failed");
            check(std::equal(expected.begin(), expected.end(), output.begin() + 8), "role copy differs");
            check(std::all_of(output.begin(), output.begin() + 8, [](uint8_t v) { return v == 0xcd; }) &&
                  std::all_of(output.begin() + 72, output.end(), [](uint8_t v) { return v == 0xcd; }), "copy overrun");
        }
}
void reject(const Fixture& fixture, Layout invalid) {
    Plan plan; std::string error;
    check(plan.open(fixture.primary.string(), fixture.layout, error), "valid fixture open failed");
    check(!plan.open(fixture.primary.string(), invalid, error), "invalid layout accepted");
    check(!error.empty() && !plan.is_open() && plan.file_count() == 0 && !plan.extent(0, 0), "failed reopen kept mappings");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        check(argc == 2, "supply a new fixture output directory");
        fs::path root(argv[1]);
        check(fs::create_directory(root), "fixture directory already exists");
        unsigned groups = 0;
        for (int mode = 0; mode < 3; ++mode) {
            auto fixture = prepare(root / ("valid" + std::to_string(mode)), mode);
            Plan plan; std::string error;
            check(plan.open(fixture.primary.string(), fixture.layout, error), "valid fixture open failed");
            check(plan.file_count() == static_cast<size_t>(mode + 1), "mapping reuse count");
            verify_copy(plan);
            for (int layer = 0; layer < 2; ++layer)
                for (int role = 0; role < 3; ++role) {
                    auto* extent = plan.extent(layer, role);
                    check(extent && extent->offset == fixture.layout.gguf_off[static_cast<size_t>(layer * 3 + role)] &&
                          extent->bytes_per_expert == (role < 2 ? 16u : 32u), "extent mismatch");
                }
            // Rename mapped files; copying still uses the validated mappings.
            for (const auto& path : fixture.paths) fs::rename(path, path.string() + ".moved");
            verify_copy(plan);
            std::vector<uint8_t> untouched(64, 0xdd);
            check(!plan.copy_blob(-1, 0, untouched.data(), 64, error), "negative layer accepted");
            check(!plan.copy_blob(2, 0, untouched.data(), 64, error), "past-end layer accepted");
            check(!plan.copy_blob(0, -1, untouched.data(), 64, error), "negative expert accepted");
            check(!plan.copy_blob(0, 3, untouched.data(), 64, error), "past-end expert accepted");
            check(!plan.copy_blob(0, 0, untouched.data(), 63, error), "short capacity accepted");
            check(!plan.copy_blob(0, 0, nullptr, 64, error), "null buffer accepted");
            check(std::all_of(untouched.begin(), untouched.end(), [](uint8_t v) { return v == 0xdd; }), "rejected call changed output");
            plan.close(); plan.close();
            check(!plan.copy_blob(0, 0, untouched.data(), 64, error), "closed copy accepted");
            ++groups;
        }
        auto fixture = prepare(root / "negative", 2);
        auto invalid = fixture.layout; invalid.native = false; reject(fixture, invalid); ++groups;
        invalid = fixture.layout; invalid.n_layers = -1; reject(fixture, invalid); ++groups;
        invalid = fixture.layout; invalid.n_expert = 0; reject(fixture, invalid); ++groups;
        invalid = fixture.layout; invalid.gguf_off.pop_back(); reject(fixture, invalid); ++groups;
        invalid = fixture.layout; invalid.gguf_file.pop_back(); reject(fixture, invalid); ++groups;
        invalid = fixture.layout; invalid.fmt.clear(); reject(fixture, invalid); ++groups;
        invalid = fixture.layout; invalid.bytes.pop_back(); reject(fixture, invalid); ++groups;
        invalid = fixture.layout; invalid.fmt[0].down_off = 65; reject(fixture, invalid); ++groups;
        invalid = fixture.layout; invalid.fmt[0].gu_row = 7; reject(fixture, invalid); ++groups;
        invalid = fixture.layout; invalid.fmt[0].gu_type = 999; reject(fixture, invalid); ++groups;
        invalid = fixture.layout; invalid.fmt[0].n_ff = 0; reject(fixture, invalid); ++groups;
        invalid = fixture.layout; invalid.fmt[0].n_embd = (std::numeric_limits<int64_t>::max)(); reject(fixture, invalid); ++groups;
        invalid = fixture.layout; invalid.n_expert = (std::numeric_limits<int64_t>::max)(); reject(fixture, invalid); ++groups;
        invalid = fixture.layout; invalid.gguf_off[2] = (std::numeric_limits<uint64_t>::max)(); reject(fixture, invalid); ++groups;
        invalid = fixture.layout; invalid.gguf_file[2] = "absent.gguf"; reject(fixture, invalid); ++groups;
        // Independently alter on-disk tensor type/shape/name and truncate payload.
        for (int variation = 0; variation < 4; ++variation) {
            auto f = prepare(root / ("badfile" + std::to_string(variation)), 0);
            std::vector<Tensor> ts;
            for (int layer = 0; layer < 2; ++layer)
                for (int role = 0; role < 3; ++role) ts.push_back(tensor(layer, role));
            if (variation == 0) ts[0].type = 0;
            if (variation == 1) ts[0].dimensions[2] = 4;
            if (variation == 2) ts[0].layer = 9;
            write(f.primary, ts);
            if (variation == 3) fs::resize_file(f.primary, fs::file_size(f.primary) - 64);
            Plan plan; std::string error;
            check(!plan.open(f.primary.string(), f.layout, error) && !plan.is_open() && !error.empty(), "bad GGUF accepted");
            ++groups;
        }
        std::cout << "PASS " << groups << " independent role-plan fixture groups\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
