#include "strata/core/dflash2_pack.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace strata::core {
bool inspect_dflash2_pack(const std::string& path, Dflash2PackInfo& out, std::string& err) try {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open pack");
    const auto end = f.tellg();
    if (end < 12) throw std::runtime_error("truncated header");
    const uint64_t file_bytes = uint64_t(end);
    f.seekg(0);
    unsigned char header[12];
    if (!f.read(reinterpret_cast<char*>(header), sizeof header)) throw std::runtime_error("truncated header");
    auto u32 = [&](int p) {
        return uint32_t(header[p]) | uint32_t(header[p+1]) << 8 |
               uint32_t(header[p+2]) << 16 | uint32_t(header[p+3]) << 24;
    };
    if (header[0] != 'D' || header[1] != 'F' || header[2] != '2' || header[3] != 'P' || u32(4) != 1)
        throw std::runtime_error("expected DF2P v1");
    const uint32_t index_bytes = u32(8);
    if (!index_bytes || index_bytes > (1u << 20) || index_bytes > file_bytes - 12)
        throw std::runtime_error("invalid index size");
    Dflash2PackInfo info;
    info.data_offset = 12 + index_bytes;
    if (info.data_offset % 4096) throw std::runtime_error("unaligned data section");
    const uint64_t payload = file_bytes - info.data_offset;
    std::string index(index_bytes, '\0');
    if (!f.read(index.data(), index_bytes)) throw std::runtime_error("truncated index");
    std::map<std::string, std::string> config;
    std::set<std::string> names;
    bool ended = false;
    for (std::istringstream lines(index);;) {
        std::string line;
        if (!std::getline(lines, line)) break;
        std::istringstream row(line);
        std::string kind;
        if (!(row >> kind)) continue;
        if (ended) throw std::runtime_error("records after end");
        if (kind == "end") { ended = true; continue; }
        std::string extra;
        if (kind == "cfg") {
            std::string key, value;
            if (!(row >> key >> value) || row >> extra || !config.emplace(key, value).second)
                throw std::runtime_error("invalid or duplicate config");
        } else if (kind == "t") {
            Dflash2Tensor t;
            if (!(row >> t.name >> t.dtype >> t.rows >> t.cols >> t.off >> t.bytes) || row >> extra ||
                !names.insert(t.name).second || t.rows <= 0 || t.cols <= 0 || t.off % 256 ||
                t.off > payload || t.bytes > payload - t.off)
                throw std::runtime_error("invalid tensor record or payload span");
            info.data_bytes = std::max(info.data_bytes, t.off + t.bytes);
            info.tensors.push_back(std::move(t));
        } else throw std::runtime_error("unknown index record");
    }
    if (!ended) throw std::runtime_error("missing index end");
    const std::pair<const char*, const char*> fixed[] = {
        {"hidden","2560"},{"layers","5"},{"heads","32"},{"kv_heads","8"},{"head_dim","128"},
        {"inter","7680"},{"window","2048"},{"block","8"},{"conv_group","16"},{"conv_kernel","2"},
        {"rank","256"},{"top_k","16"},{"taps","5,15,25,35,45"},{"vocab","248320"}};
    for (const auto& [key, value] : fixed)
        if (config[key] != value) throw std::runtime_error(std::string("unsupported ") + key);
    auto number = [&](const char* key) {
        size_t used = 0;
        const float value = std::stof(config.at(key), &used);
        if (used != config.at(key).size() || !std::isfinite(value)) throw std::runtime_error("invalid numeric config");
        return value;
    };
    const float mask = number("mask_token");
    if (mask < 0 || mask >= 248320 || std::floor(mask) != mask) throw std::runtime_error("invalid mask token");
    info.mask_token = int(mask);
    info.eps = number("eps"); info.theta = number("rope_theta");
    if (info.eps <= 0 || info.eps > .001f || info.theta <= 1 || info.theta > 1e9f)
        throw std::runtime_error("unsupported norm/rope constants");
    std::set<std::string> expected;
    auto tensor = [&](const std::string& name, const char* type, int64_t rows, int64_t cols) {
        expected.insert(name);
        auto it = std::find_if(info.tensors.begin(), info.tensors.end(), [&](const auto& t) { return t.name == name; });
        const uint64_t n = uint64_t(rows) * uint64_t(cols);
        const uint64_t bytes = std::string(type) == "q4_0" ? n / 32 * 18 : n * (std::string(type) == "f32" ? 4 : 2);
        if (it == info.tensors.end() || it->dtype != type || it->rows != rows || it->cols != cols || it->bytes != bytes)
            throw std::runtime_error("missing or incompatible tensor " + name);
    };
    tensor("fc.weight", "q4_0", 2560, 12800);
    tensor("hidden_norm.weight", "f32", 2560, 1);
    tensor("norm.weight", "f32", 2560, 1);
    for (int l = 0; l < 5; ++l) {
        const std::string p = "layers." + std::to_string(l) + ".";
        for (const char* k : {"k", "v"}) tensor(p + "self_attn." + k + "_proj.weight", "q4_0", 1024, 2560);
        tensor(p + "self_attn.q_proj.weight", "q4_0", 4096, 2560);
        tensor(p + "self_attn.o_proj.weight", "q4_0", 2560, 4096);
        for (const char* k : {"gate", "up"}) tensor(p + "mlp." + k + "_proj.weight", "q4_0", 7680, 2560);
        tensor(p + "mlp.down_proj.weight", "q4_0", 2560, 7680);
        for (const char* k : {"attention_conv", "mlp_conv"}) {
            tensor(p + k + ".kernel_projection.weight", "q4_0", 640, 2560);
            tensor(p + k + ".base_kernel", "f32", 4, 2560);
        }
        for (const char* k : {"input_layernorm", "post_attention_layernorm"}) tensor(p + k + ".weight", "f32", 2560, 1);
        for (const char* k : {"q", "k"}) tensor(p + "self_attn." + k + "_norm.weight", "f32", 128, 1);
    }
    tensor("candidate_selector.hidden_projection.weight", "q4_0", 256, 2560);
    for (const char* k : {"predecessor", "successor"}) tensor(std::string("candidate_selector.") + k + "_codebook", "bf16", 248320, 256);
    if (names != expected) throw std::runtime_error("unexpected tensor set");
    std::vector<Dflash2Tensor> spans = info.tensors;
    std::sort(spans.begin(), spans.end(), [](const auto& a, const auto& b) { return a.off < b.off; });
    uint64_t previous_end = 0;
    for (const auto& t : spans) {
        if (t.off < previous_end) throw std::runtime_error("overlapping tensor spans");
        previous_end = t.off + t.bytes;
    }
    out = std::move(info);
    return true;
} catch (const std::exception& e) {
    err = std::string("dflash2 pack: ") + e.what();
    return false;
}
}
