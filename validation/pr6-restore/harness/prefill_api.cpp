// Independently authored validation integration under the repository LICENSE.
// Source-only delivery: Main must review/build/qualify providers before execution.
#include "strata/prefill/prefill.hpp"
#include "strata/prefill/range_guard.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/validation_control.hpp"
#include "strata/core/token_admission.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/sycl_queue.hpp"
#include "strata/sycl_retirement.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern char** environ; // Target is explicitly Linux-only in the validation wrapper.
namespace {
namespace core = strata::core;
namespace pf = strata::prefill;
namespace kernels = strata::kernels;
using Clock = std::chrono::steady_clock;
static_assert(!core::detail::concurrent_doorbell_admitted());
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(std::endian::native == std::endian::little);
constexpr const char* baseline = "7b1bd9b67d58f2d653a2184b93e5ddb83f7dcc32";

void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
double ms(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
void phase(const char* name) { std::fprintf(stderr, "pr6_prefill_api phase=%s\n", name); std::fflush(stderr); }
int64_t number(const std::string& value) {
    int64_t result = 0;
    auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    require(error == std::errc{} && end == value.data() + value.size(), "invalid integer: " + value);
    return result;
}
std::string quoted(const std::string& value) {
    std::ostringstream out; out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << char(c);
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c);
        else out << char(c);
    }
    out << '"'; return out.str();
}
// SHA-256 over bounded evidence bytes. New scalar project code implementing the
// standard digest, not an inference kernel or a copied helper/template.
std::string sha256(const void* data, size_t size) {
    static constexpr uint32_t k[] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    require(size <= (1u << 24), "digest input exceeds bounded evidence budget");
    const auto* bytes = static_cast<const uint8_t*>(data);
    std::vector<uint8_t> padded(bytes, bytes + size);
    padded.push_back(0x80);
    while (padded.size() % 64 != 56) padded.push_back(0);
    for (int shift = 56; shift >= 0; shift -= 8) padded.push_back(uint8_t((uint64_t(size) * 8) >> shift));
    std::array<uint32_t, 8> h{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    for (size_t off = 0; off < padded.size(); off += 64) {
        uint32_t w[64]{};
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 4; ++j) w[i] = (w[i] << 8) | padded[off + 4 * i + j];
        for (int i = 16; i < 64; ++i) {
            const auto s0 = std::rotr(w[i-15],7) ^ std::rotr(w[i-15],18) ^ (w[i-15] >> 3);
            const auto s1 = std::rotr(w[i-2],17) ^ std::rotr(w[i-2],19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        auto v = h;
        for (int i = 0; i < 64; ++i) {
            const auto t1 = v[7] + (std::rotr(v[4],6) ^ std::rotr(v[4],11) ^ std::rotr(v[4],25)) +
                            ((v[4] & v[5]) ^ (~v[4] & v[6])) + k[i] + w[i];
            const auto t2 = (std::rotr(v[0],2) ^ std::rotr(v[0],13) ^ std::rotr(v[0],22)) +
                            ((v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]));
            for (int j = 7; j > 0; --j) v[j] = v[j-1];
            v[4] += t1; v[0] = t1 + t2;
        }
        for (int i = 0; i < 8; ++i) h[i] += v[i];
    }
    std::ostringstream out; out << std::hex << std::setfill('0');
    for (auto word : h) out << std::setw(8) << word;
    return out.str();
}
struct Options {
    std::string root, token_file, ple_file, kv, capture;
    std::vector<std::string> shards;
    int64_t count = 0, context = 0, chunk = 0, first = -1, gpu_rows = -1, ring = 0, logit = -1, device = -1;
};
Options parse(int argc, char** argv) {
    Options o; std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        require(i + 1 < argc, "every option requires a value: " + key);
        const std::string value = argv[++i];
        require(key == "--shard" || seen.insert(key).second, "duplicate option: " + key);
        if (key == "--model-root") o.root = value;
        else if (key == "--shard") o.shards.push_back(value);
        else if (key == "--tokens-file") o.token_file = value;
        else if (key == "--token-count") o.count = number(value);
        else if (key == "--context") o.context = number(value);
        else if (key == "--chunk") o.chunk = number(value);
        else if (key == "--first-chunk") o.first = number(value);
        else if (key == "--residual-gpu-tokens") o.gpu_rows = number(value);
        else if (key == "--ring-slots") o.ring = number(value);
        else if (key == "--device") o.device = number(value);
        else if (key == "--ple-file") o.ple_file = value;
        else if (key == "--kv") o.kv = value;
        else if (key == "--logit-id") o.logit = number(value);
        else if (key == "--capture-final-row") o.capture = value;
        else throw std::runtime_error("unknown option: " + key);
    }
    require(o.count >= 32768 && o.count <= 262144 && o.context >= o.count && o.context <= 262144,
            "require 32768 <= token-count <= context <= 262144 (fresh position zero only)");
    require(o.chunk == 8192 && o.first >= 0 && o.first <= o.chunk && o.gpu_rows >= 0 && o.gpu_rows <= o.count &&
            o.ring >= 8 && o.ring <= 512 && o.logit >= 0 && o.device >= 0 && o.device <= INT32_MAX,
            "explicit chunk=8192, first-chunk=0..8192, residual-gpu-tokens=0..count, ring-slots=8..512, logit-id/device>=0 required");
    require(o.kv == "fp16" || o.kv == "int8" || o.kv == "q4_0" || o.kv == "k8v4", "explicit --kv required");
    require(!o.shards.empty(), "explicit --shard paths required, metadata shard first");
    for (const auto& p : {o.root, o.token_file, o.ple_file})
        require(std::filesystem::path(p).is_absolute() && std::filesystem::exists(p), "missing absolute input path: " + p);
    std::set<std::filesystem::path> shard_paths;
    for (const auto& p : o.shards) {
        require(std::filesystem::path(p).is_absolute() && std::filesystem::is_regular_file(p), "missing absolute shard: " + p);
        require(shard_paths.insert(std::filesystem::canonical(p)).second, "duplicate shard");
    }
    require(std::filesystem::is_regular_file(o.root + "/native_experts.txt"), "native pack required; no canonical fallback");
    if (!o.capture.empty()) {
        require(std::filesystem::path(o.capture).is_absolute() && !std::filesystem::exists(o.capture),
                "capture must name a new absolute file; no overwrite");
    }
    return o;
}
std::map<std::string, std::string> environment(const Options& o) {
    const std::map<std::string, std::string> required{
        {"STRATA_PREFILL_LAYER_MAJOR", "2"}, {"STRATA_PREFILL_COMPACT", "1"},
        {"STRATA_PREFILL_MMQ", "0"}, {"STRATA_PROMPT_ATTN_XMX", "0"},
        {"STRATA_PREFILL_FIRST", std::to_string(o.first)},
        {"STRATA_PREFILL_LAYER_MAJOR_R_GPU", std::to_string(o.gpu_rows)},
        {"STRATA_PREFILL_RING", std::to_string(o.ring)}};
    const std::set<std::string> optional{"STRATA_TRACE", "STRATA_PREFILL_TIMING", "STRATA_PREFILL_TRANSFER_TIMING"};
    std::map<std::string, std::string> result;
    for (char** p = environ; *p; ++p) {
        std::string entry = *p;
        const auto equal = entry.find('=');
        if (equal == std::string::npos) continue;
        const auto key = entry.substr(0, equal), value = entry.substr(equal + 1);
        if (key.starts_with("STRATA_")) {
            require(required.contains(key) || (optional.contains(key) && value == "1"), "unsupported inherited engine variable: " + key);
        }
        bool relevant = key == "PATH" || key == "LD_LIBRARY_PATH";
        for (const auto* prefix : {"STRATA_", "SYCL_", "ONEAPI_", "ZE_", "ZES_", "UR_", "IGC_", "NEO_", "MKL_", "OMP_", "DPCPP_"})
            relevant = relevant || key.starts_with(prefix);
        if (relevant) result.emplace(key, value);
    }
    for (const auto& [key, value] : required)
        require(result.contains(key) && result.at(key) == value, "set in native child environment before exec: " + key + "=" + value);
    return result; // No setenv/unsetenv, including after runtime initialization.
}
std::vector<int64_t> tokens(const Options& o, std::string& digest) {
    const size_t bytes = pf::detail::pr6_product(size_t(o.count), 4);
    require(std::filesystem::file_size(o.token_file) == bytes, "token file must contain exactly count little-endian int32 IDs");
    std::vector<uint8_t> raw(bytes);
    std::ifstream input(o.token_file, std::ios::binary);
    require(bool(input.read(reinterpret_cast<char*>(raw.data()), std::streamsize(bytes))), "token read failed");
    require(input.peek() == std::char_traits<char>::eof(), "token file grew during read");
    digest = sha256(raw.data(), raw.size());
    std::vector<int64_t> ids(size_t(o.count), 0);
    for (size_t i = 0; i < ids.size(); ++i) {
        uint32_t id = uint32_t(raw[4*i]) | (uint32_t(raw[4*i+1]) << 8) | (uint32_t(raw[4*i+2]) << 16) | (uint32_t(raw[4*i+3]) << 24);
        require(id <= INT32_MAX, "negative token ID in file"); ids[i] = id;
    }
    return ids;
}
struct Runtime {
    sycl::queue* allocation = nullptr; // Borrowed, never destroyed here.
    sycl::queue* compute = nullptr;    // Owning DPCT queue, explicitly destroyed last.
    void* weights = nullptr; void* session_arena = nullptr; void* ple_emb = nullptr; void* ple_scratch = nullptr; void* logits = nullptr;
    core::WeightTable wt;
    core::SessionState ss;
    std::unique_ptr<core::NativeEmbed> embed;
    std::unique_ptr<core::NativeDense> dense;
    std::unique_ptr<core::NativeHead> head;
    std::unique_ptr<kernels::PleTable> ple;
    std::unique_ptr<core::FileExpertSource> experts;
    std::unique_ptr<pf::Prefill> prefill;
    std::vector<float> ple_host = std::vector<float>(kernels::NG_N_EMBD);
    bool closed = false;
    void allocate(void*& field, uint64_t bytes) {
        require(bytes && bytes <= std::numeric_limits<size_t>::max(), "invalid allocation byte count");
        field = sycl::malloc_device(size_t(bytes), *allocation);
        require(field != nullptr, "device allocation returned null");
    }
    void close() noexcept {
        if (closed) return;
        // Begin before provider calls/joins. Unknown completion exits with every
        // owner still live; known-quiescent cleanup failures terminate with 88.
        try {
            pf::detail::RetirementDeadline deadline;
            if (compute) strata::retire_queue(*compute);
            if (allocation) strata::retire_queue(*allocation);
            prefill.reset(); // Its own release retires streams before Stager joins.
            if (compute) strata::retire_queue(*compute);
            if (allocation) strata::retire_queue(*allocation);
            core::set_native_embed(nullptr);
            head.reset(); dense.reset(); embed.reset();
            core::session_release(ss); // Partial session_init leaves value-initialized QSA entries.
            delete[] ss.qsa_states; ss.qsa_states = nullptr;
            for (auto* field : {&logits, &ple_scratch, &ple_emb, &session_arena, &weights}) {
                if (*field) sycl::free(*field, *allocation);
                *field = nullptr;
            }
            experts.reset(); ple.reset();
            if (allocation) strata::retire_queue(*allocation);
            if (compute) dpct::get_current_device().destroy_queue(compute);
            compute = nullptr; allocation = nullptr; closed = true;
        } catch (...) { std::_Exit(88); }
    }
    ~Runtime() { close(); }
};
const core::WeightRef& weight(const core::WeightTable& wt, const char* name) {
    const auto* w = wt.find(name); require(w != nullptr, std::string("missing tensor: ") + name); return *w;
}
void wire_ple(Runtime& r) {
    auto& w = r.ss.ple.w;
    const auto& key = weight(r.wt, "blk.1.ple_key.weight");
    if (!key.quantized()) {
        require(key.data && key.bytes == uint64_t(2560) * 10240 * 2 && key.kind == core::WeightKind::Bf16InF32,
                "unquantized PLE key is not the retained BF16 representation");
        w.key_bf16 = static_cast<const uint16_t*>(key.data);
    } else {
        require(key.native_data && key.native_q8_1 &&
                (key.native_type == 42 || key.native_type == 18 || key.native_type == 23 || key.native_type == 8),
                "native PLE key unavailable/unsupported");
        w.key_native_data = key.native_data; w.key_native_type = key.native_type; w.key_native_q8_1 = key.native_q8_1;
    }
    auto data = [&](const char* name, int64_t elements, core::WeightKind kind, uint64_t width) -> const void* {
        const auto& x = weight(r.wt, name);
        require(x.data && !x.quantized() && x.elements == elements && x.kind == kind && x.bytes == uint64_t(elements) * width,
                std::string("PLE tensor representation mismatch: ") + name);
        return x.data;
    };
    w.value_bf16 = static_cast<const uint16_t*>(data("blk.1.ple_value.weight", int64_t(2560)*2560, core::WeightKind::Bf16InF32, 2));
    w.norm_key = static_cast<const float*>(data("blk.1.ple_norm_key.weight", 10240, core::WeightKind::F32, 4));
    w.norm_query = static_cast<const float*>(data("blk.1.ple_norm_query.weight", 10240, core::WeightKind::F32, 4));
    w.norm_conv = static_cast<const float*>(data("blk.1.ple_norm_conv.weight", 10240, core::WeightKind::F32, 4));
    const auto& conv = weight(r.wt, "blk.1.ple_conv1d.weight");
    require(conv.data && conv.elements == 4*10240 && conv.bytes == uint64_t(conv.elements)*2 &&
            (conv.kind == core::WeightKind::F16InF32 || (conv.kind == core::WeightKind::Verbatim && !conv.quantized())),
            "PLE conv requires F16 storage");
    w.conv1d_f16 = static_cast<const uint16_t*>(conv.data);
    r.allocate(r.ple_emb, kernels::NG_N_EMBD * sizeof(float));
    r.allocate(r.ple_scratch, core::ple_run_scratch_bytes());
    r.ss.ple.consts = kernels::ple_artifact_consts(); r.ss.ple.table = r.ple.get();
    r.ss.ple.token = &r.ss.ple_token; r.ss.ple.prev = r.ss.ple_prev; r.ss.ple.hist = r.ss.ple_hist;
    r.ss.ple.emb_dev = static_cast<float*>(r.ple_emb); r.ss.ple.emb_host = r.ple_host.data();
    r.ss.ple.scratch = static_cast<float*>(r.ple_scratch);
    require(r.ss.ple.ready(), "PLE readiness gate failed");
}
void finite(const std::vector<float>& values, const char* what) {
    require(std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); }), std::string(what) + " contains nonfinite values");
}
} // namespace

int main(int argc, char** argv) {
    // A throw escaping a legacy noexcept destructor cannot be caught by close().
    // Fail-stop before any static/USM teardown rather than treating it as success.
    std::set_terminate([] { std::_Exit(88); });
    try {
        const auto started = Clock::now();
        const auto o = parse(argc, argv);
        const auto env = environment(o);
        std::string token_digest;
        const auto ids = tokens(o, token_digest); // Immutable through retirement and all owner destruction.
        core::ModelGeometry g;
        std::string err;
        // Read only explicitly supplied model files. No tokenizer, search, network,
        // pack creation, fake content, decode driver, verifier or drafter.
        {
            strata::GgufFile metadata(o.shards.front());
            for (const auto& [key, wanted] : std::map<std::string, uint64_t>{{"qwen4exp.expert_count",512}, {"qwen4exp.expert_used_count",10}}) {
                const auto* v = metadata.get(key); require(v && v->is_num() && v->num() == double(wanted), "unsupported/missing model geometry: " + key);
            }
            kernels::RopeScaling rope;
            const auto* type = metadata.get("qwen4exp.rope.scaling.type");
            require(!type || (type->type == strata::MetaType::STRING && type->s == "none"), "scaled RoPE requires a separately reviewed harness configuration");
            if (const auto* base = metadata.get("qwen4exp.rope.freq_base")) {
                require(base->is_num(), "invalid RoPE base metadata"); rope.freq_base = base->num();
            }
            const auto* why = kernels::rope_scaling_invalid(rope);
            require(why == nullptr, why ? why : "invalid RoPE"); kernels::rope_scaling_set(rope);
        }
        // The retained CLI --native preset selects these existing math paths.
        // PLE postops enables Prefill's batched PLE producer; leaving its default
        // false would silently select the per-token PLE arithmetic/schedule.
        kernels::gr_set_fp32_activations(false);
        kernels::gr_set_native_mmvf(true);
        kernels::ple_set_native_bf16(true);
        kernels::ple_set_native_postops(true);
        require(kernels::cpu::cpu_avx2_ok(), "historical CPU library build requires AVX2/FMA/F16C");
        require(kernels::cpu::expert_layout_load(o.root, 48, 512, err), err);
        const auto& layout = kernels::cpu::expert_layout();
        require(layout.native && layout.n_layers == 48 && layout.n_expert == 512 && layout.fmt.size() == 48 &&
                layout.bytes.size() == 48 && layout.max_blob > 0 && layout.max_blob <= INT64_MAX, "unsupported expert layout");
        for (const auto& f : layout.fmt)
            require(kernels::native_expert_supported(f.gu_type, f.d_type, f.n_embd, f.n_ff), "unsupported native expert format");
        if (!std::filesystem::exists(o.root + "/experts.bin")) {
            // FileExpertSource uses only the role files named by this pack. They
            // must also appear in the caller's explicit shard list; no search.
            std::set<std::filesystem::path> supplied;
            for (const auto& shard : o.shards) supplied.insert(std::filesystem::canonical(shard));
            for (const auto& role : layout.gguf_file) {
                const auto path = role.empty() ? std::filesystem::path(o.shards.front()) :
                                  std::filesystem::path(o.shards.front()).parent_path() / role;
                require(supplied.contains(std::filesystem::canonical(path)), "expert role file was not explicitly supplied: " + path.string());
            }
        }
        std::set<std::string> skip;
        require(core::NativeDense::served_names(o.shards, true, skip, err), err);
        require(core::NativeDense::keep_unquantized_ple_key(o.root, skip, err), err);
        skip.insert("token_embd.weight"); skip.insert("output.weight");
        uint64_t weight_bytes = 0;
        require(core::WeightTable::pool_bytes(o.root, weight_bytes, err, &skip), err);
        std::vector<float> final_row(size_t(g.hc_dim()), 0), logits;
        // Host result/source buffers above outlive Runtime even on exception unwind.
        Runtime r;
        const auto init_started = Clock::now();
        std::string device_name, driver;
        int64_t vocabulary = 0;
        uint64_t session_bytes = 0;
        {
            pf::detail::RetirementDeadline init_deadline(std::chrono::seconds(600));
            phase("load"); dpct::select_device(int(o.device));
            r.allocation = strata::q_of(nullptr);
            require(r.allocation->is_in_order() && r.allocation->get_device().is_gpu(), "selected allocation queue is not an in-order GPU queue");
            r.compute = dpct::get_current_device().create_queue(true);
            require(strata::q_of(r.compute)->is_in_order() && r.compute->get_context() == r.allocation->get_context() &&
                    r.compute->get_device() == r.allocation->get_device(), "checked queues do not share the allocation context/device");
            device_name = r.compute->get_device().get_info<sycl::info::device::name>();
            driver = r.compute->get_device().get_info<sycl::info::device::driver_version>();
            r.allocate(r.weights, weight_bytes);
            require(r.wt.load(o.root, r.weights, weight_bytes, err, &skip), err);
            require(core::check_all(r.wt, g, err), err);
            vocabulary = weight(r.wt, "output.weight").ne1;
            require(vocabulary > 0 && vocabulary <= 1048576 && weight(r.wt,"output.weight").ne0 == g.n_embd &&
                    weight(r.wt,"token_embd.weight").ne0 == g.n_embd && weight(r.wt,"token_embd.weight").ne1 == vocabulary && o.logit < vocabulary,
                    "invalid/mismatched embedding/head vocabulary or selected logit ID");
            for (const auto id : ids) require(core::detail::embedding_token(id, vocabulary), "token outside actual vocabulary");
            // lm_head_mix validates kinds but relies on caller-owned tensor
            // extents. Admit those explicit output shapes before any head work.
            auto output_shape = [&](const char* name, int64_t ne0, int64_t ne1, core::WeightKind kind, uint64_t width) {
                const auto& x = weight(r.wt,name);
                require(x.data && x.ne0 == ne0 && x.ne1 == ne1 && x.elements == ne0*ne1 && x.kind == kind &&
                        x.bytes == uint64_t(ne0)*uint64_t(ne1)*width && !x.quantized(), std::string("output mix tensor mismatch: ") + name);
            };
            const auto& norm = weight(r.wt,"output_hc_norm.weight");
            require(norm.data && norm.elements == g.hc_dim() && norm.kind == core::WeightKind::F32 &&
                    norm.bytes == uint64_t(g.hc_dim())*4 && !norm.quantized(), "output norm mismatch");
            output_shape("output_hc_down.weight",g.hc_dim(),g.hc_lr,core::WeightKind::Bf16InF32,2);
            output_shape("output_hc_up.weight",g.hc_lr,g.hc_dim(),core::WeightKind::Bf16InF32,2);
            r.embed = std::make_unique<core::NativeEmbed>(); r.dense = std::make_unique<core::NativeDense>();
            r.head = std::make_unique<core::NativeHead>();
            require(r.embed->load(o.shards, g.n_embd, vocabulary, err), err);
            core::set_native_embed(r.embed.get());
            require(r.dense->load(o.shards, r.wt, err, true), err);
            require(r.head->load(o.shards, g.n_embd, vocabulary, err), err);
            strata::retire_queue(*r.allocation);
            // Single full session; all KV pages resident. Only storage format is
            // selected here, using existing public setters before allocation.
            core::qsa_set_kv_resident(0); core::qsa_set_kv_int8(o.kv == "int8");
            core::qsa_set_kv_q4(o.kv == "q4_0"); core::qsa_set_kv_hybrid(o.kv == "k8v4"); core::qsa_set_kv_int8_rotate(false);
            session_bytes = core::session_bytes(g, o.context, 10);
            r.allocate(r.session_arena, session_bytes);
            const auto used = core::session_init(g, o.context, 10, r.session_arena, r.ss);
            require(used && used <= session_bytes && r.ss.db == nullptr, "session initialization/arena/doorbell gate failed");
            require(core::detail::position_span(0, o.count, {core::detail::session_cells(r.ss,g)}), "actual session/QSA capacity cannot hold tokens");
            for (int64_t i = 0; i < r.ss.qsa_alloc; ++i)
                require(r.ss.qsa_states[i].kv_mode == 0 && r.ss.qsa_states[i].n_slots == r.ss.qsa_states[i].n_pages, "resident KV gate failed");
            r.ple = std::make_unique<kernels::PleTable>();
            kernels::PleIoOptions io; io.mode = kernels::PleIo::Mmap; io.io_thread = false; io.keepalive_ms = 0;
            require(r.ple->open(o.ple_file, err, io), err); wire_ple(r);
            strata::retire_queue(*r.allocation);
            core::session_zero(r.ss, g, nullptr, r.compute); strata::retire_queue(*r.compute);
            r.experts = std::make_unique<core::FileExpertSource>(); r.experts->set_gguf(o.shards.front());
            require(r.experts->open(o.root, 48, 512, err), err);
            r.allocate(r.logits, pf::detail::pr6_product(size_t(vocabulary), sizeof(float)));
            logits.resize(size_t(vocabulary));
        }
        const double initialization_ms = ms(init_started);
        int64_t callback_next = 0, callback_chunks = 0;
        double callback_copy_ms = 0;
        r.prefill = std::make_unique<pf::Prefill>();
        r.prefill->on_chunk = [&](const float* rows, int64_t count, int64_t position, std::string& error) {
            if (!rows || position != callback_next || count <= 0 || count > o.chunk ||
                !core::detail::position_span(position,count,{o.count})) { error = "harness callback span/order gate"; return false; }
            callback_next += count; ++callback_chunks;
            if (callback_next == o.count) {
                if (!r.prefill->checkpoint_ready()) { error = "final layer-major checkpoint is incoherent"; return false; }
                const auto copy_started = Clock::now();
                const size_t offset = pf::detail::pr6_product(size_t(count - 1), size_t(g.hc_dim()));
                // Only final row exported, through the documented synchronized
                // callback. Session residual is otherwise not written by run().
                strata::retire_event(r.compute->memcpy(r.ss.R, rows + offset, final_row.size()*sizeof(float)));
                callback_copy_ms += ms(copy_started);
            }
            return true;
        };
        phase("prefill_init"); const auto pf_init_started = Clock::now();
        require(r.prefill->init(r.wt,g,r.ss,r.experts.get(),nullptr,nullptr,o.chunk,r.compute,err),err);
        require(r.prefill->chunk() == o.chunk && r.ss.db == nullptr, "prefill chunk/doorbell gate failed");
        const double prefill_init_ms = ms(pf_init_started);
        phase("prefill_run"); const auto run_started = Clock::now();
        require(r.prefill->run(ids.data(),o.count,0,err),err);
        strata::retire_queue(*r.compute);
        const double prefill_run_ms = ms(run_started);
        const auto stats = r.prefill->stats();
        require(callback_next == o.count && callback_chunks > 0 && r.prefill->checkpoint_ready() &&
                stats.tokens == o.count && stats.chunks == callback_chunks && r.ss.db == nullptr, "prefill completion/statistics gate failed");
        phase("final_row_readback"); const auto row_started = Clock::now();
        strata::retire_event(r.compute->memcpy(final_row.data(),r.ss.R,final_row.size()*sizeof(float)));
        finite(final_row,"final residual");
        const double final_readback_ms = ms(row_started);
        phase("head"); const auto head_started = Clock::now();
        {
            pf::detail::RetirementDeadline head_deadline;
            require(core::lm_head_mix(r.wt,g,r.ss.block,r.compute,err) &&
                    r.head->run(r.ss.block.mixed,static_cast<float*>(r.logits),r.compute,err),err);
            strata::retire_event(r.compute->memcpy(logits.data(),r.logits,logits.size()*sizeof(float)));
        }
        finite(logits,"logits"); const double head_ms = ms(head_started);
        const size_t argmax = size_t(std::max_element(logits.begin(),logits.end()) - logits.begin());
        const auto residual_sha = sha256(final_row.data(),final_row.size()*sizeof(float));
        const auto logits_sha = sha256(logits.data(),logits.size()*sizeof(float));
        double residual_l2_squared = 0;
        for (float v : final_row) residual_l2_squared += double(v)*double(v);
        const auto residual_bounds = std::minmax_element(final_row.begin(), final_row.end());
        const auto ple_prev0 = r.ss.ple_prev[0], ple_prev1 = r.ss.ple_prev[1];
        require(ple_prev0 == ids[ids.size()-2] && ple_prev1 == ids.back(), "PLE final token window mismatch");
        const auto expert_reads = r.experts->reads(); const bool expert_gguf = r.experts->gguf_mode();
        if (!o.capture.empty()) {
            // Exclusive create: never truncate an existing comparison fixture.
            std::unique_ptr<std::FILE, decltype(&std::fclose)> file(std::fopen(o.capture.c_str(),"wbx"),&std::fclose);
            require(bool(file),"cannot exclusively create final row capture");
            require(std::fwrite(final_row.data(),sizeof(float),final_row.size(),file.get()) == final_row.size() &&
                    std::fflush(file.get()) == 0,"final row capture write failed");
        }
        phase("cleanup"); const auto cleanup_started = Clock::now(); r.close(); const double cleanup_ms = ms(cleanup_started);
        phase("complete");
        const int64_t first_len = o.first > 0 && o.first < o.chunk && o.count > 2*o.first ? o.first : o.chunk;
        const int64_t effective_gpu = o.gpu_rows == o.count ? o.count : o.gpu_rows < first_len ? 0 :
                                      first_len + (o.gpu_rows-first_len)/o.chunk*o.chunk;
        std::cout << std::setprecision(17)
          << "{\"status\":\"PREFILL_ONLY_COMPLETED_UNCOMPARED\",\"baseline_commit\":" << quoted(baseline)
          << ",\"model_root\":" << quoted(o.root) << ",\"shards\":[";
        for (size_t i=0;i<o.shards.size();++i) std::cout << (i ? "," : "") << quoted(o.shards[i]);
        std::cout << "],\"ple_file\":" << quoted(o.ple_file) << ",\"ple_io\":\"mmap\",\"kv\":" << quoted(o.kv)
          << ",\"device\":" << quoted(device_name) << ",\"driver\":" << quoted(driver)
          << ",\"tokens_file\":" << quoted(o.token_file) << ",\"token_encoding\":\"little-endian-int32\",\"tokens_sha256\":" << quoted(token_digest)
          << ",\"tokens\":" << o.count << ",\"position0\":0,\"context\":" << o.context << ",\"chunk\":" << o.chunk
          << ",\"first_chunk\":" << first_len << ",\"chunks\":" << callback_chunks << ",\"ring_slots\":" << o.ring
          << ",\"requested_gpu_residual_tokens\":" << o.gpu_rows << ",\"effective_gpu_residual_tokens\":" << effective_gpu
          << ",\"vocabulary\":" << vocabulary << ",\"weights_bytes\":" << weight_bytes << ",\"session_bytes\":" << session_bytes
          << ",\"expert_source\":" << quoted(expert_gguf ? "FileExpertSource/GGUF" : "FileExpertSource/experts.bin") << ",\"expert_reads\":" << expert_reads
          << ",\"residual_shape\":[4,2560],\"residual_encoding\":\"little-endian-fp32\",\"residual_sha256\":" << quoted(residual_sha)
          << ",\"residual_min\":" << *residual_bounds.first << ",\"residual_max\":" << *residual_bounds.second
          << ",\"residual_l2\":" << std::sqrt(residual_l2_squared) << ",\"capture_final_row\":" << quoted(o.capture) << ",\"logits_sha256\":" << quoted(logits_sha)
          << ",\"logit_id\":" << o.logit << ",\"logit\":" << logits[size_t(o.logit)] << ",\"argmax_id\":" << argmax << ",\"argmax_logit\":" << logits[argmax]
          << ",\"ple_prev\":[" << ple_prev0 << ',' << ple_prev1 << "],\"ms\":{\"initialization\":" << initialization_ms
          << ",\"prefill_init\":" << prefill_init_ms << ",\"prefill_run_including_final_callback_copy\":" << prefill_run_ms
          << ",\"final_callback_copy\":" << callback_copy_ms << ",\"engine_stats_total\":" << stats.ms_total
          << ",\"engine_host_experts\":" << stats.ms_experts_host << ",\"engine_ple\":" << stats.ms_ple
          << ",\"final_row_readback\":" << final_readback_ms << ",\"head_and_logit_readback\":" << head_ms
          << ",\"cleanup\":" << cleanup_ms << ",\"wall_to_receipt\":" << ms(started) << "},\"environment\":{";
        bool comma=false; for (const auto& [key,value] : env) { std::cout << (comma ? "," : "") << quoted(key) << ':' << quoted(value); comma=true; }
        std::cout << "},\"doorbell_initialized\":false,\"verifier_initialized\":false,\"drafter_initialized\":false,\"generation\":false,\"native_gr_mmvf\":true,\"ple_native_bf16\":true,\"ple_native_postops\":true,\"numerical_comparison\":false,\"physical256k_qualified\":false,\"historical600_requalified\":false}\n";
        require(bool(std::cout.flush()), "receipt write failed");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"pr6_prefill_api failed: %s\n",e.what()); std::fflush(stderr); return 1;
    } catch (...) { std::fputs("pr6_prefill_api failed: unknown exception\n",stderr); return 1; }
}
