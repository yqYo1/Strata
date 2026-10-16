// Exact source extraction; repository LICENSE. Host CLI checks only.
#include <charconv>
#include <climits>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
void require(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
int64_t number(const std::string& value) {
    int64_t result = 0;
    auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    require(error == std::errc{} && end == value.data() + value.size(), "invalid integer: " + value);
    return result;
}
struct Options {
    std::string root, token_file, ple_file, kv, capture;
    std::vector<std::string> shards;
    bool validation_small_input = false;
    int64_t count = 0, context = 0, chunk = 0, first = -1, gpu_rows = -1, ring = 0, logit = -1, device = -1;
};
Options parse(int argc, char** argv) {
    Options o; std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        require(key == "--shard" || seen.insert(key).second, "duplicate option: " + key);
        if (key == "--validation-small-input") {
            o.validation_small_input = true; // Bare opt-in flag; never a speed sample.
            continue;
        }
        require(i + 1 < argc, "every option requires a value: " + key);
        const std::string value = argv[++i];
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
    require(o.validation_small_input ? (o.count >= 1 && o.count < 32768) : (o.count >= 32768 && o.count <= 262144),
            "require token-count=32768..262144, or --validation-small-input with token-count=1..32767 for safety/correctness only");
    require(o.context >= 8192 && o.context >= o.count && o.context <= 262144,
            "require 8192 <= context <= 262144 and token-count <= context (fresh position zero only)");
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
int main(int argc,char** argv){try{const auto o=parse(argc,argv);
std::cout<<"{\"accepted\":true,\"count\":"<<o.count<<",\"small\":"
<<(o.validation_small_input?"true":"false")<<"}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 2;}}
