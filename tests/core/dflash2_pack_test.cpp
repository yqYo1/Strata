// Metadata admission: use the pinned stock pack's real index, never its weights.
#include "strata/core/dflash2_pack.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
int main(int argc, char** argv) try {
    if (argc != 2) throw std::runtime_error("pass the pinned Q4_0/BF16-selector pack");
    strata::core::Dflash2PackInfo info;
    std::string error;
    if (!strata::core::inspect_dflash2_pack(argv[1], info, error)) throw std::runtime_error(error);
    if (info.tensors.size() != 81 || info.data_bytes != 522449920 || info.mask_token != 248077)
        throw std::runtime_error("unexpected stock metadata");
    std::ifstream in(argv[1], std::ios::binary);
    in.seekg(12);
    std::string index(size_t(info.data_offset - 12), ' ');
    in.read(index.data(), index.size());
    struct Temp {
        fs::path path;
        ~Temp() { std::error_code ignored; fs::remove_all(path, ignored); }
    } temp{fs::temp_directory_path() / ("strata-dflash2-pack-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))};
    if (!fs::create_directory(temp.path)) throw std::runtime_error("temporary directory exists");
    const fs::path sample = temp.path / "header.pack";
    int cases = 1;
    auto check = [&](const std::string& contents, bool expected, uint64_t bytes = 522458112, unsigned version = 1) {
        const uint32_t length = uint32_t(contents.size());
        std::ofstream out(sample, std::ios::binary | std::ios::trunc);
        out.write("DF2P", 4);
        for (uint32_t v : {uint32_t(version), length})
            for (int b = 0; b < 4; ++b) out.put(char(v >> (8*b)));
        out.write(contents.data(), contents.size());
        out.close();
        fs::resize_file(sample, bytes); // sparse: no weight payload copied or read
        strata::core::Dflash2PackInfo result;
        const bool accepted = strata::core::inspect_dflash2_pack(sample.string(), result, error);
        if (accepted != expected) throw std::runtime_error("admission case " + std::to_string(cases) + ": " + error);
        ++cases;
    };
    auto changed = [&](const std::string& from, const std::string& to) {
        std::string s = index;
        const auto at = s.find(from);
        if (at == std::string::npos || from.size() != to.size()) throw std::runtime_error("invalid mutation fixture");
        s.replace(at, from.size(), to);
        return s;
    };
    check(index, true);
    check(index, false, 522458112, 2);
    check(index, false, 12);
    check(index, false, 522458111);
    check(changed("cfg layers 5", "cfg layers 0"), false);
    check(changed("cfg heads 32", "cfg heads 16"), false);
    check(changed("5,15,25,35,45", "5,15,25,35,46"), false);
    check(changed("eps 1e-06", "eps 0e-06"), false);
    check(changed("eps 1e-06", "eps 1e+06"), false);
    check(changed("eps 1e-06", "eps nan  "), false);
    check(changed("mask_token 248077", "mask_token 248320"), false);
    check(changed("fc.weight q4_0", "fc.weight bf16"), false);
    check(changed("2560 12800 0 18432000", "2560 12800 0 18432001"), false);
    check(changed("2560 12800 0 18432000", "2560 12800 1 18432000"), false);
    check(changed("18432000 10240", "00000000 10240"), false);
    check(changed("t norm.weight", "t nonm.weight"), false);
    check(changed("t norm.weight", "t norm weighx"), false);
    check(changed("end\n", "bad\n"), false);
    std::cout << "DFlash2 pack metadata: " << cases << " cases passed; no GPU execution\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
