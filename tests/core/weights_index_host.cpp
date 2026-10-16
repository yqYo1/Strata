// Metadata round-trip through the production loader. Host-only adapter stubs:
// no device runtime is linked or called, and skipped tensors must never upload.
#include "strata/core/weights.hpp"
#include <cuda_runtime.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

namespace {
int allocations = 0, live = 0, copies = 0, drains = 0;
void check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
}
cudaError_t cudaHostAlloc(void **out, size_t bytes, unsigned) {
    *out = std::malloc(bytes);
    if (!*out) return cudaErrorMemoryAllocation;
    ++allocations;
    ++live;
    return cudaSuccess;
}
cudaError_t cudaFreeHost(void *p) {
    std::free(p);
    --live;
    return cudaSuccess;
}
cudaError_t cudaMemcpy(void *, const void *, size_t, cudaMemcpyKind) {
    ++copies;
    return cudaErrorInvalidValue;
}
cudaError_t cudaDeviceSynchronize() { ++drains; return cudaSuccess; }
const char *cudaGetErrorString(cudaError_t) { return "host metadata fixture: upload is forbidden"; }

int main(int argc, char **argv) {
    try {
        check(argc == 2, "provide a private fixture directory");
        const std::filesystem::path root(argv[1]);
        check(std::filesystem::create_directory(root), "fixture directory already exists");
        const std::string row = "wide 1 0 4294967381 4294967393 4294967407 4294967419 "
                                "4294967311 -1 8 -16 32 1 1 4294967431 4294967443 4294967457 1 1";
        auto write = [&](const std::string &body) {
            std::ofstream file(root / "index.txt");
            file << "# align 256 pool 4294967552 tensors 1\n" << body << '\n';
            check(bool(file), "index fixture write failed");
        };
        write(row);
        std::string error;
        uint64_t pool = 0;
        using strata::core::WeightTable;
        check(WeightTable::pool_bytes(root.string(), pool, error), error.c_str());
        check(pool == UINT64_C(4294967552), "64-bit pool header truncated");
        const std::set<std::string> skipped{"wide"};
        check(WeightTable::pool_bytes(root.string(), pool, error, &skipped), error.c_str());
        check(pool == 0, "skipped tensor requires an arena");
        WeightTable table;
        check(table.load(root.string(), nullptr, 0, error, &skipped), error.c_str());
        const auto *r = table.find("wide");
        check(r && !r->resident && r->data == nullptr, "skipped tensor lost its metadata state");
        check(r->src_off == UINT64_C(4294967381) && r->src_bytes == UINT64_C(4294967393),
              "64-bit source extent changed");
        check(r->bytes == UINT64_C(4294967419) && r->ne0 == INT64_C(4294967311) && r->ne1 == -1,
              "destination extent or signed shape changed");
        check(r->elements == INT64_C(4294967311), "one-dimensional element count changed");
        check(r->codes_bytes == UINT64_C(4294967431) && r->scales_bytes == UINT64_C(4294967443) &&
              r->offset_bytes == UINT64_C(4294967457), "64-bit quantization planes changed");
        check(r->file_id == 1 && r->code_bits == 8 && r->code_bias == -16 && r->group_elems == 32 &&
              r->codebook_iq4nl && r->has_offset && r->scales_fp16 && r->act_kind == 1,
              "quantization metadata changed");
        check(allocations == 2 && live == 0 && copies == 0 && drains == 1,
              "skipped metadata uploaded or leaked staging");
        write(row.substr(0, row.rfind(' ')));
        WeightTable malformed;
        check(!malformed.load(root.string(), nullptr, 0, error, &skipped), "incomplete index row accepted");
        check(error.find("18 of 19 fields") != std::string::npos, "malformed index diagnostic changed");
        check(allocations == 2 && live == 0 && copies == 0, "malformed row acquired/uploaded resources");
        std::filesystem::remove_all(root);
        std::cout << "production index: >4-GiB fields, signed shape, skip metadata and malformed rejection PASS\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
