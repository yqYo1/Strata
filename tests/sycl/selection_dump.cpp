// Main-authored bounded diagnostic readback test; repository LICENSE applies.
#include <cuda_runtime.h>
#include "strata/prefill/selection_dump.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool ok, const char* error) {
    if (!ok) throw std::runtime_error(error);
}
void check(cudaError_t status) {
    require(status == cudaSuccess, "selection fixture runtime operation failed");
}
struct Owners {
    cudaStream_t stream{};
    int32_t* device{};
    int32_t boundary[2]{}; // lives through the destructor's checked drain
    ~Owners() noexcept {
        if (stream && cudaStreamSynchronize(stream) != cudaSuccess)
            strata::sycl_backend::cleanup_fail_stop("selection fixture completion uncertain");
        if (device && cudaFree(device) != cudaSuccess)
            strata::sycl_backend::cleanup_fail_stop("selection fixture free failed");
        if (stream && cudaStreamDestroy(stream) != cudaSuccess)
            strata::sycl_backend::cleanup_fail_stop("selection fixture stream retirement failed");
    }
};
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 3, "use small|full PRIVATE_OUTPUT");
        const std::string mode = argv[1], path = argv[2];
        require(mode == "small" || mode == "full", "unknown fixture mode");
        require(!std::ifstream(path).good(), "fixture output already exists");
        const int32_t rows = mode == "small" ? 17 : 8192;
        const int32_t cap = mode == "small" ? 33 : 2051;
        const size_t count = size_t(rows) * size_t(cap);
        constexpr int32_t guard = -1234567;
        Owners owned;
        check(cudaStreamCreate(&owned.stream));
        check(cudaMalloc(reinterpret_cast<void**>(&owned.device), (count + 2) * sizeof(int32_t)));
        int32_t* device = owned.device;
        owned.stream->parallel_for(sycl::range<1>(count + 2), [=](sycl::id<1> index) {
            const size_t i = index[0];
            device[i] = i == 0 || i == count + 1 ? guard : int32_t(((i - 1) * 17 + 19) % 262144);
        });
        size_t copies = 0, drains = 0;
        std::string error;
        auto copy = [&](void* dst, const void* src, size_t bytes) {
            ++copies;
            return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, owned.stream) == cudaSuccess;
        };
        auto drain = [&] { ++drains; return cudaStreamSynchronize(owned.stream) == cudaSuccess; };
        auto fatal = []() noexcept { strata::sycl_backend::cleanup_fail_stop("selection fixture readback uncertain"); };
        const bool dumped = strata::prefill::detail::selection_dump(path.c_str(), 0, 0, rows, cap,
                    device + 1, copy, drain, fatal, error);
        require(dumped, error.c_str());
        require(copies == 1 && drains == 1, "unexpected submission/retirement count");
        require(!strata::prefill::detail::selection_dump(path.c_str(), 0, 0, 0, cap,
                    device + 1, copy, drain, fatal, error), "invalid geometry accepted");
        require(copies == 1 && drains == 1, "invalid geometry submitted GPU work");
        check(cudaMemcpyAsync(&owned.boundary[0], device, sizeof(int32_t), cudaMemcpyDeviceToHost, owned.stream));
        check(cudaMemcpyAsync(&owned.boundary[1], device + count + 1, sizeof(int32_t), cudaMemcpyDeviceToHost, owned.stream));
        check(cudaStreamSynchronize(owned.stream));
        require(owned.boundary[0] == guard && owned.boundary[1] == guard, "source guard changed");
        std::ifstream file(path, std::ios::binary);
        int32_t header[4]{};
        file.read(reinterpret_cast<char*>(header), sizeof(header));
        require(file.good() && header[0] == 0 && header[1] == 0 && header[2] == rows && header[3] == cap,
                "dump header differs");
        std::vector<int32_t> values(count);
        file.read(reinterpret_cast<char*>(values.data()), std::streamsize(count * sizeof(int32_t)));
        require(file.good() && file.peek() == std::char_traits<char>::eof(), "dump payload extent differs");
        for (size_t i = 0; i < count; ++i)
            require(values[i] == int32_t((i * 17 + 19) % 262144), "dump payload differs");
        std::cout << "selection_dump mode=" << mode << " rows=" << rows << " cap=" << cap
                  << " values=" << count << " exact=PASS guards=PASS invalid_geometry=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "selection_dump fixture: " << error.what() << '\n';
        return 1;
    }
}
