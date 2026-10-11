// A finite external owner supervises this diagnostic. No model or recovery.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) try {
    if (argc != 2 || std::string(argv[1]) != "0000:05:00.0")
        throw std::runtime_error("one fixed B570 BDF is required");
    const std::string bdf = argv[1];
    sycl::device selected;
    bool found = false;
    for (const auto& d : sycl::device::get_devices(sycl::info::device_type::gpu)) {
        if (d.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        if (d.get_info<sycl::ext::intel::info::device::pci_address>() == bdf) {
            if (found) throw std::runtime_error("ambiguous PCI device");
            selected = d;
            found = true;
        }
    }
    if (!found) throw std::runtime_error("expected PCI device absent");
    const auto native = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(selected);
    const auto status = zeDeviceGetStatus(native);
    std::cout << "STATUS " << bdf << ' ' << static_cast<uint32_t>(status) << std::endl;
    if (status != ZE_RESULT_SUCCESS)
        throw std::runtime_error("device status refused; no context/queue/probe");
    // This context is newly created, never reused from the faulting process.
    sycl::context context{selected};
    sycl::queue q{context, selected, [](sycl::exception_list errors) {
        if (errors.size()) std::rethrow_exception(*errors.begin());
    }, sycl::property::queue::in_order{}};
    std::cout << "QUEUE " << bdf << std::endl;
    constexpr size_t n = 16384;
    // Once this queue exists, a failed wait must not unwind host transfer buffers.
    try {
    auto* data = sycl::malloc_device<uint32_t>(n, q);
    if (!data) throw std::runtime_error("64 KiB allocation failed");
    std::vector<uint32_t> input(n), output(n);
    struct TransferLifetime {
        bool complete = false;
        ~TransferLifetime() {
            if (!complete) {
                std::fputs("FAIL health: queue completion unconfirmed; exit before transfer buffers unwind\n", stderr);
                std::fflush(stderr);
                std::_Exit(74);
            }
        }
    } lifetime;  // destroyed before either host vector on every failure

    for (size_t i = 0; i < n; ++i)
        input[i] = uint32_t(i) * 1664525u + 1013904223u;
    q.memcpy(data, input.data(), n * sizeof(uint32_t)).wait_and_throw();
    std::cout << "H2D 65536 complete" << std::endl;
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        data[i] = (data[i] ^ 0xa5a5a5a5u) + 17u;
    }).wait_and_throw();
    std::cout << "KERNEL 16384 complete" << std::endl;
    q.memcpy(output.data(), data, n * sizeof(uint32_t)).wait_and_throw();
    std::cout << "D2H 65536 complete" << std::endl;
    lifetime.complete = true;
    for (size_t i = 0; i < n; ++i)
        if (output[i] != ((input[i] ^ 0xa5a5a5a5u) + 17u))
            throw std::runtime_error("integer mismatch at word " + std::to_string(i));
    sycl::free(data, q);
    std::cout << "PASS " << bdf << ": 1 round, 16384 exact words" << std::endl;
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        const auto at = line.find('/');
        if (at == std::string::npos) continue;
        const auto path = line.substr(at);
        if (path.find("libsycl") != std::string::npos || path.find("libur_") != std::string::npos ||
            path.find("libze_") != std::string::npos || path.find("libumf") != std::string::npos)
            std::cout << "LIBRARY " << path << '\n';
    }
    if (!maps.eof()) throw std::runtime_error("mapping identity unavailable");
    return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL health: " << e.what() << std::endl;
        std::_Exit(74);
    } catch (...) {
        std::cerr << "FAIL health: unknown queue failure" << std::endl;
        std::_Exit(74);
    }
} catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << std::endl;
    return 1;
}
