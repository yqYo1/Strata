// A bounded external controller must supervise this program on a failed driver.
// No model, graphs, virtual memory, mapped polling words or application helpers.
#include <sycl/sycl.hpp>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) try {
    if (argc != 2) throw std::runtime_error("pass the expected PCI BDF");
    const std::string bdf = argv[1];
    sycl::device selected;
    bool found = false;
    for (const auto& d : sycl::device::get_devices(sycl::info::device_type::gpu)) {
        if (d.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        if (d.get_info<sycl::ext::intel::info::device::pci_address>() == bdf) {
            selected = d;
            found = true;
            break;
        }
    }
    if (!found) throw std::runtime_error("expected Level Zero PCI device is absent");
    sycl::queue q{selected, [](sycl::exception_list errors) {
        if (errors.size()) std::rethrow_exception(*errors.begin());
    }, sycl::property::queue::in_order{}};
    std::cout << "device " << bdf << ": "
              << selected.get_info<sycl::info::device::name>() << std::endl;
    constexpr size_t n = 16384;
    auto* data = sycl::malloc_device<uint32_t>(n, q);
    if (!data) throw std::runtime_error("64 KiB allocation failed");
    std::vector<uint32_t> input(n), output(n);
    for (uint32_t round = 0; round != 3; ++round) {
        for (size_t i = 0; i < n; ++i) input[i] = uint32_t(i) * 1664525u + 1013904223u + round;
        q.memcpy(data, input.data(), n * sizeof(uint32_t)).wait_and_throw();
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            data[i] = (data[i] ^ 0xa5a5a5a5u) + 17u;
        }).wait_and_throw();
        q.memcpy(output.data(), data, n * sizeof(uint32_t)).wait_and_throw();
        for (size_t i = 0; i < n; ++i)
            if (output[i] != ((input[i] ^ 0xa5a5a5a5u) + 17u))
                throw std::runtime_error("integer mismatch at word " + std::to_string(i));
        std::cout << "round " << round << " complete" << std::endl;
    }
    q.wait_and_throw();
    sycl::free(data, q);
    std::cout << "PASS " << bdf << ": 3 rounds, 16384 exact words each" << std::endl;
    return 0;
} catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << std::endl;
    return 1;
}
