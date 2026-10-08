#include <sycl/sycl.hpp>
#include <cstdint>
#include <iostream>
#include <vector>
int main() {
    constexpr std::size_t n = 16384;
    try {
        std::cout << "queue creation start" << std::endl;
        sycl::queue q{sycl::gpu_selector_v, [](sycl::exception_list xs) {
            for (const auto &x : xs) std::rethrow_exception(x);
        }, sycl::property::queue::in_order{}};
        std::cout << "device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
        auto *d = sycl::malloc_device<std::uint32_t>(n, q);
        if (!d) throw std::runtime_error("64 KiB device allocation failed");
        std::cout << "64 KiB allocated" << std::endl;
        std::vector<std::uint32_t> in(n), out(n);
        for (std::size_t i=0; i<n; ++i) in[i]=std::uint32_t(i)*1664525u+1013904223u;
        q.memcpy(d,in.data(),n*sizeof(*d)).wait_and_throw();
        std::cout << "H2D complete" << std::endl;
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            d[i] = (d[i] ^ 0xa5a5a5a5u) + 17u;
        }).wait_and_throw();
        std::cout << "kernel complete" << std::endl;
        q.memcpy(out.data(),d,n*sizeof(*d)).wait_and_throw();
        std::cout << "D2H complete" << std::endl;
        std::uint64_t sum=0;
        for (std::size_t i=0; i<n; ++i) {
            if (out[i] != ((in[i] ^ 0xa5a5a5a5u) + 17u)) {
                std::cerr << "FAIL at " << i << std::endl;
                return 2;
            }
            sum += out[i];
        }
        sycl::free(d,q);
        std::cout << "PASS: " << n << " exact words, checksum " << sum << std::endl;
        return 0;
    } catch(const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
}
