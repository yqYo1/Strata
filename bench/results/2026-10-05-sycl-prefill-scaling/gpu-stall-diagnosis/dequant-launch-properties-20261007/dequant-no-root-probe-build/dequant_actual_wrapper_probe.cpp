// Calls the actual production wrappers. Link this one host object separately
// against the unchanged and two-property-changed iq_kernels objects.
#include <sycl/sycl.hpp>
#include "strata/kernels/iq_kernels.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) try {
    if (argc != 2) throw std::runtime_error("one new output directory required");
    const std::filesystem::path output(argv[1]);
    if (!std::filesystem::create_directory(output))
        throw std::runtime_error("refusing to overwrite probe outputs");
    sycl::queue queue(sycl::gpu_selector_v, sycl::property::queue::in_order{});
    std::fprintf(stderr, "device=%s driver=%s\n",
        queue.get_device().get_info<sycl::info::device::name>().c_str(),
        queue.get_device().get_info<sycl::info::device::driver_version>().c_str());
    constexpr size_t guard = 64;
    constexpr uint16_t guard_bits = 0x5a5a;
    constexpr std::array<uint16_t, 4> scales{0x0000, 0x3000, 0xb000, 0x2800};
    const std::array<std::array<int64_t, 2>, 4> shapes{{
        {1, 256}, {1, 2560}, {640, 2560}, {641, 2560}}};
    int cases = 0;
    for (unsigned seed : {7u, 23u}) {
        std::mt19937 random(seed);
        for (int type : {16, 17, 18, 20, 21, 22, 23, 11, 42}) {
            const int qk = type == 20 ? 32 : type == 42 ? 64 : 256;
            const size_t block_bytes = strata::kernels::iq_row_bytes(type, qk);
            if (!block_bytes) throw std::runtime_error("unknown block size");
            for (const auto& shape : shapes) {
                const int64_t rows = shape[0], columns = shape[1];
                const size_t input_bytes = strata::kernels::iq_row_bytes(type, columns) * rows;
                auto input = [&]() {
                    std::vector<uint8_t> bytes(input_bytes);
                    for (auto& value : bytes) value = uint8_t(random());
                    // Q3_K puts d at the end; the other covered types put d first.
                    // Rotate zero, positive and negative finite scales per block.
                    for (size_t at = 0; at < input_bytes; at += block_bytes) {
                        const size_t index = at + (type == 11 ? block_bytes - 2 : 0);
                        const auto scale = scales[(at / block_bytes + seed) % scales.size()];
                        bytes[index] = uint8_t(scale);
                        bytes[index + 1] = uint8_t(scale >> 8);
                    }
                    return bytes;
                };
                const auto gate_host = input(), up_host = input();
                auto* gate = sycl::malloc_device<uint8_t>(input_bytes, queue);
                auto* up = sycl::malloc_device<uint8_t>(input_bytes, queue);
                const size_t max_elements = size_t(2 * rows * columns);
                auto* destination = sycl::malloc_device<uint16_t>(max_elements + 2 * guard, queue);
                if (!gate || !up || !destination) throw std::runtime_error("device allocation failed");
                queue.memcpy(gate, gate_host.data(), input_bytes);
                queue.memcpy(up, up_host.data(), input_bytes);
                queue.wait_and_throw();
                for (bool gu : {false, true}) {
                    const size_t elements = size_t((gu ? 2 : 1) * rows * columns);
                    const size_t total = elements + 2 * guard;
                    queue.memset(destination, 0x5a, total * sizeof(uint16_t));
                    if (gu) strata::kernels::iq_dequant_gu_f16(type, gate, up, rows, columns,
                                                              destination + guard, &queue);
                    else strata::kernels::iq_dequant_f16(type, gate, rows * columns,
                                                         destination + guard, &queue);
                    queue.wait_and_throw();
                    std::vector<uint16_t> head(total);
                    queue.memcpy(head.data(), destination, total * sizeof(uint16_t)).wait_and_throw();
                    for (size_t at = 0; at < guard; ++at)
                        if (head[at] != guard_bits || head[guard + elements + at] != guard_bits)
                            throw std::runtime_error("output guard changed");
                    for (size_t at = guard; at < guard + elements; ++at)
                        if (!std::isfinite(float(sycl::bit_cast<sycl::half>(head[at]))))
                            throw std::runtime_error("nonfinite output value");
                    const auto name = "type" + std::to_string(type) + "-seed" + std::to_string(seed)
                        + "-rows" + std::to_string(rows) + "-columns" + std::to_string(columns)
                        + (gu ? "-gu.bin" : "-flat.bin");
                    std::ofstream stream(output / name, std::ios::binary | std::ios::out);
                    stream.write(reinterpret_cast<const char*>(head.data()), total * sizeof(uint16_t));
                    stream.close();
                    if (!stream) throw std::runtime_error("output write failed");
                    ++cases;
                    std::printf("{\"type\":%d,\"seed\":%u,\"rows\":%lld,\"columns\":%lld,"
                                "\"gu\":%s,\"elements\":%zu,\"guards\":%zu,"
                                "\"finite\":true,\"guards_intact\":true,\"file\":\"%s\"}\n",
                        type, seed, static_cast<long long>(rows), static_cast<long long>(columns),
                        gu ? "true" : "false", elements, guard, name.c_str());
                    std::fflush(stdout);
                }
                sycl::free(destination, queue);
                sycl::free(up, queue);
                sycl::free(gate, queue);
            }
        }
    }
    if (cases != 144) throw std::runtime_error("incomplete fixture");
    std::fprintf(stderr, "all %d cases finite with intact output guards\n", cases);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "actual-wrapper dequant probe failed: %s\n", error.what());
    return 1;
}
