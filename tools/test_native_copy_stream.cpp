// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Explicit runtime-dependent fixture; never register as an unguarded CPU CTest.
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include <sycl/sycl.hpp>
#include <cstdio>
#include <initializer_list>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void progress(const char* message) {
    std::puts(message);
    std::fflush(stdout);
}
}

int main(int argc, char** argv) {
    if (argc != 2 || (std::strcmp(argv[1], "--require-native") != 0 &&
                      std::strcmp(argv[1], "--require-generic") != 0)) {
        std::fputs("usage: test_native_copy_stream --require-native|--require-generic\n", stderr);
        return 2;
    }
    const bool want_native = std::strcmp(argv[1], "--require-native") == 0;
    const char* flag = std::getenv("STRATA_COPY_ENGINE");
    if (want_native && (!flag || std::strcmp(flag, "1") != 0)) return 2;
    namespace gpu = strata::gpu;
    strata::core::Runtime* runtime = nullptr;
    try { runtime = &strata::core::Runtime::get(); }
    catch (const std::exception& error) {
        std::fprintf(stderr, "SKIP runtime unavailable: %s\n", error.what());
        return 77;
    }
    auto& rt = *runtime;
    gpu::Stream copy = nullptr;
    gpu::Event *copied = nullptr, *read_done = nullptr, *second = nullptr, *timing = nullptr;
    unsigned char *source = nullptr, *output = nullptr, *device = nullptr;
    unsigned int* observed = nullptr;
    constexpr size_t guard = 16, payload = 1024, extent = payload + 2 * guard;
    // Under 8 KiB total. All resources remain owned across the catch/drain boundary.
    int result = 1;
    try {
        progress("BEGIN native-copy fixture allocation");
        copy = gpu::copy_stream_create();
        require(copy != nullptr, gpu::last_error());
        auto& queue = rt.stream_for_copy(copy);
        require(queue.get_context() == rt.context() && queue.get_device() == rt.device(),
                "copy queue identity mismatch");
        const bool native = rt.copy_only(&queue);
        std::printf("ROUTE native=%d\n", native ? 1 : 0);
        std::fflush(stdout);
        if (native != want_native) {
            progress("SKIP requested treatment not selected");
            result = 77;
        } else {
            source = static_cast<unsigned char*>(gpu::alloc_host(extent));
            output = static_cast<unsigned char*>(gpu::alloc_host(extent));
            observed = static_cast<unsigned int*>(gpu::alloc_host(sizeof(unsigned int)));
            device = static_cast<unsigned char*>(gpu::alloc_device(extent));
            require(source && output && observed && device, "USM allocation failed");
            require(gpu::is_host_usm(source) && gpu::is_host_usm(output), "host-USM fixture required");
            *observed = 0;
            std::memset(source, 0xA5, extent);
            for (size_t i = 0; i < payload; ++i) source[guard + i] = static_cast<unsigned char>(i * 17 + 3);
            require(gpu::event_create(&copied) && gpu::event_create(&read_done) &&
                    gpu::event_create(&second) && gpu::event_create(&timing, true), "event allocation failed");
            require(gpu::copy_async_after(device, source, extent, copy, nullptr, copied), gpu::last_error());
            require(gpu::stream_wait_event(nullptr, copied), gpu::last_error());
            // Compute observer is on the existing compute queue, never the copy-only queue.
            auto* data = device;
            auto* check = observed;
            rt.compute().single_task([=] {
                unsigned int errors = 0;
                for (size_t i = 0; i < extent; ++i) {
                    const unsigned char expected = i < guard || i >= guard + payload ? 0xA5 :
                        static_cast<unsigned char>((i - guard) * 17 + 3);
                    errors += data[i] != expected;
                }
                // Reader writes a distinguishable value: the next copy must run after it.
                data[guard] = 0xEE;
                *check = errors + 1;
            });
            require(gpu::event_record(read_done, nullptr), gpu::last_error());
            // Source stays immutable until the two copy events and the reader are drained.
            require(gpu::copy_async_after(device + guard, source + guard, payload, copy,
                                         read_done, second), gpu::last_error());
            require(gpu::stream_wait_event(nullptr, second), gpu::last_error());
            require(gpu::copy_async(output, device, extent, nullptr), gpu::last_error());
            rt.finish();
            require(*observed == 1, "compute observer saw incorrect first copy");
            require(std::memcmp(source, output, extent) == 0, "copy/reader dependency or canary mismatch");
            if (native) {
                bool refused = false;
                try { (void) rt.stream(copy); } catch (const strata::core::DeviceError&) { refused = true; }
                require(refused, "native stream accepted the kernel queue accessor");
                require(!gpu::memset_async(device, 0, extent, copy), "native stream accepted memset");
                require(!gpu::copy2d_async(device, 32, source, 32, 16, 2, copy),
                        "native stream accepted copy2d parallel_for");
                require(!gpu::launch_host(copy, [] {}), "native stream accepted host_task");
                require(!gpu::begin_capture(copy), "native stream accepted graph capture");
                require(!gpu::event_record(timing, copy), "native stream accepted profiling tag");
                require(gpu::event_record(second, copy), gpu::last_error());
            }
            rt.finish();
            result = 0;
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL native-copy fixture: %s\n", error.what());
        result = 1;
    }
    // Never release resources or report PASS while pending/async-failed work is undrained.
    try { rt.finish(); }
    catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL drain; retaining resources until process exit: %s\n", error.what());
        std::fflush(stderr);
        std::_Exit(1);
    }
    for (auto* event : {copied, read_done, second, timing}) gpu::event_destroy(event);
    for (const void* pointer : {static_cast<void*>(source), static_cast<void*>(output),
                               static_cast<void*>(observed), static_cast<void*>(device)}) gpu::free(pointer);
    if (copy) gpu::stream_destroy(copy);
    if (result == 0) progress("PASS native-copy fixture drained");
    return result;
}
