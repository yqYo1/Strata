// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/phase_cache_owner.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/runtime.hpp"
#include <exception>
#include <cstdio>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>

namespace strata::core {
namespace {
class GpuBackend final : public PhaseCacheBackend {
public:
    explicit GpuBackend(int ordinal) : ordinal_(Runtime::at(ordinal).ordinal()) {}
    bool caps(Caps& output, std::string& error) noexcept override {
        try {
            const OnDevice selected(ordinal_);
            Caps actual;
            actual.vmm = gpu::vmem_supported();
            if (!actual.vmm) { error = "phase cache: VMM aspect unavailable"; return false; }
            for (const auto& device : Runtime::get().context().get_devices())
                if (!device.has(sycl::aspect::ext_oneapi_virtual_mem)) {
                    error = "phase cache: context device lacks VMM aspect"; return false;
                }
            actual.granularity = gpu::vmem_granularity();
            namespace vm = sycl::ext::oneapi::experimental;
            actual.device_minimum = vm::get_mem_granularity(Runtime::get().device(), Runtime::get().context(),
                                                           vm::granularity_mode::minimum);
            actual.context_minimum = vm::get_mem_granularity(Runtime::get().context(), vm::granularity_mode::minimum);
            actual.maximum = Runtime::get().device().get_info<sycl::info::device::max_mem_alloc_size>();
            size_t free = 0, total = 0;
            actual.free_known = gpu::mem_info(&free, &total);
            actual.free = free;
            output = actual;
            if (!actual.vmm || !actual.granularity || !actual.device_minimum || !actual.context_minimum ||
                !actual.maximum || !actual.free_known) {
                error = "phase cache: actual Runtime VMM/free/capability query unavailable";
                return false;
            }
            if (!reported_) {
                std::fprintf(stderr, "strata phase-cache: ordinal=%d device=%s backend=%d context_devices=%zu "
                             "device_recommended=%llu device_minimum=%llu context_minimum=%llu maximum=%llu\n",
                             ordinal_, Runtime::get().device().get_info<sycl::info::device::name>().c_str(),
                             static_cast<int>(Runtime::get().context().get_backend()),
                             Runtime::get().context().get_devices().size(),
                             static_cast<unsigned long long>(actual.granularity),
                             static_cast<unsigned long long>(actual.device_minimum),
                             static_cast<unsigned long long>(actual.context_minimum),
                             static_cast<unsigned long long>(actual.maximum));
                reported_ = true;
            }
            return true;
        } catch (const std::exception& exception) { error = exception.what(); return false; }
    }
    void* reserve(uint64_t bytes, std::string& error) noexcept override {
        try {
            const OnDevice selected(ordinal_);
            void* result = gpu::vmem_reserve(static_cast<size_t>(bytes));
            if (!result) error = gpu::last_error();
            return result;
        } catch (const std::exception& exception) { error = exception.what(); return nullptr; }
    }
    bool release(void* pointer, uint64_t bytes, std::string& error) noexcept override {
        try {
            const OnDevice selected(ordinal_);
            if (gpu::vmem_free_checked(pointer, static_cast<size_t>(bytes))) return true;
            error = gpu::last_error();
        } catch (const std::exception& exception) { error = exception.what(); }
        return false;
    }
    void* physical(uint64_t bytes, std::string& error) noexcept override {
        try {
            const OnDevice selected(ordinal_);
            auto* result = gpu::vmem_new(static_cast<size_t>(bytes));
            if (!result) error = gpu::last_error();
            return result;
        } catch (const std::exception& exception) { error = exception.what(); return nullptr; }
    }
    bool attach(void* physical, void* address, uint64_t bytes, std::string& error) noexcept override {
        try {
            const OnDevice selected(ordinal_);
            if (gpu::vmem_attach(static_cast<gpu::VmemSegment*>(physical), address, static_cast<size_t>(bytes)))
                return true;
            error = gpu::last_error();
        } catch (const std::exception& exception) { error = exception.what(); }
        return false;
    }
    bool detach(void* address, uint64_t bytes, std::string& error) noexcept override {
        try {
            const OnDevice selected(ordinal_);
            if (gpu::vmem_detach(address, static_cast<size_t>(bytes))) return true;
            error = gpu::last_error();
        } catch (const std::exception& exception) { error = exception.what(); }
        return false;
    }
    void destroy_physical(void* physical) noexcept override {
        try {
            const OnDevice selected(ordinal_);
            gpu::vmem_delete(static_cast<gpu::VmemSegment*>(physical));
        } catch (...) { std::terminate(); }
    }
    bool drain(std::string& error) noexcept override {
        try {
            const OnDevice selected(ordinal_);
            Runtime::get().finish();
            return true;
        } catch (const std::exception& exception) { error = exception.what(); return false; }
    }
    bool write(void* destination, const void* source, uint64_t bytes, std::string& error) noexcept override {
        try {
            const OnDevice selected(ordinal_);
            if (gpu::copy(destination, source, static_cast<size_t>(bytes))) return true;
            error = gpu::last_error();
        } catch (const std::exception& exception) { error = exception.what(); }
        return false;
    }
    bool read(void* destination, const void* source, uint64_t bytes, std::string& error) noexcept override {
        return write(destination, source, bytes, error);
    }
private:
    int ordinal_;
    bool reported_ = false;
};
}
std::shared_ptr<PhaseCacheBackend> make_phase_cache_gpu_backend(int ordinal) {
    return std::make_shared<GpuBackend>(ordinal);
}
}  // namespace strata::core
