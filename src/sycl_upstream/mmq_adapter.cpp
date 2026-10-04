// Original prefill::mmq entry points backed by the faithful SYCL components.
#include "strata/sycl_upstream/mmq_adapter.hpp"
#include "strata/sycl_upstream/mmq_product.hpp"
#include "strata/sycl_upstream/mmq_stages.hpp"
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <algorithm>
#include <climits>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace strata::sycl_upstream {
bool mmq_device_fits(const sycl::device& d, int type, int64_t rows) {
    if (!prefill::mmq::supported(type) || rows <= 0 || rows > INT_MAX) return false;
    if (!d.is_gpu() || d.get_info<sycl::info::device::vendor_id>() != 0x8086 ||
        !d.has(sycl::aspect::ext_intel_matrix) || !d.has(sycl::aspect::fp16) ||
        !d.has(sycl::aspect::usm_device_allocations)) return false;
#ifdef STRATA_SYCL_MMQ_AOT_ARCH
    if (d.get_info<sycl::ext::oneapi::experimental::info::device::architecture>() != STRATA_SYCL_MMQ_AOT_ARCH)
        return false;
#endif
    // Local arrays in mmq_product.cpp: weight tile, A/B, and two int32 products.
    constexpr size_t local_bytes = 16 * 84 * 4 + 64 * 32 + 32 * 16 + 2 * 64 * 16 * 4;
    if (d.get_info<sycl::info::device::max_work_group_size>() < 128 ||
        d.get_info<sycl::info::device::max_work_item_sizes<1>>()[0] < 128 ||
        d.get_info<sycl::info::device::max_work_item_sizes<2>>()[1] < 128 ||
        d.get_info<sycl::info::device::local_mem_size>() < local_bytes) return false;
    const auto sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    if (std::find(sizes.begin(), sizes.end(), 16) == sizes.end() ||
        std::find(sizes.begin(), sizes.end(), 32) == sizes.end()) return false;
    namespace mx = sycl::ext::oneapi::experimental::matrix;
    for (const auto& c : d.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>()) {
        if (c.atype != mx::matrix_type::sint8 || c.btype != mx::matrix_type::sint8 ||
            c.ctype != mx::matrix_type::sint32 || c.dtype != mx::matrix_type::sint32) continue;
        const auto dimension = [](size_t value, size_t exact, size_t maximum) {
            return exact ? value == exact : maximum >= value;
        };
        if (dimension(8, c.msize, c.max_msize) && dimension(16, c.nsize, c.max_nsize) &&
            dimension(32, c.ksize, c.max_ksize)) return true;
    }
    return false;
}
} // namespace strata::sycl_upstream

namespace strata::prefill::mmq {
namespace {
using namespace sycl_upstream;
sycl::queue& queue(void* stream) {
    if (!stream) throw std::invalid_argument("SYCL MMQ requires an explicit queue");
    auto& q = *static_cast<sycl::queue*>(stream);
    if (!q.is_in_order()) throw std::invalid_argument("SYCL MMQ requires an in-order queue");
    return q;
}
int narrow(int64_t n) {
    if (n < 0 || n > INT_MAX) throw std::invalid_argument("SYCL MMQ dimension exceeds int32 range");
    return int(n);
}
struct QueueState {
    sycl::queue q;
    float* scratch = nullptr;
    size_t capacity = 0;
    int compute_units;
    explicit QueueState(const sycl::queue& q): q(q) {
        // All nine formats use the same tile resources. Cache this device check
        // and CU count once per queue rather than querying the driver per product.
        if (!mmq_device_fits(q.get_device(), GGML_TYPE_Q8_0, 1))
            throw std::invalid_argument("SYCL MMQ has no supported tile on this device");
        compute_units = q.get_device().get_info<sycl::info::device::max_compute_units>();
    }
    ~QueueState() {
        // Unlike cudaFree, sycl::free does not implicitly wait for device users.
        // Retaining a queue copy also permits destruction of the caller's queue
        // wrapper before Context. There is no such wait in run().
        try { q.wait_and_throw(); }
        catch (const std::exception& e) {
            std::fprintf(stderr, "SYCL MMQ teardown: %s\n", e.what());
            std::terminate();
        }
        if (scratch) sycl::free(scratch, q);
    }
    float* reserve(size_t bytes) {
        if (!bytes) return nullptr;
        // The normal planner's only nonzero size is CU * 64 * 16 * sizeof(float).
        // The device of this retained queue is fixed, so scratch never grows.
        if (capacity && bytes > capacity) throw std::logic_error("MMQ scratch plan changed for one device");
        if (!scratch) {
            scratch = static_cast<float*>(sycl::malloc_device(bytes, q));
            if (!scratch) throw std::bad_alloc();
            capacity = bytes;
        }
        return scratch;
    }
};
struct State {
    // Serialize host submissions so another run cannot interleave its main
    // kernel between a main/fixup pair sharing scratch. Different queues retain
    // separate scratch and their submitted device work can run concurrently.
    std::mutex mutex;
    std::vector<std::unique_ptr<QueueState>> queues;
    QueueState& get(const sycl::queue& q) {
        for (auto& entry : queues) if (entry->q == q) return *entry;
        queues.push_back(std::make_unique<QueueState>(q));
        return *queues.back();
    }
};
}
bool built() { return true; }
bool supported(int t) {
    return t >= 0 && t < GGML_TYPE_COUNT && sycl_upstream::mmq_product_supported(ggml_type(t));
}
bool fits(int t, int64_t rows) {
    if (!supported(t)) return false;
    const auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devices.empty()) return false;
    for (const auto& device : devices) if (!sycl_upstream::mmq_device_fits(device, t, rows)) return false;
    return true;
}
size_t matrix_bytes(int t, int64_t rows, int64_t cols) {
    if (t < 0 || t >= GGML_TYPE_COUNT || rows < 0 || cols < 0)
        throw std::invalid_argument("invalid MMQ matrix geometry/type");
    const auto type = ggml_type(t);
    const auto block = ggml_blck_size(type);
    const auto bytes = ggml_type_size(type);
    if (block <= 0 || !bytes) throw std::invalid_argument("invalid GGML block type");
    const auto count = size_t(cols / block);
    if (count > std::numeric_limits<size_t>::max() / bytes ||
        (count && size_t(rows) > std::numeric_limits<size_t>::max() / (count * bytes)))
        throw std::invalid_argument("MMQ matrix allocation overflow");
    return size_t(rows) * count * bytes;
}
size_t q8_bytes(int64_t rows, int64_t cols) { return sycl_upstream::mmq_q8_bytes(rows, cols); }
void quantize(const float* x, const int32_t* ids, void* xq, int t, int64_t cols, int64_t ld, int64_t rows, void* stream) {
    if (rows <= 0) return;
    sycl_upstream::mmq_quantize(queue(stream), x, ids, static_cast<sycl_upstream::MmqBlock*>(xq), ggml_type(t), cols, ld, rows);
}
Context::Context(): ctx_(new State) {}
Context::~Context() { delete static_cast<State*>(ctx_); }
void Context::run(const Product& p, void* stream) {
    if (p.n <= 0 || p.max_rows <= 0) return;
    auto& q = queue(stream);
    const sycl_upstream::MmqProduct product{p.w, ggml_type(p.type), narrow(p.w_rows), narrow(p.w_cols), p.expert_bytes,
        p.n, static_cast<const sycl_upstream::MmqBlock*>(p.xq), p.bounds, p.ids, narrow(p.total_rows), narrow(p.max_rows),
        p.dst, narrow(p.ld_dst)};
    if (!supported(p.type)) throw std::invalid_argument("unsupported SYCL MMQ product type");
    auto& state = *static_cast<State*>(ctx_);
    std::lock_guard lock(state.mutex);
    auto& entry = state.get(q);
    const auto plan = sycl_upstream::mmq_plan(product, entry.compute_units);
    sycl_upstream::mmq_product(entry.q, product, plan, entry.reserve(plan.scratch_bytes));
}
void gather_native(const void* g, const void* u, size_t half, const void* d, size_t bytes, void* gu, void* dn, void* stream) {
    sycl_upstream::mmq_gather_native(queue(stream), g, u, half, d, bytes, gu, dn);
}
bool gather_native_group(const GatherGroup& g, size_t up, size_t half, size_t down, size_t bytes,
    void* gu, size_t gs, void* dn, size_t ds, void* stream) {
    // Original admission checks precede any stream use.
    if (g.first < 0 || g.n <= g.first || g.n > kGatherGroupMax) return false;
    uintptr_t alignment = uintptr_t(gu) | uintptr_t(dn) | up | half | down | bytes | gs | ds;
    for (int e = g.first; e < g.n; ++e) alignment |= uintptr_t(g.blob[e]);
    if (alignment % 16) return false;
    return sycl_upstream::mmq_gather_native_group(queue(stream), g, up, half, down, bytes, gu, gs, dn, ds);
}
void gather_strata_q2(const uint8_t* blob, void* gu, void* dn, void* stream) {
    sycl_upstream::mmq_gather_strata_q2(queue(stream), blob, gu, dn);
}
void swiglu(const float* gu, float* h, int64_t rows, int64_t ff, bool interleaved, void* stream) {
    if (rows <= 0) return;
    sycl_upstream::mmq_swiglu(queue(stream), gu, h, rows, ff, interleaved);
}
void iota(int32_t* dst, int64_t n, void* stream) {
    if (n <= 0) return;
    sycl_upstream::mmq_iota(queue(stream), dst, n);
}
} // namespace strata::prefill::mmq
