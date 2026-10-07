// copy.inc contains the production methods; only their queue dependencies are stand-ins.
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Event { void wait() {} };
struct Queue {
    int copies = 0;
    Event memcpy(void* dst, const void* src, size_t n) {
        ++copies;
        std::memcpy(dst, src, n);
        return {};
    }
    void wait() {}
} queue;
namespace sycl { struct exception : std::runtime_error { using std::runtime_error::runtime_error; }; }
namespace dpct {
using err0 = int;
Queue& get_in_order_queue() { return queue; }
const char* get_error_string_dummy(int) { return "queue error"; }
}
namespace strata { Queue* q_of(void*) { return &queue; } }
#define DPCT_CHECK_ERROR(expression) ((void)(expression), 0)

class ExpertCache {
public:
    std::vector<uint8_t> arena;
    std::vector<uint64_t> offsets;
    int64_t blob_ = 32, slots_ = 2, backed = 2, fills_ = 0;
    bool valid() const { return !arena.empty(); }
    int64_t slots() const { return backed; }
    uint64_t slot_offset(int64_t slot) const {
        return offsets.empty() ? uint64_t(slot) * uint64_t(blob_) : offsets.at(slot);
    }
    uint8_t* device_slot(int32_t slot) {
        return valid() && slot >= 0 && slot < slots_ ? arena.data() + slot_offset(slot) : nullptr;
    }
    bool fill_slot(int32_t, const uint8_t*, void*, std::string&, int64_t = 0);
    bool fill_slot_blocking(int32_t, const uint8_t*, std::string&, int64_t = 0);
    bool fill_slot_queued(int32_t, const uint8_t*, std::string&, int64_t = 0);
    bool sync_queued(std::string&);
    bool verify_slot(int32_t, const uint8_t*, std::string&, int64_t = 0);
};
#include "copy.inc"

int main() {
    const std::vector<uint8_t> payload(64, 0xa5);
    for (int mode = 0; mode != 4; ++mode) {
        ExpertCache cache;
        cache.arena.resize(48);
        cache.offsets = {0, 32, 48};
        std::string err;
        auto copy = [&](int32_t slot, const uint8_t* src, int64_t size) {
            switch (mode) {
            case 0: return cache.fill_slot(slot, src, nullptr, err, size);
            case 1: return cache.fill_slot_blocking(slot, src, err, size);
            case 2: return cache.fill_slot_queued(slot, src, err, size);
            default: return cache.verify_slot(slot, src, err, size);
            }
        };
        assert(cache.fill_slot_blocking(1, payload.data(), err, 16));
        assert(copy(1, payload.data(), 16));
        const int copies = queue.copies;
        for (int64_t bytes : {int64_t(17), int64_t(32), int64_t(33), int64_t(0)}) {
            assert(!copy(1, payload.data(), bytes));
            assert(!err.empty() && queue.copies == copies);
        }
        assert(!copy(-1, payload.data(), 16));
        assert(!copy(2, payload.data(), 16));
        assert(!copy(1, nullptr, 16));
        assert(queue.copies == copies);
        cache.backed = 1;
        assert(!copy(1, payload.data(), 16));
        assert(queue.copies == copies);
        cache.backed = 2;
        cache.offsets.clear();
        cache.arena.resize(64);
        assert(cache.fill_slot_blocking(1, payload.data(), err));
        assert(copy(1, payload.data(), 0));
    }
    std::puts("PASS cache-copy bounds: four production copy methods, sized and uniform slots, oversized, null, invalid and unbacked slots");
}
