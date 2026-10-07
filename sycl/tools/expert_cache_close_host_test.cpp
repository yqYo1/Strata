// GPU-free lifetime test. close.inc is extracted from the production method.
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Queue {};
Queue queue;
std::vector<std::function<void()>> pending[2];
int frees = 0, drains = 0, reads = 0;
bool fail_drain = false;
struct Device {
    void queues_wait_and_throw() {
        ++drains;
        if (fail_drain) throw std::runtime_error("deferred queue failure");
        for (auto& work : pending) {
            for (auto& command : work) command();
            work.clear();
        }
    }
} device;
}
namespace dpct {
Queue& get_in_order_queue() { return queue; }
Device& get_current_device() { return device; }
}
namespace sycl {
void free(uint8_t* address, Queue&) { ++frees; delete[] address; }
}
class ExpertCache {
public:
    uint8_t* base_ = nullptr;
    struct Vmm { void release() {} };
    std::unique_ptr<Vmm> vmm_;
    std::vector<int> off_, segs_, residency_, layer_next_;
    int slots_ = 1, live_slots_ = 1, n_layers_ = 1, n_expert_ = 1;
    int blob_ = 1024, next_free_ = 1, fills_ = 1, admitted_ = 1;
    void close();
    void release_segmented() { throw std::runtime_error("not used by ordinary-USM test"); }
};
#include "close.inc"

int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string mode = argv[1];
    ExpertCache cache;
    cache.base_ = new uint8_t[1024];
    for (int i = 0; i != 1024; ++i) cache.base_[i] = 0xa5;
    auto* address = cache.base_;
    if (mode == "read" || mode == "two-queues") {
        const int count = mode == "read" ? 1 : 2;
        for (int q = 0; q != count; ++q)
            pending[q].push_back([address] {
                for (int i = 0; i != 1024; ++i) assert(address[i] == 0xa5);
                ++reads;
            });
        cache.close();
        // The old close frees the payload first; this reproduces its UAF.
        device.queues_wait_and_throw();
        assert(reads == count && frees == 1 && cache.base_ == nullptr);
    } else if (mode == "write") {
        pending[1].push_back([address] { address[0] = 0x5b; assert(address[0] == 0x5b); ++reads; });
        cache.close();
        device.queues_wait_and_throw();
        assert(reads == 1 && frees == 1);
    } else if (mode == "failed-drain") {
        fail_drain = true;
        bool threw = false;
        try { cache.close(); } catch (const std::runtime_error&) { threw = true; }
        assert(threw && frees == 0 && cache.base_ == address && address[0] == 0xa5);
        fail_drain = false;
        cache.close();
        assert(frees == 1 && cache.base_ == nullptr);
    } else if (mode == "repeat-close") {
        cache.close(); cache.close();
        assert(frees == 1 && drains == 1 && cache.base_ == nullptr);
        assert(cache.slots_ == 0 && cache.live_slots_ == 0 && cache.blob_ == 0);
    } else {
        throw std::runtime_error("unknown mode");
    }
    std::printf("PASS %s: drains=%d frees=%d reads=%d\n", mode.c_str(), drains, frees, reads);
}
