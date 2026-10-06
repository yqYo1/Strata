// The production destroy_queue method, with CPU queue objects.
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
namespace sycl {
struct queue { int marker = 123; };
}
class Device {
    using mutex_type = std::mutex;
    mutex_type m_mutex;
    std::vector<std::shared_ptr<sycl::queue>> _queues;
public:
    sycl::queue *_q_in_order, *_q_out_of_order, *owned;
    Device() {
        for (int i = 0; i < 3; ++i) _queues.push_back(std::make_shared<sycl::queue>());
        _q_in_order = _queues[0].get(); _q_out_of_order = _queues[1].get(); owned = _queues[2].get();
    }
#include "destroy.inc"
};
int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string mode = argv[1];
    Device device, other;
    sycl::queue* address = mode == "default-in" ? device._q_in_order :
                           mode == "default-out" ? device._q_out_of_order :
                           mode == "foreign" ? other.owned :
                           mode == "null" ? nullptr : device.owned;
    auto* saved = address;
    bool rejected = false;
    try { device.destroy_queue(address); } catch (const std::invalid_argument&) { rejected = true; }
    if (mode == "default-in" || mode == "default-out" || mode == "foreign") {
        // Access first: the old method has silently destroyed the default.
        assert(saved->marker == 123);
        assert(rejected && address == saved);
    } else {
        assert(!rejected && address == nullptr);
        if (mode == "repeat") device.destroy_queue(address);
    }
    assert(device._q_in_order->marker == 123 && device._q_out_of_order->marker == 123);
    std::printf("PASS %s\n", mode.c_str());
}
