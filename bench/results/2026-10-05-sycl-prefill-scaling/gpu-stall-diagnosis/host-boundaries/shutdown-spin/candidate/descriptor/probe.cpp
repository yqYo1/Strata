
#include <cassert>
#include <cstdio>
#include <stdexcept>
int default_calls=0;
bool fail_default=false;
namespace sycl { struct queue { int value; }; }
namespace dpct::cs {
using queue_ptr=sycl::queue*;
sycl::queue& get_default_queue() {
    ++default_calls;
    if (fail_default) throw std::runtime_error("injected device selection failure");
    static sycl::queue q{41}; return q;
}
}
enum class math_mode { mm_default };
#include "descriptor.inc"
int main() {
#if CONTROL
    assert(default_calls==1);
    std::puts("PASS old descriptor initialized the device before main");
#else
    assert(default_calls==0);
    assert(descriptor::get_saved_queue().value==41 && default_calls==1);
    sycl::queue explicit_queue{42};
    descriptor::set_saved_queue(&explicit_queue);
    assert(descriptor::get_saved_queue().value==42 && default_calls==1);
    descriptor::set_saved_queue(nullptr);
    fail_default=true;
    bool reported=false;
    try { descriptor::get_saved_queue(); } catch (const std::runtime_error&) { reported=true; }
    assert(reported);
    std::puts("PASS lazy default, explicit queue, and propagated initialization error");
#endif
}
