// Production queue initializers and destructor; CPU queue/USM stand-ins.
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int current_device = 0, waits = 0, destroys = 0, frees = 0;
bool fail_wait = false;
struct Queue {
    int device, marker = 123;
    void wait() { assert(marker == 123); ++waits; }
    void wait_and_throw() {
        wait();
        assert(current_device == device);
        if (fail_wait) throw std::runtime_error("deferred failure");
    }
};
Queue* default_queue = new Queue{0};
struct Device {
    void destroy_queue(Queue*& q) {
        assert(q && q->device == current_device);
        ++destroys; delete q; q = nullptr;
    }
} device;
}
namespace dpct {
using queue_ptr = Queue*;
Queue& get_in_order_queue() { return *default_queue; }
Device& get_current_device() { return device; }
}
namespace sycl {
void free(void* p, Queue& q) {
    assert(q.marker == 123);
    ++frees; delete[] static_cast<char*>(p);
}
}
struct OnDevice {
    int saved = current_device;
    explicit OnDevice(int d) { if (d >= 0) current_device = d; }
    ~OnDevice() { current_device = saved; }
};
struct Graph {};
class Verifier {
public:
    int device_ = -1;
#include "queues.inc"
    Graph* exec_[9] = {};
    Graph* commit_exec_ = nullptr;
    void* arena_ = nullptr;
    void* host_staging_ = nullptr;
    void *h_tok_ = nullptr, *h_step_ = nullptr, *h_pos_ = nullptr, *h_commit_ = nullptr,
         *h_ple_ = nullptr, *h_out_ = nullptr, *h_x_ = nullptr, *h_ids_ = nullptr,
         *h_w_ = nullptr, *h_seq_ = nullptr, *h_flag_ = nullptr, *h_ymiss_ = nullptr,
         *h_flagA_ = nullptr, *h_plan_ = nullptr, *h_flagB_ = nullptr;
    ~Verifier();
};
void unregister_live_verifier(Verifier*) {}
#include "destructor.inc"

int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string mode = argv[1];
    auto* v = new Verifier;
    if (mode != "unused") {
        v->device_ = mode == "both-remote" ? 2 : 0;
        if (mode == "copy-only" || mode == "both" || mode == "both-remote" || mode == "failed-drain")
            v->copy_ = new Queue{v->device_};
        if (mode == "compute-only" || mode == "both" || mode == "both-remote" || mode == "failed-drain")
            v->cs_ = new Queue{v->device_};
    }
    Queue* cs = v->cs_;
    Queue* cp = v->copy_;
    char* arena = nullptr;
    if (mode == "both" || mode == "both-remote" || mode == "failed-drain") {
        arena = new char[128]; v->arena_ = arena;
        v->exec_[1] = new Graph;
    }
    Graph* graph = v->exec_[1];
    if (mode == "failed-drain") {
        fail_wait = true;
        // A noexcept production destructor must fail before reclamation if
        // its drain fails. Observe that boundary without changing its code.
        std::set_terminate([] {
            assert(destroys == 0 && frees == 0);
            std::puts("PASS failed-drain: termination before reclamation");
            std::fflush(stdout);
            std::_Exit(86);
        });
    }
    delete v;
    // A verifier never owns the default queue. The old partial-init path
    // destroys it even when the second queue is distinct.
    dpct::get_in_order_queue().wait();
    if (mode == "unused") assert(destroys == 0 && frees == 0 && waits == 1);
    else if (mode == "copy-only" || mode == "compute-only")
        assert(destroys == 1 && frees == 0 && waits == 2);
    else if (mode == "both" || mode == "both-remote") assert(destroys == 2 && frees == 1 && waits == 3);
    else if (mode == "failed-drain") {
        assert(destroys == 0 && frees == 0 && waits == 2);
        // Stand-ins have no automatic members; reclaim only after the failed
        // wait was observed, for ASan leak checking of the CPU test itself.
        delete cs; delete cp; delete[] arena; delete graph;
    } else throw std::runtime_error("unknown mode");
    assert(current_device == 0);
    delete default_queue;
    std::printf("PASS %s: waits=%d destroys=%d frees=%d\n", mode.c_str(), waits, destroys, frees);
}
