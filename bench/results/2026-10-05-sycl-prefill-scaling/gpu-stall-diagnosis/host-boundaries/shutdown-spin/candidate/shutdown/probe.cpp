
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#include "strata/sycl_error.hpp"
struct Device {
    std::vector<std::function<void()>> jobs;
    bool fail = false;
    int waits = 0;
    void queues_wait_and_throw() {
        ++waits;
        if (fail) throw std::runtime_error("injected queue error");
        for (auto& job : jobs) job();
        jobs.clear();
    }
};
namespace dpct {
Device devices[2];
unsigned int device_count() { return 2; }
Device& get_device(unsigned int n) { return devices[n]; }
Device& get_in_order_queue() { return devices[0]; }
}
namespace strata::core {
void session_graphs_free(int) {}
void doorbell_free(int) {}
}
namespace sycl {
void free(float* p, Device&) { std::puts("FREE"); delete[] p; }
}
#include "drain.inc"
float *d_next, *d_logits, *d_emb, *d_parts, *sbuf, *arena;
int cli_footer() {
    int gr=0, db=0;
#include "footer.inc"
}
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    assert(argc==2);
    std::string mode=argv[1];
    d_next=new float[1]{1}; d_logits=new float[1]{2};
    d_emb=new float[1]{3}; d_parts=new float[1]{4};
    sbuf=new float[1]{5}; arena=new float[1]{6};
    unsigned int device=mode=="primary" ? 0 : 1;
    dpct::devices[device].jobs.push_back([p=arena] { assert(p[0]==6); });
    if (mode=="failed-drain") dpct::devices[1].fail=true;
    assert(cli_footer()==0);
    // Old production code leaves this read queued until after arena reclamation.
    for (auto& d : dpct::devices) for (auto& job : d.jobs) job();
    assert(dpct::devices[0].waits==1 && dpct::devices[1].waits==1);
    std::puts("PASS completion before reclamation");
}
