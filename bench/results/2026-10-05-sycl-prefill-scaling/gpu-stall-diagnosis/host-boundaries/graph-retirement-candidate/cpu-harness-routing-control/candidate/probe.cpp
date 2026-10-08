
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using Clock=std::chrono::steady_clock;
double ms_since(Clock::time_point t) {
    return std::chrono::duration<double,std::milli>(Clock::now()-t).count();
}
bool idle=false, fail_wait=false, fail_verify=false;
bool fail_head_unmap=false, fail_map=false, fail_commit=false;
int waits=0, shrinks=0, grows=0;
bool expert_mapped=true, head_mapped=true;
struct Graph {
    static inline int live=0;
    Graph() { assert(expert_mapped && head_mapped); ++live; }
    ~Graph() { assert(idle && expert_mapped && head_mapped); --live; }
};
struct Queue {
    void wait_and_throw() {
        ++waits;
        if(fail_wait) throw std::runtime_error("injected wait failure");
        idle=true;
    }
    Queue& memcpy(void* d,const void* s,size_t n) {
        std::memcpy(d,s,n); return *this;
    }
};
struct Device : Queue {
    void queues_wait_and_throw() { wait_and_throw(); }
};
namespace dpct {
Device device;
Device& get_current_device() { return device; }
Queue& get_in_order_queue() { return device; }
}
struct OnDevice { explicit OnDevice(int) {} };
struct ExpertCache {
    bool* mapped;
    uint8_t bytes[16]{};
    explicit ExpertCache(bool& m) : mapped(&m) {}
    int64_t full_bytes() const { return 16; }
    int64_t mapped_bytes() const { return *mapped ? 16 : 0; }
    uint8_t* device_slot(int) { return bytes; }
    bool shrink(int,std::string& e) {
        ++shrinks; assert(idle);
#if !CONTROL
        assert(Graph::live==0);
#endif
        if(mapped==&head_mapped && fail_head_unmap) {
            fail_head_unmap=false; e="injected partial unmap"; return false;
        }
        *mapped=false; return true;
    }
    bool grow(int64_t,std::string& e) {
        ++grows;
        if(fail_map) { e="injected map failure"; return false; }
        *mapped=true; return true;
    }
};
class MtpDrafter {
public:
    int device_=0;
    bool release_decode_weights_=true, decode_weights_suspended_=false;
    bool verify_decode_weights_=true;
    ExpertCache expert_storage_{expert_mapped}, head_storage_{head_mapped};
    std::vector<uint8_t> expert_host_=std::vector<uint8_t>(16,17);
    std::vector<uint8_t> head_host_=std::vector<uint8_t>(16,42);
    uint8_t* experts_=expert_storage_.bytes;
    uint8_t* dhead_=head_storage_.bytes;
    Graph* prefill_exec_[9]{}, *prefill_dev_exec_[9]{}, *round_exec_[9]{};
    Graph* step_exec_[9]{}, *round_exec_c_[9]{}, *step_exec_c_[9]{};
    void populate() {
        for(auto* array : {prefill_exec_,prefill_dev_exec_,round_exec_,
                           step_exec_,round_exec_c_,step_exec_c_})
            for(int i=0;i<9;++i) { assert(!array[i]); array[i]=new Graph; }
        idle=false;
    }
    void cleanup() {
        idle=true; expert_mapped=head_mapped=true;
        for(auto* array : {prefill_exec_,prefill_dev_exec_,round_exec_,
                           step_exec_,round_exec_c_,step_exec_c_})
            for(int i=0;i<9;++i) { delete array[i]; array[i]=nullptr; }
    }
    bool verify_decode_payload(std::string& e) const {
        if(fail_verify) { e="injected byte mismatch"; return false; }
        return true;
    }
    void discard_graphs_after_idle();
    bool suspend_decode_weights(std::string&);
    bool restore_decode_weights(std::string&);
};
struct Boundary {
    std::shared_ptr<Graph> input,tail;
    std::vector<std::shared_ptr<Graph>> pre,post;
};
class Verifier {
public:
    int device_=0;
    Verifier* next_=nullptr;
    Queue* cs_=&dpct::device, *copy_=&dpct::device;
    Graph* exec_[9]{};
    Boundary boundary_graphs_[9];
    struct { const uint8_t* cache_base=nullptr; } hits_;
    bool wait_commit(std::string& e) {
        if(fail_commit) { e="injected commit failure"; return false; }
        return true;
    }
    bool warm(std::string&) {
        assert(hits_.cache_base && Graph::live==0);
        for(auto& g:exec_) g=new Graph;
        return true;
    }
    void populate() {
        for(auto& g:exec_) g=new Graph;
        for(auto& b:boundary_graphs_) {
            b.input=std::make_shared<Graph>(); b.tail=std::make_shared<Graph>();
            b.pre.push_back(std::make_shared<Graph>());
            b.post.push_back(std::make_shared<Graph>());
        }
        idle=false;
    }
    bool discard_cache_graphs(std::string&);
    bool rebuild_cache_graphs(const uint8_t*,std::string&);
};
#include "functions.inc"
int main(int argc,char** argv) {
    assert(argc==2); std::string mode=argv[1],error;
    if(mode.starts_with("verify-")) {
        Verifier v; v.populate(); assert(Graph::live==45);
        if(mode=="verify-commit-error") fail_commit=true;
        if(mode=="verify-wait-error") fail_wait=true;
        if(mode=="verify-multigpu") v.next_=&v;
        if(mode=="verify-address-error") {
            assert(!v.rebuild_cache_graphs(nullptr,error) && Graph::live==45 && waits==0);
        } else if(mode=="verify-cycle") {
            assert(v.discard_cache_graphs(error) && Graph::live==0);
            expert_mapped=false; expert_mapped=true;
            uint8_t address=0;
            assert(v.rebuild_cache_graphs(&address,error) && v.hits_.cache_base==&address);
            assert(Graph::live==9);
        } else {
            assert(!v.discard_cache_graphs(error) && Graph::live==45);
        }
        fail_wait=fail_commit=false; v.next_=nullptr;
        assert(v.discard_cache_graphs(error));
    } else {
        MtpDrafter m; m.populate(); assert(Graph::live==54);
        if(mode=="disabled") m.release_decode_weights_=false;
        if(mode=="wait-error") fail_wait=true;
        if(mode=="verify-error") fail_verify=true;
        if(mode=="missing-ram") m.head_host_.pop_back();
        if(mode=="partial-unmap") fail_head_unmap=true;
        if(mode=="map-error") fail_map=true;
        if(mode=="cycle" || mode=="retained-control") {
            for(int i=0;i<3;++i) {
                assert(m.suspend_decode_weights(error));
#if CONTROL
                assert(Graph::live==54);
#else
                assert(Graph::live==0);
#endif
                assert(!expert_mapped && !head_mapped && m.decode_weights_suspended_);
                int old_waits=waits,old_shrinks=shrinks;
                assert(m.suspend_decode_weights(error) && waits==old_waits && shrinks==old_shrinks);
                assert(m.restore_decode_weights(error) && !m.decode_weights_suspended_);
                assert(std::memcmp(m.experts_,m.expert_host_.data(),16)==0);
                assert(std::memcmp(m.dhead_,m.head_host_.data(),16)==0);
#if !CONTROL
                m.populate();
#endif
            }
        } else if(mode=="partial-unmap") {
            assert(!m.suspend_decode_weights(error) && m.decode_weights_suspended_);
            assert(!expert_mapped && head_mapped && Graph::live==0);
            assert(m.restore_decode_weights(error) && !m.decode_weights_suspended_);
        } else if(mode=="map-error") {
            assert(m.suspend_decode_weights(error) && Graph::live==0);
            assert(!m.restore_decode_weights(error) && m.decode_weights_suspended_);
            fail_map=false;
            assert(m.restore_decode_weights(error));
        } else if(mode=="disabled") {
            assert(m.suspend_decode_weights(error) && waits==0 && shrinks==0 && Graph::live==54);
        } else {
            assert(!m.suspend_decode_weights(error) && shrinks==0 && Graph::live==54);
            assert(!m.decode_weights_suspended_);
        }
        m.cleanup();
    }
    assert(Graph::live==0);
    std::printf("PASS %s: waits=%d unmaps=%d maps=%d\n",mode.c_str(),waits,shrinks,grows);
}
