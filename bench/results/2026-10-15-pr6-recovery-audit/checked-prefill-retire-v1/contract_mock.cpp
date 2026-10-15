// Host-only model of required ordering; intentionally not a SYCL safety test.
// Main recipe: c++ -std=c++17 contract_mock.cpp -o /private/path/contract_mock
// then run serially under the same exclusive validation lock.
#include <cassert>
#include <stdexcept>
#include <vector>
struct Stop {};
struct Contract {
    bool ready=true, callback=false, tail=false, drained=false, stopped=false;
    int generation=1, queue_destroy=0, retries=0;
    std::vector<int> live{1,2,3};
    void retire(bool drain_ok=true, int failing_free=-1) {
        assert(!callback && !tail);
        if (!drain_ok) { stopped=true; throw Stop{}; }
        drained=true;
        for (auto& owner:live) {
            if(owner==failing_free) { stopped=true; throw Stop{}; }
            owner=0;
        }
        live.clear(); ready=false;
    }
    void batch() {
        assert(!stopped);
        if(!ready) { ready=true; ++generation; live={1,2,3}; }
    }
    void windows() { assert(!stopped); }
};
int main() {
    Contract c; c.batch(); c.batch(); assert(c.generation==1);
    c.retire(); c.windows(); assert(!c.ready); c.batch(); assert(c.generation==2);
    c.retire(); assert(!c.ready && c.live.empty());
    Contract drain_failure;
    try { drain_failure.retire(false); assert(false); } catch(Stop&) {}
    assert(drain_failure.live==std::vector<int>({1,2,3}));
    Contract partial;
    try { partial.retire(true,2); assert(false); } catch(Stop&) {}
    assert(partial.live==std::vector<int>({0,2,3}));
    assert(partial.stopped && !partial.queue_destroy && !partial.retries);
}
