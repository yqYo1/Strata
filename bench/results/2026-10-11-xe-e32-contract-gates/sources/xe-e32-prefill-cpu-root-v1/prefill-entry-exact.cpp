#include "strata/prefill/publication.hpp"
#include <cassert>
#include <limits>
#include <iostream>
namespace strata::prefill {
namespace core { struct SessionState { int64_t max_cells = 262144; }; }
struct Impl { core::SessionState* ss; };
struct Prefill { Impl* impl_; bool run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err); };
bool Prefill::run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) try {
    err.clear();
    Impl& m = *impl_;
    core::SessionState& ss = *m.ss;
    if (!publication::valid_span(n, pos0, ss.max_cells)) {
        err = "prefill: input span exceeds the physical context capacity";
        return false;
    }
    if (n == 0) return true;
    if (tokens == nullptr) { err = "prefill: missing input tokens"; return false; }
    return true; // stand-in for the GPU body after the actual guarded entry
} catch (const std::exception& e) { err = e.what(); return false; }
}
int main() {
    using namespace strata::prefill;
    core::SessionState ss; Impl impl{&ss}; Prefill pf{&impl};
    int64_t tok=1; std::string err;
    assert(pf.run(&tok,262144,0,err) && err.empty());
    assert(pf.run(&tok,1,262143,err) && err.empty());
    assert(pf.run(nullptr,0,262144,err) && err.empty());
    for (const auto& p : {std::pair<int64_t,int64_t>{1,262144},{2,262143},{0,262145},{-1,0},{1,-1},{2,std::numeric_limits<int64_t>::max()-1}}) {
        assert(!pf.run(&tok,p.first,p.second,err));
        assert(err.find("physical context capacity")!=std::string::npos);
    }
    assert(!pf.run(nullptr,1,0,err) && err=="prefill: missing input tokens");
    std::cout << "actual prefill entry: final cell, full span, empty end, invalid/overflow/null inputs passed before GPU body\n";
}
