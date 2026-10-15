// SOURCE ONLY, Main compiles/runs. Actual production host contract helper.
#include "strata/core/indexed_head_contract.hpp"
#include <array>
#include <cassert>
#include <functional>
#include <vector>
using namespace strata::core::detail;
static void rejects(const std::function<void()>& f){bool bad=false;try{f();}catch(const std::invalid_argument&){bad=true;}assert(bad);}
int main(){
    IndexedHeadShape s(2560,200000,106299,8);
    assert(s.row==2100 && s.weights==420000000 && s.ids==425196 && s.activation==8*80*36 && s.output==8ull*106299*4);
    assert(indexed_head_eligible(true,14,2560,200000,2100));
    assert(!indexed_head_eligible(false,14,2560,200000,2100));
    assert(!indexed_head_eligible(true,13,2560,200000,2100));
    assert(!indexed_head_eligible(true,14,2561,200000,2100));
    assert(!indexed_head_eligible(true,14,2560,200000,2101));
    const std::vector<int32_t> ids{199999,0,17,17,1};
    auto saved=ids;validate_indexed_head_ids(ids.data(),ids.size(),200000);assert(ids==saved);
    for(int cols=1;cols<=8;++cols)for(int subset:{1,17,33}){IndexedHeadShape a(2560,37,subset,cols);assert(a.ids==size_t(subset)*4 && a.output==size_t(cols*subset)*4);}
    for(int n:{0,-1,255,257})rejects([&]{IndexedHeadShape a(n,37,17,1);(void)a;});
    for(int n:{0,-1})rejects([&]{IndexedHeadShape a(2560,n,17,1);(void)a;});
    for(int n:{0,-1})rejects([&]{IndexedHeadShape a(2560,37,n,1);(void)a;});
    for(int n:{0,9})rejects([&]{IndexedHeadShape a(2560,37,17,n);(void)a;});
    rejects([]{indexed_head_product(UINT64_MAX,210);});
    rejects([]{validate_indexed_head_ids(nullptr,1,37);});
    rejects([&]{validate_indexed_head_ids(ids.data(),0,200000);});
    for(int32_t id:{-1,200000})rejects([&]{validate_indexed_head_ids(&id,1,200000);});
    // Duplicates can exceed full row count; subset order is never sorted/deduplicated.
    std::array<int32_t,17> duplicates{};duplicates.fill(0);validate_indexed_head_ids(duplicates.data(),17,1);
}
