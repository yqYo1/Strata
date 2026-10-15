// SOURCE ONLY, Main compiles/runs. Actual production host contract helper.
#include "strata/core/indexed_head_contract.hpp"
#include <array>
#include <stdexcept>
#include <cstring>
#include <cstdio>
#include <functional>
#include <vector>
using namespace strata::core::detail;
static void need(bool ok){if(!ok)throw std::runtime_error("host contract failed");}
static void rejects(const std::function<void()>& f){bool bad=false;try{f();}catch(const std::invalid_argument&){bad=true;}need(bad);}
static IndexedHeadMap from_bytes(const std::vector<uint8_t>& raw,int64_t full){
    FILE* f=std::tmpfile();need(f!=nullptr);
    struct Close{FILE* f;~Close(){std::fclose(f);}} close{f};
    need(std::fwrite(raw.data(),1,raw.size(),f)==raw.size());need(std::fseek(f,0,SEEK_SET)==0);
    return read_mtp_head_map(f,full);
}
static std::vector<uint8_t> bytes(std::initializer_list<int32_t> ids){std::vector<uint8_t> b(ids.size()*4);std::memcpy(b.data(),ids.begin(),b.size());return b;}
int main(){
    // A failed write on a read-only source stream sets ferror; the shared reader
    // must reject unreadable/error-bearing input rather than publish a descriptor.
    FILE* failed=std::fopen(__FILE__,"rb");need(failed!=nullptr);
    need(std::fputc('x',failed)==EOF);need(std::ferror(failed)!=0);
    rejects([&]{(void)read_mtp_head_map(failed,248320);});std::fclose(failed);
    auto absent=read_mtp_head_map(static_cast<FILE*>(nullptr),37);
    need(!mtp_indexed_map_ready(true,absent)); // absent subset: full-head fallback
    auto valid=from_bytes(bytes({36,0,17,17,1}),37);
    need(mtp_indexed_map_ready(true,valid));
    need(!mtp_indexed_map_ready(indexed_head_eligible(true,13,2560,37,2100),valid)); // noneligible legacy
    need(valid.ids==std::vector<int32_t>({36,0,17,17,1}));
    for(auto raw:std::vector<std::vector<uint8_t>>{{},{1},{1,2,3},bytes({-1}),bytes({37})})
        rejects([&]{(void)from_bytes(raw,37);});
    rejects([&]{(void)from_bytes(bytes({0,0,0}),2);}); // bounded map, before publication
    rejects([]{validate_mtp_head_capacity(38,37);});
    validate_mtp_head_capacity(37,37);
    IndexedHeadShape boundary(256,37,51,1);need(boundary.row==210); // generic duplicate count allowed

    IndexedHeadShape s(2560,200000,106299,8);
    need(s.row==2100 && s.weights==420000000 && s.ids==425196 && s.activation==8*80*36 && s.output==8ull*106299*4);
    need(indexed_head_eligible(true,14,2560,200000,2100));
    need(!indexed_head_eligible(false,14,2560,200000,2100));
    need(!indexed_head_eligible(true,13,2560,200000,2100));
    need(!indexed_head_eligible(true,14,2561,200000,2100));
    need(!indexed_head_eligible(true,14,2560,200000,2101));
    const std::vector<int32_t> ids{199999,0,17,17,1};
    auto saved=ids;validate_indexed_head_ids(ids.data(),ids.size(),200000);need(ids==saved);
    for(int cols=1;cols<=8;++cols)for(int subset:{1,17,33}){IndexedHeadShape a(2560,37,subset,cols);need(a.ids==size_t(subset)*4 && a.output==size_t(cols*subset)*4);}
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
