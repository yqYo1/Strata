// SOURCE ONLY: this is the actual production helper, not a copied implementation.
#include "files/include/strata/core/checked_pcie_fetch.hpp"
#include <array>
#include <cassert>
#include <cstring>
#include <string>
#include <vector>
using State=strata::core::detail::CheckedPcieFetch;
struct Fake {
    State state;
    std::array<uint8_t,16*16> staging{};
    std::array<std::array<uint8_t,16>,4> source{};
    std::array<const uint8_t*,4> sources{};
    int fail_copy=-1, fail_notify=0, fail_drain=0;
    int copies=0, notifications=0, drains=0, empty=0, consumers=0, pending=0;
    bool notify_pending=false;
    const char* detail="initial";
    std::vector<char> order;
    Fake(){for(size_t i=0;i<4;++i)sources[i]=source[i].data();}
    void submit(int n=4,size_t bytes=16,int64_t cap=4,void* stage=nullptr){
        state.submit(staging.data(),staging.size(),stage?stage:staging.data(),cap,16,sources.data(),n,bytes,22,
            [this](uint8_t*,const uint8_t*,size_t){const int index=copies++;order.push_back('C');if(index==fail_copy){detail="original enqueue detail";return 101;}++pending;return 0;},
            [this]{++notifications;order.push_back('N');if(fail_notify){detail="original callback detail";return 202;}notify_pending=true;return 0;},
            [this]{++empty;order.push_back('E');},
            [this](int){return detail;});
    }
    bool finish(){return state.finish([this]{++drains;order.push_back('D');if(fail_drain){detail="independent drain detail";return 303;}pending=0;notify_pending=false;return 0;},[this](int){return detail;});}
    void consume(){assert(state.can_consume());++consumers;order.push_back('X');}
};
int main(){
    for(int index:{0,1,3}){Fake f;f.fail_copy=index;f.submit();assert(f.copies==index+1 && f.notifications==0 && f.state.submitted==size_t(index));assert(!f.state.can_consume());assert(!f.state.reset());f.detail="TLS overwritten";f.submit();assert(f.copies==index+1 && f.notifications==0);assert(!f.finish());assert(f.drains==1 && f.pending==0 && f.state.drained && f.consumers==0);assert(f.state.first.code==101 && f.state.first.index==index && std::string(f.state.first.message.data())=="original enqueue detail");assert(f.state.reset());f.fail_copy=-1;f.copies=0;f.submit();assert(f.finish());f.consume();assert(f.consumers==1);}
    {Fake f;f.fail_notify=1;f.submit();assert(f.copies==4 && f.notifications==1 && f.state.first.code==202);assert(!f.finish() && f.drains==1 && f.pending==0 && !f.state.can_consume());}
    {Fake f;f.fail_copy=1;f.fail_drain=1;f.submit();assert(!f.finish());assert(f.state.first.code==101 && f.state.drain_error.code==303 && !f.state.drained);assert(!f.state.reset());assert(std::string(f.state.first.message.data())=="original enqueue detail");f.fail_drain=0;assert(!f.finish() && f.state.drained);assert(f.state.first.code==101);assert(f.state.reset());}
    {Fake f;f.fail_drain=1;f.submit();assert(!f.finish());assert(f.state.first.code==303 && !f.state.drained && !f.state.can_consume());}
    {Fake f;f.submit();assert(!f.state.can_consume());assert(f.finish());f.consume();assert(f.order==std::vector<char>({'C','C','C','C','N','D','X'}));assert(f.state.reset() && f.state.first.code==0 && f.state.drain_error.code==0 && !f.state.called);}
    for(int n:{0,-1}){Fake f;f.submit(n,0);assert(f.empty==1 && f.copies==0 && f.notifications==0);assert(f.finish());f.consume();assert(f.state.reset());}
    for(int which=0;which<5;++which){Fake f;if(which==0)f.sources[2]=nullptr;if(which==0)f.submit();if(which==1)f.submit(5);if(which==2)f.submit(4,17);if(which==3)f.submit(4,16,4,f.staging.data()+15*16);if(which==4)f.submit(4,0);assert(f.copies==0 && f.notifications==0 && f.state.first.code==22);assert(!f.finish());assert(f.consumers==0);}
    {Fake f;f.submit();f.submit();assert(f.state.first.code==22 && f.copies==4 && f.notifications==1);assert(!f.finish());}
    // A pool with no fetch still takes the existing normal checked drain.
    {Fake f;assert(f.finish());f.consume();assert(f.copies==0 && f.notifications==0 && f.drains==1);}
}
