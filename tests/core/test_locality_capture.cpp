// Main-authored DMA-lifetime and tail-boundary regression; repository LICENSE.
#include "strata/prefill/locality_capture.hpp"
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace {
void require(bool ok,const char* text) { if (!ok) throw std::runtime_error(text); }
void check_file(const std::string& path,int rows,int cap,int source_rows,int source_cap) {
    std::ifstream f(path,std::ios::binary);int32_t header[4]{};
    f.read(reinterpret_cast<char*>(header),sizeof(header));
    require(f.good() && header[0]==2 && header[1]==10000+source_rows-rows &&
        header[2]==rows && header[3]==cap,"tail header");
    std::vector<int32_t> values(size_t(rows)*cap);
    f.read(reinterpret_cast<char*>(values.data()),std::streamsize(values.size()*sizeof(int32_t)));
    require(f.good() && f.peek()==std::char_traits<char>::eof(),"exact tail file extent");
    for (size_t i=0;i<values.size();++i)
        require(values[i]==int32_t(((size_t(source_rows-rows)*source_cap+i)*17+19)%262144),"tail offset/payload");
}
}
int main(int argc,char** argv) {
    if (argc!=3) return 2;
    try {
        const std::string mode=argv[1],base=argv[2];
        const int rows=mode=="full" ? 8192 : 17,cap=mode=="full" ? 2051 : 33,step_count=7;
        constexpr int32_t guard=-1234567;
        std::vector<int32_t> ids(size_t(rows)*cap+2,guard),steps(size_t(rows)*step_count+2,guard);
        for (size_t i=1;i+1<ids.size();++i) ids[i]=int32_t(((i-1)*17+19)%262144);
        for (size_t i=1;i+1<steps.size();++i) steps[i]=int32_t(((i-1)*17+19)%262144);
        int copies=0,waits=0;void* pending=nullptr;const void* source=nullptr;size_t extent=0;
        // Emulate a partial asynchronous submission. The drain writes the
        // helper-owned destination;it must run before that host vector unwinds.
        auto copy=[&](void* dst,const void* src,size_t bytes) {
            require(!pending,"previous DMA not drained");++copies;
            pending=dst;source=src;extent=bytes;
            if (copies==2 && mode=="throw") throw std::runtime_error("partial second DMA");
            return !(copies==2 && mode=="second-fail");
        };
        auto drain=[&] {
            ++waits;if (mode=="drain-fail") return false;
            if (pending) { std::memcpy(pending,source,extent);pending=nullptr; }
            return true;
        };
        auto fatal=[] { std::_Exit(73); };
        std::string error;
        const bool ok=strata::prefill::detail::locality_capture(base.c_str(),2,10000,rows,
            mode=="geometry" ? rows-1 : rows,cap,step_count,ids.data()+1,steps.data()+1,copy,drain,fatal,error);
        require(ids.front()==guard && ids.back()==guard && steps.front()==guard && steps.back()==guard,"source guards");
        require(!pending,"host DMA escaped capture lifetime");
        if (mode=="small" || mode=="full") {
            require(ok && error.empty() && copies==2 && waits==2,"successful retirement");
            const int take=std::min(512,rows);
            check_file(base+".ids",take,cap,rows,cap);check_file(base+".steps",take,step_count,rows,step_count);
        } else if (mode=="geometry") require(!ok && copies==0 && waits==1,"refused geometry must drain inherited work");
        else if (mode=="second-fail" || mode=="throw" || mode=="file-fail")
            require(!ok && !error.empty() && copies==2 && waits==3,"partial second record must drain before return");
        else if (mode=="open-fail") require(!ok && copies==1 && waits==2,"open failure must drain inherited work");
        else return 1; // drain-fail must terminate before unwinding the DMA target.
        return 0;
    } catch (...) { return 1; }
}
