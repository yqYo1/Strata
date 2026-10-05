#include "strata/core/pinned.hpp"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <sys/mman.h>
#include <unistd.h>
#include <stdexcept>
#include <vector>

int main() try {
    const uint64_t page=sysconf(_SC_PAGESIZE);
    int checks=0;
    for (int mode=0;mode<3;++mode) {
        setenv("STRATA_SYCL_ARENA_THP",mode ? "1":"0",1);
        if(mode==2)setenv("STRATA_NO_LARGEPAGES","1",1);else unsetenv("STRATA_NO_LARGEPAGES");
        for(uint64_t bytes:std::vector<uint64_t>{1,page-1,page,page+1,(2ull<<20)-1,2ull<<20,(2ull<<20)+1,64ull*1024*1024+17}) {
            uintptr_t address=0;uint64_t length=0;
            {
                strata::core::PinnedArena arena(bytes);
                if(!arena.base||arena.capacity!=bytes)throw std::runtime_error("allocation failed");
                address=(uintptr_t)arena.base;length=((bytes+page-1)/page)*page;
                if(mode==1 && address%(2ull<<20))throw std::runtime_error("THP arena is not aligned");
                if((arena.note.find("transparent huge pages requested")!=std::string::npos)!=(mode==1))
                    throw std::runtime_error("THP advice or override mismatch");
                auto* p=(volatile unsigned char*)arena.base;p[0]=0x57;p[bytes-1]=0x92;
                std::ifstream maps("/proc/self/maps");std::string line;bool found=false;
                while(std::getline(maps,line)) {
                    unsigned long long first,last;
                    if(std::sscanf(line.c_str(),"%llx-%llx",&first,&last)==2 && first<=address && address<last) {
                        if(mode==1 && (first!=address||last!=address+length))
                            throw std::runtime_error("anonymous prefix/tail was not trimmed");
                        found=true;break;
                    }
                }
                if(!found)throw std::runtime_error("arena is absent from maps");
            }
            unsigned char resident=0;
            errno=0;
            if(mincore((void*)address,page,&resident)!=-1||errno!=ENOMEM)
                throw std::runtime_error("arena start was not unmapped");
            errno=0;
            if(mincore((void*)(address+length-page),page,&resident)!=-1||errno!=ENOMEM)
                throw std::runtime_error("arena end was not unmapped");
            ++checks;
        }
    }
    std::printf("PASS %d allocator mapping/release checks, including non-page sizes and NO_LARGEPAGES override\n",checks);
    return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
