
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#define __dpct_inline__ inline
namespace sycl {
struct int2 { uint32_t x,y; int2(uint32_t a,uint32_t b):x(a),y(b) {} };
}
namespace dpct {
#include "permute.inc"
}
#include "lookup.inc"
int main(int argc,char** argv) {
    const bool misaligned=argc==2 && std::string(argv[1])=="misaligned";
    alignas(16) int8_t storage[32]{};
    const int8_t table[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    const unsigned int first=misaligned ? 1 : 0;
    const unsigned int offsets=CONTROL ? 1 : 4;
    uint64_t compared=0;
    for (unsigned int offset=first;offset<first+offsets;++offset) {
        std::memcpy(storage+offset,table,sizeof table);
        for (uint32_t word=0;word<65536;++word) {
            const uint32_t codes=word | ((word ^ 0xffffu)<<16);
            const auto actual=iq4_table_lookup((int)codes,storage+offset);
            uint32_t even=0,odd=0;
            for (unsigned int lane=0;lane<8;++lane) {
                uint32_t value=static_cast<uint8_t>(table[(codes>>(lane*4))&15]);
                ((lane&1) ? odd : even) |= value << ((lane/2)*8);
                ++compared;
            }
            assert(actual.x==even && actual.y==odd);
        }
    }
    std::printf("PASS %llu exact codebook byte comparisons\n",(unsigned long long)compared);
}
