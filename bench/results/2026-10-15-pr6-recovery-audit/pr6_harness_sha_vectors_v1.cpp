#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
void require(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
std::string sha256(const void* data, size_t size) {
    static constexpr uint32_t k[] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    require(size <= (1u << 24), "digest input exceeds bounded evidence budget");
    const auto* bytes = static_cast<const uint8_t*>(data);
    std::vector<uint8_t> padded(bytes, bytes + size);
    padded.push_back(0x80);
    while (padded.size() % 64 != 56) padded.push_back(0);
    for (int shift = 56; shift >= 0; shift -= 8) padded.push_back(uint8_t((uint64_t(size) * 8) >> shift));
    std::array<uint32_t, 8> h{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    for (size_t off = 0; off < padded.size(); off += 64) {
        uint32_t w[64]{};
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 4; ++j) w[i] = (w[i] << 8) | padded[off + 4 * i + j];
        for (int i = 16; i < 64; ++i) {
            const auto s0 = std::rotr(w[i-15],7) ^ std::rotr(w[i-15],18) ^ (w[i-15] >> 3);
            const auto s1 = std::rotr(w[i-2],17) ^ std::rotr(w[i-2],19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        auto v = h;
        for (int i = 0; i < 64; ++i) {
            const auto t1 = v[7] + (std::rotr(v[4],6) ^ std::rotr(v[4],11) ^ std::rotr(v[4],25)) +
                            ((v[4] & v[5]) ^ (~v[4] & v[6])) + k[i] + w[i];
            const auto t2 = (std::rotr(v[0],2) ^ std::rotr(v[0],13) ^ std::rotr(v[0],22)) +
                            ((v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]));
            for (int j = 7; j > 0; --j) v[j] = v[j-1];
            v[4] += t1; v[0] = t1 + t2;
        }
        for (int i = 0; i < 8; ++i) h[i] += v[i];
    }
    std::ostringstream out; out << std::hex << std::setfill('0');
    for (auto word : h) out << std::setw(8) << word;
    return out.str();
}
int main(){
 const size_t sizes[]={0,1,2,3,7,31,55,56,57,63,64,65,119,120,121,127,128,129,255,256,257,1023,1024,4095,4096,4097,40960,131072,993280,1048576};
 for(size_t n:sizes){std::vector<uint8_t> b(n?n:1);for(size_t i=0;i<n;++i)b[i]=uint8_t((i*131u+(i>>9))&255u);std::printf("%zu %s\n",n,sha256(b.data(),n).c_str());}
 std::printf("abc %s\n",sha256("abc",3).c_str());
 std::string million(1000000,'a');std::printf("million_a %s\n",sha256(million.data(),million.size()).c_str());
 bool reject=false;try{(void)sha256(nullptr,(1u<<24)+1);}catch(const std::runtime_error&){reject=true;}
 std::printf("oversize_reject %d\n",int(reject));return reject?0:1;
}
