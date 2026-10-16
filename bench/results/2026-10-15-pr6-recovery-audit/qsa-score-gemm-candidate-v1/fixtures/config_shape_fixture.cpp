// Main-only CPU check of the actual header's parser and bounded byte arithmetic.
// The independent cell oracle is tail_causal_fixture.py, not this code.
#include "strata/prefill/sycl_qsa_score_gemm.hpp"
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <limits>
#include <initializer_list>
using namespace strata::prefill;
void require(bool ok) { if (!ok) throw std::runtime_error("config/shape fixture failed"); }
template<class F> void rejects(F f) {
    bool caught=false;try {f();}catch(const std::invalid_argument&) {caught=true;}
    require(caught);
}
int main() try {
    require(parse_qsa_score_mode(nullptr)==QsaScoreMode::off);
    require(parse_qsa_score_mode("fp16")==QsaScoreMode::fp16);
    require(parse_qsa_score_mode("bf16")==QsaScoreMode::bf16);
    for (const char* text:{"","off","OFF","0","1","FP16","BF16","fp16 "," fp16","fp16extra","bf16\n"})
        rejects([&] {parse_qsa_score_mode(text);});
    for (int64_t rows=1;rows<=256;++rows) {
        const auto bytes=qsa_score_scratch_bytes(65538,rows);
        const auto independent=uint64_t(65539)*128*2+uint64_t(rows)*512*2+uint64_t(rows)*65539*4;
        require(bytes==independent && bytes<=84152064);
    }
    for (int64_t blocks:std::initializer_list<int64_t>{-1,0,1,65539,std::numeric_limits<int64_t>::max()})
        rejects([&] {qsa_score_scratch_bytes(blocks,256);});
    for (int64_t rows:std::initializer_list<int64_t>{-1,0,257,std::numeric_limits<int64_t>::max()})
        rejects([&] {qsa_score_scratch_bytes(65538,rows);});
    require(qsa_score_scratch_bytes(65538,256)==84152064);
    std::cout<<"PASS actual header exact-parse/shape/overflow arithmetic; no SYCL runtime\n";
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
