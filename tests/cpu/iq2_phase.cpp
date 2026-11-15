// Main-authored synchronous pool phase contract; repository MIT license.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/iq2_pair_avx2.hpp"
#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
uint64_t paired(const cpu::Iq2GuAudit& a) {
    require(a.enabled && !a.overflow, "phase audit unavailable");
    uint64_t n = 0; for (unsigned i = 0; i < a.size; ++i) n += a.threads[i].pair_rows;
    return n;
}
int main(int argc, char** argv) {
    try {
        require(argc == 2 && (std::string(argv[1]) == "--pair=0" || std::string(argv[1]) == "--pair=1"), "pair argument");
        const bool enabled = std::string(argv[1]) == "--pair=1";
        const char* env = std::getenv("STRATA_IQ2_PAIR");
        require(env && std::string(env) == (enabled ? "1" : "0"), "immutable startup selection");
        auto phase = cpu::NativeGuPhase::Prefill;
        try {
            cpu::NativeGuPhaseScope decode(phase, cpu::NativeGuPhase::Decode);
            { cpu::NativeGuPhaseScope prompt(phase, cpu::NativeGuPhase::Prefill);
              require(phase == cpu::NativeGuPhase::Prefill, "nested prefill selection"); }
            require(phase == cpu::NativeGuPhase::Decode, "nested scope restore");
            throw 1;
        } catch (int) {}
        require(phase == cpu::NativeGuPhase::Prefill, "error scope leaked decode selection");
        constexpr int H = 2560, FF = 640, guard = 16;
        constexpr float sentinel = 9876.5f;
        cpu::ExpertPool pool(2, false, true);
        unsigned cases = 0;
        for (int down : {20,42}) {
            cpu::NativeFmt f; std::string error;
            require(cpu::native_fmt(22,down,H,FF,f,error), error.c_str());
            std::vector<uint8_t> weights(f.bytes+128, 0xa7);
            auto* blob = weights.data()+64;
            uint32_t seed = 20261022;
            for (size_t i = 0; i < f.bytes; ++i) { seed = seed*1664525u+1013904223u; blob[i] = uint8_t(seed>>24); }
            const auto scale = ggml_fp32_to_fp16(.01f);
            for (size_t offset = 0; offset < f.down_off; offset += sizeof(block_iq2_s))
                reinterpret_cast<block_iq2_s*>(blob+offset)->d = scale;
            const size_t block = down == 20 ? sizeof(block_iq4_nl) : sizeof(block_q2_0);
            for (size_t offset = f.down_off; offset < f.bytes; offset += block)
                std::memcpy(blob+offset, &scale, sizeof(scale));
            const auto original_weights = weights;
            std::array<std::vector<uint8_t>,2> activations;
            for (int t = 0; t < 2; ++t) {
                std::array<float,H> x{};
                for (int i = 0; i < H; ++i) x[i] = float((i*17+t*31)%257-128)/128.f;
                activations[t].resize(f.act_bytes);
                cpu::native_quant_act(f,x.data(),activations[t].data());
            }
            const auto original_activations = activations;
            for (int nt : {1,2}) {
                std::array<float,2*H+2*guard> output{}, reference{};
                for (auto selected : {cpu::NativeGuPhase::Prefill,cpu::NativeGuPhase::Decode,cpu::NativeGuPhase::Prefill}) {
                    output.fill(sentinel);
                    cpu::ExpertJobMulti job{}; job.blob = blob; job.nt = nt;
                    for (int t = 0; t < nt; ++t) { job.nact[t] = activations[t].data(); job.out[t] = output.data()+guard+t*H; }
                    const uint64_t before = paired(cpu::iq2_gu_audit_snapshot());
                    pool.run_split_multi_native(f,&job,1,selected);
                    const uint64_t delta = paired(cpu::iq2_gu_audit_snapshot())-before;
                    require(delta == (enabled && nt == 1 && selected == cpu::NativeGuPhase::Decode ? FF : 0),
                        "pair escaped its decode/NT1 scope");
                    if (cases%3 == 0) reference = output;
                    else require(std::memcmp(reference.data(),output.data(),sizeof(output)) == 0, "phase output bits differ");
                    for (int i = 0; i < guard; ++i) require(output[i] == sentinel, "front guard");
                    for (size_t i = guard+nt*H; i < output.size(); ++i) require(output[i] == sentinel, "tail or inactive row guard");
                    for (int i = 0; i < nt*H; ++i) require(std::isfinite(output[guard+i]), "nonfinite pool output");
                    require(weights == original_weights && activations == original_activations, "input mutation");
                    ++cases;
                }
            }
        }
        require(cases == 12, "incomplete phase matrix");
        std::printf("iq2_phase_scope cases=12 pair=%d prefill_pair_rows=0 decode_dispatch=PASS NT2_reference=PASS bits=PASS guards=PASS scope_restore=PASS\n", int(enabled));
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr,"iq2 phase: %s\n",error.what()); return 1; }
}
