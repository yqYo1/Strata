#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;

static void require(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "%s\n", why); std::exit(1); }
}

struct Dataset {
    cpu::NativeFmt fmt;
    std::vector<uint8_t> weights, activations;
    std::vector<float> output;
    std::vector<cpu::ExpertJobMulti> jobs;

    Dataset(strata::GgufFile& file, int layer, int count, int pattern, int seed) {
        const strata::TensorInfo* tensors[3] = {};
        const char* roles[] = {"gate", "up", "down"};
        for (int r = 0; r < 3; ++r) {
            const std::string name = "blk." + std::to_string(layer) + ".ffn_" + roles[r] + "_exps.weight";
            for (const auto& t : file.tensors()) if (t.name == name) tensors[r] = &t;
            require(tensors[r] != nullptr, "missing expert tensor");
        }
        require(tensors[0]->type == tensors[1]->type, "gate/up types differ");
        std::string err;
        require(cpu::native_fmt(tensors[0]->type, tensors[2]->type,
                               tensors[0]->shape[0], tensors[0]->shape[1], fmt, err), err.c_str());
        require(fmt.n_embd == cpu::H && fmt.n_ff == cpu::FF, "unexpected pool geometry");
        require(count >= 1 && count <= 97, "invalid expert count");
        weights.resize(count * fmt.bytes);
        const size_t sizes[] = {fmt.up_off, fmt.up_off, fmt.bytes - fmt.down_off};
        const size_t offsets[] = {0, fmt.up_off, fmt.down_off};
        for (int e = 0; e < count; ++e) for (int r = 0; r < 3; ++r) {
            const int expert = count == 3 ? (e == 0 ? 0 : e == 1 ? 7 : 31) : e;
            std::memcpy(weights.data() + e * fmt.bytes + offsets[r],
                        file.tensor_data(*tensors[r]) + expert * sizes[r], sizes[r]);
        }
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.f, .5f);
        std::vector<float> x(cpu::MAXT * cpu::H);
        for (float& v : x) v = normal(rng);
        activations.resize(cpu::MAXT * fmt.act_bytes);
        for (int t = 0; t < cpu::MAXT; ++t)
            cpu::native_quant_act(fmt, x.data() + t * cpu::H, activations.data() + t * fmt.act_bytes);
        output.resize(count * cpu::MAXT * (cpu::H + 2), .125f);
        jobs.resize(count);
        for (int e = 0; e < count; ++e) {
            auto& j = jobs[e];
            j.blob = weights.data() + e * fmt.bytes;
            j.nt = pattern > 0 ? pattern : 1 + e % (-pattern);
            require(j.nt >= 1 && j.nt <= cpu::MAXT, "invalid token count");
            for (int t = 0; t < cpu::MAXT; ++t) {
                j.nact[t] = activations.data() + t * fmt.act_bytes;
                j.out[t] = output.data() + (e * cpu::MAXT + t) * (cpu::H + 2) + 1;
            }
        }
    }

    void verify() const {
        for (int e = 0; e < (int)jobs.size(); ++e) for (int t = 0; t < cpu::MAXT; ++t) {
            const float* p = output.data() + (e * cpu::MAXT + t) * (cpu::H + 2);
            require(p[0] == .125f && p[cpu::H + 1] == .125f, "output guard overwritten");
            for (int r = 1; r <= cpu::H; ++r) {
                require(std::isfinite(p[r]), "non-finite expert output");
                if (t >= jobs[e].nt) require(p[r] == .125f, "inactive token overwritten");
            }
        }
    }
};

int main(int argc, char** argv) {
    require(argc >= 4, "usage: probe GGUF pack validate output | probe GGUF pack bench");
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string error;
    require(cpu::expert_layout_load(argv[2], 48, 512, error), error.c_str());
    strata::GgufFile file(argv[1]);
    // Construct the workers before restricting the host's affinity: topology discovery uses the allowed mask.
    const auto topology = cpu::detect_cpu_topology(true);
    cpu::ExpertPool pool(5, true, true);
    const auto previous = cpu::pin_current_thread(topology.host_core);
    require(previous.valid, "host affinity failed");
    std::printf("{\"kind\":\"ready\",\"workers\":%d,\"host_works\":true,\"host_core\":%d,\"worker_cores\":[",
                pool.workers(), topology.host_core);
    for (int i = 0; i < 5; ++i) std::printf("%s%d", i ? "," : "", topology.worker_cores.at(i));
    std::printf("]}\n");
    if (std::string(argv[3]) == "validate") {
        require(argc == 5, "validate needs output path");
        FILE* f = std::fopen(argv[4], "wb");
        require(f != nullptr, "cannot open output dump");
        uint64_t checked = 0;
        int cases = 0;
        for (int layer = 0; layer < 48; ++layer) {
            for (int nt = 1; nt <= cpu::MAXT; ++nt) {
                Dataset data(file, layer, 3, nt, 1071 + layer);
                pool.run_split_multi_native(data.fmt, data.jobs.data(), 3);
                data.verify();
                require(std::fwrite(data.output.data(), sizeof(float), data.output.size(), f) == data.output.size(), "dump failed");
                checked += data.output.size();
                ++cases;
            }
            std::printf("{\"kind\":\"layer_validated\",\"layer\":%d}\n", layer);
        }
        // Reuse one live pool across different counts, token groups and more than 96 distinct experts.
        const int counts[] = {1, 97, 13, 96, 2, 97, 12, 1, 96, 13, 97, 2};
        for (int round = 0; round < 12; ++round) {
            const int layer = round % 2 ? 1 : 2;
            Dataset data(file, layer, counts[round], round % 3 ? -8 : -4, 1091 + round);
            pool.run_split_multi_native(data.fmt, data.jobs.data(), counts[round]);
            data.verify();
            require(std::fwrite(data.output.data(), sizeof(float), data.output.size(), f) == data.output.size(), "stress dump failed");
            checked += data.output.size();
            ++cases;
        }
        require(std::fclose(f) == 0, "dump close failed");
        std::printf("{\"kind\":\"completed\",\"cases\":%d,\"finite_guard_checked_floats\":%llu}\n", cases, (unsigned long long)checked);
    } else {
        std::unique_ptr<Dataset> data;
        std::string command;
        while (std::cin >> command) {
            if (command == "QUIT") break;
            if (command == "CASE") {
                int layer, count, pattern;
                require(bool(std::cin >> layer >> count >> pattern), "bad CASE");
                data = std::make_unique<Dataset>(file, layer, count, pattern, 1103 + layer);
                pool.run_split_multi_native(data->fmt, data->jobs.data(), count);
                data->verify();
                std::printf("{\"kind\":\"case_ready\",\"layer\":%d,\"gu_type\":%d,\"down_type\":%d,\"experts\":%d,\"pattern\":%d,\"weight_bytes\":%zu}\n",
                            layer, data->fmt.gu_type, data->fmt.d_type, count, pattern, data->weights.size());
            } else if (command == "DUMP") {
                std::string path;
                require(data && bool(std::cin >> path), "bad DUMP");
                FILE* f = std::fopen(path.c_str(), "wb");
                require(f != nullptr, "cannot open case dump");
                require(std::fwrite(data->output.data(), sizeof(float), data->output.size(), f) == data->output.size(), "case dump failed");
                require(std::fclose(f) == 0, "case dump close failed");
                std::printf("{\"kind\":\"dumped\",\"bytes\":%zu}\n", data->output.size() * sizeof(float));
            } else if (command == "BENCH") {
                int repeats;
                require(data && bool(std::cin >> repeats) && repeats > 0, "bad BENCH");
                pool.run_split_multi_native(data->fmt, data->jobs.data(), data->jobs.size());
                pool.ms_multi_gu = pool.ms_multi_q = pool.ms_multi_down = 0;
                const auto start = Clock::now();
                for (int i = 0; i < repeats; ++i)
                    pool.run_split_multi_native(data->fmt, data->jobs.data(), data->jobs.size());
                const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                data->verify();
                std::printf("{\"kind\":\"timing\",\"repeats\":%d,\"wall_ms\":%.9f,\"gu_ms\":%.9f,\"quant_ms\":%.9f,\"down_ms\":%.9f}\n",
                            repeats, ms, pool.ms_multi_gu, pool.ms_multi_q, pool.ms_multi_down);
            } else require(false, "unknown command");
        }
    }
    cpu::restore_thread_affinity(previous);
}
