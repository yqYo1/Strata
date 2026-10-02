#include "strata/core/expert_cache.hpp"
#include "strata/core/graph.hpp"
#include "strata/core/pinned.hpp"
#include "strata/core/session.hpp"
#include "strata/kernels/elementwise.hpp"
#include <bit>
#include <cfenv>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
void require(bool ok, const std::string &why) {
  if (!ok)
    throw std::runtime_error(why);
}
void shared_session(bool fused = false, bool join_next = false) {
  using namespace strata::core;
  ModelGeometry g;
  g.n_embd = 19;
  g.n_layers = 3;
  Doorbell db;
  require(doorbell_init(g, 2, db) != 0, "session doorbell");
  SessionState state;
  state.db = &db;
  state.k = 2;
  state.qsa_alloc = 0;
  float *value{}, *shared{}, *parts{};
  int *count{};
  require(cudaMalloc(&value, 19 * 4) == cudaSuccess &&
              cudaMalloc(&shared, 19 * 4) == cudaSuccess &&
              cudaMalloc(&parts, 38 * 4) == cudaSuccess &&
              cudaMalloc(&count, 3 * 4) == cudaSuccess, "session buffers");
  cudaStream_t stream{};
  require(cudaStreamCreate(&stream) == cudaSuccess, "session stream");
  SessionGraphs graphs;
  graphs.execs = new cudaGraphExec_t[3]();
  graphs.posts = new cudaGraphExec_t[3]();
  graphs.routes = new cudaGraphExec_t[3]();
  graphs.shared = new cudaGraphExec_t[3]();
  graphs.shared_capacity = 3;
  graphs.parts_dev = parts;
  graphs.n = 3;
  graphs.captured = true;
  graphs.post_routes_next = join_next;
  SessionLoopScratch scratch;
  std::string err;
  require(scratch.init(38 * 4, err), err);
  if (fused) {
    graphs.post_miss_host = scratch.y_miss;
    graphs.post_add_hits_host = scratch.add_hits;
  }
  for (int layer = 0; layer < 3; ++layer) {
    auto capture = [&](int part, cudaGraphExec_t *exec) {
      require(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) ==
                  cudaSuccess, "session capture");
      if (part == 0 || part == 1) {
        auto payload = db;
        stream->parallel_for(sycl::range<1>(19), [=](sycl::id<1> i) {
          payload.d_x_f[i] = value[i];
          if (!i[0]) {
            payload.d_ids[0] = layer;
            payload.d_ids[1] = layer + 100;
            payload.d_weights[0] = 1.f;
            payload.d_weights[1] = 0.f;
            ++*payload.d_seq;
          }
        });
      }
      if (part == 0 || part == 2)
        stream->parallel_for(sycl::range<1>(19), [=](sycl::id<1> i) {
          shared[i] = 2.f * value[i] + layer;
          if (!i[0]) ++count[layer];
        });
      if (part == 3) {
        if (fused)
          strata::kernels::copy_hit_miss(parts, scratch.y_miss, parts,
                                        scratch.add_hits, 38, stream);
        stream->parallel_for(sycl::range<1>(19), [=](sycl::id<1> i) {
          value[i] = parts[i] + shared[i];
        });
        if (join_next && layer < 2) {
          auto payload = db;
          stream->parallel_for(sycl::range<1>(19), [=](sycl::id<1> i) {
            payload.d_x_f[i] = value[i];
            if (!i[0]) {
              payload.d_ids[0] = layer + 1;
              payload.d_ids[1] = layer + 101;
              payload.d_weights[0] = 1.f;
              payload.d_weights[1] = 0.f;
              ++*payload.d_seq;
            }
          });
        }
      }
      cudaGraph_t graph{};
      require(cudaStreamEndCapture(stream, &graph) == cudaSuccess &&
                  cudaGraphInstantiate(exec, graph, 0ull) == cudaSuccess,
              "session finalize");
      require(cudaGraphDestroy(graph) == cudaSuccess, "session graph release");
    };
    capture(0, &graphs.execs[layer]);
    capture(1, &graphs.routes[layer]);
    capture(2, &graphs.shared[layer]);
    capture(3, &graphs.posts[layer]);
  }
  struct Pool { int next; } pool;
  auto compute = [](void *user, const float *x, const int32_t *ids,
                    const float *weights, int64_t n, int64_t k, float *out) {
    auto &p = *static_cast<Pool *>(user);
    require(n == 19 && k == 2 && ids[0] == p.next && ids[1] == p.next + 100 &&
                weights[0] == 1.f && weights[1] == 0.f, "current routed payload");
    for (int i = 0; i < 19; ++i) {
      out[i] = 3.f * x[i] + 5.f;
      out[19 + i] = 0.f;
    }
    ++p.next;
  };
  if (join_next) {
    require(!session_loop(g, 0, 0, state, graphs, compute, nullptr, &pool,
                          false, stream, err, nullptr, &scratch), "joined posts accepted serial replay");
    std::vector<float> dump(4 * 19);
    require(!session_loop(g, 0, 0, state, graphs, compute, nullptr, &pool,
                          true, stream, err, dump.data(), &scratch), "joined posts accepted layer dump");
  }
  if (fused)
    require(!session_replay_full(g, 0, 0, state, graphs, stream, err),
            "host post graphs accepted GPU-only replay");
  for (int round = 0; round < 12; ++round) {
    std::vector<float> input(19), expected(19), result(19);
    for (int i = 0; i < 19; ++i) input[i] = i * .25f + round;
    expected = input;
    for (int layer = 0; layer < 3; ++layer)
      for (float &v : expected) v = (3.f * v + 5.f) + (2.f * v + layer);
    require(cudaMemcpy(value, input.data(), 19 * 4, cudaMemcpyHostToDevice) ==
                cudaSuccess && cudaMemset(count, 0, 3 * 4) == cudaSuccess,
            "session input");
    pool.next = 0;
    require(session_loop(g, round, 0, state, graphs, compute, nullptr, &pool,
                         join_next || round % 2 == 0, stream, err, nullptr, &scratch), err);
    int executions[3]{};
    require(cudaMemcpy(result.data(), value, 19 * 4, cudaMemcpyDeviceToHost) ==
                cudaSuccess && cudaMemcpy(executions, count, 3 * 4,
                                          cudaMemcpyDeviceToHost) == cudaSuccess,
            "session results");
    require(result == expected && pool.next == 3 && executions[0] == 1 &&
                executions[1] == 1 && executions[2] == 1,
            "shared/CPU ordering, no omitted or duplicate shared expert");
  }
  scratch.free();
  session_graphs_free(graphs);
  require(cudaStreamDestroy(stream) == cudaSuccess, "session stream release");
  cudaFree(value); cudaFree(shared); cudaFree(parts); cudaFree(count);
  doorbell_free(db);
}
void captured_handoff() {
  constexpr size_t n = 25603;
  float *misses{}, *hits{}, *out{}, *original{};
  uint32_t* flag{};
  cudaStream_t stream{};
  require(cudaStreamCreate(&stream) == cudaSuccess &&
              cudaMallocHost(&misses, n * sizeof(float)) == cudaSuccess &&
              cudaMallocHost(&flag, sizeof(uint32_t)) == cudaSuccess &&
              cudaMalloc(&hits, n * sizeof(float)) == cudaSuccess &&
              cudaMalloc(&original, n * sizeof(float)) == cudaSuccess &&
              cudaMalloc(&out, (n + 1) * sizeof(float)) == cudaSuccess,
          "post handoff buffers");
  std::vector<float> h(n), expected(n), result(n + 1, -777.f);
  require(cudaMemcpy(out, result.data(), result.size() * sizeof(float),
                     cudaMemcpyHostToDevice) == cudaSuccess, "post guard");
  {
    strata::core::CapturedGraph graph;
    std::string err;
    require(graph.begin(stream, err), err);
    strata::kernels::copy_hit_miss(out, misses, hits, flag, n, stream);
    require(graph.end(stream, err), err);
    for (int round = 0; round < 24; ++round) {
      // Earlier CPU engine initialization may enable flushing subnormals.
      std::fesetenv(FE_DFL_ENV);
      *flag = uint32_t(round % 3 != 0);
      for (size_t i = 0; i < n; ++i) {
        misses[i] = float(int(i % 101) - 50) * .03125f + round;
        h[i] = float(int(i % 47) - 23) * .0625f - round;
      }
      misses[0] = -0.f; h[0] = 0.f;
      misses[1] = std::bit_cast<float>(0x80000001u); h[1] = 0.f;
      if (!*flag) {
        // Miss-only replay must retain these bits and ignore stale hit rows.
        misses[2] = std::bit_cast<float>(0x7fc12345u);
        std::fill(h.begin(), h.end(), 900.f + round);
      }
      for (size_t i = 0; i < n; ++i)
        expected[i] = *flag ? misses[i] + h[i] : misses[i];
      require(cudaMemcpy(hits, h.data(), n * sizeof(float), cudaMemcpyHostToDevice)
                  == cudaSuccess, "changing hit rows");
      require(cudaMemcpyAsync(original, misses, n * sizeof(float), cudaMemcpyHostToDevice,
                              stream) == cudaSuccess, "original handoff copy");
      if (*flag) strata::kernels::add_inplace(original, hits, n, stream);
      require(cudaStreamSynchronize(stream) == cudaSuccess, "original handoff completion");
      std::vector<float> previous(n);
      require(cudaMemcpy(previous.data(), original, n * sizeof(float), cudaMemcpyDeviceToHost)
                  == cudaSuccess && std::memcmp(expected.data(), previous.data(), n * sizeof(float)) == 0,
              "original handoff differs from scalar float reference");
      require(graph.launch(stream, err) && graph.wait_ms(5000), err);
      require(cudaMemcpy(result.data(), out, result.size() * sizeof(float),
                         cudaMemcpyDeviceToHost) == cudaSuccess, "post results");
      if (std::memcmp(expected.data(), result.data(), n * sizeof(float)) != 0)
        for (size_t i = 0; i < n; ++i)
          if (std::bit_cast<uint32_t>(expected[i]) != std::bit_cast<uint32_t>(result[i])) {
            std::cerr << "post round " << round << " flag " << *flag << " index " << i
                      << " expected " << std::hex << std::bit_cast<uint32_t>(expected[i])
                      << " actual " << std::bit_cast<uint32_t>(result[i]) << std::dec << '\n';
            break;
          }
      require(std::memcmp(expected.data(), result.data(), n * sizeof(float)) == 0 &&
                  result.back() == -777.f, "post handoff bits or buffer guard");
    }
  }
  cudaFree(out); cudaFree(original); cudaFree(hits); cudaFreeHost(flag); cudaFreeHost(misses);
  require(cudaStreamDestroy(stream) == cudaSuccess, "post stream release");
}
int main() {
  try {
    std::string err;
    {
      strata::core::PinnedArena arena(65536,
                                      std::vector<uint64_t>{0, 32768, 65536});
      require(arena.base && arena.registered_bytes == 0,
              "ordinary expert arena");
      static_cast<unsigned char *>(arena.base)[65535] = 42;
    }
    require(cudaGetLastError() == cudaSuccess,
            "arena cleanup left runtime error");
    cudaStream_t stream{};
    require(cudaStreamCreate(&stream) == cudaSuccess, "stream");
    int *host{}, *device{};
    require(cudaMallocHost(&host, sizeof(int)) == cudaSuccess, "host");
    require(cudaMalloc(&device, sizeof(int)) == cudaSuccess, "device");
    {
      strata::core::CapturedGraph graph;
      require(graph.begin(stream, err), err);
      require(cudaMemcpyAsync(device, host, sizeof(int), cudaMemcpyHostToDevice,
                              stream) == cudaSuccess,
              "capture upload");
      stream->single_task([=] { *device = *device * 3 + 7; });
      require(cudaMemcpyAsync(host, device, sizeof(int), cudaMemcpyDeviceToHost,
                              stream) == cudaSuccess,
              "capture download");
      require(graph.end(stream, err), err);
      for (int i = 0; i < 16; ++i) {
        *host = i;
        require(graph.launch(stream, err), err);
        require(graph.wait_ms(5000), "shared graph completion");
        require(*host == i * 3 + 7, "shared graph output");
      }
    }
    require(cudaFree(device) == cudaSuccess &&
                cudaFreeHost(host) == cudaSuccess,
            "free");
    require(cudaStreamDestroy(stream) == cudaSuccess, "destroy");
    strata::core::ExpertCache cache;
    for (int reopen = 0; reopen < 2; ++reopen) {
      require(cache.open_sized({4096, 8192, 16384}, 1, 3, err), err);
      for (int round = 0; round < 8; ++round)
        for (int slot = 0; slot < 3; ++slot) {
          std::vector<uint8_t> payload(size_t(4096) << slot);
          for (size_t i = 0; i < payload.size(); ++i)
            payload[i] = uint8_t(i * 17 + round * 11 + slot + reopen);
          require(cache.fill_slot_blocking(slot, payload.data(), err,
                                           payload.size()),
                  err);
          require(cache.verify_slot(slot, payload.data(), err, payload.size()),
                  err);
        }
      cache.close();
    }
    shared_session();
    shared_session(true);
    shared_session(true, true);
    captured_handoff();
    std::cout << "SYCL shared engine: ordinary arena, captured graph and "
                 "bounded cache staging and shared/CPU session ordering passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
