#include "strata/core/expert_cache.hpp"
#include "strata/core/graph.hpp"
#include "strata/core/pinned.hpp"
#include "strata/core/session.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>
void require(bool ok, const std::string &why) {
  if (!ok)
    throw std::runtime_error(why);
}
void shared_session() {
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
      if (part == 3)
        stream->parallel_for(sycl::range<1>(19), [=](sycl::id<1> i) {
          value[i] = parts[i] + shared[i];
        });
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
  SessionLoopScratch scratch;
  std::string err;
  require(scratch.init(38 * 4, err), err);
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
                         round % 2 == 0, stream, err, nullptr, &scratch), err);
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
    std::cout << "SYCL shared engine: ordinary arena, captured graph and "
                 "bounded cache staging and shared/CPU session ordering passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
