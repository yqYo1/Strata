#include "strata/core/expert_cache.hpp"
#include "strata/core/graph.hpp"
#include "strata/core/pinned.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>
void require(bool ok, const std::string &why) {
  if (!ok)
    throw std::runtime_error(why);
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
    std::cout << "SYCL shared engine: ordinary arena, captured graph and "
                 "bounded cache staging passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
