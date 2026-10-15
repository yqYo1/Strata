// An adapter allocation can remain in its global owner map until process exit.
// Its registered extent must outlive that owner, including late static cleanup.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    const bool early = argc == 2 && std::strcmp(argv[1], "--early") == 0;
    if (argc != 1 && !early) return 2;
    void* allocation = nullptr;
    if (cudaMalloc(&allocation, 64) != cudaSuccess) return 3;
    if (cudaMemset(allocation, 0, 64) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return 4;
    if (early && cudaFree(allocation) != cudaSuccess) return 5;
    // No queued GPU work survives main. The late arm tests the real adapter's
    // existing allocation owner rather than a fixture imitation of its registry.
    std::puts(early ? "allocation_teardown early main complete" : "allocation_teardown late main complete");
    std::fflush(stdout);
    return 0;
}
