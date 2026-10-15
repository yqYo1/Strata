// Unexecuted production-linked fixture. Link the PROPOSED native_dense.cpp and
// configured production SYCL gguf/weights/native_mmvq/runtime objects; no mock.
// Usage: fixture PACK_DIR GGUF_SHARD [GGUF_SHARD ...]
#include "strata/core/native_dense.hpp"
#include "strata/core/weights.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>
#ifdef STRATA_DENSE_BANK_ALLOC_FAULT_FIXTURE
// Compile NativeDense alone with -include allocator_fault_shim.hpp. Injection arms ONLY at enter_layer_bank,
// leaving real startup allocations and payload comparisons unchanged.
static const char* allocation_mode=nullptr;
static bool allocation_armed=false;
cudaError_t strata_dense_fixture_malloc(void** out, size_t bytes) {
    if(!allocation_armed) return cudaMalloc(out,bytes);
    allocation_armed=false;
    static int fake_owner;
    const bool nonnull=std::strstr(allocation_mode,"nonnull")!=nullptr;
    *out=nonnull ? &fake_owner : nullptr;
    if(!std::strncmp(allocation_mode,"oom",3)) return cudaErrorMemoryAllocation;
    if(!std::strncmp(allocation_mode,"success",7)) return cudaSuccess;
    return cudaErrorUnknown;
}
#endif
using strata::core::NativeDense;
using strata::core::WeightRef;
static void require(bool ok, const std::string& message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message.c_str()); std::abort(); }
}
static void gpu(cudaError_t status) { require(status == cudaSuccess, "checked runtime operation"); }
static void payload(const WeightRef& ref, const std::vector<uint8_t>& source, cudaStream_t queue) {
    std::vector<uint8_t> bytes(source.size());
    gpu(cudaMemcpyAsync(bytes.data(), ref.native_data, bytes.size(), cudaMemcpyDeviceToHost, queue));
    gpu(cudaStreamSynchronize(queue)); // host destination remains alive through completion
    require(bytes == source, "packed device payload differs from original GGUF");
}
static void metadata(WeightRef a, WeightRef b) {
    // Independent audit of every metadata/provenance field; addresses of native
    // payloads change by design. The canonical pointer must remain identical.
#define SAME(f) require(a.f == b.f, "metadata changed: " #f)
    SAME(data); SAME(bytes); SAME(ne0); SAME(ne1); SAME(elements); SAME(kind);
    SAME(code_bits); SAME(code_bias); SAME(group_elems); SAME(codebook_iq4nl);
    SAME(has_offset); SAME(codes_bytes); SAME(scales_bytes); SAME(offset_bytes);
    SAME(src_off); SAME(src_bytes); SAME(file_id); SAME(scales_fp16); SAME(act_kind);
    SAME(native_q8_1); SAME(native_type); SAME(resident);
#undef SAME
}
static void run(const std::string& pack, const std::vector<std::string>& shards, cudaStream_t queue) {
    uint64_t arena_bytes = 0;
    std::string err;
    require(strata::core::WeightTable::pool_bytes(pack, arena_bytes, err), err);
    void* arena = nullptr;
    gpu(cudaMalloc(&arena, arena_bytes));
    {
        strata::core::WeightTable table;
        require(table.load(pack, arena, arena_bytes, err), err);
        // A bounded startup failure must not publish a single override.
        {
            NativeDense bounded;
            require(!bounded.load(shards, table, err, false, true, 0), "zero RAM budget accepted");
            for (const auto& [name, ref] : table.all())
                require(!ref.native_data && !ref.native_q8_1, "startup failure partially published");
        }
        NativeDense native;
        require(native.load(shards, table, err, false, true), err);
        std::map<std::string, WeightRef> original;
        std::set<int> layers;
        uint64_t total = 0, max_layer = 0;
        std::map<int, uint64_t> per_layer;
        // Audit every owned source against a NEW independent GGUF mapping,
        // including source provenance. Never obtain reference bytes via D2H.
        for (const auto& source : native.packed_sources()) {
            strata::GgufFile gguf(source.shard);
            const auto found = std::find_if(gguf.tensors().begin(), gguf.tensors().end(),
                [&](const auto& t) { return t.name == source.name; });
            require(found != gguf.tensors().end(), "source tensor missing");
            require(found->shape.size() == 2 && found->shape[0] == uint64_t(source.original.ne0) &&
                    found->shape[1] == uint64_t(source.original.ne1), "original shape");
            require(int(found->type) == source.original.native_type, "original type");
            require(found->offset == source.gguf_offset && gguf.data_start() == source.data_start &&
                    gguf.file_size() == source.file_bytes, "source provenance");
            require(source.packed.size() == strata::kernels::native_mmvq_weight_bytes(
                    found->type, found->shape[0], found->shape[1]), "original payload extent");
            require(std::memcmp(source.packed.data(), gguf.tensor_data(*found), source.packed.size()) == 0,
                    "owned RAM differs from original GGUF bytes");
            const auto* ref = table.find(source.name);
            require(ref && ref->native_data, "original table owner");
            metadata(*ref, source.original);
            payload(*ref, source.packed, queue);
            original.emplace(source.name, *ref);
            layers.insert(source.layer); total += source.packed.size();
            per_layer[source.layer] += (source.packed.size() + 255) & ~uint64_t(255);
        }
        for (auto [layer, bytes] : per_layer) max_layer = std::max(max_layer, bytes);
        require(total == native.weight_bytes() && total == native.snapshot_bytes(), "exact RAM bytes");
        require(max_layer == native.bank_bytes(), "max-layer bank bytes");
        require(original.size() == 300 && total == 2116951040ULL && max_layer == 50918400ULL,
                "target model census differs from committed independent header census");
        require(!native.enter_layer_bank({}, err), "retirement omission accepted");
        for (const auto& [name, ref] : original)
            require(table.find(name)->native_data == ref.native_data, "rejected enter changed owner");
        const NativeDense::ReaderRetirement retired{true};
#ifdef STRATA_DENSE_BANK_ALLOC_FAULT_FIXTURE
        if(allocation_mode) {
            allocation_armed=true;
            const bool entered=native.enter_layer_bank(retired,err);
            if(std::strcmp(allocation_mode,"oom-null")) {
                std::fprintf(stderr,"FAIL unexpected ordinary enter return mode=%s entered=%d\n",allocation_mode,int(entered));
                std::exit(91);
            }
            require(!entered && native.residency()==NativeDense::Residency::Full,"OOM did not retain Full");
            for(const auto& [name,ref]:original)
                require(table.find(name)->native_data==ref.native_data,"OOM changed full publication");
            require(native.layer_view().empty(),"OOM published bank view");
        }
#endif
        require(native.enter_layer_bank(retired, err), err);
        require(native.residency() == NativeDense::Residency::Bank, "bank state");
        for (const auto& [name, ref] : original) {
            const auto* withdrawn = table.find(name);
            require(withdrawn && !withdrawn->data && !withdrawn->native_data &&
                    !withdrawn->native_q8_1 && !withdrawn->resident, "normal table not withdrawn");
        }
        require(!native.activate_layer(-1, queue, retired, err) && native.layer_view().empty(),
                "invalid layer partially published");
        for (int layer : layers) {
            require(native.activate_layer(layer, queue, retired, err), err);
            auto previous = native.layer_view();
            require(!native.activate_layer(-1, queue, retired, err), "invalid layer accepted");
            require(native.layer_view().size() == previous.size(), "rejected activation changed view");
            for (const auto& [name, ref] : previous)
                require(native.layer_view().at(name).native_data == ref.native_data, "view partially republished");
            size_t count = 0;
            for (const auto& source : native.packed_sources()) if (source.layer == layer) {
                const auto& ref = native.layer_view().at(source.name);
                metadata(ref, original.at(source.name)); payload(ref, source.packed, queue); ++count;
            }
            require(count == native.layer_view().size(), "extra native view rows");
        }
        require(!native.restore_full(queue, {}, err), "restore without retirement accepted");
        require(native.residency() == NativeDense::Residency::Bank, "rejected restore changed state");
        require(native.restore_full(queue, retired, err), err);
        require(native.layer_view().empty() && native.residency() == NativeDense::Residency::Full,
                "restore state/view");
        for (const auto& source : native.packed_sources()) {
            const auto& ref = *table.find(source.name);
            metadata(ref, original.at(source.name)); payload(ref, source.packed, queue);
        }
        // Repeat catches stale original-owner validation after changed addresses.
        require(native.enter_layer_bank(retired, err), err);
        require(native.restore_full(queue, retired, err), err);
        gpu(cudaStreamSynchronize(queue));
        std::printf("PASS matrices=%zu layers=%zu snapshot=%llu bank=%llu\n", original.size(), layers.size(),
                    (unsigned long long)total, (unsigned long long)max_layer);
    } // Full NativeDense destructor before arena/stream destruction
    gpu(cudaFree(arena));
}
int main(int argc, char** argv) {
#ifdef STRATA_DENSE_BANK_ALLOC_FAULT_FIXTURE
    require(argc>=4,"fault fixture MODE PACK GGUF...");
    allocation_mode=argv[1]; --argc; ++argv;
#endif
    require(argc >= 3, "usage: fixture PACK_DIR GGUF_SHARD [GGUF_SHARD ...]");
    cudaStream_t queue = nullptr;
    gpu(cudaStreamCreate(&queue));
    run(argv[1], std::vector<std::string>(argv + 2, argv + argc), queue);
    gpu(cudaStreamSynchronize(queue)); gpu(cudaStreamDestroy(queue));
}
