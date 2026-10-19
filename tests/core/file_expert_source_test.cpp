#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/pool.hpp"

#if defined(STRATA_NATIVE_EXPERTS)
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

struct TempDirectory {
    fs::path path;

    TempDirectory() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::temp_directory_path() / ("strata-file-source-test-" + std::to_string(stamp));
        fs::create_directories(path);
    }

    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

void create_pack(const fs::path& dir, uint64_t bytes, const std::vector<std::pair<uint64_t, char>>& markers = {}) {
    const fs::path file = dir / "experts.bin";
    {
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        require((bool) out, "could not create synthetic experts.bin");
        if (bytes > 0) {
            out.seekp((std::streamoff) (bytes - 1));
            out.put('\0');
        }
        require((bool) out, "could not size synthetic experts.bin");
    }
    for (const auto& [offset, marker] : markers) {
        require(offset < bytes, "synthetic marker lies beyond experts.bin");
        std::fstream file_out(file, std::ios::binary | std::ios::in | std::ios::out);
        require((bool) file_out, "could not open synthetic experts.bin for markers");
        file_out.seekp((std::streamoff) offset);
        file_out.put(marker);
        require((bool) file_out, "could not write synthetic expert marker");
    }
}

void check_file_size_rejection(strata::core::FileExpertSource& source, const fs::path& dir, int64_t layers,
                               int64_t experts, uint64_t expected_bytes) {
    std::string err;
    source.close();
    fs::resize_file(dir / "experts.bin", expected_bytes - 1);
    require(!source.open(dir.string(), layers, experts, err), "a truncated expert file was accepted");
    require(!source.mapped() && !err.empty(), "truncated-file rejection left the source mapped or unreported");

    fs::resize_file(dir / "experts.bin", expected_bytes + 1);
    err.clear();
    require(!source.open(dir.string(), layers, experts, err), "an oversized expert file was accepted");
    require(!source.mapped() && !err.empty(), "oversized-file rejection left the source mapped or unreported");
}

void test_canonical_layout() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    constexpr int64_t layers = 2;
    constexpr int64_t experts = 3;
    const uint64_t layer_bytes = (uint64_t) experts * BLOB;
    const uint64_t total = (uint64_t) layers * layer_bytes;
    TempDirectory dir;

    std::string err;
    const bool layout_ok = expert_layout_load(dir.path.string(), layers, experts, err);
    require(layout_ok, "could not load canonical layout: " + err);
    create_pack(dir.path, total, {{0, 'a'}, {(uint64_t) BLOB, 'b'}, {layer_bytes, 'c'},
                                  {layer_bytes + (uint64_t) BLOB, 'd'}});

    FileExpertSource source;
    bool opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "could not map canonical pack: " + err);
    require(source.blobs() == layers * experts, "canonical blob count is wrong");
    const uint8_t* first = source.blob(0, 0);
    const uint8_t* second = source.blob(0, 1);
    const uint8_t* next_layer = source.blob(1, 0);
    require(first && second && next_layer, "valid canonical blob lookup failed");
    require(second - first == (ptrdiff_t) BLOB && next_layer - first == (ptrdiff_t) layer_bytes,
            "canonical expert or layer stride is wrong");
    require(first[0] == 'a' && second[0] == 'b' && next_layer[0] == 'c',
            "canonical blob lookup returned bytes from the wrong expert");
    require(source.blob(-1, 0) == nullptr && source.blob(0, -1) == nullptr &&
                source.blob(layers, 0) == nullptr && source.blob(0, experts) == nullptr,
            "canonical bounds check accepted an invalid layer or expert");
    require(source.reads() == 3, "invalid canonical lookups changed the read count");

    check_file_size_rejection(source, dir.path, layers, experts, total);
    create_pack(dir.path, total, {{layer_bytes, 'c'}});
    err.clear();
    opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "source failed to reopen after size errors: " + err);
    const uint8_t* reopened = source.blob(1, 0);
    require(reopened && source.reads() == 1 && reopened[0] == 'c', "reopened canonical source kept stale state");
    source.close();
}

#if defined(STRATA_NATIVE_EXPERTS)
void test_native_variable_layout() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    constexpr int64_t layers = 2;
    constexpr int64_t experts = 3;
    constexpr int64_t embedding = 2560;
    constexpr int64_t feed_forward = 640;
    TempDirectory dir;

    NativeFmt first_fmt, second_fmt;
    std::string err;
    const bool first_ok = native_fmt(GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, embedding, feed_forward, first_fmt, err);
    require(first_ok, "IQ3_XXS/IQ4_NL synthetic format is unavailable: " + err);
    err.clear();
    const bool second_ok = native_fmt(GGML_TYPE_IQ2_XS, GGML_TYPE_IQ4_NL, embedding, feed_forward, second_fmt, err);
    require(second_ok, "IQ2_XS/IQ4_NL synthetic format is unavailable: " + err);
    require(first_fmt.bytes != second_fmt.bytes, "chosen native formats do not exercise variable layer sizes");

    const uint64_t layer0_bytes = (uint64_t) first_fmt.bytes * experts;
    const uint64_t total = layer0_bytes + (uint64_t) second_fmt.bytes * experts;
    {
        std::ofstream metadata(dir.path / "native_experts.txt");
        require((bool) metadata, "could not create synthetic native_experts.txt");
        metadata << "0 " << GGML_TYPE_IQ3_XXS << ' ' << GGML_TYPE_IQ4_NL << " 0 " << first_fmt.bytes << '\n';
        metadata << "1 " << GGML_TYPE_IQ2_XS << ' ' << GGML_TYPE_IQ4_NL << ' ' << layer0_bytes << ' '
                 << second_fmt.bytes << '\n';
        require((bool) metadata, "could not write synthetic native_experts.txt");
    }
    create_pack(dir.path, total, {{0, 'a'}, {(uint64_t) first_fmt.bytes, 'b'},
                                  {layer0_bytes, 'c'}, {layer0_bytes + (uint64_t) second_fmt.bytes, 'd'}});

    const bool layout_ok = expert_layout_load(dir.path.string(), layers, experts, err);
    require(layout_ok, "could not load synthetic native layout: " + err);
    FileExpertSource source;
    bool opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "could not map native pack: " + err);
    require(source.blobs() == layers * experts, "native blob count is wrong");
    const uint8_t* first = source.blob(0, 0);
    const uint8_t* second = source.blob(0, 1);
    const uint8_t* next_layer = source.blob(1, 0);
    const uint8_t* next_layer_second = source.blob(1, 1);
    require(first && second && next_layer && next_layer_second, "valid native blob lookup failed");
    require(second - first == (ptrdiff_t) first_fmt.bytes && next_layer - first == (ptrdiff_t) layer0_bytes &&
                next_layer_second - next_layer == (ptrdiff_t) second_fmt.bytes,
            "native per-layer expert stride is wrong");
    require(first[0] == 'a' && second[0] == 'b' && next_layer[0] == 'c' && next_layer_second[0] == 'd',
            "native blob lookup returned bytes from the wrong expert");
    require(source.blob(-1, 0) == nullptr && source.blob(0, -1) == nullptr &&
                source.blob(layers, 0) == nullptr && source.blob(0, experts) == nullptr,
            "native bounds check accepted an invalid layer or expert");
    require(source.reads() == 4, "invalid native lookups changed the read count");

    check_file_size_rejection(source, dir.path, layers, experts, total);
    create_pack(dir.path, total, {{layer0_bytes, 'c'}});
    err.clear();
    opened = source.open(dir.path.string(), layers, experts, err);
    require(opened, "source failed to reopen after size errors: " + err);
    const uint8_t* reopened = source.blob(1, 0);
    require(reopened && source.reads() == 1 && reopened[0] == 'c', "reopened native source kept stale state");
    source.close();
}
#endif

void test_complement_plan() {
    using namespace strata::core::detail;
    std::vector<uint64_t> offsets;
    uint64_t bytes = 0;
    std::string error;
    require(make_cache_complement_plan(2, 3, {3, 5}, {{0, 1}, {1, 2}}, {}, offsets, bytes, error), error);
    require(bytes == 16 && offsets == std::vector<uint64_t>{0, kNoCacheComplement, 3, 6, 11, kNoCacheComplement},
            "wrong compact offsets for variable native layer sizes");
    const uint8_t resident[16] = {}, mapped[5] = {};
    require(cache_complement_blob_or_fallback(1, offsets, resident, mapped) == mapped,
            "GPU-resident expert lost mmap fallback");
    require(cache_complement_blob_or_fallback(4, offsets, resident, mapped) == resident + 11,
            "CPU miss did not use resident complement");
    require(!make_cache_complement_plan(2, 3, {3, 5}, {{0, 1}, {0, 1}}, {}, offsets, bytes, error)
            && offsets.empty() && bytes == 0, "duplicate pair accepted");
    require(!make_cache_complement_plan(2, 3, {3, 5}, {{0, 1}}, {{0, 1}}, offsets, bytes, error),
            "overlapping tiers accepted");
    require(!make_cache_complement_plan(2, 3, {3, 5}, {{2, 0}}, {}, offsets, bytes, error),
            "out-of-range pair accepted");
    require(!make_cache_complement_plan(2, 3, {0, 5}, {}, {}, offsets, bytes, error), "zero-size layer accepted");
}

void test_resident_lend_region() {
    using namespace strata::core::detail;
    // four slots (sizes 5, 3, 3, 5); the experts no slot holds take 10 bytes
    const std::vector<uint64_t> slots{5, 3, 3, 5};
    require(choose_resident_keep_from(slots, 10, 9, 1) == -1, "a base larger than the budget was accepted");
    require(choose_resident_keep_from(slots, 10, 10, 1) == 4, "no room: the lend region must stay on the file");
    require(choose_resident_keep_from(slots, 10, 15, 1) == 3, "the last slot fits and must be kept first");
    require(choose_resident_keep_from(slots, 10, 17, 1) == 3, "a slot that does not fit ends the lend region's walk");
    require(choose_resident_keep_from(slots, 10, 18, 1) == 2, "two slots fit");
    require(choose_resident_keep_from(slots, 10, 100, 1) == 1, "the walk must stop at the lend region's first slot");
    require(choose_resident_keep_from(slots, 10, 100, -1) == 4, "no lend region kept slots in RAM");
    require(choose_resident_keep_from(slots, 10, 100, 9) == 4, "an out-of-range lend region kept slots in RAM");
    require(choose_resident_keep_from({}, 0, 0, 0) == 0, "an empty cache");
}

void test_resident_exchange() {
    using namespace strata::core;
    using namespace strata::core::detail;
    // 2 layers x 3 experts; the GPU holds (0,1) and (1,2): the compact copy holds the other four
    std::vector<uint64_t> offsets;
    uint64_t bytes = 0;
    std::string error;
    require(make_cache_complement_plan(2, 3, {3, 5}, {{0, 1}, {1, 2}}, {}, offsets, bytes, error), error);
    const std::vector<uint64_t> before = offsets;
    // (1,0) moves into the GPU, (1,2) leaves it: (1,2) takes (1,0)'s bytes' place
    require(exchange_cache_complement(offsets, 3, 5), "a valid exchange was refused");
    require(offsets[5] == before[3] && offsets[3] == kNoCacheComplement, "the exchange did not move the place");
    for (size_t i : {0u, 1u, 2u, 4u}) require(offsets[i] == before[i], "an exchange touched another expert");
    // the copy's size and its set of places are unchanged
    std::vector<uint64_t> a, b;
    for (uint64_t o : before) if (o != kNoCacheComplement) a.push_back(o);
    for (uint64_t o : offsets) if (o != kNoCacheComplement) b.push_back(o);
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    require(a == b, "an exchange changed the compact copy's places");
    const std::vector<uint64_t> after = offsets;
    require(!exchange_cache_complement(offsets, 3, 0), "an `in` the copy does not hold was accepted");
    require(!exchange_cache_complement(offsets, 0, 2), "an `out` the copy holds already was accepted");
    require(!exchange_cache_complement(offsets, 0, 0), "a self-exchange was accepted");
    require(!exchange_cache_complement(offsets, 0, 6), "an out-of-range `out` was accepted");
    require(offsets == after, "a refused exchange changed the offsets");
    // and back: the copy is again what the plan makes for the original placement
    require(exchange_cache_complement(offsets, 5, 3), "the reverse exchange was refused");
    require(offsets == before, "exchanging back did not restore the plan");
}

void test_cgroup_memory_budget() {
    using namespace strata::core::detail;
    constexpr uint64_t GiB = 1ull << 30;
    uint64_t bytes = 0;

    CgroupMemoryStat clean_cache{40 * GiB, 32 * GiB, 0, 0, true};
    require(cgroup_available_bytes(56 * GiB, clean_cache, bytes), "valid cgroup memory.stat was rejected");
    require(bytes == 48 * GiB, "clean inactive file cache was not credited against the cgroup cap");

    CgroupMemoryStat dirty_cache{40 * GiB, 32 * GiB, 4 * GiB, 2 * GiB, true};
    require(cgroup_available_bytes(56 * GiB, dirty_cache, bytes), "valid dirty-cache counters were rejected");
    require(bytes == 42 * GiB, "dirty/writeback pages were incorrectly counted as reclaimable");

    CgroupMemoryStat oversized_inactive{10 * GiB, 20 * GiB, 0, 0, true};
    require(cgroup_available_bytes(12 * GiB, oversized_inactive, bytes),
            "valid oversized inactive-file counter was rejected");
    require(bytes == 12 * GiB, "inactive-file accounting exceeded charged current usage");

    const uint64_t max = std::numeric_limits<uint64_t>::max();
    CgroupMemoryStat large_counters{max, max, max - 1, max, true};
    require(cgroup_available_bytes(max, large_counters, bytes),
            "large memory.stat counters were rejected");
    require(bytes == 0, "dirty/writeback subtraction overflowed or escaped the usage cap");

    CgroupMemoryStat usage_over_limit{60 * GiB, 0, 0, 0, true};
    require(cgroup_available_bytes(56 * GiB, usage_over_limit, bytes),
            "valid over-limit memory counters were rejected");
    require(bytes == 0, "over-limit cgroup usage produced a positive budget");

    CgroupMemoryStat missing_stat{};
    bytes = 123;
    require(!cgroup_available_bytes(56 * GiB, missing_stat, bytes) && bytes == 0,
            "missing memory.stat counters did not fail closed");
}

}  // namespace

// Exercise the actual provider capability independently of synthetic GPU slabs.
void test_arena_resident_layer() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    constexpr int64_t layers = 2, experts = 3;
    TempDirectory dir;
    std::vector<std::pair<uint64_t, char>> markers;
    for (int64_t l = 0; l < layers; ++l) for (int64_t e = 0; e < experts; ++e) {
        const uint64_t i = uint64_t(l * experts + e);
        markers.emplace_back(i * BLOB, char('A' + i));
        markers.emplace_back((i + 1) * BLOB - 1, char('a' + i));
    }
    create_pack(dir.path, uint64_t(layers * experts) * BLOB, markers);
    std::string err;
    require(expert_layout_load(dir.path.string(), layers, experts, err), err);
    FileExpertSource file;
    require(file.open(dir.path.string(), layers, experts, err), err);
    require(!file.resident_layer(0).data, "file source unexpectedly promises an immutable view");
    ArenaExpertSource arena;
    require(arena.open(dir.path.string(), layers, experts, 2, err), err);
    require(arena.reads() == 0, "loading/view setup changed blob read counters");
    const auto first = arena.resident_layer(0);
    require(!arena.resident_layer(-1).data && !arena.resident_layer(layers).data,
            "invalid Arena view layer accepted");
    for (int64_t l = 0; l < layers; ++l) {
        const auto view = arena.resident_layer(l);
        require(view.data && view.stride == BLOB && view.bytes == uint64_t(experts) * BLOB &&
                view.experts == experts, "wrong private Arena view geometry");
        for (int64_t e = 0; e < experts; ++e) {
            const auto* pointer = view.data + uint64_t(e) * view.stride;
            const int i = int(l * experts + e);
            require(pointer == arena.blob(l, e) && pointer[0] == uint8_t('A' + i) &&
                    pointer[BLOB - 1] == uint8_t('a' + i), "Arena view does not match loaded expert");
        }
    }
    require(first.data[0] == 'A' && first.data[BLOB - 1] == 'a',
            "later source calls invalidated a resident layer view");
    arena.close();
    require(!arena.resident_layer(0).data, "closed Arena still publishes views");
    require(arena.open(dir.path.string(), layers, experts, 2, err), err);
    require(arena.resident_layer(1).data && arena.resident_layer(1).data[0] == 'D',
            "reopened Arena view snapshot is stale");
#ifdef __linux__
    arena.close();
    require(arena.open(dir.path.string(), layers, experts, 2, err, 0,
                       (dir.path / "shared-arena.bin").string()), err);
    require(!arena.resident_layer(0).data && !arena.resident_layer(1).data,
            "externally mutable MAP_SHARED Arena publishes immutable views");
    std::vector<uint8_t> copied(size_t(BLOB), 0);
    require(arena.copy_blob(1, 0, copied.data()) && copied.front() == 'D' && copied.back() == 'd',
            "shared Arena synchronous copy fallback changed");
#endif
    std::cout << "arena_resident_layer_test private=PASS file_fallback=PASS shared_exclusion="
#ifdef __linux__
              << "PASS"
#else
              << "UNRUN"
#endif
              << " blob_identity=PASS bounds=PASS close_reopen=PASS\n";
}

#ifdef STRATA_ENABLE_SYCL
// Exercise the production planner and CPU pool. The sink is a host byte reader:
// no GPU kernels, mapped aliases or invalid device addresses are involved.
void test_arena_copy_dispatch() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    constexpr int64_t experts = 3;
    TempDirectory dir;
    create_pack(dir.path, uint64_t(2 * experts) * BLOB, {{0,'A'},{BLOB,'B'},{2*BLOB,'C'}});
    std::string error;
    require(expert_layout_load(dir.path.string(), 2, experts, error), error);
    ArenaExpertSource arena;
    require(arena.open(dir.path.string(), 2, experts, 2, error), error);
    ExpertPool pool(1, false, true);
    const std::array<int32_t,6> ids{0,1,2,2,1,0}, residency{-1,-1,-1,-1,-1,-1};
    const std::vector<float> activation(2*H, 0.0f);
    int cases = 0;
    auto run = [&](ExpertSource& source, bool enabled, int mode, int fraction, int capacity, int expected,
                   int missing_callback=0) {
        struct Probe {
            bool published = false;
            int copied = 0;
            int capacity;
            std::array<const uint8_t*,3> sources{};
            std::vector<uint8_t> stage;
        } probe{false,0,capacity,{},std::vector<uint8_t>(3*BLOB,0x5a)};
        std::array<int32_t,4> counts{};
        std::array<int32_t,7> start{}, start2{};
        std::array<int32_t,6> dst{}, tok{};
        std::array<unsigned long long,6> ptr{}, ptr2{};
        GpuPlanSink sink;
        sink.counts=counts.data();sink.start=start.data();sink.start2=start2.data();
        sink.dst=dst.data();sink.tok=tok.data();sink.ptr=ptr.data();sink.ptr2=ptr2.data();
        sink.staging=reinterpret_cast<uintptr_t>(probe.stage.data());sink.staging_cap=capacity;sink.cap=6;
        sink.host_layer_copy=enabled;sink.pcie_mode=mode;sink.ctx=&probe;
        sink.publish=[](void* context){static_cast<Probe*>(context)->published=true;};
        sink.fetch=[](void* context,const uint8_t* const* sources,int n,size_t bytes){
            auto& p=*static_cast<Probe*>(context);
            require((n==0 || p.published) && n>=0 && n<=p.capacity && n<=3 && bytes==BLOB,"invalid published copy plan");
            p.copied=n;
            for(int q=0;q<n;++q){
                require(sources[q]!=nullptr,"null planned source");p.sources[q]=sources[q];
                std::memcpy(p.stage.data()+size_t(q)*bytes,sources[q],bytes);
            }
        };
        if(missing_callback==1)sink.fetch=nullptr;
        if(missing_callback==2)sink.publish=nullptr;
        ExpertDispatch dispatch;
        dispatch.pool=&pool;dispatch.src=&source;dispatch.n_expert=experts;
        dispatch.host_res=residency.data();dispatch.plan=&sink;dispatch.pcie_num=fraction;
        std::vector<float> output(6*H,123.0f);
        expert_pool_dispatch_multi(dispatch,activation.data(),ids.data(),2,3,output.data());
        require(!dispatch.failed && probe.published==(sink.publish!=nullptr) &&
                probe.copied==expected && counts[2]==expected &&
                counts[0]==0 && counts[1]==2*expected,
                "ordinary-Arena dispatch/fallback differed");
        require(dispatch.host_layer_copy_groups==expected && dispatch.pcie_experts==expected &&
                dispatch.multi_misses==3-expected && dispatch.multi_entries==2*(3-expected),
                "CPU/GPU work conservation differed");
        std::array<bool,6> covered{};
        for(int q=0;q<expected;++q){
            require(ptr2[q]==sink.staging+uint64_t(q)*BLOB && start2[q+1]-start2[q]==2,
                    "staging/group extents differ");
            require(std::memcmp(probe.stage.data()+size_t(q)*BLOB,probe.sources[q],BLOB)==0,
                    "immutable source changed while CPU pool read other experts");
            const auto view=source.resident_layer(0);
            const int64_t expert=(probe.sources[q]-view.data)/static_cast<ptrdiff_t>(BLOB);
            for(int j=start2[q];j<start2[q+1];++j){
                const int index=dst[j];
                require(index>=0 && index<6 && !covered[index] && tok[j]==index/3 && ids[index]==expert,
                        "group lost or duplicated a routed row");covered[index]=true;
            }
        }
        for(float value:output)require(value==0.0f,"zero-activation CPU/fetched rows changed");
        ++cases;
    };
    run(arena,true,0,256,3,3);
    run(arena,false,0,256,3,0);
    run(arena,true,1,256,3,0);
    run(arena,true,2,256,3,0);
    run(arena,true,0,128,3,1);
    run(arena,true,0,256,1,1);
    run(arena,true,0,256,3,0,1);
    run(arena,true,0,256,3,0,2);
    struct BadView final : ExpertSource {
        ArenaExpertSource& source;int kind;
        BadView(ArenaExpertSource& s,int k):source(s),kind(k){}
        const uint8_t* blob(int64_t l,int64_t e)override{return source.blob(l,e);}
        ResidentExpertLayer resident_layer(int64_t l)const override{
            auto v=source.resident_layer(l);
            if(kind==0)++v.stride;
            if(kind==1)--v.bytes;
            if(kind==2)++v.experts;
            if(kind==3)v.data=reinterpret_cast<const uint8_t*>(std::numeric_limits<uintptr_t>::max()-15);
            return v;
        }
    };
    for(int kind=0;kind<4;++kind){BadView bad(arena,kind);run(bad,true,0,256,3,0);}
    FileExpertSource file;require(file.open(dir.path.string(),2,experts,error),error);
    run(file,true,0,256,3,0);
    arena.close();
    require(arena.open(dir.path.string(),2,experts,2,error,0,(dir.path/"shared-copy-arena.bin").string()),error);
    run(arena,true,0,256,3,0);
    require(cases==14,"incomplete dispatch cases");
    std::cout<<"arena_copy_dispatch_test cases=14 source_bounds=PASS grouping=PASS limits=PASS cpu_fallback=PASS alias_exclusion=PASS callbacks=PASS\n";
}
#endif

int main() {
    try {
        test_complement_plan();
        test_resident_lend_region();
        test_resident_exchange();
        test_cgroup_memory_budget();
        test_canonical_layout();
#if defined(STRATA_NATIVE_EXPERTS)
        test_native_variable_layout();
#endif
        test_arena_resident_layer();
#ifdef STRATA_ENABLE_SYCL
        test_arena_copy_dispatch();
#endif
        std::cout << "file_expert_source_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "file_expert_source_test: " << error.what() << '\n';
        return 1;
    }
}
