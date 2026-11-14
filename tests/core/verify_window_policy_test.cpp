#include "strata/core/verify_window_policy.hpp"

#include <array>
#include <cstdio>
#include <stdexcept>
#include <vector>

using strata::core::VerifyInput;
using strata::core::VerifyWindowGraphCache;
using strata::core::verify_q6_columns;
using strata::core::verify_q6_centering;
using strata::core::verify_q6_native_half;
using strata::core::verify_q6_dpas_head;
using strata::core::verify_q6_dpas_dense;
using strata::core::VerifyQ6Route;

static void need(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main() try {
    // The measured KV set must stay separate from prompt, non-streamed or
    // differently sized caches, and every unmeasured target width.
    const strata::core::VerifyKvShape kv{4,65536,5120,4,2,256,2051,true};
    int kv_selected = 0;
    for (bool enabled : {false,true})
        for (auto phase : {VerifyInput::prompt,VerifyInput::decode,static_cast<VerifyInput>(255)})
            for (int queries=0;queries<=9;++queries) {
                auto shape=kv;shape.queries=queries;
                const bool expected=enabled && phase==VerifyInput::decode && (queries==2||queries==4);
                need(strata::core::verify_coalesced_kv(enabled,phase,shape)==expected,
                    "KV planner escaped measured phase/window set");
                kv_selected+=expected;
            }
    need(kv_selected==2,"wrong measured KV width count");
    for (auto member : {&strata::core::VerifyKvShape::logical_pages,
                        &strata::core::VerifyKvShape::resident_pages,
                        &strata::core::VerifyKvShape::page_size,
                        &strata::core::VerifyKvShape::kv_heads,
                        &strata::core::VerifyKvShape::head_dim,
                        &strata::core::VerifyKvShape::selection_capacity})
        for (int delta : {-1,1}) {
            auto shape=kv;shape.*member+=delta;
            need(!strata::core::verify_coalesced_kv(true,VerifyInput::decode,shape),
                "unmeasured KV geometry selected");
        }
    auto nonstreamed=kv;nonstreamed.streamed_int8=false;
    need(!strata::core::verify_coalesced_kv(true,VerifyInput::decode,nonstreamed),
        "KV planner selected a non-streamed/other-format cache");
    // Only the independently measured winning head can use the new packing.
    // Exercise negative phase/layout/owner/type controls and every window size.
    int dpas_selected=0;
    for(bool ready:{false,true}) for(bool exact:{false,true})
        for(auto phase:{VerifyInput::prompt,VerifyInput::decode,static_cast<VerifyInput>(255)})
            for(int type:{-1,13,14,15}) for(int c=0;c<=9;++c) {
                const bool expected=ready && exact && phase==VerifyInput::decode && type==14 && c==4;
                need(verify_q6_dpas_head(ready,phase,exact,type,2560,248320,c)==expected,
                    "packed target-head policy escaped measured owner/phase/layout/type/columns");
                dpas_selected+=expected;
                for(const auto& shape:{std::array<int,2>{2560,640},{2560,248319},{2561,248320},
                                      {0,248320},{2560,0},{-2560,248320}})
                    need(!verify_q6_dpas_head(ready,phase,exact,type,shape[0],shape[1],c),
                        "unmeasured packed-head geometry selected");
            }
    need(dpas_selected==1,"wrong number of packed target-head selections");
    // Independent expected set from e44575158; exclude the losing512-row
    // shape, packed full head and every prompt/C1-C3/C8 call.
    int dense_selected = 0;
    for (const auto& shape : {std::array<int,2>{6144,2560}, {2560,6144},
                              {2560,10240}, {2560,12288}})
        for (bool enabled : {false,true}) for (bool exact : {false,true})
            for (auto phase : {VerifyInput::prompt,VerifyInput::decode,static_cast<VerifyInput>(255)})
                for (int type : {-1,13,14,15}) for (int c=0;c<=9;++c) {
                    const bool expected=enabled && exact && phase==VerifyInput::decode && type==14 && c==4;
                    need(verify_q6_dpas_dense(enabled,phase,exact,type,shape[0],shape[1],c)==expected,
                         "dense DPAS escaped the independently measured phase/layout/type/shape set");
                    dense_selected += expected;
                }
    need(dense_selected==4,"wrong number of qualified dense-DPAS cells");
    for (const auto& shape : {std::array<int,2>{2560,512}, {2560,640}, {2560,248320},
                              {6144,2559}, {2561,6144}, {0,0}, {-2560,10240}})
        for (int c=0;c<=9;++c)
            need(!verify_q6_dpas_dense(true,VerifyInput::decode,true,14,shape[0],shape[1],c),
                 "unqualified dense DPAS shape selected");
    // Expected selections come from the committed component decision, not a
    // second implementation of the dispatch predicate. Bit c means columns c.
    const int eligible[][3] = {
        {2560, 10240, 0x11e}, {6144, 2560, 0x11e}, {2560, 6144, 0x11e},
        {2560, 640, 0x11a}, {2560, 512, 0x100}, {2560, 12288, 0x11e},
        {2560, 248320, 0x11e},
    };
    int selected = 0;
    for (const auto& row : eligible) for (int c = 0; c <= 9; ++c) {
        const bool expected = (row[2] & (1 << c)) != 0;
        need(verify_q6_columns(true, VerifyInput::decode, true, 14, row[0], row[1], c) == expected,
             "dispatch differs from the repeated component decision");
        selected += expected;
        need(!verify_q6_columns(true, VerifyInput::prompt, true, 14, row[0], row[1], c),
             "decode candidate reached prompt reads");
        need(!verify_q6_columns(false, VerifyInput::decode, true, 14, row[0], row[1], c),
             "disabled candidate reached decode");
        need(!verify_q6_columns(true, VerifyInput::decode, false, 14, row[0], row[1], c),
             "exact-layout candidate reached another reduction layout");
        for (int type = -1; type < 40; ++type) if (type != 14)
            need(!verify_q6_columns(true, VerifyInput::decode, true, type, row[0], row[1], c),
                 "Q6 candidate changed another weight format");
    }
    need(selected == 30, "wrong number of qualified shapes");
    // These masks are the committed V2 centering decision,independent of the
    // older column-specialization decision above. All17 other cells remain off.
    const int centered[][3] = {
        {2560, 10240, 0x10e}, {6144, 2560, 0x01a}, {2560, 6144, 0x008},
        {2560, 640, 0}, {2560, 512, 0}, {2560, 12288, 0x11e},
        {2560, 248320, 0x11e},
    };
    int centered_selected = 0;
    for (const auto& row : centered) for (int c = 0; c <= 9; ++c) {
        const bool expected = (row[2] & (1 << c)) != 0;
        need(verify_q6_centering(true, VerifyInput::decode, true, 14, row[0], row[1], c) == expected,
             "centering dispatch differs from its repeated component decision");
        centered_selected += expected;
        need(!verify_q6_centering(true, VerifyInput::prompt, true, 14, row[0], row[1], c),
             "centering reached prompt reads");
        need(!verify_q6_centering(false, VerifyInput::decode, true, 14, row[0], row[1], c),
             "disabled centering reached decode");
        need(!verify_q6_centering(true, VerifyInput::decode, false, 14, row[0], row[1], c),
             "centering reached another reduction layout");
        for (int type = -1; type < 40; ++type) if (type != 14)
            need(!verify_q6_centering(true, VerifyInput::decode, true, type, row[0], row[1], c),
                 "centering changed another weight format");
    }
    need(centered_selected == 18, "wrong number of qualified centering shapes");
    // Independently recorded3e6b836a4 winners,intersected with the incumbent.
    // The native-half component also won512x4;the incumbent does not select it.
    const int half[][3] = {
        {2560, 10240, 0x11e}, {6144, 2560, 0x11e}, {2560, 6144, 0x11e},
        {2560, 640, 0}, {2560, 512, 0x100}, {2560, 12288, 0x11e},
        {2560, 248320, 0x11e},
    };
    int half_selected = 0;
    for (const auto& row : half) for (int c = 0; c <= 9; ++c) {
        const bool expected = (row[2] & (1 << c)) != 0;
        need(verify_q6_native_half(true, VerifyInput::decode, true, 14, row[0], row[1], c) == expected,
             "native-half dispatch differs from measured incumbent intersection");
        half_selected += expected;
        need(!verify_q6_native_half(true, VerifyInput::prompt, true, 14, row[0], row[1], c) &&
             !verify_q6_native_half(false, VerifyInput::decode, true, 14, row[0], row[1], c) &&
             !verify_q6_native_half(true, VerifyInput::decode, false, 14, row[0], row[1], c),
             "native-half escaped its phase,selection or exact-layout gate");
        for (int type = -1; type < 40; ++type) if (type != 14)
            need(!verify_q6_native_half(true, VerifyInput::decode, true, type, row[0], row[1], c),
                 "native-half changed another weight format");
    }
    need(half_selected == 26, "wrong number of native-half incumbent shapes");
    for (const auto& shape : {std::array<int, 2>{-2560, 10240}, {2560, 10239},
                              {2561, 10240}, {0, 0}, {2560, -1}})
        need(!verify_q6_columns(true, VerifyInput::decode, true, 14, shape[0], shape[1], 4) &&
             !verify_q6_centering(true, VerifyInput::decode, true, 14, shape[0], shape[1], 4) &&
             !verify_q6_native_half(true, VerifyInput::decode, true, 14, shape[0], shape[1], 4),
             "unmeasured geometry reached candidate");

    // Actual cache: prompt4 -> decode4 -> prompt4 must recover the original
    // handles and route metadata. A T-only cache fails this request sequence.
    VerifyWindowGraphCache<int> cache;
    auto& prompt = cache.at(VerifyInput::prompt, 4);
    prompt.layer = {11}; prompt.resident = {12}; prompt.pcie = {13};
    prompt.q6_reference_calls = 41;
    prompt.kv_scalar_calls = 10;
    prompt.ready = true;
    auto& decode = cache.at(VerifyInput::decode, 4);
    need(decode.layer.empty() && decode.resident.empty() && decode.pcie.empty(), "phase cache alias");
    decode.layer = {21}; decode.resident = {22}; decode.pcie = {23};
    decode.q6_columns_calls = 42;
    decode.kv_coalesced_calls = 10;
    decode.q6_centered_calls = 12;
    decode.q6_dpas_head_calls = 1;
    decode.q6_dpas_dense_calls = {40,14,22,5};
    need(!decode.ready, "a partial graph cache became replayable");
    need(prompt.kv_scalar_calls==10 && prompt.kv_coalesced_calls==0 &&
         decode.kv_scalar_calls==0 && decode.kv_coalesced_calls==10 &&
         cache.at(VerifyInput::decode,1).kv_coalesced_calls==0,
         "KV route metadata crossed prompt/decode or window keys");
    decode.ready = true;
    need(cache.at(VerifyInput::prompt, 4).layer == std::vector<int>{11} &&
         cache.at(VerifyInput::prompt, 4).q6_columns_calls == 0 &&
         cache.at(VerifyInput::prompt, 4).q6_reference_calls == 41 &&
         cache.at(VerifyInput::prompt, 4).q6_centered_calls == 0 &&
         cache.at(VerifyInput::decode, 4).q6_centered_calls == 12 && prompt.ready,
         "decode changed an already captured prompt window");
    need(prompt.q6_dpas_head_calls==0 && cache.at(VerifyInput::decode,4).q6_dpas_head_calls==1,
         "packed head route leaked across phase caches");
    need(prompt.q6_dpas_dense_calls==std::array<uint64_t,4>{} &&
         cache.at(VerifyInput::decode,4).q6_dpas_dense_calls==std::array<uint64_t,4>{40,14,22,5} &&
         cache.at(VerifyInput::decode,3).q6_dpas_dense_calls==std::array<uint64_t,4>{},
         "dense DPAS capture metadata crossed phase/token keys");
    auto& decode8 = cache.at(VerifyInput::decode, 8);
    decode8.layer = {31};
    decode8.q6_columns_calls = decode8.q6_native_half_calls = 26;
    decode8.q6_capture_id = 7;
    decode8.q6_cache_hits = decode8.q6_replay_started = decode8.q6_replay_completed = 2;
    for (int i = 0; i < 26; ++i)
        decode8.record_q6(2560, 248320, 8, VerifyQ6Route::native_half);
    const auto& readonly_cache = cache;
    need(readonly_cache.at(VerifyInput::decode, 8).q6_shapes.size() == 1 &&
         readonly_cache.at(VerifyInput::decode, 8).q6_shapes[0].calls[3] == 26 &&
         readonly_cache.at(VerifyInput::prompt, 4).q6_native_half_calls == 0 &&
         readonly_cache.at(VerifyInput::decode, 4).q6_native_half_calls == 0,
         "native-half shape/capture/replay metadata crossed phase or token keys");
    need(cache.at(VerifyInput::decode, 4).layer == std::vector<int>{21}, "token cache alias");
    int refused = 0;
    for (int t : {0, 9, -1}) try { cache.at(VerifyInput::decode, t); }
        catch (const std::out_of_range&) { ++refused; }
    try { cache.at(static_cast<VerifyInput>(255), 4); }
        catch (const std::out_of_range&) { ++refused; }
    need(refused == 4, "invalid graph cache key accepted");

    // An interrupted destroy keeps the remaining owners; a retry must not
    // destroy successful predecessors twice or omit the other phase/families.
    std::vector<int> destroyed;
    bool interrupted = false;
    try {
        cache.clear([&](int handle) {
            if (handle == 12) throw std::runtime_error("intentional host-only destroy refusal");
            destroyed.push_back(handle);
        });
    } catch (const std::runtime_error&) { interrupted = true; }
    need(interrupted && destroyed == std::vector<int>{11} && prompt.resident[0] == 12 &&
         decode.layer[0] == 21, "failed cleanup released remaining graph owners");
    cache.clear([&](int handle) { destroyed.push_back(handle); });
    need(destroyed == std::vector<int>({11, 12, 13, 21, 22, 23, 31}),
         "retirement missed or double-destroyed a phase graph");
    need(prompt.layer.empty() && prompt.resident.empty() && prompt.pcie.empty() &&
         decode.layer.empty() && decode.resident.empty() && decode.pcie.empty() &&
         prompt.q6_reference_calls == 0 && decode.q6_columns_calls == 0 &&
         prompt.kv_scalar_calls == 0 && prompt.kv_coalesced_calls == 0 &&
         decode.kv_scalar_calls == 0 && decode.kv_coalesced_calls == 0 &&
         prompt.q6_centered_calls == 0 && decode.q6_centered_calls == 0 &&
         prompt.q6_dpas_head_calls==0 && decode.q6_dpas_head_calls==0 &&
         prompt.q6_dpas_dense_calls==std::array<uint64_t,4>{} &&
         decode.q6_dpas_dense_calls==std::array<uint64_t,4>{} &&
         decode8.q6_native_half_calls == 0 && decode8.q6_shapes.empty() &&
         decode8.q6_capture_id == 0 && decode8.q6_replay_started == 0 &&
         decode8.q6_replay_completed == 0 && decode8.q6_cache_hits == 0 &&
         !prompt.ready && !decode.ready,
         "retirement left stale graphs or route metadata");
    cache.clear([&](int) { throw std::runtime_error("second retirement is not empty"); });
    cache.at(VerifyInput::prompt, 4).layer = {51};
    need(cache.at(VerifyInput::decode, 4).layer.empty(), "retired decode graph survived prompt recapture");
    cache.clear([](int) {});
    // A bounded audit must report incompleteness instead of allocating without
    // limit or indexing an invalid route. Existing entries remain readable.
    for (size_t i = 0; i < decode8.max_q6_shapes; ++i)
        decode8.record_q6(2560, int(i + 1), 1, VerifyQ6Route::reference);
    decode8.record_q6(2560, 1000, 1, VerifyQ6Route::reference);
    need(!decode8.q6_routes_complete && decode8.q6_shapes.size() == decode8.max_q6_shapes,
         "route audit exceeded its declared shape budget");
    cache.clear([](int) {});
    decode8.record_q6(2560, 512, 1, static_cast<VerifyQ6Route>(255));
    need(!decode8.q6_routes_complete && decode8.q6_shapes.empty(), "invalid audit route indexed counters");
    cache.clear([](int) {});
    need(decode8.q6_routes_complete, "audit retirement kept an incomplete flag");
    std::puts("verify-window-policy: passed selections=30 centering=18 native-half=26 dpas-head=1 dpas-dense=4 phase-roundtrip=1 invalid-keys=4 retirement=1 route-budget=64");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "verify-window-policy: %s\n", error.what());
    return 1;
}
