// Original Strata fixture under the repository LICENSE. No SYCL/GPU calls.
#include "moe_window_routes.hpp"
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>

using Routes = strata::prefill::detail::MoeWindowRoutes;
static_assert(!std::is_move_constructible<Routes>::value, "published owners cannot move");
static_assert(!std::is_copy_constructible<Routes>::value, "published owners cannot copy");

static void check(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(reason);
}

static uint64_t witness(int32_t expert, int32_t row) {
    return (uint64_t(expert) << 32) | uint32_t(row);
}

static size_t verify(int32_t rows, int pattern) {
    std::vector<int32_t> ids(size_t(rows) * Routes::top_k);
    for (int32_t t = 0; t < rows; ++t)
        for (int32_t k = 0; k < Routes::top_k; ++k) {
            int32_t e = 0;
            switch (pattern) {
            case 0: e = k; break;                               // ten experts: ne==W
            case 1: e = (t * 10 + k) % Routes::experts; break;   // balanced
            case 2: e = k < 2 ? k : 2 + (t + k * 41) % 510; break; // two hot
            case 3: e = (t < 8192 ? k : 511 - k); break;        // disjoint halves
            case 4: e = (t * 37 + k * 43 + 7) % 512; break;     // unsorted slot order
            default: throw std::runtime_error("unknown pattern");
            }
            ids[size_t(t) * 10 + k] = e;
        }
    const auto expected_ids = ids;
    auto plan = Routes::build(rows, ids.data(), ids.size());
    // ID readback can be reused after build; uploads use independent owned data.
    std::fill(ids.begin(), ids.end(), -777);
    const auto& offsets = plan->offsets();
    const auto& counts = plan->counts();
    const auto& source = plan->source_rows();
    const auto& slots = plan->slot_map();
    check(plan->rows() == rows && source.size() == size_t(rows) * 10 &&
          slots.size() == source.size(), "route geometry");
    check(offsets.front() == 0 && offsets.back() == int32_t(source.size()), "prefix bounds");

    // Independent reference: scan token-slot order for EACH expert, without
    // using the implementation's cursor construction or job generation.
    for (int32_t e = 0; e < Routes::experts; ++e) {
        std::vector<int32_t> wanted;
        for (size_t i = 0; i < expected_ids.size(); ++i)
            if (expected_ids[i] == e) wanted.push_back(int32_t(i / 10));
        check(counts[size_t(e)] == int32_t(wanted.size()), "reference count");
        check(offsets[size_t(e) + 1] - offsets[size_t(e)] == counts[size_t(e)], "offset width");
        for (size_t i = 0; i < wanted.size(); ++i)
            check(source[size_t(offsets[size_t(e)]) + i] == wanted[i], "stable expert sources");
    }

    std::vector<uint64_t> down(source.size(), ~uint64_t(0));
    std::vector<int32_t> job_seen(source.size(), 0), slot_seen(source.size(), 0);
    std::array<int32_t, Routes::experts> expert_cursor{};
    int32_t last_expert = -1;
    size_t job_index = 0;
    for (const auto& job : plan->jobs()) {
        check(job.expert >= last_expert && job.expert >= 0 && job.expert < 512, "expert ordering");
        last_expert = job.expert;
        check(job.rows > 0 && job.rows <= 8192, "job scratch capacity");
        check(job.expert_row == expert_cursor[size_t(job.expert)], "job contiguous expert rows");
        check(job.global_row == offsets[size_t(job.expert)] + job.expert_row, "global Down offset");
        check(job.expert_row + job.rows <= counts[size_t(job.expert)], "expert job end");
        const auto* host_src = plan->job_source(job_index++);
        // Simulate each H2D at device_src[0] with an exact 8K allocation. This
        // catches using a global route offset as an 8K device source offset.
        std::array<int32_t, 8192> device_src{};
        std::copy_n(host_src, job.rows, device_src.begin());
        for (int32_t j = 0; j < job.rows; ++j) {
            const auto t = device_src[size_t(j)];
            const auto out = size_t(job.global_row + j);
            check(t >= 0 && t < rows, "gather input window bounds");
            ++job_seen[out];
            down[out] = witness(job.expert, t);
        }
        expert_cursor[size_t(job.expert)] += job.rows;
    }
    check(expert_cursor == counts, "all expert jobs complete");
    for (size_t i = 0; i < slots.size(); ++i) {
        const auto p = slots[i];
        check(p >= 0 && size_t(p) < slots.size(), "combine slot extent");
        ++slot_seen[size_t(p)];
        const auto e = expected_ids[i];
        check(p >= offsets[size_t(e)] && p < offsets[size_t(e) + 1], "combine expert interval");
        check(source[size_t(p)] == int32_t(i / 10), "token-slot conservation");
        check(down[size_t(p)] == witness(e, int32_t(i / 10)), "job Down -> original token-slot identity");
    }
    for (size_t i = 0; i < slots.size(); ++i)
        check(job_seen[i] == 1 && slot_seen[i] == 1, "route consumed exactly once");
    if (pattern == 0) {
        for (int32_t e = 0; e < 10; ++e) check(counts[size_t(e)] == rows, "ne==W accepted");
        check(plan->jobs().size() == size_t(10 * ((rows + 8191) / 8192)), "ne==W job split");
    }
    // Check the C++ owner keeps addresses/data alive after the submitter retires.
    const auto* upload = plan->job_source(0);
    const int32_t first = upload[0];
    auto in_flight_owner = plan;
    plan.reset();
    check(upload == in_flight_owner->job_source(0) && upload[0] == first, "upload owner stability");
    return source.size();
}

template<class F> static void refuses(F action) {
    try { action(); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid router output was accepted");
}

int main() try {
    size_t cases = 0, routes = 0, refusals = 0;
    for (int32_t rows : {1, 2, 511, 8191, 8192, 8193, 16383, 16384})
        for (int pattern = 0; pattern < 5; ++pattern) {
            routes += verify(rows, pattern); ++cases;
        }
    std::vector<int32_t> ids(20);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = int32_t(i % 10);
    for (int64_t rows : {int64_t(-1), int64_t(0), int64_t(16385), std::numeric_limits<int64_t>::max()}) {
        refuses([&] { Routes::build(rows, ids.data(), ids.size()); }); ++refusals;
    }
    refuses([&] { Routes::build(2, nullptr, ids.size()); }); ++refusals;
    for (size_t count : {size_t(0), size_t(19), size_t(21), std::numeric_limits<size_t>::max()}) {
        refuses([&] { Routes::build(2, ids.data(), count); }); ++refusals;
    }
    for (int32_t bad : {-1, 512, std::numeric_limits<int32_t>::max()}) {
        auto broken = ids; broken.back() = bad;
        refuses([&] { Routes::build(2, broken.data(), broken.size()); }); ++refusals;
    }
    auto repeated = ids; repeated.back() = repeated[10];
    refuses([&] { Routes::build(2, repeated.data(), repeated.size()); }); ++refusals;
    bool index_refused = false;
    try { Routes::build(2, ids.data(), ids.size())->job_source(size_t(-1)); }
    catch (const std::out_of_range&) { index_refused = true; }
    check(index_refused, "job index outside plan rejected"); ++refusals;
    std::cout << "{\"passed\":true,\"routing_cases\":" << cases
              << ",\"token_slots_checked\":" << routes << ",\"refusal_cases\":" << refusals
              << ",\"GPU_executed\":false,\"tensor_math_checked\":false}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
}
