// Original Strata source; repository LICENSE applies. Host metadata only.
#include "strata/prefill/moe_window_routes.hpp"
#include <cstdio>
#include <stdexcept>
#include <type_traits>
#include <vector>

using Routes = strata::prefill::detail::MoeWindowRoutes;
static_assert(!std::is_copy_constructible_v<Routes> && !std::is_move_constructible_v<Routes>);
void need(bool ok) { if (!ok) throw std::runtime_error("accumulator route contract"); }
template<class Function> void refuses(Function f) {
    bool rejected = false;
    try { f(); } catch (const std::invalid_argument&) { rejected = true; }
    need(rejected);
}
void test(int32_t rows, bool hot) {
    std::vector<int32_t> ids(size_t(rows) * 10);
    for (int32_t t = 0; t < rows; ++t)
        for (int32_t k = 0; k < 10; ++k)
            ids[size_t(t) * 10 + k] = ((hot ? 17 : t * 7) + k * 47) % 512;
    const auto plan = Routes::build_accum(rows, ids.data(), ids.size());
    need(plan->original_slots().size() == ids.size());
    std::vector<unsigned char> seen(ids.size(), 0);
    int32_t expert = -1, previous_token = -1;
    size_t covered = 0;
    for (const auto& job : plan->jobs()) {
        need(job.expert >= expert && job.rows > 0 && job.rows <= 8192);
        if (job.expert != expert) previous_token = -1;
        expert = job.expert;
        need(size_t(job.global_row) == covered);
        for (int32_t j = 0; j < job.rows; ++j) {
            const auto grouped = size_t(job.global_row + j);
            const auto original = size_t(plan->original_slots()[grouped]);
            need(original < ids.size() && !seen[original]++);
            need(size_t(plan->slot_map()[original]) == grouped);
            const int32_t token = plan->source_rows()[grouped];
            need(token >= 0 && token < rows && token == int32_t(original / 10));
            // This is the no-atomic proof: no expert has two routes to one
            // token, even across consecutive 8192-row jobs of that expert.
            need(token > previous_token && ids[original] == job.expert);
            previous_token = token;
        }
        covered += size_t(job.rows);
    }
    need(covered == ids.size());
    for (auto n : seen) need(n == 1);
    if (rows <= Routes::max_rows) {
        const auto old = Routes::build(rows, ids.data(), ids.size());
        need(old->original_slots().empty() && old->source_rows() == plan->source_rows());
        need(old->slot_map() == plan->slot_map());
    } else refuses([&] { (void)Routes::build(rows, ids.data(), ids.size()); });
    refuses([&] { (void)Routes::build_accum(rows, ids.data(), ids.size() - 1); });
    ids[1] = ids[0];
    refuses([&] { (void)Routes::build_accum(rows, ids.data(), ids.size()); });
    ids[0] = 512;
    refuses([&] { (void)Routes::build_accum(rows, ids.data(), ids.size()); });
    std::printf("routes rows=%d hot=%d inverse=PASS unique=PASS bounded_jobs=PASS legacy=PASS\n", rows, int(hot));
}
int main() {
    try {
        for (int32_t rows : {1, 8192, 8193, 16384, 32768, 65536, 65537, 78452, 81920}) {
            test(rows, false); test(rows, true);
        }
        refuses([] { (void)Routes::build_accum(Routes::max_accum_rows + 1, nullptr, 0); });
        refuses([] { (void)Routes::build_accum(0, nullptr, 0); });
        std::puts("moe_accum_routes PASS GPU=UNRUN");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "moe_accum_routes failed: %s\n", error.what());
        return 1;
    }
}
