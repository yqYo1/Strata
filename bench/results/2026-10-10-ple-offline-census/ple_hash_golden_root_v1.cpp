#include "strata/kernels/ngram.hpp"
#include "ple_oracle_vectors.inc"
#include <array>
#include <cstdio>
#include <vector>
int main() {
    namespace k = strata::kernels;
    const auto c = k::ple_artifact_consts();
    size_t checked = 0;
    for (const auto& test : k::ple_oracle::kHashCases) {
        std::vector<uint32_t> actual(size_t(test.n_tokens) * k::PLE_N_HEADS);
        k::ngram_rows(test.tokens, test.prev, test.n_tokens, c, actual.data());
        for (size_t i=0; i<actual.size(); ++i) {
            if (actual[i] != test.rows[i]) {
                std::fprintf(stderr, "golden mismatch case=%s row=%zu actual=%u reference=%u\n", test.name, i, actual[i], test.rows[i]);
                return 1;
            }
            ++checked;
        }
    }
    std::printf("{\"passed\":true,\"golden_cases\":%d,\"row_ids_checked\":%zu,\"GPU_submitted\":false}\n", k::ple_oracle::kHashCaseCount, checked);
}
