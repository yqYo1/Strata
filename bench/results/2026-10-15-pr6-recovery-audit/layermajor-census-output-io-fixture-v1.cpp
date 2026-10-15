#include "layermajor_shape_census.hpp.in"
#include <cassert>
#include <cstring>
#include <memory>

int main() {
    using C = strata::prefill::detail::LayerMajorShapeCensus;
    // Artificial completed metadata exercises only the output failure contract.
    // It is not inference/census qualification or a valid routing fixture.
    auto c = std::make_unique<C>(1, 0, 1);
    c->retired = C::layers;
    FILE* out = std::fopen("/dev/full", "w");
    assert(out);
    c->complete(out);
    assert(c->finalized && c->failure);
    assert(std::strcmp(c->failure, "output_io_incomplete") == 0);
    std::fclose(out);
    std::puts("PASS: output I/O failure remains unqualified");
}
