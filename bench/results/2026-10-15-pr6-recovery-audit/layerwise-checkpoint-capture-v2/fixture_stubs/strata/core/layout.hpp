#pragma once
// CPU CONTRACT FIXTURE ONLY. Never place this directory on a product build path.
#include <cstdint>
namespace strata::core {
struct ModelGeometry {
    int64_t n_layers = 48, qsa_interval = 4;
    int64_t n_qsa_layers() const { return n_layers / qsa_interval; }
    int64_t n_gdn_layers() const { return n_layers - n_qsa_layers(); }
};
inline bool is_qsa_layer(const ModelGeometry& g, int64_t l) {
    return l % g.qsa_interval == g.qsa_interval - 1;
}
}
