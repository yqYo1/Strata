// Included in strata::kernels::cpu, outside the anonymous namespace.
bool iq256_iq3s_rowpair_gu_rows(int type, const uint8_t* blob, size_t row, size_t up_off,
                               int n, const void* act, float* ff, int r0, int r1) {
    // This v1 intentionally accepts only the actual 2560 / 640 packed GU geometry.
    // Caller still owns/proves full blob and activation extents and disjoint output.
    if (type != 21 || n != 2560 || row != 1100 || up_off != 640*size_t(1100) ||
        r0 < 0 || r1 < r0 || r1 > 640) return false;
    if (r0 == r1) return true;
    if (!blob || !act || !ff) return false;
    int r = r0;
    for (; r1 - r >= 2; r += 2) {
        float g[2], u[2];
        iq3s_pair_dot<2>(blob + size_t(r)*row, row, act, g);
        iq3s_pair_dot<2>(blob + up_off + size_t(r)*row, row, act, u);
        ff[r] = (g[0] / (1.f + std::exp(-g[0]))) * u[0];
        ff[r+1] = (g[1] / (1.f + std::exp(-g[1]))) * u[1];
    }
    if (r < r1) {
        float g, u;
        iq3s_pair_dot<1>(blob + size_t(r)*row, row, act, &g);
        iq3s_pair_dot<1>(blob + up_off + size_t(r)*row, row, act, &u);
        ff[r] = (g / (1.f + std::exp(-g))) * u;
    }
    return true;
}
