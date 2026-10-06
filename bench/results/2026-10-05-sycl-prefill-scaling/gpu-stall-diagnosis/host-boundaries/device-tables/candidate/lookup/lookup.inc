__dpct_inline__ sycl::int2 iq4_table_lookup(int q4, int8_t *iq4nl_values) {
    // Copy bytes into word objects: the codebook's int8_t storage neither
    // promises uint32_t alignment nor permits a uint32_t aliasing load.
    uint32_t table32[4];
    __builtin_memcpy(table32, iq4nl_values, sizeof table32);
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = 0x32103210 | ((q4 & 0x88888888) >> 1);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low =
            dpct::byte_level_permute(table32[0], table32[1], q4 >> shift);
        const uint32_t high =
            dpct::byte_level_permute(table32[2], table32[3], q4 >> shift);
        tmp[i] = dpct::byte_level_permute(low, high,
                                          low_high_selection_indices >> shift);
    }
    return sycl::int2(dpct::byte_level_permute(tmp[0], tmp[1], 0x6420),
                      dpct::byte_level_permute(tmp[0], tmp[1], 0x7531));
}
