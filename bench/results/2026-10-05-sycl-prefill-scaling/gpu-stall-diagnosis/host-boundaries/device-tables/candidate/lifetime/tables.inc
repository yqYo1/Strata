inline dpct::global_memory<int8_t, 1>& iq4nl_storage() {
    // Retain the table for captured graphs; initialize only at its first use.
    static auto* table = new dpct::global_memory<int8_t, 1>(sycl::range(16),
        {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113});
    return *table;
}
inline dpct::constant_memory<float, 2>& s2_codes_storage() {
    // Captured kernels retain the codebook; avoid creating a GPU before main.
    static auto* table = new dpct::constant_memory<float, 2>(256, 4);
    return *table;
}
