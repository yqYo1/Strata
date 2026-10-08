inline dpct::global_memory<int8_t, 1>& iq4nl_values = *new dpct::global_memory<int8_t, 1>(sycl::range(16), {-127, -104, -83, -65, -49, -35, -22, -10, 1,
                                   13, 25, 38, 53, 69, 89, 113});
inline dpct::constant_memory<float, 2>& c_codes = *new dpct::constant_memory<float, 2>(256, 4);
