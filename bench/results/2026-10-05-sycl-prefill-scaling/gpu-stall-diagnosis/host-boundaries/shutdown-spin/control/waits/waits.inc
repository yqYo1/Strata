__dpct_inline__ void wait_flag_ge_kernel(const volatile uint32_t *flag,
                                         uint32_t value) {
    for (uint32_t spin = 0; spin < strata::kSpinMax && strata::sys_load(flag) < value; ++spin) strata_spin_pause();
    /*
    DPCT1078: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}
__dpct_inline__ void wait_flag_ge_or_kernel(const volatile uint32_t *flag,
                                            uint32_t value,
                                            const volatile uint32_t *skip) {
    if (strata::sys_load(skip) == value) return;
    for (uint32_t spin = 0; spin < strata::kSpinMax && strata::sys_load(flag) < value; ++spin) strata_spin_pause();
    /*
    DPCT1078: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}
__dpct_inline__ void doorbell_wait_kernel(const volatile uint32_t *flag,
                                          const volatile uint32_t *seq) {
    const uint32_t want = strata::sys_load(seq);
    for (uint32_t spin = 0; spin < strata::kSpinMax && strata::sys_load(flag) != want; ++spin) strata_spin_pause();
    /*
    DPCT1078: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}
