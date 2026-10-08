void wait_flag_ge(const uint32_t*, uint32_t, void*) {
    throw std::logic_error("SYCL wait_flag_ge is disabled; use host completion before submission");
}
void wait_flag_ge_or(const uint32_t*, uint32_t, const uint32_t*, void*) {
    throw std::logic_error("SYCL wait_flag_ge_or is disabled; use host completion before submission");
}
void doorbell_wait(const uint32_t*, const uint32_t*, void*) {
    throw std::logic_error("SYCL doorbell_wait is disabled; use per-layer host completion");
}
