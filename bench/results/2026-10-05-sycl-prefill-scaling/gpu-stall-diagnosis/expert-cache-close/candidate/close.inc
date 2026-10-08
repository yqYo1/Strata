void ExpertCache::close() {
#if defined(STRATA_USE_HIP)
    if (blocking_staging_) (void) cudaFreeHost(blocking_staging_);
    blocking_staging_ = nullptr;
    blocking_staging_bytes_ = 0;
#endif
    off_.clear();
    if (!segs_.empty()) {
        release_segmented();
    } else if (base_ != nullptr) {
        // sycl::free does not synchronize like CUDA's ordinary cudaFree.
        // Cache fills/readers can use other queues, so drain every queue on
        // this device before reclaiming their payload. A failed wait must not
        // proceed to free memory which may still be in use.
        dpct::get_current_device().queues_wait_and_throw();
        sycl::free(base_, dpct::get_in_order_queue());
        base_ = nullptr;
    }
    residency_.clear();
    slots_ = 0;
    live_slots_ = 0;
    n_layers_ = 0;
    n_expert_ = 0;
    blob_ = 0;
    next_free_ = 0;
    fills_ = 0;
    admitted_ = 0;
    layer_next_.clear();
}
