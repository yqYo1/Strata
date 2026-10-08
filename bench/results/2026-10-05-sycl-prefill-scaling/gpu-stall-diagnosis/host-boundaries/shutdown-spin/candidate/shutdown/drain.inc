static void drain_sycl_devices() {
    // Another device's copy queue may still use the primary's allocations.
    for (unsigned int device = 0; device < dpct::device_count(); ++device)
        dpct::get_device(device).queues_wait_and_throw();
}
