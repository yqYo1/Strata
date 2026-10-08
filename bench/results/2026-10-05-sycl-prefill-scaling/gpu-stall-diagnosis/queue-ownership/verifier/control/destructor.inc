Verifier::~Verifier() try {
    // SYCL port: at process exit the Level Zero context can already be gone (serve mode's end), and a destructor
    // that throws aborts the process (exit 139); the waits and frees below are best-effort then.
    unregister_live_verifier(this);
    if (cs_) cs_->wait();
    if (copy_) copy_->wait();
    if (host_staging_) sycl::free(host_staging_, dpct::get_in_order_queue());
    for (auto& e : exec_)
        if (e) delete (e);
    if (commit_exec_) delete (commit_exec_);
    if (cs_) dpct::get_current_device().destroy_queue(cs_);
    if (copy_) {
        copy_->wait(); dpct::get_current_device().destroy_queue(copy_);
    }
    if (arena_) sycl::free(arena_, dpct::get_in_order_queue());
    void* hosts[] = {h_tok_, h_step_, h_pos_, h_commit_, h_ple_, h_out_, h_x_, h_ids_, h_w_, h_seq_, h_flag_, h_ymiss_,
                     h_flagA_, h_plan_, h_flagB_};
    for (void* h : hosts)
        if (h) sycl::free(h, dpct::get_in_order_queue());
} catch (...) {
}
