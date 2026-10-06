void MtpDrafter::discard_graphs_after_idle() {
    auto discard = [](auto& graphs) {
        for (auto& graph : graphs) {
            delete graph;
            graph = nullptr;
        }
    };
    discard(prefill_exec_);
    discard(prefill_dev_exec_);
    discard(round_exec_);
    discard(step_exec_);
    discard(round_exec_c_);
    discard(step_exec_c_);
}
bool MtpDrafter::suspend_decode_weights(std::string& err) try {
    if (!release_decode_weights_ || decode_weights_suspended_) return true;
    const OnDevice on_device(device_);
    const auto started = Clock::now();
    dpct::get_current_device().queues_wait_and_throw();
    const double wait_ms = ms_since(started);
    if (expert_host_.size() != (size_t) expert_storage_.full_bytes() ||
        head_host_.size() != (size_t) head_storage_.full_bytes()) {
        err = "mtp: decode weights have no complete immutable RAM source"; return false;
    }
    const auto verify_started = Clock::now();
    if (!verify_decode_payload(err)) return false;
    const double verify_ms = ms_since(verify_started);
    const auto graph_started = Clock::now();
    // Backend command lists can retain allocation pointers as well as VAs.
    // Drop those references before destroying physical allocations and
    // recapture after restoration. Same-address copies alone cannot check
    // backend residency; the Oct 7 repeated lease crashed in that path.
    discard_graphs_after_idle();
    const double graph_ms = ms_since(graph_started);
    const int64_t physical = expert_storage_.mapped_bytes() + head_storage_.mapped_bytes();
    // A partially successful unmap must also be restored before any consumer.
    decode_weights_suspended_ = true;
    const auto unmap_started = Clock::now();
    if (!expert_storage_.shrink(0, err) ||
        (head_storage_.full_bytes() && !head_storage_.shrink(0, err))) return false;
    const double unmap_ms = ms_since(unmap_started);
    std::fprintf(stderr, "strata mtp decode release: %lld physical bytes, experts and head; K/V retained; "
                         "total %.3f ms, wait %.3f ms, unmap %.3f ms, verify %.3f ms, verified=%d, graph_drop %.3f ms\n",
                 (long long) physical, ms_since(started), wait_ms, unmap_ms, verify_ms, verify_decode_weights_, graph_ms);
    return true;
}
catch (const std::exception& e) {
    err = std::string("mtp decode release: ") + e.what(); return false;
}
bool MtpDrafter::restore_decode_weights(std::string& err) try {
    if (!decode_weights_suspended_) return true;
    const OnDevice on_device(device_);
    const auto started = Clock::now();
    if (!expert_storage_.grow(expert_storage_.full_bytes(), err) ||
        (head_storage_.full_bytes() && !head_storage_.grow(head_storage_.full_bytes(), err))) return false;
    const double map_ms = ms_since(started);
    if (expert_storage_.device_slot(0) != experts_ ||
        (head_storage_.full_bytes() && head_storage_.device_slot(0) != dhead_)) {
        err = "mtp: decode restoration changed a captured virtual address"; return false;
    }
    auto& q = dpct::get_in_order_queue();
    const auto copy_started = Clock::now();
    q.memcpy(experts_, expert_host_.data(), expert_host_.size());
    if (!head_host_.empty()) q.memcpy(dhead_, head_host_.data(), head_host_.size());
    q.wait_and_throw();
    const double copy_ms = ms_since(copy_started);
    const auto verify_started = Clock::now();
    if (!verify_decode_payload(err)) return false;
    const double verify_ms = ms_since(verify_started);
    decode_weights_suspended_ = false;
    std::fprintf(stderr, "strata mtp decode restore: %llu payload bytes from RAM, same addresses; "
                         "total %.3f ms, map %.3f ms, copy %.3f ms, verify %.3f ms, verified=%d\n",
                 (unsigned long long) (expert_host_.size() + head_host_.size()), ms_since(started),
                 map_ms, copy_ms, verify_ms, verify_decode_weights_);
    return true;
}
catch (const std::exception& e) {
    err = std::string("mtp decode restoration: ") + e.what(); return false;
}
bool Verifier::discard_cache_graphs(std::string& err) try {
    if (next_) {
        err = "verify cache graph release requires a single GPU";
        return false;
    }
    const OnDevice on(device_);
    if (!wait_commit(err)) return false;
    cs_->wait_and_throw();
    copy_->wait_and_throw();
    for (auto& graph : exec_) { delete graph; graph = nullptr; }
    for (auto& graphs : boundary_graphs_) {
        graphs.input.reset(); graphs.tail.reset();
        graphs.pre.clear(); graphs.post.clear();
    }
    return true;
}
catch (const std::exception& e) {
    err = std::string("verify cache graph release: ") + e.what();
    return false;
}
bool Verifier::rebuild_cache_graphs(const uint8_t* address, std::string& err) try {
    if (!address) {
        err = "verify cache graph rebuild requires a restored cache";
        return false;
    }
    const OnDevice on(device_);
    if (!discard_cache_graphs(err)) return false;
    hits_.cache_base = address;
    // Recording submits nodes to a graph, rather than executing the window.
    // Full-model checks still verify that recapture preserves persistent state.
    return warm(err);
}
catch (const std::exception& e) {
    err = std::string("verify cache graph rebuild: ") + e.what();
    return false;
}
