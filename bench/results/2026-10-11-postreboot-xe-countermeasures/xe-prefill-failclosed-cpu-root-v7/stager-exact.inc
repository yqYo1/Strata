struct Stager {
    // D-5: the pinned ring's depth (STRATA_STAGER_RING, default 16) - how far the host copies can run ahead of the
    // DMAs of the unpinned experts' blobs
    int kRing = 16;
    // `from` set: a transient source (the GGUF read in place) copies the blob of (l, e) itself (copy_blob), since its
    // blob() pointers do not outlast a chunk's queue
    struct Job { const uint8_t* src; size_t bytes; core::ExpertSource* from = nullptr; int32_t l = 0, e = 0; };
    std::atomic<bool> failed{false}, cancelled{false};
    publication::Failure failure;
    std::vector<uint8_t*> buf;
    std::vector<char> pinned;
    std::vector<std::vector<uint8_t>> pageable;   // the fallback when no more RAM can be pinned
    std::vector<strata::gpu::Event*> dma_done;
    std::vector<Job> jobs;
    std::unique_ptr<std::atomic<int>[]> ready;
    size_t ready_cap = 0;
    // gen << 32 | n << 16 | next index: a claim is a CAS on the generation it woke for (a thread late from the
    // previous layer can never take a job of this one - the expert pool's issue #29 lesson)
    std::atomic<uint64_t> head{0};
    std::atomic<int> issued{0}, active{0};
    uint32_t gen = 0;
    bool quit = false;
    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::thread> threads;
    int device = 0;

    bool init(size_t blob_bytes, int nthreads) {
        if (const char* v = std::getenv("STRATA_STAGER_RING")) kRing = (int) std::clamp<long>(std::strtol(v, nullptr, 10), 2, 256);
        buf.assign((size_t) kRing, nullptr);
        pinned.assign((size_t) kRing, 0);
        dma_done.assign((size_t) kRing, nullptr);
        pageable.resize(kRing);
        for (int i = 0; i < kRing; ++i) {
            pinned[i] = (char) (!force_pageable() && strata::gpu::alloc_host((void**) &buf[i], blob_bytes));
            if (!pinned[i]) {
                pageable[(size_t) i].resize(blob_bytes);
                buf[i] = pageable[(size_t) i].data();
            }
            if (!strata::gpu::event_create(&dma_done[i])) return false;
        }
        device = strata::core::current_device();
        for (int t = 0; t < nthreads; ++t) threads.emplace_back([this] { work(); });
        return true;
    }
    ~Stager() {
        finish();
        { std::lock_guard<std::mutex> lk(mu); quit = true; }
        cv.notify_all();
        for (auto& t : threads) t.join();
        for (int i = 0; i < kRing; ++i) {
            if (dma_done[i]) strata::gpu::event_destroy(dma_done[i]);
            if (buf[i] && pinned[i]) strata::gpu::free(buf[i]);
        }
    }
    void work() {
        const strata::core::OnDevice on_device(device);   // the stage's GPU, on this worker thread too
        uint32_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return quit || gen != seen; });
                if (quit) return;
                seen = gen;
            }
            for (;;) {
                active.fetch_add(1, std::memory_order_acq_rel);
                const int j = claim(seen);
                if (j < 0) { active.fetch_sub(1, std::memory_order_acq_rel); break; }
                const int b = j % kRing;
                if (j >= kRing)   // job j - kRing's DMA from this buffer is queued: sleep until it is (upstream #1057:
                                  // with --mmap-experts the 32 spinning stagers starved the launching thread)
                    for (int v = issued.load(std::memory_order_acquire); v <= j - kRing;
                         v = issued.load(std::memory_order_acquire)) {
                        if (cancelled.load(std::memory_order_acquire)) break;
                        if (stager_sleep()) issued.wait(v, std::memory_order_acquire);   // STRATA_STAGER_SLEEP=0: spin
                        else std::this_thread::yield();
                    }
                if (cancelled.load(std::memory_order_acquire)) { active.fetch_sub(1, std::memory_order_acq_rel); break; }
                try {
                // and done - for a generation's first kRing jobs that is the previous generation's last DMA from
                // the buffer, which nothing else waits for when a chunk ends without a sync (no MTP) or the DMA
                // was a ring entry the routing skipped (an event never recorded returns at once)
                publication::require(strata::gpu::event_sync(dma_done[b]), "strata::gpu::event_sync(dma_done[b])", strata::gpu::last_error);
                const Job& jb = jobs[(size_t) j];
                if (jb.from != nullptr) {
                    if (!jb.from->copy_blob(jb.l, jb.e, buf[b])) failed.store(true, std::memory_order_release);
                } else {
                    std::memcpy(buf[b], jb.src, jb.bytes);
                }
                } catch (...) { failure.capture(); failed.store(true, std::memory_order_release); cancel(); }
                ready[(size_t) j].store(1, std::memory_order_release);
                active.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
    }
    int claim(uint32_t g) {
        uint64_t cur = head.load(std::memory_order_acquire);
        for (;;) {
            if ((uint32_t) (cur >> 32) != g) return -1;
            const int n = (int) ((cur >> 16) & 0xffff), j = (int) (cur & 0xffff);
            if (j >= n) return -1;
            if (head.compare_exchange_weak(cur, cur + 1, std::memory_order_acq_rel, std::memory_order_acquire)) return j;
        }
    }
    /// A layer's jobs; the previous layer's are finished (finish()).
    void start(std::vector<Job>&& js) {
        // A worker exception makes this instance terminal. Reconstruct only
        // after the owning Prefill has drained both queues and released it.
        failure.rethrow();
        if (js.empty()) return;
        cancelled.store(false);
        failed.store(false);
        std::lock_guard<std::mutex> lk(mu);
        jobs = std::move(js);
        if (ready_cap < jobs.size()) {
            ready_cap = jobs.size() * 2;
            ready.reset(new std::atomic<int>[ready_cap]);
        }
        for (size_t i = 0; i < jobs.size(); ++i) ready[i].store(0, std::memory_order_relaxed);
        issued.store(0);
        ++gen;
        head.store((uint64_t) gen << 32 | (uint64_t) jobs.size() << 16, std::memory_order_release);
        cv.notify_all();
    }
    /// Job j's bytes, in a pinned buffer (waits for the copy).
    const uint8_t* wait(int j) {
        while (!ready[(size_t) j].load(std::memory_order_acquire)) {
            if (cancelled.load(std::memory_order_acquire)) { failure.rethrow(); throw std::runtime_error("stager aborted"); }
            std::this_thread::yield();
        }
        failure.rethrow();
        if (failed.load(std::memory_order_acquire)) throw std::runtime_error("stager copy or event failed");
        return buf[j % kRing];   // a failed copy_blob leaves `failed` set: the chunk ends in an error
    }
    /// The launching thread queued job j's DMA on `copy`: its buffer is free once that is done.
    void issued_one(int j, strata::gpu::Stream copy) {
        publication::require(strata::gpu::event_record(dma_done[j % kRing], copy), "strata::gpu::event_record(dma_done[j % kRing], copy)", strata::gpu::last_error);
        publication::publish_staged(issued, j + 1);
    }
    void cancel() {
        cancelled.store(true, std::memory_order_release);
        issued.store(1 << 30, std::memory_order_release);
        issued.notify_all();
    }
    /// No job is running after this (the end of a layer, or an early return in the middle of one).
    void finish() {
        head.store((uint64_t) gen << 32, std::memory_order_release);   // n = 0: nothing more to claim
        issued.store(1 << 30, std::memory_order_release);
        issued.notify_all();
        while (active.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    }
};
