template <class T> class HostVec {
public:
    HostVec() = default;
    HostVec(const HostVec&) = delete;
    HostVec& operator=(const HostVec&) = delete;
    ~HostVec() { release(); }
    void resize(size_t n) {
        if (n > cap_) {
            release();
            if (!force_pageable() && strata::gpu::alloc_host(&pinned_, n * sizeof(T))) cap_ = n;
            else { pinned_ = nullptr; fallback_.resize(n); cap_ = n; }
        }
        n_ = n;
    }
    T* data() { return pinned_ != nullptr ? pinned_ : fallback_.data(); }
    size_t size() const { return n_; }
    T& operator[](size_t i) { return data()[i]; }
    void from_device(const T* src, strata::gpu::Stream s) {   // the first size() values, queued on `s`
        if (pinned_ != nullptr) strata::kernels::copy_i32_from_mapped(pinned_, src, (int64_t) n_, s);
        else publication::require(strata::gpu::copy_async(fallback_.data(), src, n_ * sizeof(T), s), "HostVec from_device", strata::gpu::last_error);
    }
    void to_device(T* dst, strata::gpu::Stream s) {
        if (pinned_ != nullptr) strata::kernels::copy_i32_from_mapped(dst, pinned_, (int64_t) n_, s);
        else publication::require(strata::gpu::copy_async(dst, fallback_.data(), n_ * sizeof(T), s), "HostVec to_device", strata::gpu::last_error);
    }
private:
    static_assert(sizeof(T) == 4, "copied as int32");
    void release() {
        if (pinned_ != nullptr) strata::gpu::free(pinned_);
        pinned_ = nullptr;
        fallback_.clear();
        cap_ = n_ = 0;
    }
    T* pinned_ = nullptr;
    std::vector<T> fallback_;
    size_t cap_ = 0, n_ = 0;
};
