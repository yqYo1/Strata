// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <vector>
#include <utility>

namespace strata::core {
// Host-only staging shared by NativeHead and its CPU fault fixture. A failed or
// throwing submission/wait has no completion guarantee. Retain both allocations
// and the original source; do not retry, free, or unwind those owners.
class HeadUploadOwner {
public:
    using Payload = std::vector<uint8_t>;
    HeadUploadOwner() = default;
    HeadUploadOwner(const HeadUploadOwner&) = delete;
    HeadUploadOwner& operator=(const HeadUploadOwner&) = delete;
    ~HeadUploadOwner() { if (uncertain_) std::terminate(); }
    static std::shared_ptr<const Payload> snapshot(const void* source, size_t bytes) {
        auto payload = std::make_shared<Payload>(bytes);
        std::memcpy(payload->data(), source, bytes);
        return payload;
    }
    template<class Alloc, class Free, class Upload, class Convert>
    bool stage(std::shared_ptr<const void> source_owner, const void* source,
               size_t bytes, size_t scratch_bytes, Alloc alloc, Free free,
               Upload upload, Convert convert) {
        if (uncertain_ || weights_ || scratch_ || !source_owner || !source) return false;
        source_owner_ = std::move(source_owner);
        source_ = source; bytes_ = bytes; scratch_bytes_ = scratch_bytes;
        try {
            if (!alloc(&weights_, bytes) || !alloc(&scratch_, scratch_bytes)) {
                clear(free); return false; // no submission has occurred
            }
            uncertain_ = true; // set BEFORE a callback can submit or throw
            if (!upload(weights_, source_, bytes)) return false;
            uncertain_ = false;
            uncertain_ = true;
            if (!convert(weights_)) return false;
            uncertain_ = false;
            source_owner_.reset(); source_ = nullptr;
            return true;
        } catch (...) {
            if (!uncertain_) clear(free);
            throw;
        }
    }
    template<class Free> bool clear(Free free) {
        if (uncertain_) return false;
        auto release = [&](void*& pointer) {
            if (!pointer) return;
            // free may throw before or after releasing storage. Never retry an
            // uncertain release, and clear each confirmed pointer individually.
            uncertain_ = true;
            free(pointer);
            pointer = nullptr;
            uncertain_ = false;
        };
        release(scratch_);
        release(weights_);
        source_owner_.reset(); source_ = nullptr;
        bytes_ = scratch_bytes_ = 0;
        return true;
    }
    bool uncertain() const { return uncertain_; }
    void* weights() const { return weights_; }
    void* scratch() const { return scratch_; }
    const void* source() const { return source_; }
    size_t bytes() const { return bytes_; }
    size_t scratch_bytes() const { return scratch_bytes_; }
private:
    std::shared_ptr<const void> source_owner_;
    const void* source_ = nullptr;
    void* weights_ = nullptr;
    void* scratch_ = nullptr;
    size_t bytes_ = 0, scratch_bytes_ = 0;
    bool uncertain_ = false;
};
}  // namespace strata::core
