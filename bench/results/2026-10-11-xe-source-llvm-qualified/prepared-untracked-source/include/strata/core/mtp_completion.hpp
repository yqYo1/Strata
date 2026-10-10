// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <cstdint>
#include <string>
#include <utility>

namespace strata::core {

// Ownership of MTP's mapped records, outputs and upload storage. A rejected
// enqueue may have submitted part of a graph: mark pending BEFORE calling it.
// Checked queue completion releases ownership, but never repairs a failed state.
class MtpCompletion {
public:
    bool healthy(std::string& err) const {
        if (error_.empty()) return true;
        err = error_; return false;
    }
    bool submit(std::string& err) {
        if (!healthy(err)) return false;
        pending_ = true;
        ++submissions_;
        return true;
    }
    void fail(std::string error) {
        if (error_.empty()) error_ = std::move(error);
    }
    bool pending() const { return pending_; }
    bool faulted() const { return !error_.empty(); }
    uint64_t submissions() const { return submissions_; }
    bool host_ready() const { return !pending_ && error_.empty(); }

    // Always check the queues: other production consumers can also use stream().
    // A failed check never clears pending or becomes success on a later drain.
    template<class Wait, class Error>
    bool complete(Wait wait, Error error, std::string& err) {
        bool done = false;
        try { done = wait(); }
        catch (...) {
            pending_ = true;
            fail("mtp: completion callback threw");
            throw;
        }
        if (!done) {
            pending_ = true;
            fail(error());
            err = error_;
            return false;
        }
        pending_ = false;
        return healthy(err);
    }
private:
    bool pending_ = false;
    uint64_t submissions_ = 0;
    std::string error_;
};
} // namespace strata::core
