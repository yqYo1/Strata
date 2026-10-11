// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <string>
#include <utility>

namespace strata::core {

// Host-side ownership of an asynchronous state write. A completion event is usable only after it was
// recorded for this transaction; an earlier event must never stand in for a rejected recording.
// Failure poisons the state even after a later drain succeeds: device and host state may have diverged.
// Reconstruct the verifier/session after a failure rather than publishing that state as valid.
class CommitTransaction {
public:
    bool begin(std::string& err) {
        if (!error_.empty()) { err = error_; return false; }
        if (pending_) { err = "verify: a commit is still pending"; return false; }
        pending_ = true;   // set before any submission that might modify session state
        recorded_ = false;
        return true;
    }

    void recorded() { recorded_ = true; }
    void fail(std::string error) {
        if (error_.empty()) error_ = std::move(error);
    }
    bool pending() const { return pending_; }

    // The callbacks are the real event/queue operations in production and fault cutpoints in host tests.
    // Preserve pending on failed completion; the next attempt drains the queue rather than reusing an
    // event whose wait failed. A successful drain clears ownership, but never clears a recorded fault.
    template <class EventWait, class QueueWait, class ErrorText>
    bool wait(EventWait event_wait, QueueWait queue_wait, ErrorText error_text, std::string& err) {
        if (pending_) {
            if (!(recorded_ ? event_wait() : queue_wait())) {
                recorded_ = false;
                fail(error_text());
                err = error_;
                return false;
            }
            pending_ = false;
            recorded_ = false;
        }
        if (!error_.empty()) { err = error_; return false; }
        return true;
    }

private:
    bool pending_ = false;
    bool recorded_ = false;
    std::string error_;
};

}  // namespace strata::core
