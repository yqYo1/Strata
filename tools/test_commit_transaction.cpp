// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Host-only fault injection through the verifier's production commit state helper; no GPU or model.
#include "strata/core/commit_transaction.hpp"

#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void expect(bool condition, const char* what) {
    if (!condition) throw std::runtime_error(what);
}
}  // namespace

int main() try {
    using strata::core::CommitTransaction;
    const auto failure = [] { return std::string("injected completion failure"); };
    std::string err;

    // Normal asynchronous completion uses its recorded event; a second transaction cannot use that event
    // before a fresh recording, even though the external event object may still contain the earlier event.
    {
        CommitTransaction tx;
        expect(tx.begin(err) && tx.pending(), "pending must precede submission");
        expect(!tx.begin(err) && tx.pending(), "pending ownership cannot be overwritten");
        tx.recorded();
        int events = 0, queues = 0;
        expect(tx.wait([&] { ++events; return true; }, [&] { ++queues; return true; }, failure, err),
               "successful event completion");
        expect(!tx.pending() && events == 1 && queues == 0, "event completion releases ownership");
        expect(tx.begin(err), "new transaction after successful completion");
        expect(tx.wait([&] { ++events; return true; }, [&] { ++queues; return true; }, failure, err),
               "unrecorded transaction drains queue");
        expect(events == 1 && queues == 1, "earlier event cannot confirm a new transaction");
    }

    // Rejected launch or event recording may follow an accepted state-writing submission. Completion must
    // still be checked via the queue; the original fault must survive that drain and block reuse.
    for (const char* cut : {"injected launch failure", "injected recording failure"}) {
        CommitTransaction tx;
        expect(tx.begin(err), "begin rejected operation");
        tx.fail(cut);
        int events = 0, queues = 0;
        expect(!tx.wait([&] { ++events; return true; }, [&] { ++queues; return true; }, failure, err),
               "drained failure cannot become success");
        expect(!tx.pending() && events == 0 && queues == 1 && err == cut, "queue drain preserves original fault");
        expect(!tx.begin(err), "failed state cannot be reused");
        expect(!tx.wait([&] { ++events; return true; }, [&] { ++queues; return true; }, failure, err),
               "later boundary cannot forget a drained failure");
        expect(events == 0 && queues == 1, "already drained fault needs no additional queue operation");
    }

    // Event wait failure retains ownership. A later wait must drain the queue, not reuse that event.
    {
        CommitTransaction tx;
        expect(tx.begin(err), "begin failing event wait");
        tx.recorded();
        int events = 0, queues = 0;
        expect(!tx.wait([&] { ++events; return false; }, [&] { ++queues; return true; }, failure, err),
               "failed event wait reports failure");
        expect(tx.pending() && events == 1 && queues == 0, "failed event wait must retain ownership");
        expect(!tx.begin(err), "cannot overwrite outstanding state after a failed event wait");
        expect(!tx.wait([&] { ++events; return true; }, [&] { ++queues; return true; }, failure, err),
               "successful later queue drain still reports invalid state");
        expect(!tx.pending() && events == 1 && queues == 1, "retry uses queue completion");
    }

    // Synchronous wait failure and the failed-record-plus-failed-drain cutpoint both keep pending. The
    // verifier fail-stops the latter; the helper must not falsely advertise that its async users retired.
    for (bool recording_failed : {false, true}) {
        CommitTransaction tx;
        expect(tx.begin(err), "begin failing queue wait");
        if (recording_failed) tx.fail("injected recording failure");
        int events = 0, queues = 0;
        expect(!tx.wait([&] { ++events; return true; }, [&] { ++queues; return false; }, failure, err),
               "failed queue drain reports failure");
        expect(tx.pending() && events == 0 && queues == 1, "failed queue drain retains ownership");
        expect(err == (recording_failed ? "injected recording failure" : "injected completion failure"),
               "first failure remains recorded");
        expect(!tx.begin(err), "unconfirmed state cannot be reused");
        expect(!tx.wait([&] { ++events; return true; }, [&] { ++queues; return true; }, failure, err),
               "a later successful drain does not validate failed state");
        expect(!tx.pending() && events == 0 && queues == 2, "only checked queue completion releases ownership");
    }
    std::cout << "commit transaction fault injection passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
