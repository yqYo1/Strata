// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Host-only failure boundary of the actual production template. No Runtime or SYCL operation is called.
#include "parity_device.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <vector>

namespace {
enum class Case { happy, standard, unknown };
constexpr size_t kBytes = 1024;
constexpr uint64_t kChecksum = 130560;   // four complete 0..255 sequences
char stderr_buffer[4096];               // static lifetime, including the atexit witness

const char* name(Case which) {
    switch (which) {
        case Case::happy: return "happy";
        case Case::standard: return "standard";
        case Case::unknown: return "unknown";
    }
    return "invalid";
}
void exit_witness() { std::fputs("ATEXIT\n", stderr); }

struct CallerStaging {
    const char* label;
    std::vector<unsigned char> bytes;
    explicit CallerStaging(const char* which) : label(which), bytes(kBytes, 0xa5) {}
    ~CallerStaging() {
        std::fprintf(stderr, "STAGING_DTOR case=%s bytes=%zu\n", label, bytes.size());
    }
};

int exercise(Case which) {
    CallerStaging staging(name(which));
    int callbacks = 0;
    const char* operation = which == Case::happy ? "host happy" :
                            which == Case::standard ? "host standard" : "host unknown";
    std::fprintf(stderr, "START case=%s\n", name(which));
    // The production header supplies checked_transfer and transfer_fatal; neither body is duplicated here.
    strata::parity::checked_transfer(operation, [&] {
        ++callbacks;
        for (size_t i = 0; i < staging.bytes.size(); ++i)
            staging.bytes[i] = static_cast<unsigned char>(i % 256);
        std::fprintf(stderr, "CALLBACK case=%s count=%d\n", name(which), callbacks);
        if (which == Case::standard) {
            std::fputs("PRE_FAILURE case=standard\n", stderr);
            throw std::runtime_error("injected standard failure");
        }
        if (which == Case::unknown) {
            std::fputs("PRE_FAILURE case=unknown\n", stderr);
            throw 17;
        }
        std::fputs("CALLBACK_RETURN case=happy\n", stderr);
    });
    // Failure callbacks must terminate inside production before this statement or the caller destructor.
    std::fprintf(stderr, "AFTER_CALL case=%s\n", name(which));
    if (which != Case::happy) {
        std::fputs("UNEXPECTED_FAILURE_RETURN\n", stderr);
        return 4;
    }
    uint64_t checksum = 0;
    for (size_t i = 0; i < staging.bytes.size(); ++i) {
        if (staging.bytes[i] != static_cast<unsigned char>(i % 256)) {
            std::fputs("HAPPY_CHECK_FAILED staging\n", stderr);
            return 4;
        }
        checksum += staging.bytes[i];
    }
    if (callbacks != 1 || checksum != kChecksum) {
        std::fputs("HAPPY_CHECK_FAILED callback count or checksum\n", stderr);
        return 4;
    }
    std::fprintf(stderr, "HAPPY_OK callback_count=%d checksum=%llu\n", callbacks,
                 static_cast<unsigned long long>(checksum));
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    Case which;
    if (argc != 2) {
        std::fputs("usage: parity_failure_host happy|standard|unknown\n", stderr);
        return 2;
    }
    if (std::strcmp(argv[1], "happy") == 0) which = Case::happy;
    else if (std::strcmp(argv[1], "standard") == 0) which = Case::standard;
    else if (std::strcmp(argv[1], "unknown") == 0) which = Case::unknown;
    else {
        std::fputs("usage: parity_failure_host happy|standard|unknown\n", stderr);
        return 2;
    }
    // No failure callback flushes stderr. The small buffered pre-failure markers become visible only if
    // the actual production fatal path flushes them before _Exit (which does not perform stdio cleanup).
    if (std::setvbuf(stderr, stderr_buffer, _IOFBF, sizeof(stderr_buffer)) != 0) {
        std::fputs("HARNESS_SETUP_FAILED stderr buffering\n", stderr);
        std::fflush(stderr);
        return 2;
    }
    if (std::atexit(exit_witness) != 0) {
        std::fputs("HARNESS_SETUP_FAILED atexit registration\n", stderr);
        std::fflush(stderr);
        return 2;
    }
    try {
        const int result = exercise(which);
        std::fprintf(stderr, "MAIN_RETURN case=%s status=%d\n", name(which), result);
        std::fflush(stderr);
        return result;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ESCAPED_TO_CALLER standard: %s\n", error.what());
        std::fflush(stderr);
        return 3;
    } catch (...) {
        std::fputs("ESCAPED_TO_CALLER unknown\n", stderr);
        std::fflush(stderr);
        return 3;
    }
}
