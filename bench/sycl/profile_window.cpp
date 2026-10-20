// Main-authored recording-window harness; repository LICENSE applies.
// Reuse the maintained fixture without changing its operations or lifetimes.
#define main strata_profile_fixture_main
#include "../../tests/sycl/compat.cpp"
#undef main

#include <chrono>
#include <cerrno>
#include <iomanip>
#include <sstream>
#include <poll.h>
#include <time.h>
#include <unistd.h>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include "strata/sycl/runtime.hpp"

namespace {
uint64_t raw_ns() {
    timespec value{};
    if (clock_gettime(CLOCK_MONOTONIC_RAW,&value)!=0)
        throw std::runtime_error("profile window raw clock");
    return uint64_t(value.tv_sec)*1000000000ULL+uint64_t(value.tv_nsec);
}

void clock_samples(const char* phase,ze_device_handle_t device) {
    for (int i=0;i<4;++i) {
        uint64_t host=0,ticks=0;
        const auto before=raw_ns();
        const auto status=zeDeviceGetGlobalTimestamps(device,&host,&ticks);
        const auto after=raw_ns();
        if (status!=ZE_RESULT_SUCCESS) throw std::runtime_error("profile window paired clocks");
        std::cout << "profile_clock {\"phase\":\"" << phase << "\",\"before_ns\":" << before
                  << ",\"host_ns\":" << host << ",\"device_ticks\":" << ticks
                  << ",\"after_ns\":" << after << "}\n";
    }
}

void checkpoint(const char* phase) {
    std::cout << "profile_window phase=" << phase << " pid=" << getpid()
              << " raw_ns=" << raw_ns() << '\n' << std::flush;
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    std::string input;
    for (;;) {
        const auto remaining=std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline-std::chrono::steady_clock::now()).count();
        if (remaining<=0) throw std::runtime_error("profile window control deadline");
        pollfd descriptor{STDIN_FILENO,POLLIN,0};
        const int status=poll(&descriptor,1,static_cast<int>(remaining));
        if (status<0 && errno==EINTR) continue;
        if (status<=0 || !(descriptor.revents&POLLIN))
            throw std::runtime_error("profile window control unavailable");
        char byte=0;
        if (read(STDIN_FILENO,&byte,1)!=1) throw std::runtime_error("profile window control EOF");
        if (byte=='\n') break;
        if (input.size()>=16) throw std::runtime_error("profile window command extent");
        input+=byte;
    }
    if (input!="continue") throw std::runtime_error("profile window command");
}
}

int main(int argc,char** argv) {
    try {
        const auto runtime=strata::sycl_backend::runtime_for();
        const auto device=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(runtime->device());
        ze_device_properties_t properties{};
        properties.stype=ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2;
        if (zeDeviceGetProperties(device,&properties)!=ZE_RESULT_SUCCESS || !properties.timerResolution ||
            !properties.timestampValidBits || properties.timestampValidBits>64 ||
            !properties.kernelTimestampValidBits || properties.kernelTimestampValidBits>64)
            throw std::runtime_error("profile window timer properties");
        std::ostringstream uuid;
        for (const auto byte:properties.uuid.id) uuid << std::hex << std::setfill('0') << std::setw(2) << unsigned(byte);
        std::cout << "profile_timer {\"vendor_id\":" << properties.vendorId
                  << ",\"device_id\":" << properties.deviceId << ",\"uuid\":\"" << uuid.str()
                  << "\",\"frequency_hz\":" << properties.timerResolution
                  << ",\"timestamp_bits\":" << properties.timestampValidBits
                  << ",\"kernel_timestamp_bits\":" << properties.kernelTimestampValidBits << "}\n";
        clock_samples("ready",device);
        checkpoint("ready");
        const int status=strata_profile_fixture_main(argc,argv);
        std::cout << "profile_window fixture_status=" << status << '\n' << std::flush;
        clock_samples("drained",device);
        checkpoint("drained");
        return status;
    } catch (const std::exception& error) {
        std::cerr << "profile window: " << error.what() << '\n';
        return 125;
    }
}
