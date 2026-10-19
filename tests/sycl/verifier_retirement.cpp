#include "strata/core/verify.hpp"
#include "strata/sycl/runtime.hpp"

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace strata::core {
// Access only to install small real queue/USM owners without loading a model.
// The tested submit, finish, query and destruction methods are production code.
struct VerifierRetirementTestAccess {
    static constexpr size_t arena_bytes = 16 * 64;
    static void prepare(Verifier& v, size_t slot_bytes = 64, int slots = 2) {
        if (!slot_bytes || slot_bytes > (std::numeric_limits<size_t>::max()-128)/16)
            throw std::runtime_error("test extent overflow");
        if (cudaStreamCreate(&v.copy_) != cudaSuccess ||
            cudaStreamCreate(&v.cs_) != cudaSuccess ||
            cudaMalloc(&v.arena_, 16*slot_bytes+128) != cudaSuccess ||
            cudaMallocHost(reinterpret_cast<void**>(&v.h_flagB_), sizeof(uint32_t)) != cudaSuccess)
            throw std::runtime_error("test owner admission");
        if (cudaMemset(v.arena_,0xa5,16*slot_bytes+128)!=cudaSuccess)
            throw std::runtime_error("test device guards");
        v.staging_ = static_cast<uint8_t*>(v.arena_)+64;
        v.staging_blob_bytes_ = slot_bytes;
        v.sink_.staging = reinterpret_cast<uintptr_t>(v.staging_);
        v.sink_.staging_cap = slots;
        v.head_logits_ = reinterpret_cast<float*>(v.staging_);
        v.R_ = reinterpret_cast<float*>(v.staging_);
        v.n_vocab_ = 16;
        v.max_t_ = 2;
        v.last_t_ = 1;
        *v.h_flagB_ = 0;
    }
    static void submit(Verifier& v, const uint8_t* const* sources, int count, size_t bytes) {
        Verifier::fetch_dma(&v, sources, count, bytes);
    }
    static bool finish(Verifier& v, std::string& error) { return v.finish_expert_copies(error); }
    static cudaError_t status(const Verifier& v) { return v.copy_submit_status_; }
    static uint32_t flag(const Verifier& v) { return *v.h_flagB_; }
    static bool failed(const Verifier& v) { return v.released_.load(); }
    static void* stage(Verifier& v) { return v.staging_; }
    static void shift_stage(Verifier& v, intptr_t offset) {
        v.sink_.staging = reinterpret_cast<uintptr_t>(v.staging_) + offset;
    }
    static void poison(Verifier& v) {
        // The existing runtime fixture uses this host-only failure mechanism.
        // No invalid device address, stalled kernel or device fault is induced.
        v.copy_->submit([](sycl::handler& h) {
            h.host_task([] { throw std::runtime_error("intentional verifier host-task failure"); });
        });
        try { v.copy_->wait_and_throw(); } catch (const std::exception&) {}
        bool rejected = false;
        try { strata::sycl_backend::runtime_for()->check(); }
        catch (const std::exception&) { rejected = true; }
        if (!rejected) throw std::runtime_error("async error did not become sticky");
    }
};
} // namespace strata::core

namespace {
using V = strata::core::Verifier;
using Access = strata::core::VerifierRetirementTestAccess;
int marker_fd = -1;
void need(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t code) { need(code == cudaSuccess, cudaGetErrorString(code)); }
void mark(char value) { need(write(marker_fd, &value, 1) == 1, "marker write"); }

void large_copy_case() {
    constexpr size_t slot=2662400, total=16*slot;
    std::vector<uint8_t> source(total), output(total);
    for(size_t i=0;i<total;++i)source[i]=uint8_t(i*3+7+(i>>8)+(i/slot)*29);
    std::array<const uint8_t*,16> inputs{};
    for(size_t i=0;i<inputs.size();++i)inputs[i]=source.data()+i*slot;
    // All borrowed ordinary host bytes outlive the production verifier owner.
    auto verifier=std::make_unique<V>();Access::prepare(*verifier,slot,16);
    Access::submit(*verifier,inputs.data(),16,slot);
    need(Access::status(*verifier)==cudaSuccess && Access::flag(*verifier)==0,"large submit/refusal");
    std::string error;need(Access::finish(*verifier,error),"large retirement");
    check(cudaMemcpy(output.data(),Access::stage(*verifier),total,cudaMemcpyDeviceToHost));
    need(output==source && Access::flag(*verifier)==1,"large payload or completion differs");
    const auto stats=verifier->expert_copy_stats();
    need(stats.submitted_blobs==16 && stats.completed_blobs==16 &&
         stats.submitted_bytes==total && stats.completed_bytes==total,"large copy accounting differs");
    std::array<uint8_t,128> guards{};
    auto* stage=static_cast<uint8_t*>(Access::stage(*verifier));
    check(cudaMemcpy(guards.data(),stage-64,64,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(guards.data()+64,stage+total,64,cudaMemcpyDeviceToHost));
    for(auto value:guards)need(value==0xa5,"large copy changed outer guards");
}

[[noreturn]] void child(const std::string& mode, int fd) {
    marker_fd = fd;
    need(std::atexit([] { (void)write(marker_fd, "X", 1); }) == 0, "atexit hook");
    struct Owner {
        ~Owner() { (void)write(marker_fd, "U", 1); }
    } owner;
    std::array<uint8_t, 128> source{}, output{};
    for (size_t i = 0; i < source.size(); ++i) source[i] = uint8_t(i * 3 + 7);
    const uint8_t* inputs[] = {source.data(), source.data() + 64};
    // Borrowed source storage outlives verifier destruction on every error.
    auto verifier = std::make_unique<V>();
    Access::prepare(*verifier);
    std::string error;
    if (mode == "success") {
        Access::submit(*verifier, inputs, 2, 64);
        need(Access::status(*verifier) == cudaSuccess, "valid copies refused");
        need(Access::flag(*verifier) == 0, "completion published before checked drain");
        need(Access::finish(*verifier, error), "valid copies did not finish");
        need(Access::flag(*verifier) == 1, "completion missing after drain");
        check(cudaMemcpy(output.data(), Access::stage(*verifier), output.size(), cudaMemcpyDeviceToHost));
        need(output == source, "ordinary source copy payload/guards differ");
        verifier.reset();
        mark('S');
    } else if (mode == "large") {
        verifier.reset();large_copy_case();mark('L');
    } else if (mode == "empty") {
        Access::submit(*verifier, nullptr, 0, 0);
        need(Access::status(*verifier) == cudaSuccess && Access::flag(*verifier) == 0,
             "empty batch published before checked drain");
        need(Access::finish(*verifier, error) && Access::flag(*verifier) == 1,
             "empty batch did not publish after checked drain");
        verifier.reset();
        mark('Z');
    } else if (mode == "partial") {
        Access::submit(*verifier, inputs, 1, 64);
        // A failed follow-up leaves the earlier real copy borrowed until finish.
        const uint8_t* invalid[] = {nullptr};
        Access::submit(*verifier, invalid, 1, 64);
        need(Access::status(*verifier) == cudaErrorInvalidValue, "submit failure lost");
        need(!Access::finish(*verifier, error) && !error.empty(), "failed submit became success");
        need(Access::flag(*verifier) == 0, "failed batch published completion");
        check(cudaMemcpy(output.data(), Access::stage(*verifier), 64, cudaMemcpyDeviceToHost));
        need(std::memcmp(output.data(), source.data(), 64) == 0, "earlier copy not retired");
        const auto stats=verifier->expert_copy_stats();
        need(stats.submitted_blobs==1 && stats.completed_blobs==1 &&
             stats.submitted_bytes==64 && stats.completed_bytes==64,"partial copy accounting differs");
        need(Access::failed(*verifier), "failed copy did not latch terminal state");
        std::array<float, 16> logits; logits.fill(42.0f);
        need(!verifier->copy_logits(0, logits.data()), "failed window exposed old logits");
        for (auto value : logits) need(value == 42.0f, "refused logits copy changed output");
        need(!verifier->final_R(0) && !verifier->final_R_all(), "failed window exposed residuals");
        int32_t token = 0, result = -123;
        need(!verifier->run(1, &token, 0, nullptr, nullptr, &result, error) &&
             error.find("restart") != std::string::npos && result == -123, "failed verifier was reused");
        need(!verifier->commit(1, error) && error.find("cannot commit") != std::string::npos,
             "failed verifier accepted commit");
        std::FILE* sink = std::tmpfile(); need(sink != nullptr, "temporary output");
        const bool logged = verifier->window_logprobs(&token, 1, 0, -1, sink, error);
        const auto written = std::ftell(sink); std::fclose(sink);
        need(!logged && written == 0, "failed window exposed old logprobs");
        verifier.reset();
        mark('P');
    } else if (mode == "bounds") {
        // Check actual refusal before an invalid copy can enter a real queue.
        struct Input { int count; size_t bytes; intptr_t offset; };
        const Input cases[] = {{-1,64,0}, {3,64,0}, {1,0,0}, {1,65,0},
                               {1,64,-1}, {2,64,Access::arena_bytes-64}};
        std::array<uint8_t, Access::arena_bytes> guards{};
        verifier.reset();
        for (const auto& input : cases) {
            verifier = std::make_unique<V>(); Access::prepare(*verifier);
            check(cudaMemset(Access::stage(*verifier), 0x5a, guards.size()));
            Access::shift_stage(*verifier, input.offset);
            Access::submit(*verifier, inputs, input.count, input.bytes);
            need(Access::status(*verifier) == cudaErrorInvalidValue, "invalid span accepted");
            need(!Access::finish(*verifier, error), "invalid span became success");
            check(cudaMemcpy(guards.data(), Access::stage(*verifier), guards.size(), cudaMemcpyDeviceToHost));
            for (auto value : guards) need(value == 0x5a, "refused copy changed staging");
            verifier.reset();
        }
        mark('B');
    } else {
        Access::poison(*verifier);
        if (mode == "query") {
            need(!verifier->release_gpu_waits(100), "query error was called completion");
            mark('Q');
        } else if (mode == "destructor") {
            mark('D');
            verifier.reset(); // Must stop before returning, atexit or unwind.
            mark('A');
        } else if (mode == "copy-error") {
            Access::submit(*verifier, inputs, 1, 64);
            need(Access::status(*verifier) != cudaSuccess && Access::flag(*verifier) == 0,
                 "failed runtime submit was published as success");
            mark('E');
            (void)Access::finish(*verifier, error); // Unprovable retirement must stop.
            mark('A');
        } else throw std::runtime_error("unknown test case");
    }
    std::_Exit(0); // Intentionally poisoned cases retain owners until process exit.
}

void isolated(const char* mode, const char* expected, int exit_code) {
    int fds[2]; need(pipe(fds) == 0, "pipe");
    const pid_t pid = fork(); need(pid >= 0, "fork");
    if (pid == 0) {
        close(fds[0]);
        const auto fd = std::to_string(fds[1]);
        execl("/proc/self/exe", "sycl_verifier_retirement_test", "--case", mode, fd.c_str(), nullptr);
        std::_Exit(99);
    }
    close(fds[1]);
    std::string observed; char value;
    for (;;) {
        const auto n = read(fds[0], &value, 1);
        if (n == 1) observed += value;
        else if (n < 0 && errno == EINTR) continue;
        else { need(n == 0, "pipe read"); break; }
    }
    close(fds[0]); int status = 0;
    need(waitpid(pid, &status, 0) == pid, "reap child");
    need(WIFEXITED(status) && WEXITSTATUS(status) == exit_code, "wrong child outcome");
    need(observed == expected, "release, return, atexit or unwind preceded failed retirement");
    std::printf("case=%s expected_exit=%d markers=%s PASS\n", mode, exit_code, observed.c_str());
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 4 && std::string(argv[1]) == "--case") child(argv[2], std::stoi(argv[3]));
        need(argc == 1, "usage: sycl_verifier_retirement_test");
        // The coordinator initializes no SYCL runtime before fork+exec.
        isolated("success", "S", 0);
        isolated("large", "L", 0);
        isolated("empty", "Z", 0);
        isolated("partial", "P", 0);
        isolated("bounds", "B", 0);
        isolated("query", "Q", 0);
        isolated("destructor", "D", EXIT_FAILURE);
        isolated("copy-error", "E", EXIT_FAILURE);
        std::puts("fixture_status=PASS cases=8");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
