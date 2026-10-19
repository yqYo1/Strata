#include "strata/core/verify.hpp"
#include "strata/sycl/runtime.hpp"

#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
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
    static void prepare_early(Verifier& v, int count, size_t bytes, uint8_t* result) {
        if (bytes % sizeof(uint32_t))
            throw std::runtime_error("early test word alignment");
        if (cudaMallocHost(reinterpret_cast<void**>(&v.h_plan_), 4*sizeof(int32_t)) != cudaSuccess ||
            cudaMallocHost(reinterpret_cast<void**>(&v.h_flagA_), sizeof(uint32_t)) != cudaSuccess ||
            cudaEventCreateWithFlags(&v.work_done_, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&v.pcie_copy_ready_, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&v.pcie_done_, cudaEventDisableTiming) != cudaSuccess)
            throw std::runtime_error("early test owner admission");
        v.sink_.counts = v.h_plan_;
        std::memset(v.h_plan_, 0, 4*sizeof(int32_t));
        v.sink_.counts[2] = count;
        v.early_pcie_ = v.overlap_gpu_ = true;
        v.cur_layer_.store(0);
        const auto finalize = [&](std::vector<cudaGraphExec_t>& target) {
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t exec = nullptr;
            if (cudaStreamEndCapture(v.cs_, &graph) != cudaSuccess ||
                cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess ||
                cudaGraphDestroy(graph) != cudaSuccess)
                throw std::runtime_error("early test graph finalization");
            target.push_back(exec);
        };
        if (cudaStreamBeginCapture(v.cs_, cudaStreamCaptureModeRelaxed) != cudaSuccess)
            throw std::runtime_error("early resident capture");
        // Models resident work sharing the copied kernel's queue/scratch.
        v.cs_->single_task([=] { result[bytes] = uint8_t(0x3c); });
        finalize(v.work_[1]);
        if (cudaStreamBeginCapture(v.cs_, cudaStreamCaptureModeRelaxed) != cudaSuccess)
            throw std::runtime_error("early copied capture");
        auto* stage = v.staging_;
        v.cs_->parallel_for(sycl::range<1>(bytes/sizeof(uint32_t)), [=](sycl::id<1> i) {
            uint32_t value = 0;
            for (unsigned j=0;j<sizeof(uint32_t);++j)
                value |= uint32_t(stage[i[0]*sizeof(uint32_t)+j] ^ result[bytes]) << (8*j);
            // A separate test queue can inspect these words while the copied
            // graph is held. Atomic reads/writes keep that negative assertion
            // defined even if a broken wait lets the copied graph start early.
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                sycl::memory_scope::device, sycl::access::address_space::global_space>
                word(reinterpret_cast<uint32_t*>(result)[i[0]]);
            word.store(value);
        });
        finalize(v.pcie_work_[1]);
        reset_early(v);
    }
    static void reset_early(Verifier& v) {
        v.work_launched_ = v.pcie_launched_ = v.pcie_copy_recorded_ = false;
        v.work_status_ = v.pcie_status_ = v.copy_submit_status_ = cudaSuccess;
        *v.h_flagA_ = *v.h_flagB_ = 0;
    }
    static void publish(Verifier& v) { Verifier::publish_plan(&v); }
    static void warm_early(Verifier& v) {
        if (cudaGraphLaunch(v.work_[1][0],v.cs_)!=cudaSuccess ||
            cudaGraphLaunch(v.pcie_work_[1][0],v.cs_)!=cudaSuccess ||
            cudaStreamSynchronize(v.cs_)!=cudaSuccess)
            throw std::runtime_error("early test graph warmup");
    }
    static sycl::queue& copy_queue(Verifier& v) { return *v.copy_; }
    static cudaError_t early_query(Verifier& v) { return cudaEventQuery(v.pcie_done_); }
    static void launch_unrecorded(Verifier& v, int groups) { v.launch_pcie(groups); }
    static cudaError_t early_status(const Verifier& v) { return v.pcie_status_; }
    static cudaError_t early_done(Verifier& v) { return cudaEventSynchronize(v.pcie_done_); }
    static cudaError_t resident_done(Verifier& v) { return cudaEventSynchronize(v.work_done_); }
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

// Test-only host gate: no runtime calls, device spin, or unbounded wait inside
// the host task. Always release before the verifier drains its queues, including
// exception paths. Automatic release is a failed assertion, never a pass.
class CopyGate {
    struct State {
        std::mutex mutex;
        std::condition_variable changed;
        bool entered=false, open=false, expired=false;
    };
    std::shared_ptr<State> state_=std::make_shared<State>();
public:
    explicit CopyGate(sycl::queue& queue) {
        const auto state=state_;
        queue.submit([=](sycl::handler& h) {
            h.host_task([=] {
                std::unique_lock lock(state->mutex);
                state->entered=true;
                state->changed.notify_all();
                if (!state->changed.wait_for(lock,std::chrono::seconds(2),[&] { return state->open; })) {
                    state->expired=true;
                    state->open=true;
                }
                state->changed.notify_all();
            });
        });
    }
    ~CopyGate() { release(); }
    void wait_entered() {
        std::unique_lock lock(state_->mutex);
        need(state_->changed.wait_for(lock,std::chrono::seconds(2),[&] { return state_->entered; }),
             "copy gate host task did not start");
        need(!state_->open,"copy gate expired during admission");
    }
    bool held() const {
        std::lock_guard lock(state_->mutex);
        return state_->entered && !state_->open && !state_->expired;
    }
    void release() {
        std::lock_guard lock(state_->mutex);
        state_->open=true;
        state_->changed.notify_all();
    }
};

void require_untouched(sycl::queue& queue,uint8_t* output,size_t bytes,uint32_t* observed) {
    auto event=queue.single_task([=] {
        uint32_t untouched=1;
        for (size_t i=0;i<bytes/sizeof(uint32_t);++i) {
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                sycl::memory_scope::device, sycl::access::address_space::global_space>
                word(reinterpret_cast<uint32_t*>(output)[i]);
            if (word.load()!=0xe7e7e7e7u) untouched=0;
        }
        *observed=untouched;
    });
    strata::sycl_backend::runtime_for()->wait(event);
    need(*observed==1,"copied output changed while DMA gate was held");
}

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

void early_case(const std::string& mode) {
    const bool empty = mode == "early-empty", bad_plan = mode == "early-plan";
    const bool unrecorded = mode == "early-unrecorded", replay = mode == "early-replay";
    const size_t slot = mode == "early-large" ? 2662400 : 64;
    const int count = empty ? 0 : mode == "early-large" ? 16 : 2;
    const size_t bytes = (count ? count : 1)*slot;
    std::vector<uint8_t> source(bytes), observed(bytes);
    std::array<const uint8_t*,16> inputs{};
    for (int i=0;i<count;++i) inputs[i]=source.data()+size_t(i)*slot;
    void* output_raw = nullptr;
    check(cudaMalloc(&output_raw, bytes+129));
    check(cudaMemset(output_raw, 0xe7, bytes+129));
    auto* result = static_cast<uint8_t*>(output_raw)+64;
    uint32_t* probe_raw=nullptr;
    check(cudaMallocHost(reinterpret_cast<void**>(&probe_raw),sizeof(uint32_t)));
    const auto free_probe=[](uint32_t* p) { (void)cudaFreeHost(p); };
    std::unique_ptr<uint32_t,decltype(free_probe)> probe(probe_raw,free_probe);
    auto probe_queue=strata::sycl_backend::runtime_for()->make_queue(false);
    auto verifier = std::make_unique<V>();
    Access::prepare(*verifier, slot, 16);
    Access::prepare_early(*verifier, count, bytes, result);
    if (mode=="early") {
        // Compile/initialize both synthetic graphs before the finite gate.
        // These direct warmups do not increment production early counters.
        Access::warm_early(*verifier);
        check(cudaMemset(output_raw,0xe7,bytes+129));
        require_untouched(probe_queue,result,bytes,probe.get());
    }
    // Declared after the verifier so any exception releases the host gate
    // before checked verifier destruction tries to drain the copy queue.
    std::unique_ptr<CopyGate> gate;
    if (mode=="early") {
        gate=std::make_unique<CopyGate>(Access::copy_queue(*verifier));
        gate->wait_entered();
    }
    const int rounds = replay ? 2 : 1;
    for (int round=0;round<rounds;++round) {
        // The preceding graph and explicit source retirement completed before
        // reusing these ordinary host bytes or either event in a later round.
        for(size_t i=0;i<bytes;++i) source[i]=uint8_t(i*3+7+(i>>8)+round*31);
        if(round) Access::reset_early(*verifier);
        Access::publish(*verifier);
        if (gate) {
            check(Access::resident_done(*verifier));
            need(gate->held(),"resident positive control consumed the copy gate deadline");
        }
        if (unrecorded) Access::launch_unrecorded(*verifier, count);
        else Access::submit(*verifier, empty ? nullptr : inputs.data(), bad_plan ? count-1 : count,
                            empty ? 0 : slot);
        const auto before = verifier->early_pcie_stats();
        if (!empty && !bad_plan && !unrecorded) {
            need(before.launches==uint64_t(round+1) && before.groups==uint64_t(count*(round+1)),
                 "early graph not submitted before CPU callback retirement");
            if (gate) {
                need(gate->held(),"early submission blocked until the copy gate expired");
                const auto until=std::chrono::steady_clock::now()+std::chrono::milliseconds(50);
                do {
                    need(Access::early_query(*verifier)==cudaErrorNotReady,
                         "early graph completed while DMA gate was held");
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                } while (std::chrono::steady_clock::now()<until);
                // Do not call cudaMemcpy here: that wrapper waits every stream
                // and would release the gate only through its failure deadline.
                require_untouched(probe_queue,result,bytes,probe.get());
                need(gate->held(),"output probe consumed the copy gate deadline");
                gate->release();
            }
            check(Access::early_done(*verifier));
        } else {
            need(before.launches==0 && before.groups==0,"empty/refused plan launched a copied kernel");
            check(Access::resident_done(*verifier));
        }
        need(Access::flag(*verifier)==0,"early enqueue published source retirement");
        if (bad_plan) need(Access::status(*verifier)==cudaErrorInvalidValue,"plan mismatch accepted");
        if (unrecorded) need(Access::early_status(*verifier)==cudaErrorInvalidValue,"unrecorded event accepted");
        check(cudaMemcpy(observed.data(),result,bytes,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<bytes;++i) {
            const auto expected=uint8_t(i*3+7+(i>>8)+round*31);
            need(source[i]==expected,"early source bytes changed");
            need(observed[i]==(empty||bad_plan||unrecorded ? uint8_t(0xe7) : uint8_t(expected^0x3c)),
                 "early copy/resident ordering or payload differs");
        }
        std::string error;
        need(Access::finish(*verifier,error)==!bad_plan,"early checked source retirement differs");
        const auto copied=verifier->expert_copy_stats();
        const auto expected=uint64_t(empty||bad_plan||unrecorded ? 0 : count*(round+1));
        need(copied.submitted_blobs==expected && copied.completed_blobs==expected &&
             copied.submitted_bytes==expected*slot && copied.completed_bytes==expected*slot,
             "early copy conservation differs");
        if(!bad_plan) need(Access::flag(*verifier)==1,"early retirement flag missing");
        std::array<uint8_t,128> guards{};
        check(cudaMemcpy(guards.data(),static_cast<uint8_t*>(output_raw),64,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(guards.data()+64,result+bytes+1,64,cudaMemcpyDeviceToHost));
        for(auto value:guards)need(value==0xe7,"early output outer guard changed");
        auto* stage=static_cast<uint8_t*>(Access::stage(*verifier));
        check(cudaMemcpy(guards.data(),stage-64,64,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(guards.data()+64,stage+16*slot,64,cudaMemcpyDeviceToHost));
        for(auto value:guards)need(value==0xa5,"early staging outer guard changed");
    }
    verifier.reset();
    check(cudaFree(output_raw));
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
    if (mode.rfind("early",0)==0) {
        verifier.reset();
        early_case(mode);
        if(mode=="early")mark('F');
        else if(mode=="early-large")mark('G');
        else if(mode=="early-empty")mark('H');
        else if(mode=="early-replay")mark('I');
        else if(mode=="early-plan")mark('J');
        else if(mode=="early-unrecorded")mark('K');
        else throw std::runtime_error("unknown early case");
    } else if (mode == "success") {
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
        if(argc==2 && std::string(argv[1])=="--early") {
            isolated("early", "F", 0);
            isolated("early-large", "G", 0);
            isolated("early-empty", "H", 0);
            isolated("early-replay", "I", 0);
            isolated("early-plan", "J", 0);
            isolated("early-unrecorded", "K", 0);
            std::puts("early_fixture_status=PASS cases=6");
            return 0;
        }
        need(argc == 1, "usage: sycl_verifier_retirement_test [--early]");
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
