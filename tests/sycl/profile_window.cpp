// Main-authored collector qualification fixture; repository LICENSE.
// No inference code or third-party kernel implementation is copied here.
// Graph construction uses the installed SYCL command_graph interface. The
// external owner controls collector state at each stdin gate, never getenv.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/command_graph.hpp>

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace graph_api = sycl::ext::oneapi::experimental;
using namespace std::chrono_literals;

class StrataWindowProduce;
class StrataWindowConsume;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void gate(const char* phase) {
    std::cout << "COMPONENT_READY " << phase << '\n' << std::flush;
    require(std::cin.get() == 'X', "Main collector-window gate");
}

struct Release {
    std::mutex mutex;
    std::condition_variable cv;
    bool released = false;
    bool timer_released = false;

    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        cv.notify_all();
    }
};

struct Fixture {
    static constexpr size_t n = 257, guard = 17, total = n + 2 * guard;
    static constexpr int sentinel = -1234567;
    using Graph = graph_api::command_graph<graph_api::graph_state::modifiable>;
    using Executable = graph_api::command_graph<graph_api::graph_state::executable>;

    std::mutex error_mutex;
    std::vector<std::exception_ptr> errors;
    sycl::queue queue;
    int* device = nullptr;
    std::unique_ptr<Graph> graph;
    std::unique_ptr<Executable> executable;
    std::array<int, total> input{}, output{};
    std::shared_ptr<Release> release = std::make_shared<Release>();
    std::thread timer;

    Fixture()
        : queue(sycl::gpu_selector_v,
                [this](sycl::exception_list list) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    for (auto error : list) errors.push_back(error);
                },
                {sycl::property::queue::in_order{},
                 sycl::property::queue::enable_profiling{}}) {}

    void drain() {
        queue.wait_and_throw();
        std::lock_guard<std::mutex> lock(error_mutex);
        if (!errors.empty()) std::rethrow_exception(errors.front());
    }

    void validate(int replays) {
        queue.memcpy(output.data(), device, sizeof(output));
        drain();
        for (size_t i = 0; i < total; ++i) {
            int expected = input[i];
            if (i >= guard && i < guard + n)
                for (int r = 0; r < replays; ++r)
                    expected = (expected * 3 + 1) * 5 - 2;
            require(output[i] == expected, "numerical result or allocation guard");
        }
    }

    void run() {
        const auto name = queue.get_device().get_info<sycl::info::device::name>();
        require(name.find("B570") != std::string::npos, "unexpected GPU");
        std::cout << "device=" << name << '\n';
        gate("runtime-ready");
        device = sycl::malloc_device<int>(total, queue);
        require(device != nullptr, "device allocation");
        input.fill(sentinel);
        for (size_t i = 0; i < n; ++i) input[guard + i] = int(i) - 128;
        queue.memcpy(device, input.data(), sizeof(input));
        drain();

        graph = std::make_unique<Graph>(queue.get_context(), queue.get_device());
        graph->begin_recording(queue);
        int* const data = device;
        queue.parallel_for<StrataWindowProduce>(sycl::range<1>(n), [=](sycl::id<1> i) {
            data[guard + i[0]] = data[guard + i[0]] * 3 + 1;
        });
        queue.parallel_for<StrataWindowConsume>(sycl::range<1>(n), [=](sycl::id<1> i) {
            data[guard + i[0]] = data[guard + i[0]] * 5 - 2;
        });
        graph->end_recording(queue);
        executable = std::make_unique<Executable>(graph->finalize());
        gate("captured-paused");

        for (int iteration = 1; iteration <= 3; ++iteration) {
            queue.ext_oneapi_graph(*executable);
            validate(iteration);
        }
        std::cout << "case clean_replays=3 kernels=6 readbacks=3 numerical=PASS guards=PASS\n";
        gate("replay-drained");

        // The owner resumes collection before permitting this phase. A bounded
        // host task delays GPU work without a device spin loop or host writes
        // racing a device allocation. A timer releases it if the owner stalls.
        const auto pending_release = release;
        timer = std::thread([pending_release] {
            std::unique_lock<std::mutex> lock(pending_release->mutex);
            if (!pending_release->cv.wait_for(lock, 2s,
                                             [&] { return pending_release->released; })) {
                pending_release->timer_released = true;
                pending_release->released = true;
                pending_release->cv.notify_all();
            }
        });
        queue.submit([pending_release](sycl::handler& handler) {
            handler.host_task([pending_release] {
                std::unique_lock<std::mutex> lock(pending_release->mutex);
                require(pending_release->cv.wait_for(lock, 4s,
                            [&] { return pending_release->released; }), "host task release deadline");
            });
        });
        const auto late = queue.ext_oneapi_graph(*executable);
        const bool pending = late.get_info<sycl::info::event::command_execution_status>() !=
                             sycl::info::event_command_status::complete;
        std::cout << "case late_graph_pending=" << pending << '\n';
        gate("late-enqueued");
        release->release();
        timer.join();
        validate(4);
        std::cout << "case late_replays=1 kernels=2 numerical=PASS guards=PASS timer_fallback="
                  << release->timer_released << '\n';
        gate("late-drained");

        drain();
        executable.reset();
        graph.reset();
        sycl::free(device, queue);
        device = nullptr;
        gate("before-exit");
        std::cout << "fixture_status=PASS mode=profile-window\n";
    }
};

int main(int argc, char**) {
    std::cout << std::unitbuf;
    try {
        require(argc == 1, "profile-window fixture takes no arguments");
        gate("before-runtime");
        // Keep all asynchronous operands and graph/queue owners alive on a
        // failure. The supervised process exits rather than freeing uncertain
        // in-flight resources; ordinary successful cleanup is checked above.
        auto* fixture = new Fixture;
        fixture->run();
        delete fixture;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fixture_status=FAIL " << error.what() << '\n' << std::flush;
        std::_Exit(1);
    }
}
