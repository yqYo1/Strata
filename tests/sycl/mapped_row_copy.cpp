// Independently authored for Strata; repository LICENSE.
// Compare the existing production entry with unselected row-copy components.
#include "strata/kernels/elementwise.hpp"
#include "mapped_row_copy.hpp"
#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <time.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {
using namespace strata::sycl_backend;
using Copy = void (*)(float*, const float*, int64_t, int64_t,
                       const int32_t*, const int32_t*, void*);
constexpr size_t Guard = 16;
constexpr uint32_t Sentinel = 0x7fc5a39b;
struct Variant { const char* name; Copy copy; };
const std::array<Variant, 5> Variants{{
    {"production", strata::kernels::copy_rows_from_mapped},
    {"row256e1", strata::kernels::component::mapped_row_copy<256, 1>},
    {"row256e4", strata::kernels::component::mapped_row_copy<256, 4>},
    {"tile256e4", strata::kernels::component::mapped_row_copy<256, 4, true>},
    {"tile64e4", strata::kernels::component::mapped_row_copy<64, 4, true>}
}};
void need(bool value, const char* why) {
    if (!value) throw std::runtime_error(why);
}
void drain(const std::shared_ptr<Runtime>& rt) noexcept {
    // Never unwind a source/metadata owner when GPU completion is uncertain.
    try { rt->wait(); } catch (...) { std::abort(); }
}
struct Stamp { int64_t wall, cpu; };
int64_t now(clockid_t id) {
    timespec t{};
    need(clock_gettime(id, &t) == 0 && t.tv_sec >= 0 &&
         t.tv_sec <= INT64_MAX / 1000000000 - 1, "unavailable/overflowing clock");
    return int64_t(t.tv_sec) * 1000000000 + t.tv_nsec;
}
Stamp stamp() { return {now(CLOCK_MONOTONIC), now(CLOCK_THREAD_CPUTIME_ID)}; }
uint32_t pattern(size_t i) {
    // Copy bits: include both zero signs, infinities, quiet/signaling NaNs and
    // subnormals. No floating-point reference arithmetic changes these bits.
    constexpr uint32_t special[] = {0, 0x80000000, 0x7f800000, 0xff800000,
                                    0x7fc12345, 0x7f812345, 1, 0x80000001};
    return i % 19 < 8 ? special[i % 19] : uint32_t(i * 2654435761U);
}
class Buffers {
    std::shared_ptr<Runtime> rt_;
    size_t rows_, width_, begin_, words_, banks_;
    MemoryKind source_kind_;
    std::vector<uint32_t> input_, output_, source_check_;
    std::vector<int32_t> ids_, ids_check_;
    std::array<int32_t, 2 * Guard + 1> counts_{}, counts_check_{};
    std::vector<bool> hit_;
    Allocation source_, destination_, ids_device_, count_device_;
public:
    Buffers(std::shared_ptr<Runtime> rt, size_t rows, size_t width,
            size_t offset, MemoryKind kind, size_t banks)
        : rt_(std::move(rt)), rows_(rows), width_(width), begin_(Guard + offset),
          words_(begin_ + rows * width + Guard), banks_(banks), source_kind_(kind),
          input_(words_ * banks), output_(words_, Sentinel), source_check_(words_ * banks),
          ids_(rows + 2 * Guard, -1234567), ids_check_(ids_.size()), hit_(rows),
          source_(rt_, input_.size() * 4, kind), destination_(rt_, words_ * 4, MemoryKind::Device),
          ids_device_(rt_, ids_.size() * 4, MemoryKind::Device),
          count_device_(rt_, counts_.size() * 4, MemoryKind::Device) {
        need(rows && rows <= 40 && width && width <= 2561 && offset <= 1 &&
             (banks == 1 || banks == 64), "unsupported fixture extent");
        for (size_t bank = 0; bank < banks; ++bank) {
            auto* p = input_.data() + bank * words_;
            std::fill(p, p + words_, Sentinel);
            for (size_t i = 0; i < rows * width; ++i)
                p[begin_ + i] = pattern(i + bank * 104729);
        }
        if (kind == MemoryKind::Host) std::memcpy(source_.data(), input_.data(), input_.size() * 4);
        else rt_->wait(rt_->compute().memcpy(source_.data(), input_.data(), input_.size() * 4));
        for (const auto* pointer : {destination_.data(), ids_device_.data(), count_device_.data()})
            need(sycl::get_pointer_type(pointer, rt_->context()) == sycl::usm::alloc::device,
                 "destination/metadata must be device USM");
        counts_.fill(-1234567);
        reset();
    }
    ~Buffers() noexcept { drain(rt_); }
    Buffers(const Buffers&) = delete;
    Buffers& operator=(const Buffers&) = delete;
    void reset() {
        drain(rt_);
        std::fill(output_.begin(), output_.end(), Sentinel);
        rt_->wait(rt_->compute().memcpy(destination_.data(), output_.data(), words_ * 4));
    }
    // Every metadata producer and consumer uses this same in-order queue.
    // Previous readers complete before CPU staging storage is changed.
    void hits(int mode) {
        drain(rt_);
        std::fill(hit_.begin(), hit_.end(), false);
        std::fill(ids_.begin(), ids_.end(), -1234567);
        int count = 0;
        if (mode == 1) ids_[Guard + count++] = int32_t(rows_ - 1);
        if (mode == 2) {
            ids_[Guard + count++] = int32_t(rows_ - 1);
            if (rows_ >= 2) ids_[Guard + count++] = 0;
            if (rows_ >= 3) ids_[Guard + count++] = int32_t(rows_ - 1); // duplicate,unsorted
        }
        if (mode == 3) for (size_t i = rows_; i; --i) ids_[Guard + count++] = int32_t(i - 1);
        if (mode == 4 || mode == 20) {
            need(rows_ == 40, "timed hit shape");
            for (int i = 0; i < mode; ++i) ids_[Guard + count++] = (i * 13 + 7) % 40;
        }
        need(count >= 0 && size_t(count) <= rows_, "metadata count bounds");
        for (int i = 0; i < count; ++i) hit_[size_t(ids_[Guard + i])] = true;
        counts_[Guard] = count;
        rt_->compute().memcpy(ids_device_.data(), ids_.data(), ids_.size() * 4);
        rt_->wait(rt_->compute().memcpy(count_device_.data(), counts_.data(), counts_.size() * 4));
    }
    void launch(size_t variant, size_t bank = 0) {
        need(variant < Variants.size() && bank < banks_, "fixture selection bounds");
        Variants[variant].copy(destination_.as<float>() + begin_,
            source_.as<float>() + bank * words_ + begin_, rows_, width_,
            ids_device_.as<int32_t>() + Guard, count_device_.as<int32_t>() + Guard,
            &rt_->compute());
    }
    void wait() { rt_->wait(); }
    void verify(size_t bank, bool all_inputs = true) {
        rt_->wait(rt_->compute().memcpy(output_.data(), destination_.data(), words_ * 4));
        for (size_t i = 0; i < words_; ++i) {
            uint32_t expected = Sentinel;
            if (i >= begin_ && i < begin_ + rows_ * width_)
                expected = hit_[(i - begin_) / width_] ? 0 : input_[bank * words_ + i];
            need(output_[i] == expected, "destination bits/tail/canary mismatch");
        }
        if (!all_inputs) return;
        rt_->compute().memcpy(source_check_.data(), source_.data(), input_.size() * 4);
        rt_->compute().memcpy(ids_check_.data(), ids_device_.data(), ids_.size() * 4);
        rt_->wait(rt_->compute().memcpy(counts_check_.data(), count_device_.data(), counts_.size() * 4));
        need(source_check_ == input_ && ids_check_ == ids_ && counts_check_ == counts_,
             "source/metadata/canary changed");
        need(sycl::get_pointer_type(source_.data(), rt_->context()) ==
             (source_kind_ == MemoryKind::Host ? sycl::usm::alloc::host : sycl::usm::alloc::device),
             "actual source placement differs");
    }
};
void host_checks() {
    for (size_t v = 0; v < Variants.size(); ++v) {
        Variants[v].copy(nullptr, nullptr, 0, 2560, nullptr, nullptr, nullptr);
        Variants[v].copy(nullptr, nullptr, 40, 0, nullptr, nullptr, nullptr);
        for (const auto shape : {std::array<int64_t, 2>{-1, 4}, {4, -1}, {INT64_MAX, 2}}) {
            bool rejected = false;
            try { Variants[v].copy(nullptr, nullptr, shape[0], shape[1], nullptr, nullptr, nullptr); }
            catch (const std::invalid_argument&) { rejected = true; }
            need(rejected, "invalid dimensions must fail before queue access");
        }
        if (v) {
            for (int64_t rows : {INT64_MAX, INT64_MAX / 4}) {
                bool rejected = false;
                try { Variants[v].copy(nullptr, nullptr, rows, 1, nullptr, nullptr, nullptr); }
                catch (const std::invalid_argument&) { rejected = true; }
                need(rejected, "byte/global extent must fail before queue access");
            }
        }
    }
}
std::shared_ptr<Runtime> checked_runtime() {
    auto rt = runtime_for();
    const auto& device = rt->device();
    const auto name = device.get_info<sycl::info::device::name>();
    need(Runtime::devices().size() == 1 && device.is_gpu() &&
         device.get_info<sycl::info::device::vendor_id>() == 0x8086 &&
         name.find("B570") != std::string::npos &&
         device.get_info<sycl::info::device::max_work_group_size>() >= 256 &&
         device.has(sycl::aspect::usm_host_allocations),
         "one Intel B570 with host-USM and 256-lane groups required");
    std::cout << "DEVICE name=" << name << " host_usm=1 metadata=device\n";
    return rt;
}
void checks(bool small, std::shared_ptr<Runtime>& keepalive) {
    host_checks();
    auto rt = checked_runtime();
    keepalive = rt;
    const std::vector<size_t> widths = small ? std::vector<size_t>{17, 2560} :
        std::vector<size_t>{1, 3, 63, 64, 65, 255, 256, 257, 1023, 1024, 1025, 2559, 2560, 2561};
    size_t launches = 0;
    for (auto kind : {MemoryKind::Host, MemoryKind::Device})
        for (size_t offset = small ? 1 : 0; offset <= 1; ++offset)
            for (size_t width : widths) {
                const size_t rows = width == 1 ? 1 : width < 100 ? 3 : 40;
                Buffers b(rt, rows, width, offset, kind, 1);
                for (int mode = 0; mode < 4; ++mode) {
                    b.hits(mode);
                    for (size_t v = 0; v < Variants.size(); ++v) {
                        b.reset(); b.launch(v); b.verify(0); ++launches;
                    }
                }
            }
    std::cout << "CHECK passed=1 small=" << small << " variants=5 launches=" << launches
              << " metadata=device source=host_usm,device bitwise=1 guards=1\n";
}
void timing(int process, std::shared_ptr<Runtime>& keepalive) {
    constexpr size_t Batch = 4096;
    constexpr int Samples = 5;
    constexpr int64_t Floor = 10000000; // whole dependent batch,not one launch
    const std::array<std::array<size_t, 5>, 3> orders{{{{0,1,2,3,4}}, {{4,3,2,1,0}}, {{2,0,4,1,3}}}};
    need(process >= 0 && process < 3, "independent process index required");
    auto rt = checked_runtime();
    keepalive = rt;
    std::vector<int64_t> overhead;
    for (int i = 0; i < 1000; ++i) {
        const auto a = stamp(), b = stamp();
        need(b.wall >= a.wall && b.cpu >= a.cpu, "clock monotonicity");
        overhead.push_back(b.wall - a.wall);
    }
    std::sort(overhead.begin(), overhead.end());
    const int64_t timer = overhead[overhead.size()/2];
    std::cout << "CLOCK median_stamp_ns=" << timer << " floor_ns=" << Floor
              << " batch=" << Batch << " process=" << process << '\n';
    for (auto kind : {MemoryKind::Host, MemoryKind::Device})
        for (size_t banks : {size_t(1), size_t(64)}) {
            Buffers b(rt, 40, 2560, 0, kind, banks);
            for (int hits : {4, 20}) {
                b.hits(hits);
                for (size_t v : orders[size_t(process)]) {
                    b.reset(); b.launch(v); b.verify(0); // correctness/JIT outside timing
                    for (int sample = -2; sample < Samples; ++sample) {
                        const auto a = stamp();
                        for (size_t i = 0; i < Batch; ++i) b.launch(v, i % banks);
                        const auto submitted = stamp();
                        b.wait();
                        const auto end = stamp();
                        const auto wall = end.wall - a.wall;
                        need(wall >= Floor && wall >= 300 * timer && submitted.cpu >= a.cpu &&
                             end.cpu >= submitted.cpu &&
                             submitted.wall >= a.wall && end.wall >= submitted.wall,
                             "ineligible batch:timer floor/overhead/clock");
                        b.verify((Batch - 1) % banks, sample == Samples - 1);
                        if (sample >= 0)
                            std::cout << "SAMPLE process=" << process << " variant=" << Variants[v].name
                                      << " source=" << (kind == MemoryKind::Host ? "host_usm" : "device")
                                      << " rows=40 width=2560 hits=" << hits << " banks=" << banks
                                      << " sample=" << sample << " launches=" << Batch
                                      << " wall_ns=" << wall << " cpu_ns=" << end.cpu-a.cpu
                                      << " submit_wall_ns=" << submitted.wall-a.wall
                                      << " wait_wall_ns=" << end.wall-submitted.wall
                                      << " submit_cpu_ns=" << submitted.cpu-a.cpu
                                      << " wait_cpu_ns=" << end.cpu-submitted.cpu << '\n';
                    }
                }
            }
        }
    std::cout << "TIMING passed=1 samples=200 model_speed=0 physical_traffic=0\n";
}
} // namespace
int main(int argc, char** argv) {
    try {
        // Immutable startup choice. The owner inspects the live native child
        // after all work completes; the handshake is outside every timed batch.
        const char* value = std::getenv("STRATA_COPY_COMPONENT_GATE");
        const bool gate = value && std::strcmp(value, "1") == 0;
        need(!value || gate, "unsupported component gate setting");
        need(argc == 2 || argc == 3, "use --check-small|--check|--time 0|1|2");
        const std::string mode(argv[1]);
        std::shared_ptr<Runtime> keepalive;
        if (mode == "--check-small" || mode == "--check") {
            need(argc == 2, "check takes no other arguments");
            checks(mode == "--check-small", keepalive);
        } else {
            need(mode == "--time" && argc == 3 && std::strlen(argv[2]) == 1 &&
                 argv[2][0] >= '0' && argv[2][0] <= '2', "use --time 0|1|2");
            timing(argv[2][0] - '0', keepalive);
        }
        if (gate) {
            std::cout << "STRATA_COPY_GATE pid=" << getpid() << std::endl;
            std::string answer;
            need(bool(std::getline(std::cin, answer)) && answer == "continue",
                 "component owner did not release the completed native child");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mapped-row fixture failed: " << error.what() << '\n';
        return 1;
    }
}
