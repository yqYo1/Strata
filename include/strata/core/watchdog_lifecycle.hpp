// Host-only watchdog lifetime and diagnostic helpers. No device dependencies.
#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace strata::core {
enum class VerifyPhase { inactive, preparing, running, finishing, completed, failed };
struct VerifyWindowView {
    uint64_t generation = 0;
    VerifyPhase phase = VerifyPhase::inactive;
    int tokens = 0;
    int64_t position = 0;
};
inline bool verify_rescue_eligible(VerifyPhase phase) { return phase == VerifyPhase::running; }
class VerifyWindowSnapshot {
    mutable std::mutex mutex_;
    VerifyWindowView value_;
public:
    void begin(int tokens, int64_t position) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++value_.generation;
        value_.tokens = tokens; value_.position = position;
        value_.phase = VerifyPhase::preparing;
    }
    void phase(VerifyPhase phase) {
        std::lock_guard<std::mutex> lock(mutex_); value_.phase = phase;
    }
    template<class Fn> bool if_running(Fn&& fn) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!verify_rescue_eligible(value_.phase)) return false;
        fn(); // Only a bounded host signal here; never call a driver or wait.
        return true;
    }
    VerifyWindowView read() const {
        std::lock_guard<std::mutex> lock(mutex_); return value_;
    }
};
// Destruction of a registered object MUST start with remove(). The closing slot
// cannot be reused until all callbacks that acquired it have returned.
template<class T, std::size_t Capacity> class CallbackRegistry {
    struct Entry { T* object = nullptr; std::size_t leases = 0; bool closing = false; };
    std::array<Entry, Capacity> entries_{};
    std::mutex mutex_;
    std::condition_variable cv_;
    class Lease {
        CallbackRegistry& owner_; Entry& entry_;
    public:
        Lease(CallbackRegistry& owner, Entry& entry) : owner_(owner), entry_(entry) {}
        ~Lease() {
            std::lock_guard<std::mutex> lock(owner_.mutex_);
            --entry_.leases; owner_.cv_.notify_all();
        }
    };
public:
    bool add(T* object) {
        if (!object) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& e : entries_) if (e.object == object) return !e.closing;
        for (auto& e : entries_) if (!e.object && !e.closing) { e.object = object; return true; }
        return false;
    }
    void remove(T* object) {
        if (!object) return;
        std::unique_lock<std::mutex> lock(mutex_);
        for (auto& e : entries_) if (e.object == object) {
            e.closing = true;
            cv_.wait(lock, [&] { return e.leases == 0; });
            e.object = nullptr; e.closing = false;
            return;
        }
    }
    template<class Fn> void visit(Fn&& fn) {
        for (auto& e : entries_) {
            T* object;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!e.object || e.closing) continue;
                object = e.object; ++e.leases;
            }
            Lease lease(*this, e);
            fn(*object);
        }
    }
};
// A wakeable, scoped periodic observer. Declare after the objects it observes.
class ScopedWatchdog {
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stopped_ = false;
    std::thread thread_;
public:
    ScopedWatchdog() = default;
    ScopedWatchdog(const ScopedWatchdog&) = delete;
    ScopedWatchdog& operator=(const ScopedWatchdog&) = delete;
    template<class Fn> void start(Fn tick, std::chrono::milliseconds period = std::chrono::seconds(1)) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (stopped_ || thread_.joinable()) throw std::logic_error("watchdog can only be started once");
        thread_ = std::thread([this, period, tick = std::move(tick)]() mutable {
            std::unique_lock<std::mutex> lock(mutex_);
            while (!cv_.wait_for(lock, period, [this] { return stopped_; })) {
                lock.unlock(); tick(); lock.lock();
            }
        });
    }
    void stop() {
        { std::lock_guard<std::mutex> lock(mutex_); stopped_ = true; }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }
    ~ScopedWatchdog() { stop(); }
};
// All HOST accesses to a host-written mapped flag use the same atomic_ref.
// This establishes host/host ordering only; device visibility still uses the
// existing mapped allocation, fences and backend completion protocol.
static_assert(std::atomic_ref<uint32_t>::is_always_lock_free, "mapped rescue flags require lock-free host atomics");
static_assert(std::atomic_ref<uint32_t>::required_alignment <= 64, "mapped flag allocation alignment");
inline uint32_t host_flag_load(uint32_t* flag) {
    return std::atomic_ref<uint32_t>(*flag).load(std::memory_order_seq_cst);
}
inline void host_flag_store(uint32_t* flag, uint32_t value) {
    std::atomic_ref<uint32_t>(*flag).store(value, std::memory_order_seq_cst);
}
inline void host_flag_raise(uint32_t* flag, uint32_t value) {
    std::atomic_ref<uint32_t> word(*flag);
    uint32_t current = word.load(std::memory_order_seq_cst);
    while (current < value && !word.compare_exchange_weak(current, value, std::memory_order_seq_cst)) {}
}
} // namespace strata::core
