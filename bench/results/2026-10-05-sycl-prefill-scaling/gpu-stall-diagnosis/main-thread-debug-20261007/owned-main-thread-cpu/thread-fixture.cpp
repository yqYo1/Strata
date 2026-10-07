#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unistd.h>
std::atomic<bool> ready{false};
[[gnu::noinline]] void main_wait() { for (;;) ::pause(); }
int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "bad-pc") == 0) {
        reinterpret_cast<void(*)()>(1)(); // Intentional CPU crash fixture only.
        return 9;
    }
    const bool crash = argc == 2 && std::strcmp(argv[1], "worker-abort") == 0;
    std::thread worker([crash] {
        while (!ready.load(std::memory_order_acquire)) std::this_thread::yield();
        if (crash) {
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            std::abort();
        }
        for (;;) ::pause();
    });
    ready.store(true, std::memory_order_release);
    main_wait();
}
