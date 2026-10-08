#include "strata/host_wait.hpp"
#include <atomic>
#include <cassert>
#include <fstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
int main(int argc, char** argv) {
    assert(argc==2);
    unsigned calls=0;
    assert(strata::wait_host_ready([&]{++calls;return true;},[]{return false;},"ready"));
    assert(calls==1);
    calls=0;
    assert(!strata::wait_host_ready([&]{++calls;return true;},[]{return true;},"cancel"));
    assert(calls==0); // cancellation takes priority over a fresh readiness query
    const auto start=std::chrono::steady_clock::now();
    calls=0;
    assert(strata::wait_host_ready([&]{++calls;return std::chrono::steady_clock::now()-start>=std::chrono::milliseconds(20);},
        []{return false;},"pending",std::chrono::seconds(1)));
    std::printf("pending readiness queries=%u elapsed_us=%lld\n",calls,
        (long long)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count());
    std::atomic<bool> stop{false};
    std::thread cancel([&]{std::this_thread::sleep_for(std::chrono::milliseconds(5));stop.store(true,std::memory_order_release);});
    assert(!strata::wait_host_ready([]{return false;},[&]{return stop.load(std::memory_order_acquire);},"cancel pending",std::chrono::seconds(1)));
    cancel.join();
    const pid_t child=fork();assert(child>=0);
    if(child==0) {
        struct Guard { const char* path; ~Guard(){std::ofstream(path)<<"unsafe cleanup";} } guard{argv[1]};
        strata::wait_host_ready([]{return false;},[]{return false;},"timeout pending",std::chrono::milliseconds(20));
        return 9;
    }
    int status=0;assert(waitpid(child,&status,0)==child);
    assert(WIFEXITED(status) && WEXITSTATUS(status)==1);
    assert(access(argv[1],F_OK)!=0); // timeout must not destroy possibly-active GPU buffers
}
