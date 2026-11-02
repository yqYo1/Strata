// Main-authored control/lifetime regression; repository LICENSE.
#include "strata/sycl/checked_worker.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace sb = strata::sycl_backend;
namespace {
void need(bool ok, const char *message) {
  if (!ok) {
    std::fprintf(stderr, "checked worker host: %s\n", message);
    std::abort();
  }
}

void worker_case(int mode) {
  bool alive = true, finished = false, result = true, main_caught = false;
  int accepted = 0, drains = 0;
  try {
    struct Owner {
      bool &alive;
      bool &finished;
      ~Owner() {
        need(finished, "main unwound a captured owner before joining");
        alive = false;
      }
    } owner{alive, finished};
    std::thread worker;
    sb::CheckedWorkerJoin join(worker);
    worker = std::thread([&] {
      result = sb::checked_worker([&] {
        need(alive, "work used a retired owner");
        ++accepted;
        if (mode == 2) throw std::runtime_error("after accepted work");
        if (mode == 3) throw 17;
        if (mode == 4) throw std::bad_alloc();
        return mode == 0;
      }, [&] {
        need(alive && accepted == 1, "drain followed owner retirement");
        ++drains;
      }, "host accepted-work fixture");
      finished = true;
    });
    throw std::runtime_error("main commit/draft exception");
  } catch (const std::runtime_error &) {
    main_caught = true;
  }
  need(main_caught && finished && !alive && accepted == 1,
       "main exception did not retire the worker before its owner");
  need(result == (mode == 0) && drains == (mode == 0 ? 0 : 1),
       "success added a drain or failure escaped without exactly one drain");
}

int exit_marker = -1;
void failed_drain_case() {
  int fd[2];
  need(pipe(fd) == 0, "pipe failed");
  const pid_t child = fork();
  need(child >= 0, "fork failed");
  if (child == 0) {
    close(fd[0]);
    exit_marker = fd[1];
    need(std::atexit([] { (void)write(exit_marker, "X", 1); }) == 0,
         "atexit registration failed");
    struct Owner {
      int fd;
      ~Owner() { (void)write(fd, "U", 1); }
    } owner{fd[1]};
    bool result = true;
    std::thread worker;
    sb::CheckedWorkerJoin join(worker);
    worker = std::thread([&] {
      result = sb::checked_worker([&]() -> bool {
        (void)write(fd[1], "W", 1);
        throw std::bad_alloc();
      }, [&] {
        (void)write(fd[1], "D", 1);
        throw std::runtime_error("injected drain failure");
      }, "host failed-drain fixture");
    });
    worker.join();
    (void)write(fd[1], result ? "T" : "F", 1);
    std::_Exit(37);
  }
  close(fd[1]);
  std::string observed;
  char value;
  for (;;) {
    const auto count = read(fd[0], &value, 1);
    if (count == 1) observed += value;
    else if (count < 0 && errno == EINTR) continue;
    else { need(count == 0, "pipe read failed"); break; }
  }
  close(fd[0]);
  int status = 0;
  need(waitpid(child, &status, 0) == child, "failed child was not reaped");
  need(WIFEXITED(status) && WEXITSTATUS(status) == EXIT_FAILURE,
       "drain failure did not fail-stop");
  need(observed == "WD", "failed drain resumed,released or unwound an owner");
}
} // namespace

int main() {
  for (int mode = 0; mode < 5; ++mode) worker_case(mode);
  failed_drain_case();
  std::puts("checked_worker_host=PASS success_no_drain=1 failed_work_drains=4 main_unwind_joins=5 failed_drain_retains_owners=1");
}
