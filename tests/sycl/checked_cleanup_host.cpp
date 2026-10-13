#include "strata/sycl/checked_cleanup.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace {
int exit_marker = -1;

void require(bool ok, const char *message) {
  if (!ok) {
    std::fprintf(stderr, "%s\n", message);
    std::abort();
  }
}

void child_case(bool drain_fails, bool release_fails) {
  int fd[2];
  require(pipe(fd) == 0, "pipe failed");
  const pid_t child = fork();
  require(child >= 0, "fork failed");
  if (child == 0) {
    close(fd[0]);
    exit_marker = fd[1];
    require(std::atexit([] { (void)write(exit_marker, "X", 1); }) == 0,
            "atexit registration failed");
    struct UnwindOwner {
      int output;
      ~UnwindOwner() { (void)write(output, "U", 1); }
    } owner{fd[1]};
    strata::sycl_backend::checked_cleanup(
        [&] {
          (void)write(fd[1], "D", 1);
          if (drain_fails)
            throw std::runtime_error("injected completion failure");
        },
        [&] {
          (void)write(fd[1], "R", 1);
          if (release_fails)
            throw std::runtime_error("injected release failure");
        },
        "injected failed allocation retirement");
    (void)write(fd[1], "A", 1);
    std::_Exit(37); // The failure path must not resume here.
  }
  close(fd[1]);
  std::string observed;
  char c;
  for (;;) {
    const auto count = read(fd[0], &c, 1);
    if (count == 1) observed += c;
    else if (count < 0 && errno == EINTR) continue;
    else { require(count == 0, "read failed"); break; }
  }
  close(fd[0]);
  int status = 0;
  require(waitpid(child, &status, 0) == child, "child was not reaped");
  require(WIFEXITED(status), "child did not exit normally");
  require(WEXITSTATUS(status) == EXIT_FAILURE, "failed retirement did not fail-stop");
  require(observed == (drain_fails ? "D" : "DR"),
          "failed retirement released, resumed or unwound an owner");
}
} // namespace

int main() {
  int state = 0;
  strata::sycl_backend::checked_cleanup(
      [&] { require(state == 0, "wrong drain order"); state = 1; },
      [&] { require(state == 1, "release without drain"); state = 2; },
      "successful retirement");
  require(state == 2, "successful retirement did not release exactly once");
  child_case(true, false);
  child_case(false, true);
  std::puts("checked drain/free order and fail-stop before release/unwind: PASS");
  return 0;
}
