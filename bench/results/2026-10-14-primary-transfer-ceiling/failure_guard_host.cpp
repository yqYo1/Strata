#include "transfer_failure_guard.hpp"
#include <stdexcept>
#include <string>

struct HostCopySource {
  ~HostCopySource() { std::fputs("host-copy-source-destroyed\n", stderr); }
};

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  try {
    HostCopySource source;
    TransferFailureGuard guard;
    const std::string mode = argv[1];
    if (mode == "pending-error") throw std::runtime_error("pending copy");
    guard.closed();
    if (mode == "closed-error") throw std::runtime_error("completed copy");
    if (mode != "normal") return 2;
    return 0;
  } catch (const std::exception&) {
    std::fputs("outer-catch\n", stderr);
    return 7;
  }
}
