// Authored for Strata; repository LICENSE (MIT) applies. Source only / UNRUN.
// R810 numerics_chain.cpp is compiled unchanged except -Dmain=r810_fixture_main.
// Its Runtime is private to this executable. No fixture Runtime enters benchmark.
#include "evidence.hpp"
int r810_fixture_main(int argc, char **argv);
int main(int argc, char **argv) {
  try {
    r816::libraries("before_runtime");
    const int result=r810_fixture_main(argc,argv);
    // Original PASS lines follow CheckedDrain::finish(). Original return zero
    // plus this marker and normal Owner closure are all mandatory. A zero count
    // of async errors by itself is not a drain/completion proof.
    r816::libraries("after_fixture_return");
    std::cout << "{\"kind\":\"fixture_return\",\"exit_code\":" << result
      << ",\"pid\":" << getpid() << "}" << std::endl;
    return result;
  } catch (const std::exception &e) {
    std::cerr << "R816_EVIDENCE_FAILED: " << e.what() << std::endl;
    return 87;
  }
}
