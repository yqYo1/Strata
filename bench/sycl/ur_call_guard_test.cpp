// Main-authored host-only mutation controls; repository LICENSE applies.
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const char* name = argv[1];
  if (std::strcmp(name, "setenv") && std::strcmp(name, "unsetenv") &&
      std::strcmp(name, "putenv") && std::strcmp(name, "clearenv")) return 2;
  std::printf("mutation-case-begin:%s\n", name);
  if (std::fflush(stdout)) return 3;
  char value[] = "STRATA_MUTATION_TEST=1";
  if (!std::strcmp(name, "setenv")) setenv("STRATA_MUTATION_TEST", "1", 1);
  else if (!std::strcmp(name, "unsetenv")) unsetenv("STRATA_MUTATION_TEST");
  else if (!std::strcmp(name, "putenv")) putenv(value);
  else clearenv();
  return 99; // Returning from a write means the interposition did not protect it.
}
