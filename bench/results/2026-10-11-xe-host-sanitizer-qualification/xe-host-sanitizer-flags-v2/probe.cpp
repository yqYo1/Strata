#include <sycl/sycl.hpp>
void kernel(sycl::queue &q, int *p) { q.single_task([=] { p[0] = p[1]; }); }
int host(int *p, unsigned i) { return p[i] + 1; }
