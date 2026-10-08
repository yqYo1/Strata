
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
int initializations=0;
namespace sycl {
template<int D=1> struct range { range(int) {} };
range(int)->range<1>;
}
namespace dpct {
template<class T,int D> struct global_memory {
    global_memory(sycl::range<1>,std::initializer_list<T>) { ++initializations; }
};
template<class T,int D> struct constant_memory {
    constant_memory(int,int) { ++initializations; }
};
}
#include "tables.inc"
int main() {
#if CONTROL
    assert(initializations==2);
    delete &iq4nl_values; delete &c_codes;
    std::puts("PASS old table constructors ran before main");
#else
    assert(initializations==0);
    auto* a=&iq4nl_storage(); auto* b=&s2_codes_storage();
    assert(initializations==2 && a==&iq4nl_storage() && b==&s2_codes_storage());
    delete a; delete b; // CPU stand-ins only: no captured users remain.
    std::puts("PASS table constructors delayed until use and retained across calls");
#endif
}
