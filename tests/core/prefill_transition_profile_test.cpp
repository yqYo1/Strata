// Authored for Strata; repository LICENSE (MIT) applies.
// Host-only observer contracts: no SYCL runtime and no GPU work.
#include "strata/sycl/prefill_transition_profile.hpp"
#include <stdexcept>
#include <thread>
#include <type_traits>
namespace tp = strata::sycl_backend::prefill_transition_profile;
namespace hp = strata::sycl_backend::cpu_wait_profile;
static_assert(!std::is_copy_constructible_v<tp::Scope>);
static_assert(!std::is_move_constructible_v<tp::Part>);
static_assert(std::is_nothrow_destructible_v<tp::Scope>);
static_assert(std::is_nothrow_destructible_v<tp::Part>);
void require(bool value) { if (!value) throw std::runtime_error("transition observer contract"); }
void work() { volatile unsigned n=1; for(unsigned i=0;i<100000;++i)n=n*1664525u+1013904223u; }
int main() try {
    unsigned cases=0;
    { // Disabled observation must neither install TLS nor count/clock work.
        tp::Scope scope(false,"disabled"); { tp::Part part(tp::allocate); work(); }
        require(!tp::active && !hp::active && !scope.finish(true).available); ++cases;
    }
    { // Part wall is disjoint; nested waits are a subinterval, not added service.
        tp::Scope scope(true,"valid");
        { tp::Part part(tp::allocate,3,256); work(); }
        { tp::Part part(tp::drain,1); hp::Wait wait(hp::compat_queue);
          std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
        std::thread child([] { require(!tp::active && !hp::active); tp::Part p(tp::publish); work(); });
        child.join(); work();
        auto s=scope.finish(true);
        require(s.available && s.complete && !tp::active && !hp::active &&
            s.cells[tp::allocate].calls==1 && s.cells[tp::allocate].items==3 &&
            s.cells[tp::allocate].logical_bytes==256 && s.cells[tp::publish].calls==0 &&
            s.cells[tp::drain].wait_wall>0 && s.cells[tp::drain].wait_cpu>=0 &&
            s.other_wall>=0 && s.other_cpu>=0);
        require(scope.finish(false).complete); ++cases; // Idempotent closure.
    }
    { tp::Scope scope(true,"nested_parts");
      {tp::Part a(tp::allocate);tp::Part b(tp::submit);work();}
      require(!scope.finish(true).available); ++cases; }
    { tp::Scope scope(true,"part_inside_wait");
      {hp::Wait w(hp::compat_queue);tp::Part p(tp::submit);work();}
      require(!scope.finish(true).available); ++cases; }
    { tp::Scope scope(true,"exception");
      try{tp::Part p(tp::submit);work();throw 1;}catch(int){}
      auto s=scope.finish(true);require(s.available && !s.complete && s.cells[tp::submit].throws==1);++cases; }
    { tp::Scope scope(true,"caught_wait_exception");
      {tp::Part p(tp::drain);try{hp::Wait w(hp::compat_queue);work();throw 1;}catch(int){}}
      auto s=scope.finish(true);require(s.available && !s.complete && s.wait_errors==1);++cases; }
    { tp::Scope outer(true,"outer");
      {tp::Scope inner(true,"nested_scope");require(!inner.finish(true).available);}
      require(!outer.finish(true).available);++cases; }
    { hp::Scope outer(true); auto* original=hp::active;
      {tp::Scope inner(true,"existing_wait_scope");require(!inner.finish(true).available && hp::active==original);}
      require(original->invalid); ++cases; }
    { tp::Scope scope(true,"open_part");
      {tp::Part p(tp::submit);require(!scope.finish(true).available);}
      require(!tp::active && !hp::active);++cases; }
    { tp::Scope scope(true,"overflow");
      {tp::Part p(tp::submit,UINT64_MAX,UINT64_MAX);work();}
      {tp::Part p(tp::submit,1,1);work();}
      require(!scope.finish(true).available);++cases; }
    { tp::Scope scope(true,"emit");
      {tp::Part p(tp::release,1);work();}
      scope.emit(true); require(scope.finish(true).complete); ++cases; }
    require(!tp::active && !hp::active);
    std::printf("prefill transition host: cases=%u passed=1\n",cases); return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
