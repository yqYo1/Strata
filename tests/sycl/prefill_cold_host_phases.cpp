#include "cold_host_phases.hpp"

#include <chrono>
#include <thread>

namespace {
using Phases = strata::prefill::detail::ColdHostPhases;
using Boundary = Phases::Boundary;
namespace wait_profile = strata::sycl_backend::cpu_wait_profile;

void timed_wait() {
    wait_profile::blocking(wait_profile::compat_queue, [] {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    });
}

void complete_phases() {
    timed_wait();
    Phases::mark(Boundary::allocated_plan);
    {
        wait_profile::Host region(wait_profile::upload_submit);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    Phases::mark(Boundary::provisioned);
    timed_wait();
    Phases::mark(Boundary::layer_passes_finished);
}
}

int main() {
    {
        wait_profile::Scope ledger(true);
        Phases disabled(false);
        complete_phases();
        disabled.finish(true);
        ledger.complete(true);
    }
    {
        wait_profile::Scope ledger(true);
        Phases observer(true);
        complete_phases();
        observer.finish(true);
        ledger.complete(true);
    }
    {
        wait_profile::Scope ledger(true);
        Phases observer(true);
        Phases::mark(Boundary::provisioned);
        observer.finish(true);
        ledger.complete(true);
    }
    {
        wait_profile::Scope ledger(true);
        Phases observer(true);
        {
            wait_profile::Host region(wait_profile::upload_submit);
            Phases::mark(Boundary::allocated_plan);
        }
        observer.finish(true);
        ledger.complete(true);
    }
    {
        wait_profile::Scope ledger(true);
        Phases observer(true);
        complete_phases();
        observer.finish(false);
        ledger.complete(false);
    }
}
