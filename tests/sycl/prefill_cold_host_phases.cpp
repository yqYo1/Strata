#include "cold_host_phases.hpp"

#include <chrono>
#include <cstring>
#include <thread>

namespace {
using Phases = strata::prefill::detail::ColdHostPhases;
using Boundary = Phases::Boundary;
using Layer = Phases::LayerBoundary;
using Window = Phases::WindowBoundary;
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

void layer_phases(int64_t count, bool misorder = false) {
    wait_profile::Scope ledger(true);
    Phases observer(true, true);
    Phases::mark(Boundary::allocated_plan);
    Phases::mark(Boundary::provisioned);
    Phases::set_layers(count);
    for (int64_t layer = 0; layer < 2; ++layer) {
        Phases::mark_layer(layer, Layer::begin);
        Phases::mark_layer(layer, misorder ? Layer::drain : Layer::run);
        timed_wait();
        Phases::mark_layer(layer, Layer::drain);
        Phases::mark_layer(layer, Layer::end);
    }
    Phases::mark(Boundary::layer_passes_finished);
    observer.finish(true);
    ledger.complete(true);
}

void window_phases(bool omit_tail, bool gap) {
    wait_profile::Scope ledger(true);
    Phases observer(true, false, true);
    Phases::mark(Boundary::allocated_plan);
    Phases::mark(Boundary::provisioned);
    Phases::set_layers(2, 7);
    const auto window = [](int64_t layer, int64_t first, int64_t rows) {
        Phases::mark_window(layer, first, rows, Window::begin);
        timed_wait();
        Phases::mark_window(layer, first, rows, Window::after_moe);
        timed_wait();
        Phases::mark_window(layer, first, rows, Window::end);
    };
    for (int64_t layer = 0; layer < 2; ++layer) {
        Phases::mark_layer(layer, Layer::begin);
        Phases::mark_layer(layer, Layer::run);
        window(layer, 0, 4);
        if (!omit_tail) window(layer, gap ? 5 : 4, gap ? 2 : 3);
        Phases::mark_layer(layer, Layer::drain);
        Phases::mark_layer(layer, Layer::end);
    }
    Phases::mark(Boundary::layer_passes_finished);
    observer.finish(true);
    ledger.complete(true);
}
}

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--windows") == 0) {
        window_phases(false, false);
        window_phases(true, false);
        window_phases(false, true);
        return 0;
    }
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
    layer_phases(2);
    layer_phases(2, true);
    layer_phases(3);
}
