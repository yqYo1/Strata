#include <cstdint>
#include <iostream>
#include "collector_options.h"
int main() {
    CollectorOptions host;
    host.host_timing = 1;
    host.chrome_call_logging = 1;
    host.log_to_file = 1;
    host.output_dir_path = 1;
    host.DeriveFlags();
    std::cout << "host-only api=" << host.api_tracing
              << " kernel=" << host.kernel_tracing
              << " device=" << host.device_timing
              << " metric=" << host.metric_query << "\n";
    if (!host.api_tracing || host.kernel_tracing || host.device_timing ||
        host.device_timeline || host.kernel_submission || host.chrome_kernel_logging ||
        host.chrome_device_logging || host.metric_query || host.metric_stream ||
        host.stall_sampling) return 1;
    CollectorOptions bare;
    bare.DeriveFlags();
    std::cout << "option-free api=" << bare.api_tracing
              << " kernel=" << bare.kernel_tracing
              << " device=" << bare.device_timing << "\n";
    if (!bare.host_timing || !bare.device_timing || !bare.kernel_tracing) return 2;
}
