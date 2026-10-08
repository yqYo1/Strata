#include <cstdint>
#include <iostream>
#include "collector_options.h"
int main() {
    CollectorOptions o;
    o.device_timing=1; o.chrome_device_logging=1;
    o.log_to_file=1; o.output_dir_path=1; o.DeriveFlags();
    std::cout << "device=" << o.device_timing << " kernel=" << o.kernel_tracing
              << " api=" << o.api_tracing << " metrics=" << o.metric_query << "\n";
    return !(o.device_timing && o.chrome_device_logging && o.kernel_tracing &&
             !o.api_tracing && !o.host_timing && !o.chrome_call_logging &&
             !o.metric_query && !o.metric_stream && !o.stall_sampling &&
             !o.conditional_collection);
}
