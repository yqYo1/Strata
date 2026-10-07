#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/root_group.hpp>
#include <cstdio>
#include <string>
int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("device %s\n",q.get_device().get_info<sycl::info::device::name>().c_str());
    size_t matches=0;
    for (const auto& id:sycl::get_kernel_ids()) {
        const std::string name=id.get_name();
        if (name.find("dequant_flat_kernel_d891c8")==std::string::npos &&
            name.find("dequant_gu_kernel_2397ca")==std::string::npos) continue;
        const auto bundle=sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(),{q.get_device()},{id});
        const auto k=bundle.get_kernel(id);
        const auto maximum=k.ext_oneapi_get_info<sycl::ext::oneapi::experimental::info::kernel_queue_specific::max_num_work_groups>(q,sycl::range<3>(1,1,32),0);
        std::printf("kernel %s max_workgroups_installed_query %zu actual_flat 6400 actual_GU 12800\n",name.c_str(),maximum);
        ++matches;
    }
    std::printf("queried %zu kernels; no kernel submitted\n",matches);
    return matches==2?0:1;
}
