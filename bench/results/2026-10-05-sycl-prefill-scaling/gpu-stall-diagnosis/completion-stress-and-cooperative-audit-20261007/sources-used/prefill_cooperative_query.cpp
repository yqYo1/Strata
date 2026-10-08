#include <sycl/sycl.hpp>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

int main(int argc,char** argv) try {
    if (argc!=2) throw std::runtime_error("one target metadata path required");
    sycl::queue q(sycl::gpu_selector_v,sycl::property::queue::in_order{});
    std::map<std::string,sycl::kernel_id> ids;
    for (const auto& id:sycl::get_kernel_ids()) ids.emplace(id.get_name(),id);
    std::ifstream input(argv[1]);if (!input) throw std::runtime_error("missing query targets");
    size_t queried=0,exceeds=0;
    for (std::string line;std::getline(input,line);) {
        std::istringstream row(line);
        std::string name;size_t x,y,z,observed;
        if (!(row>>name>>x>>y>>z>>observed)) throw std::runtime_error("invalid target row");
        const auto found=ids.find(name);if (found==ids.end()) throw std::runtime_error("target kernel absent: "+name);
        const auto bundle=sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(),{q.get_device()},{found->second});
        const auto kernel=bundle.get_kernel(found->second);
        const auto maximum=kernel.ext_oneapi_get_info<sycl::ext::oneapi::experimental::info::kernel_queue_specific::max_num_work_groups>(q,sycl::range<3>(x,y,z),0);
        ++queried;exceeds+=observed>maximum;
        std::printf("{\"kernel\":\"%s\",\"sycl_local\":[%zu,%zu,%zu],\"dynamic_local_bytes\":0,"
                    "\"max_cooperative_groups\":%zu,\"observed_groups\":%zu,\"exceeds\":%s}\n",
                    name.c_str(),x,y,z,maximum,observed,observed>maximum?"true":"false");
        std::fflush(stdout);
    }
    std::printf("{\"stage\":\"PASS\",\"queried\":%zu,\"exceeds\":%zu,\"kernels_submitted\":0}\n",queried,exceeds);
    return queried?0:1;
} catch (const std::exception& e) {std::fprintf(stderr,"resource query failed: %s\n",e.what());return 1;}
