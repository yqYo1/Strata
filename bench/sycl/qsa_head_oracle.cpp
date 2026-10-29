// Authored for Strata; repository LICENSE applies.
// Reuse the independent host/state oracle with explicit component entrypoints.
#include "qsa-head.hpp"
#include <dlfcn.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>

namespace strata::kernels {
#define HEAD_ARGUMENTS const uint16_t *query, const QsaAttnPools &pools, \
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape, \
    uint16_t *output, int64_t queries, const QsaPromptBf16Bounds &bounds, \
    int64_t live_cells, const QsaBf16PackedScratch &scratch
#define HEAD_VALUES query,pools,ids,steps,cap,shape,output,queries,bounds,live_cells,scratch
bool head_candidate(HEAD_ARGUMENTS, void *stream) {
  return qsa_head_candidate::batch(HEAD_VALUES,stream);
}
bool head_original(HEAD_ARGUMENTS, void *stream) {
  return qsa_head_original::batch(HEAD_VALUES,stream);
}
bool head_state(HEAD_ARGUMENTS, bool candidate, QsaPrefixStateProbe probe, void *stream) {
  return candidate ? qsa_head_candidate::capture(HEAD_VALUES,stream,probe) :
                     qsa_head_original::capture(HEAD_VALUES,stream,probe);
}
#undef HEAD_ARGUMENTS
#undef HEAD_VALUES
}
#define STRATA_QSA_COMPARE_PREFIX33 1
#define STRATA_QSA_COMPARE_PREFIX_STATE 1
#define STRATA_QSA_REAL_HEAD_STATE 1
#define qsa_bf16_prefix_batch head_candidate
#define qsa_bf16_prefix_optimized_batch head_original
#define qsa_bf16_prefix_state_probe_batch head_state
#define main qsa_head_existing_oracle
#include "../../tests/sycl/qsa_bf16_prefix.cpp"
#undef main
#undef qsa_bf16_prefix_batch
#undef qsa_bf16_prefix_optimized_batch
#undef qsa_bf16_prefix_state_probe_batch

static void properties(const char *path) {
  void *loader=dlopen(path,RTLD_NOLOAD|RTLD_LOCAL|RTLD_NOW);
  require(loader,"qualified Level Zero loader not loaded");
  const auto query=reinterpret_cast<decltype(&zeKernelGetProperties)>(dlsym(loader,"zeKernelGetProperties"));
  require(query,"Level Zero kernel property API missing");
  auto &queue=rt->compute();int count=0;
  for (const auto &id:sycl::get_kernel_ids()) {
    const std::string name=id.get_name();
    const bool candidate=name.find("qsa_head_candidate")!=std::string::npos;
    if (!candidate && name.find("qsa_head_original")==std::string::npos) continue;
    auto bundle=sycl::get_kernel_bundle<sycl::bundle_state::executable>(queue.get_context(),{queue.get_device()},{id});
    auto kernel=bundle.get_kernel(id);
    const auto sg=kernel.get_info<sycl::info::kernel_device_specific::compile_sub_group_size>(queue.get_device());
    if (sg!=16) continue; // The separately generated prefix pack has no SG16 requirement.
    const size_t wg=128,declared=20036;
    const auto maximum=kernel.get_info<sycl::info::kernel_device_specific::work_group_size>(queue.get_device());
    ze_kernel_properties_t p{};p.stype=ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
    require(query(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(kernel),&p)==ZE_RESULT_SUCCESS,"kernel properties failed");
    require(maximum>=wg && p.requiredSubgroupSize==16 && p.maxNumSubgroups>=wg/16,"compiled group/subgroup limits");
    require((!p.requiredGroupSizeX || p.requiredGroupSizeX==wg) &&
        (!p.requiredGroupSizeY || p.requiredGroupSizeY==1) && (!p.requiredGroupSizeZ || p.requiredGroupSizeZ==1) &&
        (!p.requiredNumSubGroups || p.requiredNumSubGroups==wg/16),"compiled required shape differs");
    require(size_t(p.localMemSize)+declared+4096<=queue.get_device().get_info<sycl::info::device::local_mem_size>(),"conservative local budget");
    std::cout<<"HEAD_PROPERTY arm="<<(candidate?"candidate":"original")<<" name="<<name
      <<" wg="<<wg<<" subgroup="<<sg<<" arguments="<<p.numKernelArgs
      <<" maximum="<<maximum<<" static_local="<<p.localMemSize<<" declared_local="<<declared
      <<" spill_descriptor="<<p.spillMemSize<<'\n';++count;
  }
  require(count==4,"exactly four normal/state attention kernels required");
  dlclose(loader);
}
int main(int argc,char **argv) {
  try {
    require(argc>=2,"explicit mode required");const std::string mode=argv[1];
    if (mode=="--component" || mode=="--smoke") return qsa_head_existing_oracle(argc,argv);
    require(mode=="--tiny" || mode=="--properties","unknown mode");
    gate(true,"before-runtime");rt=sycl_backend::runtime_for();Drain pending;
    require(rt->device().get_info<sycl::info::device::name>()=="Intel(R) Arc(TM) B570 Graphics","qualified B570 required");
    gate(true,"runtime-ready");
    if (mode=="--properties") {require(argc==3,"pinned Level Zero loader path required");properties(argv[2]);}
    else {require(argc==2,"tiny arguments");run(17,17,1,false,false);}
    rt->wait();gate(true,"after-cleanup");
    std::cout<<"HEAD_DONE mode="<<mode<<" passed=1\n";return 0;
  } catch(const std::exception &error) {drain();std::cerr<<"HEAD_ERROR "<<error.what()<<'\n';return 1;}
}
