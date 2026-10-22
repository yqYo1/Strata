// Main-authored capability gate; repository LICENSE applies.
// Reuse only this project's JSON/progress helpers, never its inventory main.
#define main metric_inventory_reference_main
#include "metric_inventory.cpp"
#undef main

int main(int argc,char**) {
  ze_context_handle_t context=nullptr;zet_metric_query_pool_handle_t pool=nullptr;
  zet_metric_query_handle_t query=nullptr;ze_device_handle_t target=nullptr;
  bool activated=false,closed=true;ze_result_t result=ZE_RESULT_SUCCESS;
  const char* failed="none";
  auto cleanup=[&] {
    if (!closed) return; // Never retry a failed destroy or retire its parent.
    if (query) {if (call("zetMetricQueryDestroy",[&]{return zetMetricQueryDestroy(query);})!=ZE_RESULT_SUCCESS) {closed=false;return;}query=nullptr;}
    if (pool) {if (call("zetMetricQueryPoolDestroy",[&]{return zetMetricQueryPoolDestroy(pool);})!=ZE_RESULT_SUCCESS) {closed=false;return;}pool=nullptr;}
    if (activated) {if (call("zetContextActivateMetricGroups/clear",[&]{return zetContextActivateMetricGroups(context,target,0,nullptr);})!=ZE_RESULT_SUCCESS) {closed=false;return;}activated=false;}
    if (context) {if (call("zeContextDestroy",[&]{return zeContextDestroy(context);})!=ZE_RESULT_SUCCESS) {closed=false;return;}context=nullptr;}
  };
  try {
    std::cout.setf(std::ios::unitbuf);require(argc==1,"no command options");
    const char* enabled=std::getenv("ZET_ENABLE_METRICS");require(enabled && std::string(enabled)=="1","metrics enabled before initialization");
    checked("zeInit",[]{return zeInit(ZE_INIT_FLAG_GPU_ONLY);});
    uint32_t drivers=0;checked("zeDriverGet/count",[&]{return zeDriverGet(&drivers,nullptr);});
    require(drivers>0 && drivers<=32,"driver cap");const auto dc=drivers;
    std::vector<ze_driver_handle_t> dh(drivers);checked("zeDriverGet/list",[&]{return zeDriverGet(&drivers,dh.data());});
    require(drivers==dc,"driver count changed");ze_driver_handle_t selected=nullptr;unsigned matches=0;
    for (auto driver:dh) {
      uint32_t devices=0;checked("zeDeviceGet/count",[&]{return zeDeviceGet(driver,&devices,nullptr);});
      require(devices<=32,"device cap");if (!devices) continue;
      const auto capacity=devices;std::vector<ze_device_handle_t> dev(devices);
      checked("zeDeviceGet/list",[&]{return zeDeviceGet(driver,&devices,dev.data());});require(devices==capacity,"device count changed");
      for (auto device:dev) {
        ze_device_properties_t prop{};prop.stype=ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        checked("zeDeviceGetProperties",[&]{return zeDeviceGetProperties(device,&prop);});
        if (prop.type==ZE_DEVICE_TYPE_GPU && prop.vendorId==0x8086 && prop.deviceId==0xe20c) {target=device;selected=driver;++matches;}
      }
    }
    require(matches==1,"unique E20C required");
    uint32_t groups=0;checked("zetMetricGroupGet/count",[&]{return zetMetricGroupGet(target,&groups,nullptr);});
    require(groups>0 && groups<=128,"group cap");const auto gc=groups;
    std::vector<zet_metric_group_handle_t> gh(groups);checked("zetMetricGroupGet/list",[&]{return zetMetricGroupGet(target,&groups,gh.data());});
    require(groups==gc,"group count changed");zet_metric_group_handle_t group=nullptr;unsigned group_matches=0;
    for (auto candidate:gh) {
      zet_metric_group_properties_t gp{};gp.stype=ZET_STRUCTURE_TYPE_METRIC_GROUP_PROPERTIES;
      checked("zetMetricGroupGetProperties",[&]{return zetMetricGroupGetProperties(candidate,&gp);});
      if (quoted(gp.name)=="\"ComputeBasic\"" && gp.samplingType==ZET_METRIC_GROUP_SAMPLING_TYPE_FLAG_EVENT_BASED) {
        require(gp.domain==3 && gp.metricCount==83,"selected group changed since inventory");group=candidate;++group_matches;
      }
    }
    require(group_matches==1,"unique event-based ComputeBasic required");
    ze_context_desc_t cd{};cd.stype=ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    ze_context_handle_t created_context=nullptr;
    checked("zeContextCreate",[&]{return zeContextCreate(selected,&cd,&created_context);});
    require(created_context,"null created context");context=created_context;
    failed="zetContextActivateMetricGroups";
    activated=true; // Clear the owned context's request on either result.
    result=call(failed,[&]{return zetContextActivateMetricGroups(context,target,1,&group);});
    if (result==ZE_RESULT_SUCCESS) {
      zet_metric_query_pool_desc_t pd{};pd.stype=ZET_STRUCTURE_TYPE_METRIC_QUERY_POOL_DESC;
      pd.type=ZET_METRIC_QUERY_POOL_TYPE_PERFORMANCE;pd.count=1;
      failed="zetMetricQueryPoolCreate";
      zet_metric_query_pool_handle_t created_pool=nullptr;
      result=call(failed,[&]{return zetMetricQueryPoolCreate(context,target,group,&pd,&created_pool);});
      if (result==ZE_RESULT_SUCCESS) {
        require(created_pool,"null created query pool");pool=created_pool;failed="zetMetricQueryCreate";
        zet_metric_query_handle_t created_query=nullptr;
        result=call(failed,[&]{return zetMetricQueryCreate(pool,0,&created_query);});
        if (result==ZE_RESULT_SUCCESS) {require(created_query,"null created query");query=created_query;}
      }
    }
    cleanup();libraries();
    std::cout<<"{\"kind\":\"summary\",\"status\":\""<<(result==ZE_RESULT_SUCCESS ? "setup_ready" : "api_error")
      <<"\",\"last_api\":\""<<failed<<"\",\"result\":"<<uint32_t(result)<<",\"objects_closed\":"<<(closed ? "true" : "false")
      <<",\"GPU_commands\":0,\"collection_qualified\":false}\n";
    return !closed ? 2 : result==ZE_RESULT_SUCCESS ? 0 : 3;
  } catch (const std::exception& error) {cleanup();std::cerr<<"metric query setup failure: "<<error.what()<<'\n';return 2;}
}
