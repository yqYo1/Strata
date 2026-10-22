// Main-authored read-only Level Zero inventory; repository LICENSE applies.
// No context, queue, allocation, activation, query, streamer or command API.
#include <level_zero/ze_api.h>
#include <level_zero/zet_api.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <link.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool ok,const char* why) { if (!ok) throw std::runtime_error(why); }
template<size_t N> std::string quoted(const char (&s)[N]) {
  const char* end=std::find(s,s+N,'\0');require(end!=s+N,"unterminated driver property");
  std::string out="\"";constexpr char hex[]="0123456789abcdef";
  for (const char* p=s;p!=end;++p) {
    const auto c=static_cast<unsigned char>(*p);
    if (c=='"' || c=='\\') {out+='\\';out+=char(c);}
    else if (c<32) {out+="\\u00";out+=hex[c>>4];out+=hex[c&15];}
    else out+=char(c);
  }
  return out+'"';
}
template<class Call> ze_result_t call(const char* name,Call fn) {
  std::cout<<"{\"kind\":\"begin\",\"api\":\""<<name<<"\"}\n";
  const auto result=fn();
  std::cout<<"{\"kind\":\"return\",\"api\":\""<<name<<"\",\"result\":"<<uint32_t(result)<<"}\n";
  return result;
}
template<class Call> void checked(const char* name,Call fn) {
  require(call(name,fn)==ZE_RESULT_SUCCESS,name);
}
int loaded(dl_phdr_info* info,size_t,void*) {
  if (info->dlpi_name && info->dlpi_name[0]=='/') {
    std::string path=info->dlpi_name;
    require(path.find_first_of("\"\\\n\r\t")==std::string::npos,"unexpected ELF path");
    std::cout<<"{\"kind\":\"elf\",\"path\":\""<<path<<"\"}\n";
  }
  return 0;
}
void libraries() {dl_iterate_phdr(loaded,nullptr);}
}

int main(int argc,char** argv) {
  try {
    std::cout.setf(std::ios::unitbuf);
    require(argc==2 && (std::string(argv[1])=="--device-only" || std::string(argv[1])=="--metrics"),"explicit inventory mode required");
    const bool metrics=std::string(argv[1])=="--metrics";
    const char* enabled=std::getenv("ZET_ENABLE_METRICS");
    require(enabled && std::string(enabled)==(metrics ? "1" : "0"),"immutable startup metric mode");
    checked("zeInit",[]{return zeInit(ZE_INIT_FLAG_GPU_ONLY);});
    uint32_t drivers=0;checked("zeDriverGet/count",[&]{return zeDriverGet(&drivers,nullptr);});
    require(drivers>0 && drivers<=32,"driver count cap");
    std::vector<ze_driver_handle_t> dh(drivers);const uint32_t driver_capacity=drivers;
    checked("zeDriverGet/list",[&]{return zeDriverGet(&drivers,dh.data());});
    require(drivers==driver_capacity,"driver inventory changed");
    ze_device_handle_t target=nullptr;unsigned matches=0;
    for (auto driver:dh) {
      uint32_t devices=0;checked("zeDeviceGet/count",[&]{return zeDeviceGet(driver,&devices,nullptr);});
      require(devices<=32,"device count cap");if (!devices) continue;
      const uint32_t capacity=devices;std::vector<ze_device_handle_t> handles(devices);
      checked("zeDeviceGet/list",[&]{return zeDeviceGet(driver,&devices,handles.data());});
      require(devices==capacity,"device inventory changed");
      for (auto device:handles) {
        ze_device_properties_t prop{};prop.stype=ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        checked("zeDeviceGetProperties",[&]{return zeDeviceGetProperties(device,&prop);});
        std::cout<<"{\"kind\":\"device\",\"name\":"<<quoted(prop.name)<<",\"vendor\":"<<prop.vendorId
          <<",\"device\":"<<prop.deviceId<<",\"type\":"<<prop.type<<"}\n";
        if (prop.type==ZE_DEVICE_TYPE_GPU && prop.vendorId==0x8086 && prop.deviceId==0xe20c) {target=device;++matches;}
      }
    }
    require(matches==1 && target,"unique E20C GPU required");
    uint32_t groups=0,total=0;
    if (metrics) {
      const auto result=call("zetMetricGroupGet/count",[&]{return zetMetricGroupGet(target,&groups,nullptr);});
      if (result!=ZE_RESULT_SUCCESS) {
        libraries();std::cout<<"{\"kind\":\"summary\",\"status\":\"api_error\",\"result\":"<<uint32_t(result)<<",\"complete\":false}\n";return 3;
      }
      require(groups<=128,"group count cap");
      if (groups) {
        const uint32_t capacity=groups;std::vector<zet_metric_group_handle_t> handles(groups);
        checked("zetMetricGroupGet/list",[&]{return zetMetricGroupGet(target,&groups,handles.data());});
        require(groups==capacity,"metric group inventory changed");
        for (uint32_t g=0;g<groups;++g) {
          zet_metric_group_properties_t gp{};gp.stype=ZET_STRUCTURE_TYPE_METRIC_GROUP_PROPERTIES;
          checked("zetMetricGroupGetProperties",[&]{return zetMetricGroupGetProperties(handles[g],&gp);});
          std::cout<<"{\"kind\":\"group\",\"group\":"<<g<<",\"name\":"<<quoted(gp.name)<<",\"description\":"<<quoted(gp.description)
            <<",\"domain\":"<<gp.domain<<",\"sampling\":"<<gp.samplingType<<",\"metrics\":"<<gp.metricCount<<"}\n";
          uint32_t count=0;checked("zetMetricGet/count",[&]{return zetMetricGet(handles[g],&count,nullptr);});
          require(count==gp.metricCount && count<=1024 && total+count<=8192,"metric count consistency/cap");
          total+=count;if (!count) continue;
          const uint32_t mc=count;std::vector<zet_metric_handle_t> members(count);
          checked("zetMetricGet/list",[&]{return zetMetricGet(handles[g],&count,members.data());});
          require(count==mc,"metric member inventory changed");
          for (uint32_t m=0;m<count;++m) {
            zet_metric_properties_t mp{};mp.stype=ZET_STRUCTURE_TYPE_METRIC_PROPERTIES;
            checked("zetMetricGetProperties",[&]{return zetMetricGetProperties(members[m],&mp);});
            std::cout<<"{\"kind\":\"metric\",\"group\":"<<g<<",\"metric\":"<<m<<",\"name\":"<<quoted(mp.name)
              <<",\"description\":"<<quoted(mp.description)<<",\"component\":"<<quoted(mp.component)<<",\"tier\":"<<mp.tierNumber
              <<",\"metric_type\":"<<mp.metricType<<",\"result_type\":"<<mp.resultType<<",\"units\":"<<quoted(mp.resultUnits)<<"}\n";
          }
        }
      }
    }
    libraries();std::cout<<"{\"kind\":\"summary\",\"status\":\"success\",\"complete\":true,\"groups\":"<<groups
      <<",\"metrics\":"<<total<<",\"GPU_commands\":0}\n";return 0;
  } catch (const std::exception& error) {
    std::cerr<<"metric inventory failure: "<<error.what()<<'\n';return 2;
  }
}
