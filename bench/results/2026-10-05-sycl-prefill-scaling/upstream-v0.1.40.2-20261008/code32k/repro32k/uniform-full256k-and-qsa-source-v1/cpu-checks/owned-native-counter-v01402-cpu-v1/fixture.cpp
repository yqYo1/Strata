#include <cstdint>
#include <csignal>
#include <cstdlib>
namespace NEO {
struct InOrderExecEventData { uint64_t counterValue; uint32_t counterOffset; uint32_t hostPartitions; uint32_t devicePartitions; };
struct SharableEventDataHelper { InOrderExecEventData *eventDataPtr; };
struct InOrderExecEventHelper { SharableEventDataHelper sharableEventDataHelper; bool dataAssigned; uint64_t *baseHostCpuAddress; };
}
namespace L0 {
template <typename T> struct EventImp {
 NEO::InOrderExecEventHelper inOrderExecHelper;
 bool heapfullCbEventWithProfiling;
 __attribute__((noinline)) int queryCounterBasedEventStatus(long timeSinceWait) { raise(SIGTRAP); return timeSinceWait; }
};
}
int main(int argc,char **argv) {
 uint64_t counters[2]={999, static_cast<uint64_t>(std::strtoull(argv[1],nullptr,10))};
 NEO::InOrderExecEventData data{17,8,1,1};
 L0::EventImp<unsigned long> event{{{&data},true,counters},false};
 return event.queryCounterBasedEventStatus(0);
}
