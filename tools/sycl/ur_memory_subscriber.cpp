// Main-authored diagnostic using the installed XPTI/UR public headers.
// Repository LICENSE applies. No inference code or API argument is modified.
#include <xpti/xpti_trace_framework.h>
#include <unified-runtime/ur_api.h>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <execinfo.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

namespace {
constexpr uint64_t limit = 25000;
std::atomic<uint64_t> records{0}, invalid{0}, unsupported{0}, lost{0}, active{0};
thread_local bool inside = false;

void line(const char* text, size_t bytes) noexcept {
    // One short write preserves individual lines in the owned stderr pipe.
    // A partial/interrupted write invalidates collection rather than retrying
    // and interleaving the remainder with another thread's record.
    if (::write(STDERR_FILENO,text,bytes) != static_cast<ssize_t>(bytes)) ++lost;
}

struct Event {
    const char* kind = "invalid";
    const char* domain = "unknown";
    const void* pointer = nullptr;
    const void* parent = nullptr;
    size_t bytes = 0;
};

template<class Params> Event allocation(const void* raw,const char* domain,bool ok) noexcept {
    const auto& p=*static_cast<const Params*>(raw);
    Event e; e.kind="alloc"; e.domain=domain;
    if (!p.psize || (ok && (!p.pppMem || !*p.pppMem))) {e.kind="invalid";return e;}
    e.bytes=*p.psize;
    // Failed calls need not initialize their output; never dereference it.
    if(ok) e.pointer=**p.pppMem;
    return e;
}

Event decode(const xpti::function_with_args_t& call,bool ok) noexcept {
    if(!call.args_data) return {};
    switch(static_cast<ur_function_t>(call.function_id)) {
    case UR_FUNCTION_USM_DEVICE_ALLOC:
        return allocation<ur_usm_device_alloc_params_t>(call.args_data,"device",ok);
    case UR_FUNCTION_USM_HOST_ALLOC:
        return allocation<ur_usm_host_alloc_params_t>(call.args_data,"host",ok);
    case UR_FUNCTION_USM_SHARED_ALLOC:
        return allocation<ur_usm_shared_alloc_params_t>(call.args_data,"shared",ok);
    case UR_FUNCTION_USM_FREE: {
        const auto& p=*static_cast<const ur_usm_free_params_t*>(call.args_data);
        if(!p.ppMem)return {};
        return {"free","usm",*p.ppMem,nullptr,0};
    }
    case UR_FUNCTION_MEM_BUFFER_CREATE: {
        const auto& p=*static_cast<const ur_mem_buffer_create_params_t*>(call.args_data);
        if(!p.psize || (ok && (!p.pphBuffer || !*p.pphBuffer)))return {};
        return {"alloc","buffer",ok ? **p.pphBuffer : nullptr,nullptr,*p.psize};
    }
    case UR_FUNCTION_MEM_RETAIN: {
        const auto& p=*static_cast<const ur_mem_retain_params_t*>(call.args_data);
        if(!p.phMem)return {};
        return {"retain","buffer",*p.phMem,nullptr,0};
    }
    case UR_FUNCTION_MEM_RELEASE: {
        const auto& p=*static_cast<const ur_mem_release_params_t*>(call.args_data);
        if(!p.phMem)return {};
        return {"release","buffer",*p.phMem,nullptr,0};
    }
    case UR_FUNCTION_MEM_BUFFER_PARTITION: {
        const auto& p=*static_cast<const ur_mem_buffer_partition_params_t*>(call.args_data);
        if(!p.phBuffer || (ok && (!p.pphMem || !*p.pphMem)))return {};
        // Alias handles do not create another physical/requested allocation.
        return {"alias","buffer",ok ? **p.pphMem : nullptr,*p.phBuffer,0};
    }
    default:return {"unsupported","unknown",nullptr,nullptr,0};
    }
}

void callback(uint16_t type,xpti::trace_event_data_t*,xpti::trace_event_data_t*,
              uint64_t instance,const void* raw) noexcept {
    if(type!=static_cast<uint16_t>(xpti::trace_point_type_t::function_with_args_end) || !raw)return;
    const auto& call=*static_cast<const xpti::function_with_args_t*>(raw);
    const char* name=call.function_name;
    if(!name || (std::strncmp(name,"urUSM",5) && std::strncmp(name,"urMem",5) &&
                 std::strncmp(name,"urPhysicalMem",13) && std::strncmp(name,"urVirtualMem",12) &&
                 std::strncmp(name,"urEnqueueUSM",12)))return;
    // Ignore read-only memory queries; keep unknown mutating entrypoints so a
    // future API cannot silently make the requested-byte ledger look complete.
    if(std::strstr(name,"Get") || std::strstr(name,"Memcpy") || std::strstr(name,"Fill") ||
       std::strstr(name,"Prefetch") || std::strstr(name,"Advise"))return;
    if(inside){++lost;return;}
    inside=true;
    struct Exit {~Exit(){inside=false;}} exit;
    const auto number=++records;
    if(number>limit){++lost;return;}
    const auto result=call.ret_data ? *static_cast<const ur_result_t*>(call.ret_data)
                                   : UR_RESULT_ERROR_UNKNOWN;
    Event e=decode(call,result==UR_RESULT_SUCCESS);
    if(!call.ret_data || !std::strcmp(e.kind,"invalid"))++invalid;
    if(!std::strcmp(e.kind,"unsupported"))++unsupported;
    timespec now{};
    if(clock_gettime(CLOCK_MONOTONIC,&now))++lost;
    char output[1536];
    int size=std::snprintf(output,sizeof(output),
        "strata UR memory: seq=%llu ns=%llu tid=%ld instance=%llu api=%s result=%u kind=%s domain=%s ptr=%p parent=%p requested_bytes=%zu stack=",
        static_cast<unsigned long long>(number),static_cast<unsigned long long>(now.tv_sec)*1000000000ull+now.tv_nsec,
        syscall(SYS_gettid),static_cast<unsigned long long>(instance),name,static_cast<unsigned>(result),
        e.kind,e.domain,e.pointer,e.parent,e.bytes);
    if(size<0 || size>=static_cast<int>(sizeof(output)-2)){++lost;return;}
    if(result==UR_RESULT_SUCCESS && !std::strcmp(e.kind,"alloc")) {
        void* frames[32];const int count=backtrace(frames,32);
        for(int i=0;i<count;++i){
            const int n=std::snprintf(output+size,sizeof(output)-size,"%s%p",i?",":"",frames[i]);
            if(n<0 || n>=static_cast<int>(sizeof(output)-size-2)){++lost;return;}
            size+=n;
        }
    }
    output[size++]='\n';line(output,static_cast<size_t>(size));
}

__attribute__((destructor)) void summary() noexcept {
    char text[256];const int n=std::snprintf(text,sizeof(text),
        "strata UR memory summary: active=%llu records=%llu invalid=%llu unsupported=%llu lost=%llu scope=requested_api_bytes_not_physical_residency\n",
        static_cast<unsigned long long>(active.load()),static_cast<unsigned long long>(records.load()),
        static_cast<unsigned long long>(invalid.load()),static_cast<unsigned long long>(unsupported.load()),
        static_cast<unsigned long long>(lost.load()));
    if(n>0 && n<static_cast<int>(sizeof(text)))line(text,static_cast<size_t>(n));
}
}

XPTI_CALLBACK_API void xptiTraceInit(unsigned int major,unsigned int minor,const char*,const char* stream) {
    if(!stream || std::strcmp(stream,"ur.call"))return;
    if(UR_MAKE_VERSION(major,minor)!=UR_API_VERSION_CURRENT){++invalid;return;}
    void* frames[1];(void)backtrace(frames,1); // warm the host unwinder before callbacks
    const auto id=xptiRegisterStream(stream);
    if(xptiRegisterCallback(id,static_cast<uint16_t>(xpti::trace_point_type_t::function_with_args_end),callback)
       !=xpti::result_t::XPTI_RESULT_SUCCESS){++invalid;return;}
    ++active;
    const char text[]="strata UR memory: activated stream=ur.call format=1 limit=25000\n";
    line(text,sizeof(text)-1);
}
XPTI_CALLBACK_API void xptiTraceFinish(const char*) {}
