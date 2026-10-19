#pragma once

// Main-authored test-only gate, from the installed Intel LLVM v7.1.1 interop
// signatures and Level Zero1.17.24 event contract; no external implementation.
// https://oneapi-src.github.io/level-zero-spec/level-zero/1.17.24/core/PROG.html
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include "strata/sycl/checked_cleanup.hpp"
#include "strata/sycl/runtime.hpp"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace strata::test {
inline void event_need(bool value,const char* message) {
    if(!value) throw std::runtime_error(message);
}
inline void ze_check(ze_result_t value,const char* message) {
    if(value!=ZE_RESULT_SUCCESS)
        throw std::runtime_error(std::string(message)+" result="+std::to_string(unsigned(value)));
}

class HostEventGate {
    std::shared_ptr<sycl_backend::Runtime> runtime_;
    std::vector<sycl::queue> consumers_;
    ze_event_pool_handle_t pool_=nullptr;
    ze_event_handle_t native_=nullptr;
    std::optional<sycl::event> imported_;
    std::mutex mutex_;
    std::condition_variable changed_;
    bool requested_=false, expired_=false;
    ze_result_t signal_result_=ZE_RESULT_NOT_READY;
    std::thread signaller_;
    void destroy_native() noexcept {
        if(native_ && zeEventDestroy(native_)!=ZE_RESULT_SUCCESS)
            sycl_backend::cleanup_fail_stop("native test event destruction");
        native_=nullptr;
        if(pool_ && zeEventPoolDestroy(pool_)!=ZE_RESULT_SUCCESS)
            sycl_backend::cleanup_fail_stop("native test event pool destruction");
        pool_=nullptr;
    }
public:
    HostEventGate(std::shared_ptr<sycl_backend::Runtime> runtime,std::vector<sycl::queue> consumers)
        :runtime_(std::move(runtime)),consumers_(std::move(consumers)) {
        for(const auto& queue:consumers_)
            event_need(queue.get_backend()==sycl::backend::ext_oneapi_level_zero &&
                queue.get_context()==runtime_->context() && queue.get_device()==runtime_->device(),
                "native gate context/device/backend mismatch");
        const auto context=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(runtime_->context());
        auto device=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(runtime_->device());
        ze_event_pool_desc_t pool{};
        pool.stype=ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
        pool.flags=ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
        pool.count=1;
        ze_check(zeEventPoolCreate(context,&pool,1,&device,&pool_),"native gate pool creation");
        try {
            ze_event_desc_t event{};
            event.stype=ZE_STRUCTURE_TYPE_EVENT_DESC;
            event.signal=event.wait=ZE_EVENT_SCOPE_FLAG_HOST;
            ze_check(zeEventCreate(pool_,&event,&native_),"native gate event creation");
            event_need(zeEventQueryStatus(native_)==ZE_RESULT_NOT_READY,"native gate initially signalled");
            // Armed before import or any dependent submit. This is an ordinary
            // host thread; it never waits on a SYCL queue or calls a host task.
            signaller_=std::thread([this] {
                std::unique_lock lock(mutex_);
                expired_=!changed_.wait_for(lock,std::chrono::seconds(2),[&] {return requested_;});
                lock.unlock();
                const auto status=zeEventHostSignal(native_);
                lock.lock();signal_result_=status;
            });
        } catch(...) {destroy_native();throw;}
    }
    HostEventGate(const HostEventGate&)=delete;
    HostEventGate& operator=(const HostEventGate&)=delete;
    ~HostEventGate() {
        sycl_backend::checked_cleanup([&] {
            (void)release(); // Signal and join BEFORE any dependent queue drain.
            for(auto& queue:consumers_) queue.wait_and_throw();
            runtime_->check();
        },[&] {imported_.reset();destroy_native();},"native test gate retirement");
    }
    bool held() {
        {
            std::lock_guard lock(mutex_);
            if(requested_ || expired_) return false;
        }
        // No native/SYCL call may hold the timer's mutex: even a blocking
        // backend query must leave the fallback signaller independent.
        const auto status=zeEventQueryStatus(native_);
        if(status!=ZE_RESULT_NOT_READY) ze_check(status,"native gate status query failed");
        return status==ZE_RESULT_NOT_READY;
    }
    const sycl::event& import(long long& elapsed_us) {
        event_need(!imported_ && held(),"native gate import admission");
        const auto begin=std::chrono::steady_clock::now();
        imported_=sycl::make_event<sycl::backend::ext_oneapi_level_zero>(
            {native_,sycl::ext::oneapi::level_zero::ownership::keep},runtime_->context());
        event_need(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(*imported_)==native_,
            "native gate imported handle differs");
        elapsed_us=std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now()-begin).count();
        event_need(elapsed_us<=100000 && held(),"native event import exceeded100ms or timer expired");
        return *imported_;
    }
    bool release() {
        bool manual;
        {
            std::lock_guard lock(mutex_);
            manual=!requested_ && !expired_;
            requested_=true;
            changed_.notify_all();
        }
        if(signaller_.joinable()) signaller_.join();
        ze_check(signal_result_,"native gate host signal failed");
        ze_check(zeEventQueryStatus(native_),"native gate signal was not observable");
        return manual;
    }
};
} // namespace strata::test
