#include "strata/sycl_upstream/memory.hpp"
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>
#include <cstdlib>
#include <unistd.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace strata::sycl_upstream {
namespace {
void ze_check(ze_result_t result, const char* operation) {
    if (result == ZE_RESULT_SUCCESS) return;
    if (result == ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY || result == ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)
        throw std::bad_alloc();
    throw std::runtime_error(std::string(operation) + " failed: Level Zero result " + std::to_string(result));
}
uintptr_t end_of(uintptr_t start, size_t bytes) {
    if (!start || !bytes || bytes > std::numeric_limits<uintptr_t>::max() - start)
        throw std::invalid_argument("invalid memory range");
    return start + bytes;
}
}
struct Memory::Impl {
    struct Backing {
        ze_context_handle_t context;
        void* base;
        size_t bytes;
        bool read_only;
        Backing(ze_context_handle_t context, size_t bytes, bool read_only)
            : context(context), base(nullptr), bytes(bytes), read_only(read_only) {}
        Backing(const Backing&) = delete;
        Backing& operator=(const Backing&) = delete;
        void release() {
            if (base) {
                ze_check(zeMemFree(context, base), "release imported mapping");
                ze_memory_allocation_properties_t props{ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES};
                ze_check(zeMemGetAllocProperties(context, base, &props, nullptr), "query released import");
                if (props.type != ZE_MEMORY_TYPE_UNKNOWN)
                    throw std::runtime_error("driver retained a released external mapping; caller storage is still in use");
                base = nullptr;
            }
        }
        ~Backing() {
            try { release(); }
            catch (const std::exception& e) { std::fprintf(stderr, "SYCL imported memory teardown: %s\n", e.what()); std::terminate(); }
        }
    };
    struct Entry {
        Info info;
        bool retiring = false;
        sycl::context context;
        void* owned = nullptr;
        std::vector<std::shared_ptr<Backing>> backings;
        Entry(Info info, const sycl::context& context): info(info), context(context) {}
        void release_owned() { if (owned) { sycl::free(owned, context); owned = nullptr; } }
        ~Entry() {
            try { release_owned(); }
            catch (const std::exception& e) { std::fprintf(stderr, "SYCL memory teardown: %s\n", e.what()); std::terminate(); }
        }
    };
    Runtime& runtime;
    sycl::context context;
    sycl::device device;
    ze_context_handle_t native_context = nullptr;
    bool can_import = false;
    std::string import_error = "Level Zero external host mapping is not supported";
    size_t page;
    mutable std::mutex mutex;
    std::map<uintptr_t, std::shared_ptr<Entry>> entries;
    // Weak ownership lets a registration retain a mapping shared with an
    // adjacent registration without overlapping native page-table mappings.
    std::map<uintptr_t, std::weak_ptr<Backing>> imports;
    explicit Impl(Runtime& runtime): runtime(runtime), context(runtime.context()), device(runtime.device()) {
        const long queried = sysconf(_SC_PAGESIZE);
        if (queried <= 0) throw std::runtime_error("cannot query system page size");
        page = size_t(queried);
        if (device.get_backend() == sycl::backend::ext_oneapi_level_zero) {
            native_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(context);
            auto driver = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device.get_platform());
            uint32_t count = 0;
            ze_check(zeDriverGetExtensionProperties(driver, &count, nullptr), "query import extension");
            std::vector<ze_driver_extension_properties_t> extensions(count);
            ze_check(zeDriverGetExtensionProperties(driver, &count, extensions.data()), "query import extension");
            for (const auto& e : extensions)
                if (!std::strcmp(e.name, ZE_EXTERNAL_MEMORY_MAPPING_EXT_NAME) &&
                    e.version >= ZE_EXTERNAL_MEMMAP_SYSMEM_EXT_VERSION_1_0) can_import = true;
            ze_driver_properties_t properties{ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES};
            ze_check(zeDriverGetProperties(driver, &properties), "query import driver version");
            // Intel encodes this version as 0x01030000 + NEO_VERSION_BUILD.
            // Older releases cache freed external mappings (NEO-19425). Reject
            // before touching caller storage; 39758 is the first tested fixed build.
            if (device.get_info<sycl::info::device::vendor_id>() == 0x8086 &&
                properties.driverVersion < 0x01030000u + 39758u) {
                can_import = false;
                import_error = "external host registration requires Intel compute-runtime 26.35.39758.10 or newer "
                               "(older drivers retain released mappings; NEO-19425)";
            }
        }
    }
    auto containing(uintptr_t pointer) const {
        auto it = entries.upper_bound(pointer);
        if (it == entries.begin()) return entries.end();
        --it;
        return pointer - it->first < it->second->info.bytes ? it : entries.end();
    }
    void require_unused(uintptr_t first, uintptr_t last) {
        auto it = entries.lower_bound(first);
        if (it != entries.end() && it->first < last) throw std::invalid_argument("memory range already owned or registered");
        if (it != entries.begin()) {
            --it;
            if (it->second->info.bytes > first - it->first)
                throw std::invalid_argument("memory range overlaps an existing allocation or registration");
        }
    }
    void prune_imports() { std::erase_if(imports, [](const auto& p) { return p.second.expired(); }); }
    std::shared_ptr<Backing> import(uintptr_t first, size_t bytes, bool read_only) {
        auto backing = std::make_shared<Backing>(native_context, bytes, read_only);
        ze_external_memmap_sysmem_ext_desc_t ext{ZE_STRUCTURE_TYPE_EXTERNAL_MEMMAP_SYSMEM_EXT_DESC,
                                               nullptr, reinterpret_cast<void*>(first), bytes};
        ze_host_mem_alloc_desc_t desc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, &ext,
                                     read_only ? ZE_HOST_MEM_ALLOC_FLAG_MEM_READ_ONLY : 0u};
        ze_check(zeMemAllocHost(native_context, &desc, bytes, page, &backing->base), "import external host memory");
        if (backing->base != reinterpret_cast<void*>(first)) throw std::runtime_error("import changed the host address");
        ze_memory_allocation_properties_t properties{ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES};
        ze_check(zeMemGetAllocProperties(native_context, backing->base, &properties, nullptr), "query imported allocation");
        // 26.35 retains HOST for compatibility (Intel commit 2b6562547d);
        // the newer extension specification also permits HOST_IMPORTED.
        if (properties.type != ZE_MEMORY_TYPE_HOST && properties.type != ZE_MEMORY_TYPE_HOST_IMPORTED)
            throw std::runtime_error("driver did not recognize the imported host allocation");
        imports.emplace(first, backing);
        return backing;
    }
    void release(void* pointer, Kind kind) {
        if (!pointer) return;
        runtime.check_memory_operation();
        std::shared_ptr<Entry> entry;
        {
            std::lock_guard lock(mutex);
            const auto it = entries.find(reinterpret_cast<uintptr_t>(pointer));
            if (it == entries.end() || it->second->info.kind != kind || it->second->retiring)
                throw std::invalid_argument("release requires the original pointer and matching memory kind");
            entry = it->second;
            entry->retiring = true;
        }
        try { runtime.synchronize_device(); }
        catch (...) { std::lock_guard lock(mutex); entry->retiring = false; throw; }
        std::lock_guard lock(mutex);
        try {
            entry->release_owned();
            for (const auto& backing : entry->backings) if (backing.use_count() == 1) backing->release();
        } catch (...) { entry->retiring = false; throw; }
        entries.erase(reinterpret_cast<uintptr_t>(pointer));
        entry.reset();
        prune_imports();
    }
    void* allocate(size_t bytes, Kind kind) {
        runtime.check_memory_operation();
        if (!bytes) return nullptr;
        auto entry = std::make_shared<Entry>(Info{nullptr, bytes, kind, false}, context);
        entry->owned = kind == Kind::device ? sycl::aligned_alloc_device(256, bytes, device, context)
                                          : sycl::aligned_alloc_host(64, bytes, context);
        if (!entry->owned) throw std::bad_alloc();
        entry->info.base = entry->owned;
        std::lock_guard lock(mutex);
        const auto first = reinterpret_cast<uintptr_t>(entry->owned);
        require_unused(first, end_of(first, bytes));
        entries.emplace(first, entry);
        return entry->owned;
    }
};
Memory::Memory(Runtime& runtime): impl_(std::make_unique<Impl>(runtime)) {}
Memory::~Memory() {
    try {
        if (!impl_->entries.empty()) {
            impl_->runtime.synchronize_device();
            impl_->entries.clear();
        }
    } catch (const std::exception& e) { std::fprintf(stderr, "SYCL memory domain teardown: %s\n", e.what()); std::terminate(); }
}
void* Memory::allocate_device(size_t bytes) { return impl_->allocate(bytes, Kind::device); }
void* Memory::allocate_host(size_t bytes) { return impl_->allocate(bytes, Kind::host); }
void Memory::free_device(void* p) { impl_->release(p, Kind::device); }
void Memory::free_host(void* p) { impl_->release(p, Kind::host); }
void Memory::unregister_host(void* p) { impl_->release(p, Kind::registered_host); }
size_t Memory::page_size() const { return impl_->page; }
void Memory::register_host(void* pointer, size_t bytes, bool read_only) {
    impl_->runtime.check_memory_operation();
    if (!impl_->can_import) throw std::runtime_error(impl_->import_error);
    const auto first = reinterpret_cast<uintptr_t>(pointer), last = end_of(first, bytes);
    const auto page = impl_->page;
    const auto begin_page = first - first % page;
    if (last > std::numeric_limits<uintptr_t>::max() - (page - 1)) throw std::invalid_argument("rounded memory range overflows");
    const auto end_page = ((last + page - 1) / page) * page;
    auto entry = std::make_shared<Impl::Entry>(Info{pointer, bytes, Kind::registered_host, read_only}, impl_->context);
    std::lock_guard lock(impl_->mutex);
    impl_->require_unused(first, last);
    impl_->prune_imports();
    uintptr_t cursor = begin_page;
    while (cursor < end_page) {
        auto next = impl_->imports.upper_bound(cursor);
        std::shared_ptr<Impl::Backing> existing;
        if (next != impl_->imports.begin()) {
            auto previous = std::prev(next);
            auto backing = previous->second.lock();
            if (backing && cursor - previous->first < backing->bytes) existing = std::move(backing);
        }
        if (existing) {
            if (existing->read_only != read_only)
                throw std::invalid_argument("registration permissions differ on shared imported pages");
            cursor = std::min(end_page, reinterpret_cast<uintptr_t>(existing->base) + existing->bytes);
            entry->backings.push_back(std::move(existing));
        } else {
            const uintptr_t stop = next == impl_->imports.end() ? end_page : std::min(end_page, next->first);
            // Do not import a USM allocation owned outside this memory domain.
            ze_memory_allocation_properties_t properties{ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES};
            ze_check(zeMemGetAllocProperties(impl_->native_context, reinterpret_cast<void*>(cursor), &properties, nullptr), "query external range");
            if (properties.type != ZE_MEMORY_TYPE_UNKNOWN) {
                void* native_base = nullptr; size_t native_size = 0;
                zeMemGetAddressRange(impl_->native_context, reinterpret_cast<void*>(cursor), &native_base, &native_size);
                throw std::invalid_argument("external range is already a native allocation: cursor=" + std::to_string(cursor) +
                    " native_base=" + std::to_string(reinterpret_cast<uintptr_t>(native_base)) + " native_bytes=" + std::to_string(native_size));
            }
            entry->backings.push_back(impl_->import(cursor, stop - cursor, read_only));
            cursor = stop;
        }
    }
    impl_->entries.emplace(first, entry);
}
std::optional<Memory::Info> Memory::info(const void* pointer) const {
    std::lock_guard lock(impl_->mutex);
    auto it = impl_->containing(reinterpret_cast<uintptr_t>(pointer));
    return it == impl_->entries.end() ? std::nullopt : std::optional<Info>(it->second->info);
}
bool Memory::overlaps(const void* pointer,size_t bytes) const {
    const auto first=reinterpret_cast<uintptr_t>(pointer),last=end_of(first,bytes);
    std::lock_guard lock(impl_->mutex);
    const auto next=impl_->entries.lower_bound(first);
    if(next!=impl_->entries.end() && next->first<last) return true;
    if(next==impl_->entries.begin()) return false;
    const auto previous=std::prev(next);
    return previous->second->info.bytes>first-previous->first;
}
void* Memory::device_alias(void* pointer) const {
    std::lock_guard lock(impl_->mutex);
    auto it = impl_->containing(reinterpret_cast<uintptr_t>(pointer));
    if (it == impl_->entries.end() || it->second->info.kind == Kind::device || it->second->retiring)
        throw std::invalid_argument("pointer has no live host mapping");
    return pointer;
}
Memory::Stats Memory::stats() const {
    std::lock_guard lock(impl_->mutex);
    Stats result;
    for (const auto& [base, entry] : impl_->entries) {
        if (entry->info.kind == Kind::device) result.device_bytes += entry->info.bytes;
        else if (entry->info.kind == Kind::host) result.host_bytes += entry->info.bytes;
        else result.registered_bytes += entry->info.bytes;
    }
    for (const auto& [base, weak] : impl_->imports) if (auto b = weak.lock()) {
        result.imported_bytes += b->bytes;
        ++result.imported_ranges;
    }
    return result;
}
Memory::Available Memory::available() const {
    auto unsupported=[](const char* why) -> void {
        throw sycl::exception(sycl::make_error_code(sycl::errc::feature_not_supported),why);
    };
    auto query=[&](ze_result_t result) {
        if (result!=ZE_RESULT_SUCCESS) unsupported("device free-memory Sysman query unavailable");
    };
    if (impl_->device.get_backend()!=sycl::backend::ext_oneapi_level_zero)
        unsupported("free-memory query requires Level Zero");
    // Standalone Sysman has its own handles. Never reinterpret a core handle,
    // and never change the application's process-wide initialization variables.
    if (const char* legacy=std::getenv("ZES_ENABLE_SYSMAN"); legacy && std::strcmp(legacy,"0"))
        unsupported("standalone Sysman free-memory query requires ZES_ENABLE_SYSMAN unset or 0");
    static const auto drivers=[&] {
        query(zesInit(0));
        uint32_t n=0; query(zesDriverGet(&n,nullptr));
        std::vector<zes_driver_handle_t> result(n);
        if(n) query(zesDriverGet(&n,result.data()));
        result.resize(n);return result;
    }();
    ze_device_properties_t properties{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
    query(zeDeviceGetProperties(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(impl_->device),&properties));
    zes_uuid_t uuid{};static_assert(sizeof(uuid.id)==sizeof(properties.uuid.id));
    std::memcpy(uuid.id,properties.uuid.id,sizeof(uuid.id));
    zes_device_handle_t device=nullptr;ze_bool_t sub=0;uint32_t sub_id=0;
    for(auto driver:drivers) {
        zes_device_handle_t matched=nullptr;
        if(zesDriverGetDeviceByUuidExp(driver,uuid,&matched,&sub,&sub_id)==ZE_RESULT_SUCCESS) {device=matched;break;}
    }
    if(!device) unsupported("no Sysman device matching the compute device UUID");
    uint32_t n=0;query(zesDeviceEnumMemoryModules(device,&n,nullptr));
    std::vector<zes_mem_handle_t> modules(n);
    if(n) query(zesDeviceEnumMemoryModules(device,&n,modules.data()));
    modules.resize(n);uint64_t free=0;bool found=false,root_module=false,tile_module=false;
    for(auto module:modules) {
        zes_mem_properties_t p{ZES_STRUCTURE_TYPE_MEM_PROPERTIES};query(zesMemoryGetProperties(module,&p));
        if(p.location!=ZES_MEM_LOC_DEVICE || (sub && (!p.onSubdevice || p.subdeviceId!=sub_id))) continue;
        root_module|=!p.onSubdevice;tile_module|=bool(p.onSubdevice);
        zes_mem_state_t state{ZES_STRUCTURE_TYPE_MEM_STATE};query(zesMemoryGetState(module,&state));
        if(state.free>std::numeric_limits<uint64_t>::max()-free) unsupported("free-memory sum overflow");
        free+=state.free;found=true;
    }
    const auto total=impl_->device.get_info<sycl::info::device::global_mem_size>();
    if(!found || (root_module && tile_module) || !total || free>total || total>std::numeric_limits<size_t>::max())
        unsupported("inconsistent device memory capacity or missing device memory modules");
    return {size_t(free),size_t(total)};
}
} // namespace strata::sycl_upstream
