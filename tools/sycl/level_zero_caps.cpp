#include <iostream>
#include <level_zero/ze_api.h>
#include <stdexcept>
#include <vector>
void check(ze_result_t r) {
  if (r != ZE_RESULT_SUCCESS)
    throw std::runtime_error("Level Zero error " + std::to_string(r));
}
void show(const char *n, ze_memory_access_cap_flags_t f) {
  std::cout << n << " flags=" << f
            << " RW=" << bool(f & ZE_MEMORY_ACCESS_CAP_FLAG_RW)
            << " atomic=" << bool(f & ZE_MEMORY_ACCESS_CAP_FLAG_ATOMIC)
            << " concurrent=" << bool(f & ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT)
            << " concurrent_atomic="
            << bool(f & ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT_ATOMIC) << '\n';
}
int main() {
  try {
    check(zeInit(0));
    uint32_t n = 0;
    check(zeDriverGet(&n, nullptr));
    std::vector<ze_driver_handle_t> drivers(n);
    check(zeDriverGet(&n, drivers.data()));
    for (auto driver : drivers) {
      uint32_t m = 0;
      check(zeDeviceGet(driver, &m, nullptr));
      std::vector<ze_device_handle_t> ds(m);
      check(zeDeviceGet(driver, &m, ds.data()));
      for (auto d : ds) {
        ze_device_properties_t p{};
        p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        check(zeDeviceGetProperties(d, &p));
        std::cout << p.name << '\n';
        ze_device_memory_access_properties_t a{};
        a.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES;
        check(zeDeviceGetMemoryAccessProperties(d, &a));
        show("host", a.hostAllocCapabilities);
        show("device", a.deviceAllocCapabilities);
        show("shared_single", a.sharedSingleDeviceAllocCapabilities);
        show("shared_cross", a.sharedCrossDeviceAllocCapabilities);
        show("shared_system", a.sharedSystemAllocCapabilities);
      }
    }
  } catch (std::exception const &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
