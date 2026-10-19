#pragma once

#include <cstddef>
#include <string>

namespace strata::core {

// A partial upload is not a usable residency snapshot. The caller must stop
// before any prompt/window reads the table when this returns false. Copy is a
// synchronous upload returning a runtime status (zero means success); keeping
// it injectable lets host tests cover failures without faulting a GPU context.
template <class Copy>
[[nodiscard]] bool publish_residency_tables(std::size_t tables, const char* phase, Copy&& copy, std::string& err) {
    for (std::size_t table = 0; table < tables; ++table) {
        const auto status = copy(table);
        if (status != 0) {
            err = std::string("residency publication during ") + phase + " failed at table " +
                  std::to_string(table) + " (runtime status " + std::to_string(static_cast<int>(status)) + ")";
            return false;
        }
    }
    return true;
}

} // namespace strata::core
