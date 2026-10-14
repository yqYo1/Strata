// UNCOMPILED/UNRUN. Main-owned object-size receipt, not RSS.
#include "strata/core/decode_route_ledger.hpp"
#include <cstdio>
int main() {
    using namespace strata::core;
    using namespace strata::kernels::cpu;
    std::printf("request_state=%zu native_service=%zu slot=%zu clock_scope=%zu output_cap=%zu request_cap=98304 combined_cap=262144\n",
        sizeof(DecodeRouteLedger),sizeof(NativeServiceBudget),sizeof(NativeServiceBudget::Slot),
        sizeof(ServiceClockScope),DecodeLedgerOutput::kCap);
}
