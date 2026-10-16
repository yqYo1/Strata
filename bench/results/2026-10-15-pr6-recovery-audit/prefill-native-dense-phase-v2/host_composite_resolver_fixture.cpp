// Unexecuted host fake-table fixture. Tests the actual proposed inline resolver.
// Do not link weights.cpp/layout.cpp: their two lookup functions are stubbed.
#include "strata/core/layout.hpp"
#include <cstdlib>
#include <map>
using namespace strata::core;
static std::map<std::string,WeightRef> fake_canonical;
const WeightRef* strata::core::WeightTable::find(const std::string& name) const {
    const auto it=fake_canonical.find(name);
    return it==fake_canonical.end()?nullptr:&it->second;
}
std::string strata::core::LayerView::name(const char* suffix) const {
    return "blk."+std::to_string(layer())+"."+suffix;
}
int main() {
    WeightTable table;
    WeightRef norm, withdrawn, native;
    int norm_owner, bank_owner;
    norm.data=&norm_owner;native.native_data=&bank_owner;
    fake_canonical.emplace("blk.1.norm",norm);
    fake_canonical.emplace("blk.1.projection",withdrawn);
    LayerView::Overlay overlay{{"blk.1.projection",native}};
    LayerView active(table,1,&overlay), ordinary(table,1), next(table,2,&overlay);
    if(active.get("projection")->native_data!=&bank_owner ||
       active.get("norm")->data!=&norm_owner || active.get("absent") ||
       ordinary.get("projection")->native_data || next.get("projection") ||
       fake_canonical.at("blk.1.projection").native_data) std::abort();
}
