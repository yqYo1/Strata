#include <algorithm>
#include <cstdint>
#include <vector>
#include <cassert>
// A verifier may accept past max_new or an EOS. Only the predictions that
// reach the caller may advance the saved running state; the final emitted
// prediction remains the next input and is not itself consumed.
int visible_output_prefix(int accepted, int64_t remaining, const int32_t* targets,
                          const std::vector<int64_t>& eos_ids) {
    if (accepted < 0 || remaining < 1 || targets == nullptr) return 0;
    int keep = (int) std::min<int64_t>((int64_t) accepted + 1, remaining);
    for (int i = 0; i < keep; ++i) {
        if (std::find(eos_ids.begin(), eos_ids.end(), (int64_t) targets[i]) != eos_ids.end()) {
            keep = i + 1;
            break;
        }
    }
    return keep;
}

int main() {
      const int32_t t[] = {11,22,33,44};
      assert(visible_output_prefix(3,4,t,{}) == 4);
      assert(visible_output_prefix(3,1,t,{}) == 1);
      assert(visible_output_prefix(3,2,t,{}) == 2);
      assert(visible_output_prefix(3,4,t,{11}) == 1);
      assert(visible_output_prefix(3,4,t,{22}) == 2);
      assert(visible_output_prefix(3,4,t,{44}) == 4);
      assert(visible_output_prefix(3,2,t,{33}) == 2);
      assert(visible_output_prefix(0,INT64_MAX,t,{}) == 1);
      assert(visible_output_prefix(3,0,t,{}) == 0);
      assert(visible_output_prefix(-1,4,t,{}) == 0);
      assert(visible_output_prefix(3,4,nullptr,{}) == 0);
      // Reproduced32K limit: two visible predictions in the final accepted
      // four-row window must commit two, leaving32831 consumed, not32833.
      assert(32829 + visible_output_prefix(3,2,t,{}) == 32831);
    }
