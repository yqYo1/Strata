#pragma once
// CPU CONTRACT FIXTURE ONLY. These are small host tag buffers, NOT real tensors,
// session allocation, GPU queue behavior or the real conversation validator.
#include "strata/core/layout.hpp"
#include <vector>
#include <string>
#include <cstdint>
#include <cstddef>
namespace strata::core {
struct ConversationImageKey {
    int64_t start = 0;
    uint64_t hash = 0;
    bool operator==(const ConversationImageKey&) const = default;
};
struct ConversationCheckpoint {
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> imgs;
    std::vector<uint8_t> gdn, ple, tails, dead, block_pos;
    uint64_t used = 0;
    std::vector<ConversationCheckpoint> stage_parts;
};
struct QsaState {
    void* idx_tail = nullptr;
    void* idx_dead = nullptr;
    void* idx_block_pos = nullptr;
};
struct SessionState {
    int64_t layer_lo = 0, layer_hi = 48, qsa_ord0 = 0, gdn_ord0 = 0;
    int64_t qsa_alloc = 12, gdn_alloc = 36, max_cells = 262144;
    float* gdn_state = nullptr;
    float* ple_hist = nullptr;
    QsaState* qsa_states = nullptr;
};
struct ConversationStateSizes {
    size_t gdn = 0, ple = 0, tail = 0, dead = 0, block_pos = 0;
};
inline ConversationStateSizes fixture_sizes{36*32,32,32,32,sizeof(int32_t)};
inline bool conversation_session_sizes(const ModelGeometry&, const SessionState& ss,
                                       ConversationStateSizes& z, std::string&) {
    // 32 bytes = independently interpreted Tag in the standalone fixture.
    z = fixture_sizes;
    if (!ss.ple_hist) z.ple = 0;
    return true;
}
inline bool conversation_checkpoint_validate(const ConversationCheckpoint& c, const SessionState& ss,
                                              const ModelGeometry& g, std::string& e) {
    ConversationStateSizes z;
    conversation_session_sizes(g,ss,z,e);
    if (!ss.gdn_state || !ss.qsa_states || c.ids.empty() || c.ids.size() > uint64_t(ss.max_cells) ||
        c.gdn.size() != z.gdn || c.ple.size() != z.ple || c.tails.size() != 12*z.tail ||
        c.dead.size() != 12*z.dead || c.block_pos.size() != 12*z.block_pos) {
        e = "CPU stub checkpoint payload invalid"; return false;
    }
    int64_t previous = -1;
    for (const auto& im : c.imgs) {
        if (im.start <= previous || im.start < 0 || uint64_t(im.start) >= c.ids.size()) {
            e = "CPU stub image prefix invalid"; return false;
        }
        previous = im.start;
    }
    for (int i=0;i<12;++i)
        if (!ss.qsa_states[i].idx_tail || !ss.qsa_states[i].idx_dead || !ss.qsa_states[i].idx_block_pos)
            return false;
    return true;
}
}
