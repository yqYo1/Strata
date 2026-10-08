from pathlib import Path
import json,hashlib
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out=base/'upstream-e8ca-refresh-20261007'
g=root/'sycl/src/program/generate.cpp';s=g.read_text();old='} else if (native_pack && o.expert_cache > 0 && !profile.empty()) {'
assert s.count(old)==1
s=s.replace(old,'} else if (native_pack && o.expert_cache > 0 && !o.expert_cache_per_layer && !profile.empty()) {')
g.write_text(s)
p=root/'sycl/src/core/expert_cache.cpp';s=p.read_text();old='bool ExpertCache::fill_slot(int32_t slot, const uint8_t *host_blob,'
assert s.count(old)==1
helper='''namespace {
bool slot_copy_fits(const ExpertCache& cache, int32_t slot, uint64_t bytes,
                    uint64_t max_blob, std::string& err) {
    if (!cache.valid() || slot < 0 || slot >= cache.slots()) {
        err = "ExpertCache: copy slot is outside the backed arena";
        return false;
    }
    const uint64_t begin = cache.slot_offset(slot);
    const uint64_t end = cache.slot_offset((int64_t) slot + 1);
    if (bytes == 0 || bytes > max_blob || end < begin || bytes > end - begin) {
        err = "ExpertCache: copy of " + std::to_string(bytes) +
              " bytes exceeds slot " + std::to_string(slot) + " capacity";
        return false;
    }
    return true;
}
}  // namespace

'''
s=s.replace(old,helper+old)
old='    const size_t n = (size_t) (bytes > 0 && bytes <= blob_ ? bytes : blob_);'
assert s.count(old)==3
s=s.replace(old,'    const size_t n = (size_t) (bytes > 0 ? bytes : blob_);\n    if (!slot_copy_fits(*this, slot, n, (uint64_t) blob_, err)) return false;')
old='    const int64_t nb = bytes > 0 && bytes <= blob_ ? bytes : blob_;'
assert s.count(old)==1
s=s.replace(old,'    const int64_t nb = bytes > 0 ? bytes : blob_;\n    if (!slot_copy_fits(*this, slot, (uint64_t) nb, (uint64_t) blob_, err)) return false;\n    if (host_blob == nullptr) { err = "ExpertCache::verify_slot: the host blob is null"; return false; }')
p.write_text(s)
r={'scope':'Correct the explicit-count reconciliation fallthrough introduced locally; profile-order slot sizing must never service per-layer admission. Reject copies larger than the actual backed slot, before forming a pointer or enqueueing; no GPU arithmetic changes','cpu_crash':{'cache_slots_reported':770,'allocation_bytes':1596723200,'destination_offset':1594547200,'copy_bytes':2662400,'overrun_bytes':486400},'sources':{str(q.relative_to(root)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [g,p]}}
(out/'cache-slot-bounds-reconciliation.json').write_text(json.dumps(r,indent=2)+'\n')
