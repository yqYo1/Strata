"""Generate a host aggregation overlay for pinned MIT Intel PTI, without GPU edits.

The failed model profile reached PrintKernelsTable's header/total but not its
rows. Qualify this proposed postprocessor separately before replacing the
collector; this source alone establishes neither a cause nor a model result.
"""
import hashlib

REVISION = '4b6d74294cdc72766c00a9211fe2e4261ee94244'
HEADER = 'tools/unitrace/src/levelzero/ze_collector.h'

# Preserve the original first map entry for each printed name and merge later
# entries in the same order as the nested loop. All six fields are integers;
# this changes name lookup cost, not clock collection, naming or summation order.
REPLACEMENT = '''  void AggregateDeviceTimeStats() const {
    // Caller holds global_device_time_stats_mutex_, as in the original PTI.
    using Iterator = decltype(global_device_time_stats_->begin());
    std::map<std::string, Iterator> representatives;
    for (auto it = global_device_time_stats_->begin(); it != global_device_time_stats_->end();) {
      std::string name;
      if (it->first.tile_ >= 0) {
        name = "Tile #" + std::to_string(it->first.tile_) + ": ";
      }
      name += GetZeKernelCommandName(it->first.kernel_command_id_, it->first.group_count_,
                                    it->first.mem_size_, options_.verbose);
      auto inserted = representatives.emplace(name, it);
      if (inserted.second) {
        ++it;
        continue;
      }
      auto& keep = inserted.first->second->second;
      const auto& next = it->second;
      keep.append_time_ += next.append_time_;
      keep.submit_time_ += next.submit_time_;
      keep.execute_time_ += next.execute_time_;
      if (keep.min_time_ > next.min_time_) keep.min_time_ = next.min_time_;
      if (keep.max_time_ < next.max_time_) keep.max_time_ = next.max_time_;
      keep.call_count_ += next.call_count_;
      it = global_device_time_stats_->erase(it);
    }
  }

'''


def overlay(original):
    """Replace only this complete method, with exact pinned-body admission."""
    if hashlib.sha256(original).hexdigest() != ORIGINAL_SHA256:
        raise ValueError('collector header differs from the pinned official revision')
    start = original.index(b'  void AggregateDeviceTimeStats() const {\n')
    end = original.index(b'  static std::vector<std::string> ParseFilterList(', start)
    return original[:start] + REPLACEMENT.encode() + original[end:]


# Filled from the immutable official blob, not the checked-out worktree.
ORIGINAL_SHA256 = 'a28f6564ac1318c73dc36d4ad31d1929516d0b6f2710bfb3ff5327f3da38e834'
