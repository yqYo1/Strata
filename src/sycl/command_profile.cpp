// Authored for Strata; repository LICENSE (MIT) applies.
#include "strata/sycl/command_profile.hpp"
#include "strata/sycl/launch.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace strata::sycl_backend::command_profile {
thread_local Trace* active = nullptr;
namespace {
constexpr size_t Kinds = size_t(Kind::count), Bins = 8, PerBin = 4;
constexpr size_t Frames = 4, PendingMax = 192, RecordsMax = Frames * PendingMax;
const char* kind_name(Kind kind) {
  constexpr const char* names[] = {"none","upload","dequant_gu","dequant_down",
      "gather","gu","swiglu","down","combine","write","accumulate","job_weights","status"};
  return names[size_t(kind)];
}
unsigned row_bin(int64_t rows) {
  if (rows < 16) return 0;
  unsigned b = 1;
  while (b < 7 && rows >= (int64_t(16) << b)) ++b;
  return b;
}
bool by_job(Kind kind) {
  return kind==Kind::gather || kind==Kind::gu || kind==Kind::swiglu || kind==Kind::down || kind==Kind::accumulate;
}
bool by_expert(Kind kind) { return kind==Kind::dequant_gu || kind==Kind::dequant_down; }
// Independent deterministic FNV rank over IDs, not copied implementation code.
uint64_t rank(uint64_t seed, uint64_t id) {
  uint64_t h = seed;
  for (unsigned i=0;i<8;++i) { h ^= (id >> (i*8)) & 255; h *= 1099511628211ULL; }
  return h;
}
struct Choice { uint64_t hash = UINT64_MAX; int64_t id = -1; };
void consider(std::array<Choice,PerBin>& choices, uint64_t hash, int64_t id) {
  for (const auto& item:choices) if (item.id==id) return;
  Choice value{hash,id};
  for (auto& item:choices)
    if (value.hash < item.hash || (value.hash==item.hash && value.id < item.id)) std::swap(value,item);
}
bool contains(const std::array<Choice,PerBin>& choices, int64_t id) {
  return std::any_of(choices.begin(),choices.end(),[&](auto x){return x.id==id;});
}
struct Group {
  uint32_t eligible=0, selected=0, service_samples=0, queue_delay_samples=0;
  uint64_t service=0, queue_delay=0;
};
struct Window {
  int32_t layer=-1; int64_t first_row=0;
  std::array<std::array<Group,Bins>,Kinds> groups{};
};
struct Record {
  uint64_t submitted=0,start=0,end=0;
  Meta meta{};
  uint8_t window=0,bin=0;
};
struct Pending { std::optional<sycl::event> event; Record record; };
struct Invalid { Record record; uint64_t query_wall=0; unsigned violations=0; };
} // namespace

struct Trace::Impl {
  void* stream;
  unsigned expected, windows=0, current=0;
  size_t used=0, records=0, peak=0;
  size_t invalid_total=0, invalid_saved=0;
  size_t submit_clock_inversions=0;
  uint64_t timer_resolution=0, query_wall_ns=0;
  bool complete=true;
  const char* error="none";
  Meta meta{};
  std::array<std::array<Choice,PerBin>,Bins> jobs;
  std::array<Choice,PerBin> experts;
  std::array<Window,Frames> window;
  std::array<Pending,PendingMax> pending;
  std::array<Record,RecordsMax> record;
  std::array<Invalid,32> invalid;
  void fail(const char* message) noexcept { complete=false; if (!std::strcmp(error,"none")) error=message; }
};

bool requested() {
  static const bool result=[] {
    const char* value=std::getenv("STRATA_SYCL_PREFILL_COMMAND_PROFILE");
    if (!value || !std::strcmp(value,"0")) return false;
    if (!std::strcmp(value,"1")) return true;
    throw std::invalid_argument("STRATA_SYCL_PREFILL_COMMAND_PROFILE must be0 or1");
  }();
  return result;
}
Trace::Trace(void* stream, unsigned expected) : impl_(new Impl) {
  static_assert(sizeof(Impl) <= 128*1024, "fixed host record budget");
  if (!stream || expected<1 || expected>Frames) throw std::invalid_argument("command profile stream/frame count");
  auto& q=queue_for(stream);
  if (!q.has_property<sycl::property::queue::enable_profiling>() ||
      q.get_backend()!=sycl::backend::ext_oneapi_level_zero || !q.get_device().is_gpu())
    throw std::invalid_argument("command profile requires actual profiled LevelZero GPU queue");
  impl_->stream=stream; impl_->expected=expected;
  impl_->timer_resolution=q.get_device().get_info<sycl::info::device::profiling_timer_resolution>();
}
Trace::~Trace() noexcept {
  auto& x=*impl_;
  if (active==this) { active=nullptr; x.fail("unclosed_frame"); }
  if (x.used) x.fail("unretired_frame");
  if (x.windows!=x.expected) x.fail("frame_count");
  for (size_t i=0;i<x.records;++i) {
    const auto& r=x.record[i];const auto& m=r.meta;
    std::fprintf(stderr,"strata command sample: window=%u kind=%s bin=%u expert=%d m=%d n=%d k=%d format=%d row=%lld submit=%llu start=%llu end=%llu\n",
        unsigned(r.window),kind_name(m.kind),unsigned(r.bin),m.expert,m.rows,m.n,m.k,m.format,
        (long long)m.source_row,
        (unsigned long long)r.submitted,(unsigned long long)r.start,(unsigned long long)r.end);
  }
  for (size_t i=0;i<x.invalid_saved;++i) {
    const auto& bad=x.invalid[i];const auto& r=bad.record;const auto& m=r.meta;
    std::fprintf(stderr,"strata command invalid: window=%u kind=%s bin=%u expert=%d m=%d n=%d k=%d format=%d row=%lld submit=%llu start=%llu end=%llu query_wall_ns=%llu violations=%u\n",
        unsigned(r.window),kind_name(m.kind),unsigned(r.bin),m.expert,m.rows,m.n,m.k,m.format,
        (long long)m.source_row,(unsigned long long)r.submitted,(unsigned long long)r.start,
        (unsigned long long)r.end,(unsigned long long)bad.query_wall,bad.violations);
  }
  for (unsigned w=0;w<x.windows;++w)
    for (size_t k=1;k<Kinds;++k) for (size_t b=0;b<Bins;++b) {
      const auto& g=x.window[w].groups[k][b];if (!g.eligible) continue;
      if (!g.selected) x.fail("unsampled_eligible_stratum");
      std::fprintf(stderr,"strata command group: window=%u layer=%d first_row=%lld kind=%s bin=%zu eligible=%u selected=%u service_samples=%u queue_delay_samples=%u sample_service_ns=%llu sample_queue_delay_ns=%llu\n",
          w,x.window[w].layer,(long long)x.window[w].first_row,kind_name(Kind(k)),b,g.eligible,g.selected,
          g.service_samples,g.queue_delay_samples,
          (unsigned long long)g.service,(unsigned long long)g.queue_delay);
    }
  std::fprintf(stderr,"strata command profile: version=2 service_complete=%d submit_clock_qualified=%d submit_clock_inversions=%zu windows=%u expected=%u records=%zu peak_events=%zu cap=%zu timer_resolution_ns=%llu max_query_wall_ns=%llu error=%s invalid=%zu invalid_saved=%zu scope=selected_first_windows hardware_busy=UNMEASURED\n",
      int(x.complete),int(x.complete && !x.submit_clock_inversions),x.submit_clock_inversions,
      x.windows,x.expected,x.records,x.peak,PendingMax,
      (unsigned long long)x.timer_resolution,(unsigned long long)x.query_wall_ns,x.error,x.invalid_total,x.invalid_saved);
  // On an early error the enclosing layer-major checked lease has retired
  // before this request owner is destroyed. Never query its unfinished events.
}
Frame Trace::begin(int64_t layer, int64_t first_row) {
  auto& x=*impl_;
  if (first_row || (layer!=0 && layer!=15 && layer!=31 && layer!=47)) return Frame();
  if (active || x.used || x.windows>=x.expected) { x.fail("frame_state"); return Frame(); }
  x.current=x.windows++;
  x.window[x.current].layer=int32_t(layer);x.window[x.current].first_row=first_row;
  x.jobs={};x.experts={};x.meta={};
  active=this;
  return Frame(this);
}
Frame::~Frame() noexcept {
  if (!trace_) return;
  if (!retired_) trace_->impl_->fail("missing_checked_retirement");
  if (active==trace_) active=nullptr;
}
void Frame::consider(int32_t expert, int64_t row, int64_t rows) noexcept {
  if (!trace_) return;
  auto& x=*trace_->impl_;
  if (expert<0 || row<0 || rows<=0 || rows>8192) { x.fail("plan_shape"); return; }
  const uint64_t seed=1469598103934665603ULL ^ uint64_t(x.window[x.current].layer*8193+17);
  command_profile::consider(x.jobs[row_bin(rows)],rank(seed,uint64_t(row)),row);
  command_profile::consider(x.experts,rank(seed^0x9e3779b97f4a7c15ULL,uint64_t(expert)),expert);
}
Tag Frame::tag(Kind kind, int32_t expert, int64_t row, int64_t rows,
               int64_t n, int64_t k, int32_t format) noexcept {
  Meta m;
  m.kind=kind;m.expert=expert;m.format=format;m.source_row=row;
  if (trace_) {
    auto& x=*trace_->impl_;
    if (rows<0 || rows>INT32_MAX || n<0 || n>INT32_MAX || k<0 || k>INT32_MAX || kind==Kind::none || size_t(kind)>=Kinds) {
      x.fail("tag_shape");m.kind=Kind::none;
    } else {
      m.rows=int32_t(rows);m.n=int32_t(n);m.k=int32_t(k);
      m.selected=by_job(kind) ? contains(x.jobs[row_bin(rows)],row) :
                 by_expert(kind) ? contains(x.experts,expert) : true;
    }
  }
  return Tag(trace_,m);
}
Tag::Tag(Trace* trace, Meta meta) noexcept : trace_(trace) {
  if (trace_) { previous_=trace_->impl_->meta;trace_->impl_->meta=meta; }
}
Tag::~Tag() noexcept { if (trace_) trace_->impl_->meta=previous_; }
void observe_impl(Trace& trace, void* stream, const sycl::event& event) noexcept {
  auto& x=*trace.impl_;
  if (stream!=x.stream || x.meta.kind==Kind::none || x.current>=Frames) { x.fail("unexpected_stream_or_tag");return; }
  const unsigned bin=by_job(x.meta.kind) ? row_bin(x.meta.rows) : 0;
  auto& group=x.window[x.current].groups[size_t(x.meta.kind)][bin];
  ++group.eligible;
  if (!x.meta.selected) return;
  if (x.used==PendingMax) { x.fail("event_capacity");return; }
  try {
    auto& slot=x.pending[x.used];
    slot.record.meta=x.meta;slot.record.window=uint8_t(x.current);slot.record.bin=uint8_t(bin);
    slot.event.emplace(event);
    ++group.selected;++x.used;x.peak=std::max(x.peak,x.used);
  } catch (...) { x.fail("event_retention_exception"); }
}
void Frame::after_checked_drain() noexcept {
  if (!trace_) return;
  auto& x=*trace_->impl_;
  retired_=true;
  if (active==trace_) active=nullptr;
  try {
    // Validate completion for the whole held set before any timestamp query.
    for (size_t i=0;i<x.used;++i)
      if (x.pending[i].event->get_info<sycl::info::event::command_execution_status>()!=sycl::info::event_command_status::complete) {
        x.fail("event_incomplete_after_drain");return;
      }
    uint64_t previous_end=0;
    for (size_t i=0;i<x.used;++i) {
      auto& p=x.pending[i];auto& r=p.record;
      const auto before=std::chrono::steady_clock::now();
      r.submitted=p.event->get_profiling_info<sycl::info::event_profiling::command_submit>();
      r.start=p.event->get_profiling_info<sycl::info::event_profiling::command_start>();
      r.end=p.event->get_profiling_info<sycl::info::event_profiling::command_end>();
      const uint64_t wall=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-before).count();
      x.query_wall_ns=std::max(x.query_wall_ns,wall);
      // Intel LLVM estimates command_submit separately from GPU timestamps.
      // An inversion disqualifies queue latency, not positive ordered GPU
      // service. Preserve the raw tuple; never clamp timestamps or subtract
      // unsigned values before checking order. Original strict failures stay
      // failed in Git. This version2 policy is prospective.
      if (r.start<r.submitted) ++x.submit_clock_inversions;
      const unsigned violations=(unsigned(r.end<=r.start)<<1) |
          (unsigned(r.end>r.start && r.end-r.start<x.timer_resolution)<<2) |
          (unsigned(wall>=1000000000ULL)<<3) |
          (unsigned(i && r.start<previous_end)<<4);
      previous_end=r.end;
      if (violations) {
        x.fail("timestamp_order_granularity_or_query_delay");++x.invalid_total;
        if (x.invalid_saved<x.invalid.size()) x.invalid[x.invalid_saved++]={r,wall,violations};
        continue;
      }
      if (x.records==RecordsMax) { x.fail("record_capacity");continue; }
      x.record[x.records++]=r;
      auto& g=x.window[r.window].groups[size_t(r.meta.kind)][r.bin];
      g.service+=r.end-r.start;++g.service_samples;
      if (r.start>=r.submitted) { g.queue_delay+=r.start-r.submitted;++g.queue_delay_samples; }
    }
    for (size_t i=0;i<x.used;++i) x.pending[i].event.reset();
    x.used=0;
  } catch (...) { x.fail("profiling_query_exception"); }
}
} // namespace strata::sycl_backend::command_profile
