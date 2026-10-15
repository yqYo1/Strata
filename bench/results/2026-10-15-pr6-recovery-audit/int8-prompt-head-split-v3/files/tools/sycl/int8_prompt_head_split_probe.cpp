// UNBUILT / UNTESTED v3 fixed-environment single-arm capture proposal.
// Derived from matrix-direct-probe-v1; original proposal remains unchanged.
// Product entrypoint, independent FP64 oracle, guards, immutable owners.
// No model capture; no fallback. Admission alone does not prove dispatch.
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/sycl/runtime.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include "strata/core/mtp_expert_source.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <sstream>
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
using namespace strata;
using namespace strata::kernels;
using sycl_backend::Allocation;
using sycl_backend::MemoryKind;
using sycl_backend::Runtime;
constexpr size_t Guard = 16;
constexpr int Heads = 24, KvHeads = 2, Dim = 256, PageSize = 4;
constexpr int LogicalCells = 262144, LogicalPages = LogicalCells / PageSize;
constexpr size_t LiveBudget = size_t(512) << 20;
constexpr double OracleTolerance = 1e-5;
constexpr float OutputSentinel = 12345.25f;

void require(bool ok, const std::string& message) {
  if (!ok) throw std::invalid_argument(message);
}
size_t multiply(uint64_t a, uint64_t b) {
  const uint64_t maximum = std::min<uint64_t>(SIZE_MAX, std::numeric_limits<std::ptrdiff_t>::max());
  require(!b || a <= maximum / b, "geometry byte/element multiplication overflow");
  return size_t(a * b);
}
size_t add(size_t a, size_t b) {
  require(b <= SIZE_MAX - a, "geometry byte addition overflow");
  return a + b;
}
struct Geometry {
  int64_t nq, cap, physical_pages;
  size_t query_elements, selection_elements, step_elements, pool_rows, live_bound;
  Geometry(int64_t queries, int64_t capacity, int64_t physical)
      : nq(queries), cap(capacity), physical_pages(physical) {
    require(nq > 0 && nq <= 65535 && cap > 0 && cap <= INT32_MAX &&
            physical_pages > 0 && physical_pages <= LogicalPages, "unsupported fixture geometry");
    query_elements = multiply(multiply(nq, Heads), Dim);
    selection_elements = multiply(nq, cap);
    step_elements = multiply(nq, kStepCount);
    pool_rows = multiply(physical_pages, PageSize * KvHeads);
    // Conservative combined host/device/reference bound. Includes guarded
    // arrays and all format representations even though only one is populated.
    live_bound = add(multiply(query_elements, 32), multiply(selection_elements, 16));
    live_bound = add(live_bound, multiply(step_elements, 16));
    live_bound = add(live_bound, multiply(LogicalPages, 16));
    live_bound = add(live_bound, multiply(multiply(pool_rows, Dim), 48));
    live_bound = add(live_bound, size_t(1) << 20);
    require(live_bound <= LiveBudget, "fixture exceeds conservative 512 MiB live-byte budget");
  }
};
uint16_t half(float x) { return std::bit_cast<uint16_t>(_Float16(x)); }
double decode(uint16_t x) { return double(std::bit_cast<_Float16>(x)); }
template<class T> uint64_t hash(const std::vector<T>& values) {
  uint64_t result = 14695981039346656037ull;
  const auto* bytes = reinterpret_cast<const uint8_t*>(values.data());
  for (size_t i = 0; i < values.size() * sizeof(T); ++i) { result ^= bytes[i]; result *= 1099511628211ull; }
  return result;
}

int64_t capacity(int width) {
  require(width >= 0 && width <= 2051, "fixture width must be 0..2051");
  return std::max(1, width);
}
struct Input {
  Geometry g;
  int format, width;
  bool dense, nonresident, zero_scales;
  std::vector<float> query;
  std::vector<int32_t> table, ids, steps;
  std::vector<uint16_t> k16, v16, ks, vs;
  std::vector<int8_t> k8, v8;
  std::vector<uint8_t> k4, v4;
  Input(int mode, int64_t nq, int count, bool hot = false, bool missing = false, bool zero = false, int64_t explicit_cap = 0)
      : g(nq, explicit_cap ? explicit_cap : capacity(count), hot ? (capacity(count) + 3) / 4 + 1 : 5),
        format(mode), width(count), dense(hot), nonresident(missing), zero_scales(zero),
        query(g.query_elements), table(LogicalPages, -1), ids(g.selection_elements, -1), steps(g.step_elements, 0) {
    require(format >= 0 && format <= 3 && width >= 0 && width <= g.cap,
            "invalid input format/width");
    if (!nonresident) {
      if (dense) {
        const int pages = int(g.physical_pages) - 1;
        for (int i = 0; i < pages; ++i) table[size_t(i)] = (i + 7) % pages;
        table.back() = pages;
      } else {
        table[0] = 3; table[1] = 1; table[17] = 4;
        table[65534] = 0; table[65535] = 2;
      }
    }
    for (size_t i = 0; i < query.size(); ++i) query[i] = float(std::sin(double(i) * .019));
    for (int64_t b = 0; b < g.nq; ++b) {
      steps[size_t(b) * kStepCount + kStepWidth] = width;
      steps[size_t(b) * kStepCount + kStepPos] = LogicalCells - 1;
      steps[size_t(b) * kStepCount + kStepNKv] = LogicalCells;
      steps[size_t(b) * kStepCount + kStepNBid] = LogicalPages;
      for (int i = 0; i < width; ++i) {
        int cell = i == width - 1 ? LogicalCells - 1 : dense ? i : i * 3 + int(b % 2);
        if (!dense && i % 11 == 9 && i != width - 1) cell = -7;
        ids[size_t(b * g.cap + i)] = cell;
      }
    }
    const size_t elements = multiply(g.pool_rows, Dim), scales = multiply(g.pool_rows, 4);
    if (format == 0) { k16.resize(elements); v16.resize(elements); }
    if (format == 1 || format == 3) { k8.resize(elements); ks.resize(scales); }
    if (format == 1) { v8.resize(elements); vs.resize(scales); }
    if (format == 2) k4.resize(multiply(g.pool_rows, 144));
    if (format == 2 || format == 3) v4.resize(multiply(g.pool_rows, 144));
    for (size_t i = 0; i < elements; ++i) {
      if (!k16.empty()) { k16[i] = half(zero_scales ? 0.f : float(std::sin(double(i) * .073))); v16[i] = half(zero_scales ? 0.f : float(std::cos(double(i) * .017))); }
      if (!k8.empty()) k8[i] = int8_t(int(i * 7 % 255) - 127);
      if (!v8.empty()) v8[i] = int8_t(int(i * 11 % 255) - 127);
    }
    for (size_t i = 0; i < scales; ++i) {
      if (!ks.empty()) ks[i] = half(zero_scales || i % 17 == 5 ? 0.f : .007f + float(i % 7) * .0001f);
      if (!vs.empty()) vs[i] = half(zero_scales || i % 19 == 7 ? 0.f : .008f + float(i % 11) * .0002f);
    }
    for (size_t i = 0; i < g.pool_rows * 8; ++i) {
      if (!k4.empty()) {
        const uint16_t scale = half(zero_scales ? 0.f : .11f + float(i % 7) * .002f);
        std::memcpy(k4.data() + i * 18, &scale, 2);
        for (int j = 0; j < 16; ++j) k4[i * 18 + 2 + size_t(j)] = uint8_t((i * 7 + size_t(j) * 13) % 256);
      }
      if (!v4.empty()) {
        const uint16_t scale = half(zero_scales ? 0.f : -.09f - float(i % 9) * .003f);
        std::memcpy(v4.data() + i * 18, &scale, 2);
        for (int j = 0; j < 16; ++j) v4[i * 18 + 2 + size_t(j)] = uint8_t((i * 11 + size_t(j) * 19) % 256);
      }
    }
    // Full host address validation precedes runtime creation for a caller
    // constructing inputs before obtaining its runtime.
    for (const int cell : ids) {
      require(cell < LogicalCells, "selected logical cell exceeds page table");
      if (cell < 0) continue;
      const int page = table[size_t(cell / PageSize)];
      require(page < g.physical_pages, "physical page exceeds small pool");
    }
  }
  double load(bool value, size_t row, int dimension) const {
    if (format == 0) return decode((value ? v16 : k16)[row * Dim + size_t(dimension)]);
    if (format == 1 || (format == 3 && !value))
      return double((value ? v8 : k8)[row * Dim + size_t(dimension)]) *
             decode((value ? vs : ks)[row * 4 + size_t(dimension / 64)]);
    const auto& data = value ? v4 : k4;
    const size_t block = row * 8 + size_t(dimension / 32);
    uint16_t scale;
    std::memcpy(&scale, data.data() + block * 18, 2);
    const uint8_t bits = data[block * 18 + 2 + size_t(dimension % 16)];
    return double(int(dimension % 32 < 16 ? bits & 15 : bits >> 4) - 8) * decode(scale);
  }
  int64_t row(int64_t batch, int head, int column) const {
    const int cell = ids[size_t(batch * g.cap + column)];
    if (cell < 0) return -1;
    const int page = table[size_t(cell / PageSize)];
    return page < 0 ? -1 : (int64_t(page) * KvHeads + head / 12) * PageSize + cell % PageSize;
  }
};

std::vector<double> oracle(const Input& input) {
  std::vector<double> result(input.g.query_elements, 0.);
  for (int64_t b = 0; b < input.g.nq; ++b)
    for (int h = 0; h < Heads; ++h) {
      const int raw_width = input.steps[size_t(b) * kStepCount + kStepWidth];
      const int count = raw_width >= 0 && raw_width <= input.g.cap ? raw_width : 0;
      std::vector<double> weights(size_t(count), -INFINITY);
      double maximum = -INFINITY, denominator = 0.;
      for (int j = 0; j < count; ++j) {
        const int64_t row = input.row(b, h, j);
        if (row < 0) continue;
        double dot = 0.;
        for (int d = 0; d < Dim; ++d)
          dot += double(input.query[size_t((b * Heads + h) * Dim + d)]) * input.load(false, size_t(row), d);
        weights[size_t(j)] = dot / 16.;
        maximum = std::max(maximum, weights[size_t(j)]);
      }
      for (double& value : weights) { value = std::isfinite(value) ? std::exp(value - maximum) : 0.; denominator += value; }
      if (denominator == 0.) continue;
      for (int d = 0; d < Dim; ++d) {
        double numerator = 0.;
        for (int j = 0; j < count; ++j) {
          const int64_t row = input.row(b, h, j);
          if (row >= 0) numerator += weights[size_t(j)] * input.load(true, size_t(row), d);
        }
        result[size_t((b * Heads + h) * Dim + d)] = numerator / denominator;
      }
    }
  return result;
}

[[noreturn]] void fail_stop(const char* context) noexcept {
  std::fprintf(stderr, "matrix-direct probe: cannot prove completion (%s); aborting with owners live\n", context);
  std::fflush(stderr); sycl_backend::cleanup_fail_stop(context);
}
void drain(const std::shared_ptr<Runtime>& runtime) noexcept {
  try { runtime->wait(); } catch (...) { fail_stop("checked runtime drain"); }
}
// Declared after all input/device owners, before the first upload/submission.
struct Lease {
  std::shared_ptr<Runtime> runtime;
  ~Lease() noexcept { drain(runtime); }
};
template<class T> struct Buffer {
  std::shared_ptr<Runtime> runtime;
  std::vector<T> initial;
  Allocation memory;
  Buffer(std::shared_ptr<Runtime> rt, const std::vector<T>& values)
      : runtime(std::move(rt)), initial(add(values.size(), 2 * Guard), T(91)),
        memory(runtime, multiply(initial.size(), sizeof(T)), MemoryKind::Device) {
    std::copy(values.begin(), values.end(), initial.begin() + Guard);
  }
  T* base() { return memory.as<T>(); }
  T* data() { return base() + Guard; }
  void upload() {
    try { runtime->wait(runtime->compute().memcpy(base(), initial.data(), memory.size())); }
    catch (...) { fail_stop("upload submission/wait"); }
  }
  std::vector<T> read() {
    std::vector<T> result(initial.size());
    try { runtime->wait(runtime->compute().memcpy(result.data(), base(), memory.size())); }
    catch (...) { fail_stop("readback submission/wait"); }
    return result;
  }
  void immutable() {
    const auto current = read();
    require(std::memcmp(current.data(), initial.data(), memory.size()) == 0, "input payload/guard modified");
  }
  void guards(const std::vector<T>& values) const {
    require(values.size() == initial.size(), "output readback size");
    require(std::memcmp(values.data(), initial.data(), Guard * sizeof(T)) == 0 &&
            std::memcmp(values.data() + values.size() - Guard,
                        initial.data() + initial.size() - Guard, Guard * sizeof(T)) == 0,
            "output prefix/suffix guard modified");
  }
};
struct DeviceInput {
  Buffer<float> query, output;
  Buffer<int32_t> table, ids, steps;
  Buffer<uint16_t> k16, v16, ks, vs;
  Buffer<int8_t> k8, v8;
  Buffer<uint8_t> k4, v4;
  DeviceInput(std::shared_ptr<Runtime> rt, const Input& input)
      : query(rt, input.query), output(rt, std::vector<float>(input.g.query_elements, OutputSentinel)),
        table(rt, input.table), ids(rt, input.ids), steps(rt, input.steps),
        k16(rt, input.k16), v16(rt, input.v16), ks(rt, input.ks), vs(rt, input.vs),
        k8(rt, input.k8), v8(rt, input.v8), k4(rt, input.k4), v4(rt, input.v4) {}
  void upload() {
    query.upload(); output.upload(); table.upload(); ids.upload(); steps.upload();
    k16.upload(); v16.upload(); ks.upload(); vs.upload(); k8.upload(); v8.upload(); k4.upload(); v4.upload();
  }
  QsaAttnPools pools(int format) {
    QsaAttnPools p;
    p.page_table = table.data();
    if (format == 0) { p.k_pool = k16.data(); p.v_pool = v16.data(); }
    if (format == 1 || format == 3) { p.k_q = k8.data(); p.k_scale = ks.data(); }
    if (format == 1) { p.v_q = v8.data(); p.v_scale = vs.data(); }
    if (format == 2) p.k_q4 = k4.data();
    if (format == 2 || format == 3) p.v_q4 = v4.data();
    return p;
  }
  void immutable() {
    query.immutable(); table.immutable(); ids.immutable(); steps.immutable();
    k16.immutable(); v16.immutable(); ks.immutable(); vs.immutable();
    k8.immutable(); v8.immutable(); k4.immutable(); v4.immutable();
  }
};
// Environment is read/validated once before runtime initialization. No process
// environment writer is present in this probe, including its host-only path.
std::string quoted(const std::string& s) {
  std::ostringstream out; out << '"';
  for (unsigned char c : s) {
    if (c=='"' || c=='\\') out << '\\' << char(c);
    else if (c<32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
    else out << char(c);
  }
  out << '"'; return out.str();
}
std::string env_value(const char* key) {
  const char* p=std::getenv(key);
  require(!p || (std::string(p)!="@unset" && std::string(p)!="@empty"),"reserved marker in actual route environment");
  return p ? std::string(p) : "@unset";
}
std::string expectation(const std::string& value) {
  if (value=="@empty") return "";
  require(value.empty() || value[0]!='@' || value=="@unset", "unknown env expectation marker");
  return value;
}
bool hex64(const std::string& s) {
  return s.size()==64 && std::all_of(s.begin(),s.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');});
}
struct Options {
  std::string route, cases, dir, direct, split, source, binary, fixture;
  int format=1;
  size_t budget=size_t(8)<<20;
};
Options parse(int argc, char** argv) {
  Options o; std::vector<std::string> seen;
  for (int i=1;i<argc;i+=2) {
    require(i+1<argc,"every option requires a value");
    const std::string key=argv[i], value=argv[i+1];
    require(std::find(seen.begin(),seen.end(),key)==seen.end(),"duplicate CLI option"); seen.push_back(key);
    if (key=="--route") o.route=value;
    else if (key=="--case-set") o.cases=value;
    else if (key=="--capture-dir") o.dir=value;
    else if (key=="--expect-direct") o.direct=expectation(value);
    else if (key=="--expect-split") o.split=expectation(value);
    else if (key=="--source-id") o.source=value;
    else if (key=="--binary-id") o.binary=value;
    else if (key=="--fixture-id") o.fixture=value;
    else if (key=="--format" || key=="--capture-budget") {
      uint64_t n=0; const auto p=std::from_chars(value.data(),value.data()+value.size(),n);
      require(!value.empty() && p.ec==std::errc{} && p.ptr==value.data()+value.size(),"invalid unsigned CLI integer");
      if (key=="--format") { require(n<=3,"format must be 0..3"); o.format=int(n); }
      else { require(n>=1024 && n<=(size_t(16)<<20),"capture budget must be 1 KiB..16 MiB"); o.budget=size_t(n); }
    } else require(false,"unknown CLI option");
  }
  for (const std::string& key : {"--route","--case-set","--capture-dir","--expect-direct","--expect-split","--source-id","--binary-id","--fixture-id"})
    require(std::find(seen.begin(),seen.end(),key)!=seen.end(),"missing required CLI option "+key);
  require(o.route=="legacy" || o.route=="32" || o.route=="64","route must be legacy/32/64");
  require(o.cases=="full" || o.cases=="isolation" || o.cases=="maxcount","case-set must be full/isolation/maxcount");
  require(hex64(o.source)&&hex64(o.binary)&&hex64(o.fixture),"provenance IDs must be SHA256 hex");
  if (o.cases=="full" || o.cases=="maxcount") require(o.format==1 && o.direct=="1" &&
      o.split==(o.route=="legacy" ? "@unset" : o.route),"full case route intent mismatch");
  const std::string predicted=o.format==1 && o.direct=="1" && (o.split=="32"||o.split=="64") ? o.split : "legacy";
  require(predicted==o.route,"route intent does not match legacy dispatch semantics");
  // Runtime has not been created. Only targeted read-only validation is used.
  require(env_value("STRATA_SYCL_PROMPT_DIRECT")==o.direct,"Direct exec environment mismatch");
  require(env_value("STRATA_SYCL_PROMPT_HEAD_SPLIT")==o.split,"split exec environment mismatch");
  const auto native=env_value("STRATA_SYCL_PROMPT_NATIVE_KV_MIN");
  require(native=="@unset" || native=="0","independent Native KV option must be unset or exact0 at exec");
  return o;
}
std::string sha(const void* data, size_t bytes) {
  core::detail::MtpExpertSha256 h; h.update(static_cast<const uint8_t*>(data),bytes);
  std::ostringstream out; out << std::hex << std::setfill('0');
  for (auto b : h.finish()) out << std::setw(2) << unsigned(b);
  return out.str();
}
template<class T> std::string identity(const std::vector<T>& v, const char* dtype) {
  return "{\"dtype\":"+quoted(dtype)+",\"count\":"+std::to_string(v.size())+
         ",\"sha256\":"+quoted(sha(v.data(),v.size()*sizeof(T)))+"}";
}
std::string inputs_json(const Input& in) {
  return "{\"q\":"+identity(in.query,"f32")+",\"table\":"+identity(in.table,"i32")+
    ",\"ids\":"+identity(in.ids,"i32")+",\"steps\":"+identity(in.steps,"i32")+
    ",\"k16\":"+identity(in.k16,"u16")+",\"v16\":"+identity(in.v16,"u16")+
    ",\"k8\":"+identity(in.k8,"i8")+",\"v8\":"+identity(in.v8,"i8")+
    ",\"ks\":"+identity(in.ks,"u16")+",\"vs\":"+identity(in.vs,"u16")+
    ",\"k4\":"+identity(in.k4,"u8")+",\"v4\":"+identity(in.v4,"u8")+"}";
}
struct Fd {
  int value=-1;
  explicit Fd(int v=-1):value(v){}
  ~Fd(){if(value>=0) ::close(value);}
  Fd(const Fd&)=delete; Fd& operator=(const Fd&)=delete;
};
class Capture {
  Fd dir_; size_t remaining_, written_=0;
public:
  Capture(const std::string& path, size_t budget):remaining_(budget) {
    require(!path.empty()&&path.front()=='/'&&path.back()!='/',"capture path must be absolute without trailing slash");
    std::vector<std::string> parts;
    for (size_t a=1;a<path.size();) {
      const size_t b=path.find('/',a); const auto p=path.substr(a,b==std::string::npos ? path.size()-a : b-a);
      require(!p.empty()&&p!="."&&p!="..","unsafe capture path component"); parts.push_back(p);
      if (b==std::string::npos) break; a=b+1;
    }
    require(!parts.empty(),"capture path cannot be root");
    Fd parent(::open("/",O_RDONLY|O_DIRECTORY|O_CLOEXEC)); require(parent.value>=0,"open root directory");
    for (size_t i=0;i+1<parts.size();++i) {
      const int next=::openat(parent.value,parts[i].c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW);
      require(next>=0,"capture parent missing/not directory/symlink");
      ::close(parent.value); parent.value=next;
    }
    struct stat st{}; require(::fstat(parent.value,&st)==0 && st.st_uid==::geteuid() && !(st.st_mode&0022),
                              "capture parent must be owned and not writable by group/other");
    require(::mkdirat(parent.value,parts.back().c_str(),0700)==0,"capture directory must be NEW; no overwrite");
    dir_.value=::openat(parent.value,parts.back().c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW);
    require(dir_.value>=0 && ::fsync(parent.value)==0,"open/fsync new capture directory");
  }
  size_t written() const {return written_;}
  void save(const std::string& name, const void* data, size_t bytes) {
    require(!name.empty() && std::all_of(name.begin(),name.end(),[](char c){return (c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='_'||c=='-'||c=='.';}),"unsafe capture filename");
    require(bytes<=remaining_,"capture byte budget exhausted");
    Fd file(::openat(dir_.value,name.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600));
    require(file.value>=0,"new capture file open failed; no overwrite");
    const auto* p=static_cast<const uint8_t*>(data); size_t done=0;
    while (done<bytes) {
      const ssize_t n=::write(file.value,p+done,bytes-done);
      if (n<0 && errno==EINTR) continue;
      require(n>0,"capture write failed"); done+=size_t(n);
    }
    require(::fsync(file.value)==0,"capture fsync failed");
    const int fd=file.value; file.value=-1; require(::close(fd)==0,"capture close failed");
    require(::fsync(dir_.value)==0,"capture directory fsync failed");
    remaining_-=bytes; written_+=bytes;
  }
  void text(const std::string& name, const std::string& data) {save(name,data.data(),data.size());}
};
struct Declined : std::runtime_error {using std::runtime_error::runtime_error;};
bool attention(const std::shared_ptr<Runtime>& rt,const Input& in,DeviceInput& dev) {
  const auto s=qsa_real_shapes();
  return qsa_prompt_attn_batch(dev.query.data(),dev.pools(in.format),dev.ids.data(),dev.steps.data(),
                               in.g.cap,s,dev.output.data(),in.g.nq,&rt->compute());
}
void run_case(const std::shared_ptr<Runtime>& rt, Capture& cap, const Input& in,
              const std::string& id, std::vector<std::string>& cases) {
  const auto fingerprint=inputs_json(in);
  const auto ref=oracle(in); // independent scalar FP64 over actual stored inputs
  DeviceInput dev(rt,in); Lease lease{rt}; dev.upload();
  const bool admitted=attention(rt,in,dev); drain(rt); const auto out=dev.output.read();
  const auto outname=id+".out.f32", refname=id+".ref.f64";
  cap.save(outname,out.data(),out.size()*sizeof(float));
  cap.save(refname,ref.data(),ref.size()*sizeof(double));
  // Save arrays before checks. Any failed/missing receipt remains unusable for
  // comparison, but useful first-failure arrays are retained for Main review.
  dev.immutable();
  const bool guards=std::memcmp(out.data(),dev.output.initial.data(),Guard*sizeof(float))==0 &&
    std::memcmp(out.data()+out.size()-Guard,dev.output.initial.data()+out.size()-Guard,Guard*sizeof(float))==0;
  bool good=true; double worst=0.; size_t nonfinite=0,untouched=0;
  if (admitted) for (size_t i=0;i<ref.size();++i) {
    const float value=out[Guard+i]; const double error=std::abs(double(value)-ref[i]);
    if (std::isfinite(error)) worst=std::max(worst,error);
    if (!std::isfinite(value)) ++nonfinite;
    if (value==OutputSentinel) ++untouched;
    if (!std::isfinite(value)||!std::isfinite(ref[i])||error>OracleTolerance*(1.+std::abs(ref[i]))) good=false;
  }
  const bool unchanged=!admitted && std::memcmp(out.data(),dev.output.initial.data(),dev.output.memory.size())==0;
  std::ostringstream record; record << std::boolalpha << std::setprecision(17)
    << "{\"schema\":\"head-split-case-v2\",\"case_id\":" << quoted(id)
    << ",\"format\":" << in.format << ",\"nq\":" << in.g.nq << ",\"cap\":" << in.g.cap
    << ",\"width\":" << in.width << ",\"step_width0\":" << in.steps[kStepWidth]
    << ",\"physical_pages\":" << in.g.physical_pages << ",\"inputs\":" << fingerprint
    << ",\"admitted\":" << admitted << ",\"fallback_calls\":0,\"input_immutable\":true"
    << ",\"guard_count\":16,\"output_guards\":" << guards << ",\"declined_unchanged\":" << unchanged
    << ",\"output_file\":" << quoted(outname) << ",\"output_count\":" << out.size()
    << ",\"output_sha256\":" << quoted(sha(out.data(),out.size()*sizeof(float)))
    << ",\"reference_file\":" << quoted(refname) << ",\"reference_count\":" << ref.size()
    << ",\"reference_sha256\":" << quoted(sha(ref.data(),ref.size()*sizeof(double)))
    << ",\"oracle_abs_rel\":1e-5,\"oracle_pass\":" << (admitted&&good)
    << ",\"nonfinite\":" << nonfinite << ",\"untouched\":" << untouched << ",\"max_abs\":" << worst
    << ",\"status\":" << quoted(!admitted ? "UNMEASURED_DECLINED" : guards&&good ? "PASS" : "FAIL") << "}\n";
  cap.text(id+".json",record.str()); std::cout << record.str() << std::flush;
  require(guards,"output guards changed");
  if (!admitted) { require(unchanged,"declined output changed"); throw Declined("matrix call declined; no completion receipt"); }
  require(good,"all-output FP64 oracle gate failed"); cases.push_back(id);
}
// Production-reachable active selection count, not physical256K residency.
// All earlier selected positions are negative: only the final loop tile loads KV.
Input maximum_count_input(int width) {
  require(width==2050 || width==2051,"maximum-count width contract");
  Input in(1,1,width);
  std::fill(in.ids.begin(),in.ids.end(),-7);
  std::fill(in.table.begin(),in.table.end(),-1);
  in.ids[size_t(width)-1]=LogicalCells-1;
  in.table[LogicalPages-1]=2;
  require(in.g.cap==width && in.g.physical_pages==5 && in.g.pool_rows==40,
          "maximum-count small-pool geometry");
  require(in.ids.size()==size_t(width) && in.steps[kStepWidth]==width &&
          in.ids.back()==262143 && in.table[65535]==2,"maximum-count terminal slot");
  require(in.row(0,0,width-1)==19 && in.row(0,12,width-1)==23,
          "maximum-count physical rows");
  require(in.k8.size()==40*256 && in.v8.size()==40*256 &&
          in.ks.size()==40*4 && in.vs.size()==40*4,
          "maximum-count INT8/per64 scale extents");
  require(std::all_of(in.ids.begin(),in.ids.end()-1,[](int id){return id<0;}),
          "maximum-count preceding slots must be masked");
  return in;
}
void host_contracts() {
  for(int width : {2050,2051}) {
    const auto in=maximum_count_input(width);
    require(in.query.size()==24*256 && in.steps.size()==4,"maximum-count Q/steps extents");
    require(in.table.size()==65536 && in.width==width,"maximum-count logical table");
    require((width+31)/32==65 && (width+63)/64==33 && width%32==width-2048 && width%64==width-2048,
            "maximum-count final partial chunks");
  }
  for (size_t nq : {size_t(1),size_t(3),size_t(65535)}) for (size_t batch : {size_t(0),nq-1}) {
    std::array<unsigned char,Heads*Dim> owners{};
    for (size_t kv=0;kv<KvHeads;++kv) for (size_t tile=0;tile<2;++tile) {
      const size_t group=(batch*KvHeads+kv)*2+tile;
      require(group%2==tile && (group/2)%KvHeads==kv && group/(2*KvHeads)==batch,"group decode");
      for (size_t r=0;r<(tile?4u:8u);++r) for(size_t sg=0;sg<4;++sg)
        for(size_t j=0;j<4;++j) for(size_t c=0;c<16;++c)
          ++owners[(kv*12+tile*8+r)*Dim+(sg*4+j)*16+c];
    }
    require(std::all_of(owners.begin(),owners.end(),[](auto n){return n==1;}),"24-head ownership");
  }
  const auto zero=oracle(Input(1,1,65,false,true));
  require(std::all_of(zero.begin(),zero.end(),[](double n){return n==0.;}),"missing oracle");
  Input high(1,3,65,false,false,false,LogicalCells);
  require(high.ids[64]==262143 && high.ids[size_t(high.g.cap)+64]==262143,"cap stride/high logical ID");
  auto s=qsa_real_shapes(); void* invalid=reinterpret_cast<void*>(uintptr_t(1));
  require(qsa_prompt_attn_batch(nullptr,{},nullptr,nullptr,0,s,nullptr,0,invalid),"empty early gate");
  for (int64_t capacity : {int64_t(0),int64_t(-1),int64_t(INT32_MAX)+1})
    require(!qsa_prompt_attn_batch(nullptr,{},nullptr,nullptr,capacity,s,nullptr,1,invalid),"capacity early gate");
}
void extreme_query(Input& in,int tile) {
  for(int64_t b=0;b<in.g.nq;++b) for(int h=0;h<Heads;++h) for(int d=0;d<Dim;++d) {
    const bool large=(h%12<8?0:1)==tile;
    const float m=float(1.+.125*std::sin(double((b*Heads+h)*Dim+d)*.013));
    in.query[size_t((b*Heads+h)*Dim+d)]=std::ldexp(m,large?20:-20);
  }
}
} // namespace
int main(int argc,char** argv) {
  try {
    static_assert(kStepCount==4); // Offline fingerprint extent contract.
    static_assert(sizeof(float)==4 && sizeof(double)==8 && std::endian::native==std::endian::little);
    static_assert(std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559);
    if(argc==2 && std::string(argv[1])=="--help") {
      std::cout << "Single fixed-env arm: --route legacy|32|64 --case-set full|isolation|maxcount --capture-dir ABS_NEW --expect-direct VALUE --expect-split VALUE --source-id SHA256 --binary-id SHA256 --fixture-id SHA256 [--format 0..3] [--capture-budget BYTES]\n"
                   "Use @unset/@empty for env absence/empty. Also --host-only. No route mutation or benchmark.\n"; return 0;
    }
    std::cout << std::boolalpha << std::setprecision(17);
    if(argc==2 && std::string(argv[1])=="--host-only") {host_contracts();std::cout<<"HOST_PASS runtime_created=false\n";return 0;}
    const auto o=parse(argc,argv); // env is validated read-only before runtime
    Capture cap(o.dir,o.budget); host_contracts();
    // Take the targeted environment snapshot before runtime creation as well.
    const std::string native=env_value("STRATA_SYCL_PROMPT_NATIVE_KV_MIN"), q4=env_value("STRATA_PROMPT_ATTN_Q4");
    auto rt=sycl_backend::runtime_for();
    std::ostringstream start; start << "{\"schema\":\"head-split-start-v2\",\"route\":" << quoted(o.route)
      << ",\"case_set\":" << quoted(o.cases) << ",\"format\":" << o.format
      << ",\"source_sha256\":" << quoted(o.source) << ",\"binary_sha256\":" << quoted(o.binary)
      << ",\"fixture_sha256\":" << quoted(o.fixture) << ",\"expected_route_only\":true,\"route_observed\":false"
      << ",\"environment\":{\"STRATA_SYCL_PROMPT_DIRECT\":" << quoted(o.direct)
      << ",\"STRATA_SYCL_PROMPT_HEAD_SPLIT\":" << quoted(o.split)
      << ",\"STRATA_SYCL_PROMPT_NATIVE_KV_MIN\":" << quoted(native) << ",\"STRATA_PROMPT_ATTN_Q4\":" << quoted(q4) << "}"
      << ",\"device\":" << quoted(rt->device().get_info<sycl::info::device::name>())
      << ",\"driver\":" << quoted(rt->device().get_info<sycl::info::device::driver_version>())
      << ",\"capture_budget\":" << o.budget << "}\n";
    cap.text("start.json",start.str());
    std::vector<std::string> cases;
    if(o.cases=="maxcount") {
      for(int width : {2050,2051})
        run_case(rt,cap,maximum_count_input(width),"maxcount_"+std::to_string(width)+"_logical262143",cases);
    } else if(o.cases=="isolation") run_case(rt,cap,Input(o.format,1,65),"isolation_format"+std::to_string(o.format),cases);
    else {
      for(int w : {0,1,31,32,33,63,64,65,2051}) {
        run_case(rt,cap,Input(1,3,w),"sparse_tail_"+std::to_string(w),cases);
        run_case(rt,cap,Input(1,1,w,true),"dense_tail_"+std::to_string(w),cases);
      }
      run_case(rt,cap,Input(1,1,65,false,true),"all_missing",cases);
      run_case(rt,cap,Input(1,3,65,true,false,true),"zero_scales",cases);
      run_case(rt,cap,Input(1,3,65,false,false,false,LogicalCells),"cap262144_logical262143",cases);
      for(int tile : {0,1}) {Input in(1,3,65,true);extreme_query(in,tile);run_case(rt,cap,in,"large_query_tile"+std::to_string(tile),cases);}
      Input subq(1,3,65,true);
      for(size_t i=0;i<subq.query.size();++i) subq.query[i]=std::numeric_limits<float>::denorm_min()*float(int(i%15)-7);
      run_case(rt,cap,subq,"subnormal_query",cases);
      Input subs(1,3,65,true);
      for(size_t i=0;i<subs.ks.size();++i) subs.ks[i]=uint16_t(1+i%3);
      for(size_t i=0;i<subs.vs.size();++i) subs.vs[i]=uint16_t(1+i%3);
      run_case(rt,cap,subs,"fp16_subnormal_scales",cases);
      Input zero(1,3,65,true);std::fill(zero.query.begin(),zero.query.end(),0.f);run_case(rt,cap,zero,"zero_query",cases);
      for(int width : {2050,2051})
        run_case(rt,cap,maximum_count_input(width),"maxcount_"+std::to_string(width)+"_logical262143",cases);
      Input bad(1,3,65);
      for(int w : {-1,66}) {for(int b=0;b<3;++b) bad.steps[size_t(b)*kStepCount+kStepWidth]=w;
        run_case(rt,cap,bad,w<0?"negative_width":"width_gt_cap",cases);}
    }
    drain(rt);
    const size_t expected=o.cases=="full" ? 30 : o.cases=="maxcount" ? 2 : 1;require(cases.size()==expected,"case count mismatch");
    std::ostringstream done;done << "{\"schema\":\"head-split-complete-v2\",\"status\":\"CLOSED_PASS\",\"cases\":[";
    for(size_t i=0;i<cases.size();++i) {if(i)done<<',';done<<quoted(cases[i]);}
    done << "],\"bytes_before_completion\":" << cap.written() << ",\"all_admitted\":true,\"fallback_calls\":0}\n";
    cap.text("complete.json",done.str());std::cout << done.str() << std::flush;return 0;
  } catch(const Declined& e) {std::cerr<<"UNMEASURED_DECLINED: "<<e.what()<<'\n';return 2;}
  catch(const std::exception& e) {std::cerr<<"FAIL fixed-env head-split v3: "<<e.what()<<'\n';return 1;}
}
