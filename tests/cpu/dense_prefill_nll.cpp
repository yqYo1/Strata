// SPDX-FileCopyrightText: 2026 yayoi
// SPDX-License-Identifier: MIT
#include "strata/program/dense_prefill_nll.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace nll = strata::program::dense_prefill_nll;
namespace fs = std::filesystem;
static void need(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
static const std::string recipe(64, 'a');
struct PrivateDirectory {
    fs::path path;
    PrivateDirectory() {
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        for (int i = 0; i < 100; ++i) {
            path = fs::temp_directory_path() / ("strata-nll-host-" + std::to_string(nonce) + "-" + std::to_string(i));
            if (fs::create_directory(path)) return;
        }
        throw std::runtime_error("cannot create private fixture directory");
    }
    ~PrivateDirectory() { std::error_code ec; fs::remove_all(path, ec); }
};
static std::string bytes(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    need(bool(in), "read fixture");
    return {std::istreambuf_iterator<char>(in), {}};
}
static uint32_t u32(const std::string& s, size_t p) {
    uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i) value |= uint32_t(static_cast<unsigned char>(s.at(p+i))) << (8*i);
    return value;
}
static double f64(const std::string& s, size_t p) {
    uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) value |= uint64_t(static_cast<unsigned char>(s.at(p+i))) << (8*i);
    double result; std::memcpy(&result, &value, sizeof(result)); return result;
}
// Independent CPU FP64 reference: an online log-add-exp recurrence relative
// to the target, rather than the production max/sum/Kahan reduction.
static double oracle(const std::vector<float>& x, size_t target) {
    double total = -std::numeric_limits<double>::infinity();
    for (float v : x) {
        const double next = double(v) - double(x[target]);
        if (!std::isfinite(total)) total = next;
        else total = std::max(total, next) + std::log1p(std::exp(-std::abs(total-next)));
    }
    return total;
}
struct FaultIO : nll::Stdio {
    static inline int writes = 0, fail_write = 0;
    static inline bool fail_seek = false, fail_flush = false, fail_close = false;
    static void reset() { writes = fail_write = 0; fail_seek = fail_flush = fail_close = false; }
    static size_t write(const void* p, size_t n, std::FILE* f) {
        if (++writes == fail_write) return n ? n-1 : 0;
        return nll::Stdio::write(p, n, f);
    }
    static int seek(std::FILE* f, long p) { return fail_seek ? -1 : nll::Stdio::seek(f, p); }
    static int flush(std::FILE* f) { return fail_flush ? -1 : nll::Stdio::flush(f); }
    static int close(std::FILE* f) { const int r = nll::Stdio::close(f); return fail_close ? -1 : r; }
};
static void reductions() {
    std::vector<std::vector<float>> cases = {{0,0}, {10000,10000}, {-10000,-10000},
        {-2,0,1,9}, {-1000,1000}, {std::numeric_limits<float>::max(),std::numeric_limits<float>::max()},
        {-std::numeric_limits<float>::max(),std::numeric_limits<float>::max()}};
    std::vector<float> wide(248320);
    for (size_t j = 0; j < wide.size(); ++j) wide[j] = float(int(j % 149)-74) / 16;
    cases.push_back(std::move(wide));
    for (const auto& x : cases) for (size_t t : {size_t(0), x.size()-1}) {
        double actual = -1;
        need(nll::target_loss(x.data(), x.size(), t, actual), "finite reduction");
        const double expected = oracle(x, t);
        need(std::abs(actual-expected) <= 2e-11 * std::max(1.0,std::abs(expected)), "FP64 independent oracle");
    }
    double value;
    const float equal[] = {10000,10000};
    need(nll::target_loss(equal, 2, 0, value) && std::abs(value-std::log(2.0)) < 1e-15, "offset cancellation");
    for (float bad : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()}) {
        const float row[] = {0,bad}; need(!nll::target_loss(row,2,0,value), "reject nonfinite row");
    }
    need(!nll::target_loss(nullptr,2,0,value) && !nll::target_loss(equal,2,2,value), "reject invalid reduction");
}
static void complete_and_budget(const fs::path& root, bool full) {
    const size_t count = full ? nll::Writer::max_input : 8;
    std::vector<int64_t> input(count);
    for (size_t i = 0; i < count; ++i) input[i] = int64_t(i%2);
    std::string error; const auto path = root / (full ? "physical.bin" : "chunks.bin");
    {
        nll::Writer writer;
        need(writer.begin(path.string(),input.data(),input.size(),2,true,13,recipe,error), "begin valid");
        for (size_t start = 0; start < count-1;) {
            const size_t rows = std::min(size_t(full ? 8192 : 3),count-1-start);
            need(writer.chunk(int64_t(start),int64_t(rows),error), "contiguous callback");
            for (size_t p = start; p < start+rows; ++p) {
                writer.row()[0] = writer.row()[1] = 10000.0f;
                need(writer.append(int64_t(p), input[p+1], error), "write exact gold row");
            }
            start += rows;
        }
        need(writer.finish(0.25,error), "complete full coverage");
        need(!writer.finish(0.25,error), "reject double finish");
    }
    const auto raw = bytes(path); const auto meta = bytes(path.string()+".meta");
    need(raw.size() == 64+12*(count-1) && raw.size()+meta.size() < nll::Writer::raw_limit, "bounded exact file sizes");
    need(raw.substr(0,8) == "STNLLV2\n" && u32(raw,8)==2 && u32(raw,12)==64 &&
         u32(raw,16)==count && u32(raw,20)==2 && u32(raw,24)==count-1 && u32(raw,28)==uint32_t(input[0]), "LE header and coverage");
    for (size_t p = 0; p < count-1; ++p) {
        need(u32(raw,64+12*p)==uint32_t(input[p+1]), "all token identities");
        need(std::abs(f64(raw,68+12*p)-std::log(2.0)) < 1e-15, "binary64 loss roundtrip");
    }
    need(meta.find("recipe_sha256="+recipe+"\n") != std::string::npos &&
         meta.find("complete=1\n") != std::string::npos, "receipt identity and publication");
    std::string error2; nll::Writer collision;
    need(!collision.begin(path.string(),input.data(),count,2,true,13,recipe,error2), "no clobber completed capture");
    need(bytes(path)==raw && bytes(path.string()+".meta")==meta, "existing files unchanged");
}
static void invalid_and_failures(const fs::path& root) {
    const int64_t input[] = {0,1,0}; std::string error;
    int serial = 0;
    auto next = [&]() { return (root/("bad-"+std::to_string(++serial))).string(); };
    for (int bad = 0; bad < 8; ++bad) {
        const auto path = next();
        { nll::Writer w;
          need(w.begin(path,input,3,2,false,-1,recipe,error), "begin invalid scenario");
          w.row()[0]=w.row()[1]=0;
          if (bad==0) need(!w.chunk(1,1,error), "reject gap");
          if (bad==1) need(!w.chunk(0,3,error), "reject beyond final row");
          if (bad==2) { need(w.chunk(0,2,error), "chunk"); need(!w.append(1,0,error), "reject skipped row"); }
          if (bad==3) { need(w.chunk(0,2,error), "chunk"); need(!w.append(0,0,error), "reject wrong target"); }
          if (bad==4) { need(w.chunk(0,2,error), "chunk"); need(!w.finish(0,error), "reject incomplete coverage"); }
          if (bad==5) { need(w.chunk(0,1,error), "chunk"); need(w.append(0,1,error), "append"); need(!w.chunk(0,1,error), "reject overlap"); }
          if (bad==6) { need(w.chunk(0,1,error), "chunk"); need(!w.chunk(1,1,error), "reject unconsumed chunk"); }
          if (bad==7) { need(w.chunk(0,1,error), "chunk"); w.row()[0]=std::numeric_limits<float>::quiet_NaN(); need(!w.append(0,1,error), "reject NaN"); }
        }
        need(!fs::exists(path+".meta"), "failed diagnostic never published");
    }
    for (int failure = 1; failure <= 8; ++failure) {
        const auto path=next(); FaultIO::reset();
        { nll::BasicWriter<FaultIO> w;
          if (failure<=5) FaultIO::fail_write=failure;
          if (failure==6) FaultIO::fail_seek=true;
          if (failure==7) FaultIO::fail_flush=true;
          if (failure==8) FaultIO::fail_close=true;
          bool ok = w.begin(path,input,2,2,true,13,recipe,error);
          if (ok) { w.row()[0]=w.row()[1]=0; ok=w.chunk(0,1,error) && w.append(0,1,error); }
          if (ok) ok=w.finish(0,error);
          need(!ok, "injected write/seek/flush/close failure rejects completion");
        }
        need(!fs::exists(path+".meta"), "I/O error never publishes");
    }
    FaultIO::reset();
    { nll::Writer w; need(!w.begin(next(),input,262145,2,true,13,recipe,error), "max input checked before input read"); }
    { nll::Writer w; need(!w.begin(next(),input,3,2,true,13,"bad",error), "recipe digest required"); }
    { const int64_t bad[]={0,2}; nll::Writer w; need(!w.begin(next(),bad,2,2,true,13,recipe,error), "invalid gold ID"); }
    const auto path=next();
    { nll::Writer w; need(w.begin(path,input,2,2,true,13,recipe,error), "begin publish collision");
      need(w.chunk(0,1,error), "chunk"); w.row()[0]=w.row()[1]=0; need(w.append(0,1,error), "append");
      {std::ofstream existing(path+".meta"); existing<<"existing";}
      need(!w.finish(0,error), "no-clobber publish race"); }
    need(bytes(path+".meta")=="existing", "publication preserves preexisting receipt");
}
int main() {
    try { PrivateDirectory d; reductions(); complete_and_budget(d.path,false); complete_and_budget(d.path,true);
          invalid_and_failures(d.path); std::cout<<"dense_prefill_nll_host PASS\n"; return 0; }
    catch (const std::exception& e) { std::cerr<<"dense_prefill_nll_host FAIL: "<<e.what()<<'\n'; return 1; }
}
