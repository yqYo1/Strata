#!/usr/bin/env python3
"""Derive a standalone fixture proposal without compiling/running/applying it."""
from pathlib import Path
P=Path('/home/yayoi/orca/workspaces/Strata/sycl')
A=P/'bench/results/2026-10-15-pr6-recovery-audit/int8-prompt-head-split-v1'
old=(P/'bench/results/2026-10-15-pr6-recovery-audit/matrix-direct-probe-v1/files/tools/sycl/matrix_direct_probe.cpp').read_text()
prefix=old[:old.index('enum class Mode')]
body=old[old.index('uint16_t half('):old.index('void set_arm(')]
body=body.replace('bool hot = false, bool missing = false, bool zero = false)', 'bool hot = false, bool missing = false, bool zero = false, int64_t explicit_cap = 0)')
body=body.replace(': g(nq, capacity(count), hot ?', ': g(nq, explicit_cap ? explicit_cap : capacity(count), hot ?')
prefix=prefix.replace('// Synthetic component fixture. Uses the unchanged product matrix entrypoint;\n// declined calls are never filled by a fallback. No model or source capture.', '''// UNBUILT / UNTESTED head-split component fixture proposal.
// Derived from matrix-direct-probe-v1; original proposal remains unchanged.
// Product entrypoint, independent FP64 oracle, guards, immutable owners.
// No model capture; no fallback. Admission alone does not prove dispatch.''')
# No unused benchmark parser/timing is carried into this correctness fixture.
prefix=prefix.replace('using Clock = std::chrono::steady_clock;\n', '')
prefix=prefix.replace('#include <vector>\n', '#include <vector>\n#include <utility>\n')
tail=r'''struct Arm { const char* direct; const char* split; const char* expected; };
constexpr std::array<Arm, 3> Arms{{{"1", nullptr, "legacy_direct"},
                                {"1", "32", "head_split_32"},
                                {"1", "64", "head_split_64"}}};
void set_arm(const Arm& arm) {
  require(setenv("STRATA_SYCL_PROMPT_DIRECT", arm.direct, 1) == 0, "Direct env");
  require((arm.split ? setenv("STRATA_SYCL_PROMPT_HEAD_SPLIT", arm.split, 1)
                     : unsetenv("STRATA_SYCL_PROMPT_HEAD_SPLIT")) == 0, "split env");
}
bool attention(const std::shared_ptr<Runtime>& rt, const Input& in, DeviceInput& dev) {
  const auto shapes = qsa_real_shapes();
  return qsa_prompt_attn_batch(dev.query.data(), dev.pools(in.format), dev.ids.data(), dev.steps.data(),
                               in.g.cap, shapes, dev.output.data(), in.g.nq, &rt->compute());
}
void describe(const Input& in, const char* label) {
  std::cout << "{\"type\":\"input\",\"case\":\"" << label << "\",\"format\":" << in.format
            << ",\"nq\":" << in.g.nq << ",\"cap\":" << in.g.cap << ",\"width\":" << in.width
            << ",\"step_width0\":" << in.steps[kStepWidth]
            << ",\"physical_pages\":" << in.g.physical_pages << ",\"logical_cells\":262144"
            << ",\"live_byte_bound\":" << in.g.live_bound
            << ",\"q_fnv64\":" << hash(in.query) << ",\"table_fnv64\":" << hash(in.table)
            << ",\"ids_fnv64\":" << hash(in.ids) << ",\"steps_fnv64\":" << hash(in.steps)
            << ",\"k16_fnv64\":" << hash(in.k16) << ",\"v16_fnv64\":" << hash(in.v16)
            << ",\"k8_fnv64\":" << hash(in.k8) << ",\"v8_fnv64\":" << hash(in.v8)
            << ",\"ks_fnv64\":" << hash(in.ks) << ",\"vs_fnv64\":" << hash(in.vs)
            << ",\"k4_fnv64\":" << hash(in.k4) << ",\"v4_fnv64\":" << hash(in.v4) << "}" << std::endl;
}
// Returns false for any decline, which the controller must count as unmeasured.
bool validate(const std::shared_ptr<Runtime>& rt, const Input& in, const char* label,
              bool include64) {
  describe(in, label);
  const auto ref = oracle(in);
  DeviceInput dev(rt, in);
  Lease lease{rt}; // declared after owners, before submission
  dev.upload();
  std::array<std::vector<float>, 3> outputs;
  std::array<bool, 3> admitted{}, good{true, true, true};
  const int arms = include64 ? 3 : 2;
  bool measured = true;
  for (int a=0; a<arms; ++a) {
    set_arm(Arms[size_t(a)]);
    dev.output.upload();
    admitted[size_t(a)] = attention(rt, in, dev);
    drain(rt);
    outputs[size_t(a)] = dev.output.read();
    dev.output.guards(outputs[size_t(a)]);
    dev.immutable();
    double maximum = 0.;
    size_t nonfinite = 0, untouched = 0;
    measured = measured && admitted[size_t(a)];
    if (!admitted[size_t(a)]) {
      require(std::memcmp(outputs[size_t(a)].data(), dev.output.initial.data(), dev.output.memory.size()) == 0,
              "declined call modified output");
    } else {
      for (size_t i=0; i<ref.size(); ++i) {
        const float value = outputs[size_t(a)][Guard+i];
        const double error = std::abs(double(value)-ref[i]);
        maximum = std::max(maximum, error);
        if (!std::isfinite(value)) ++nonfinite;
        if (value == OutputSentinel) ++untouched;
        if (!std::isfinite(value) || error > OracleTolerance*(1.+std::abs(ref[i]))) good[size_t(a)] = false;
      }
    }
    std::cout << "{\"type\":\"arm\",\"case\":\"" << label << "\",\"arm\":" << a
              << ",\"expected_route\":\"" << Arms[size_t(a)].expected << "\",\"route_observed\":false"
              << ",\"admitted\":" << admitted[size_t(a)] << ",\"fallback_calls\":0"
              << ",\"outputs_checked\":" << (admitted[size_t(a)] ? ref.size() : 0)
              << ",\"max_abs\":" << maximum << ",\"nonfinite\":" << nonfinite << ",\"untouched\":" << untouched
              << ",\"output_fnv64\":" << hash(outputs[size_t(a)]) << ",\"oracle_abs_rel\":1e-5"
              << ",\"oracle_pass\":" << (admitted[size_t(a)] && good[size_t(a)])
              << ",\"input_immutable\":true,\"output_guards\":true}" << std::endl;
  }
  bool pairs_good = true;
  for (int a=1; a<arms; ++a) {
    const bool comparable = admitted[0] && admitted[size_t(a)];
    const bool bitwise = comparable && std::memcmp(outputs[0].data(), outputs[size_t(a)].data(),
                                                    outputs[0].size()*sizeof(float)) == 0;
    double maximum = 0.; bool close = true;
    if (comparable) for (size_t i=0; i<ref.size(); ++i) {
      const double base = outputs[0][Guard+i], value = outputs[size_t(a)][Guard+i];
      maximum = std::max(maximum, std::abs(value-base));
      if (!std::isfinite(value) || !std::isfinite(base) ||
          std::abs(value-base) > OracleTolerance*(1.+std::abs(base))) close = false;
    }
    const bool pair_pass = comparable && good[0] && good[size_t(a)] && (a==1 ? bitwise : close);
    pairs_good = pairs_good && (!comparable || pair_pass);
    std::cout << "{\"type\":\"pair\",\"case\":\"" << label << "\",\"candidate\":" << a
              << ",\"comparable\":" << comparable << ",\"bitwise\":" << bitwise
              << ",\"bitwise_required\":" << (a==1) << ",\"max_abs\":" << maximum
              << ",\"status\":\"" << (!comparable ? "UNMEASURED_DECLINED" : pair_pass ? "PASS" : "FAIL") << "\"}" << std::endl;
  }
  require(good[0] && good[1] && (!include64 || good[2]) && pairs_good, "oracle or pair gate failed");
  return measured;
}
// Dispatch isolation checks numerical invariance. Observed kernel names remain
// a separate Main-owned diagnostic; equal outputs alone cannot identify a route.
bool isolation(const std::shared_ptr<Runtime>& rt, const Input& in, const Arm& option) {
  DeviceInput dev(rt, in); Lease lease{rt}; dev.upload();
  std::vector<float> baseline;
  bool measured=true, same=true;
  for (int i=0; i<2; ++i) {
    set_arm(i ? option : Arm{option.direct, nullptr, "legacy"});
    dev.output.upload();
    const bool admitted=attention(rt, in, dev);
    drain(rt);
    auto out=dev.output.read(); dev.output.guards(out); dev.immutable();
    measured=measured && admitted;
    std::cout << "{\"type\":\"isolation_arm\",\"format\":" << in.format
              << ",\"arm\":" << i << ",\"admitted\":" << admitted
              << ",\"fallback_calls\":0,\"output_fnv64\":" << hash(out) << "}" << std::endl;
    if (!admitted) require(std::memcmp(out.data(), dev.output.initial.data(), dev.output.memory.size())==0,
                           "isolation declined output changed");
    if (!i) baseline=std::move(out);
    else same=std::memcmp(baseline.data(), out.data(), baseline.size()*sizeof(float))==0;
  }
  std::cout << "{\"type\":\"isolation\",\"format\":" << in.format << ",\"direct\":\"" << option.direct
            << "\",\"split\":\"" << (option.split ? option.split : "<unset>")
            << "\",\"route_observed\":false,\"bitwise\":" << same << ",\"status\":\""
            << (!measured ? "UNMEASURED_DECLINED" : same ? "PASS" : "FAIL") << "\"}" << std::endl;
  require(!measured || same, "off/invalid/other-mode output changed");
  return measured;
}
void host_contracts() {
  // Prove proposed ownership enumeration; this does not instrument GPU stores.
  for (size_t nq : {size_t(1), size_t(3), size_t(65535)}) {
    // For largest nq, enumerate heads/bands for first and last batches only.
    for (size_t batch : {size_t(0), nq-1}) {
      std::array<unsigned char, Heads*Dim> owners{};
      for (size_t kv=0; kv<KvHeads; ++kv) for (size_t tile=0; tile<2; ++tile) {
        const size_t group=(batch*KvHeads+kv)*2+tile;
        require(group%2==tile && (group/2)%KvHeads==kv && group/(2*KvHeads)==batch, "group decode");
        for (size_t r=0; r<(tile ? 4u : 8u); ++r)
          for (size_t sg=0; sg<4; ++sg) for (size_t j=0; j<4; ++j) for (size_t c=0; c<16; ++c)
            ++owners[(kv*12+tile*8+r)*Dim+(sg*4+j)*16+c];
      }
      require(std::all_of(owners.begin(), owners.end(), [](auto n){return n==1;}), "24-head coverage");
    }
  }
  const auto zero=oracle(Input(1, 1, 65, false, true));
  require(std::all_of(zero.begin(), zero.end(), [](double n){return n==0.;}), "all-missing oracle");
  Input high(1,3,65,false,false,false,LogicalCells);
  require(high.g.cap==262144 && high.ids[64]==262143 && high.ids[size_t(high.g.cap)+64]==262143,
          "high logical ID and per-query capacity stride");
  for (int width : {0,1,31,32,33,63,64,65,2051}) { Input in(1,3,width); (void)in; }
  // Existing early product gates only, before queue lookup.
  auto shapes=qsa_real_shapes(); void* invalid=reinterpret_cast<void*>(uintptr_t(1));
  require(qsa_prompt_attn_batch(nullptr,{},nullptr,nullptr,0,shapes,nullptr,0,invalid), "nq0 early return");
  for (int64_t cap : {int64_t(0),int64_t(-1),int64_t(INT32_MAX)+1})
    require(!qsa_prompt_attn_batch(nullptr,{},nullptr,nullptr,cap,shapes,nullptr,1,invalid), "capacity early refusal");
  std::cout << "{\"type\":\"host_contracts\",\"status\":\"PASS\",\"runtime_created\":false}" << std::endl;
}
// Common max must come from both tiles in turn, independently for each KV head.
void extreme_query(Input& in, int large_tile) {
  for (int64_t b=0; b<in.g.nq; ++b) for (int h=0; h<Heads; ++h) for (int d=0; d<Dim; ++d) {
    const bool large=(h%12 < 8 ? 0 : 1)==large_tile;
    const float mantissa=float(1.+.125*std::sin(double((b*Heads+h)*Dim+d)*.013));
    in.query[size_t((b*Heads+h)*Dim+d)]=std::ldexp(mantissa, large ? 20 : -20);
  }
}
} // namespace
int main(int argc, char** argv) {
  try {
    const std::string option=argc==2 ? argv[1] : "--validate";
    require(argc<=2 && (option=="--validate" || option=="--validate32" || option=="--host-only" || option=="--help"), "invalid CLI");
    if (option=="--help") {
      std::cout << "sycl_int8_prompt_head_split_probe [--host-only|--validate32|--validate|--help]\n"
                   "--validate32 excludes CH64; --validate adds CH64 tolerance gate. No benchmark/fallback.\n"; return 0;
    }
    std::cout << std::boolalpha << std::setprecision(17);
    host_contracts();
    if (option=="--host-only") return 0;
    require(setenv("STRATA_SYCL_PROMPT_NATIVE_KV_MIN","0",1)==0,"native independent option off");
    auto rt=sycl_backend::runtime_for();
    std::cout << "device=" << rt->device().get_info<sycl::info::device::name>()
              << " driver=" << rt->device().get_info<sycl::info::device::driver_version>() << std::endl;
    const bool include64=option=="--validate"; bool measured=true;
    for (int width : {0,1,31,32,33,63,64,65,2051}) {
      measured=validate(rt,Input(1,3,width),"sparse_tail",include64) && measured;
      measured=validate(rt,Input(1,1,width,true),"dense_tail",include64) && measured;
    }
    measured=validate(rt,Input(1,1,65,false,true),"all_missing",include64) && measured;
    measured=validate(rt,Input(1,3,65,true,false,true),"zero_scales",include64) && measured;
    measured=validate(rt,Input(1,3,65,false,false,false,LogicalCells),"cap262144_logical262143",include64) && measured;
    for (int tile : {0,1}) {
      Input in(1,3,65,true); extreme_query(in,tile);
      measured=validate(rt,in,tile ? "large_query_tile1" : "large_query_tile0",include64) && measured;
    }
    Input subq(1,3,65,true);
    for (size_t i=0;i<subq.query.size();++i)
      subq.query[i]=std::numeric_limits<float>::denorm_min()*float(int(i%15)-7);
    measured=validate(rt,subq,"subnormal_query",include64) && measured;
    Input subscale(1,3,65,true);
    for (size_t i=0;i<subscale.ks.size();++i) subscale.ks[i]=uint16_t(1+i%3);
    for (size_t i=0;i<subscale.vs.size();++i) subscale.vs[i]=uint16_t(1+i%3);
    measured=validate(rt,subscale,"fp16_subnormal_scales",include64) && measured;
    Input qzero(1,3,65,true); std::fill(qzero.query.begin(),qzero.query.end(),0.f);
    measured=validate(rt,qzero,"zero_query",include64) && measured;
    Input invalid(1,3,65);
    for (int width : {-1,66}) {
      for (int b=0;b<3;++b) invalid.steps[size_t(b)*kStepCount+kStepWidth]=width;
      measured=validate(rt,invalid,width<0 ? "negative_width" : "width_gt_cap",include64) && measured;
    }
    // Direct strings other than exact "1" and split values other than exact
    // "32"/"64" remain on the old dispatch. Preserve legacy Direct prefix rule.
    for (const char* split : {"", "0", "off", "032", "+32", "32x", "64 ", "64x", "128"})
      measured=isolation(rt,Input(1,1,65),Arm{"1",split,"legacy"}) && measured;
    for (const char* direct : {"0", "", "10", "1x"})
      for (const char* split : {"32","64"})
        measured=isolation(rt,Input(1,1,65),Arm{direct,split,"legacy"}) && measured;
    for (int mode : {0,2,3}) for (const char* split : {"32","64"})
      measured=isolation(rt,Input(mode,1,65),Arm{"1",split,"legacy_other_mode"}) && measured;
    drain(rt);
    std::cout << "{\"type\":\"completed\",\"all_measured\":" << measured
              << ",\"status\":\"" << (measured ? "PASS" : "UNMEASURED_DECLINED") << "\"}" << std::endl;
    return measured ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "FAIL int8_prompt_head_split_probe: " << error.what() << std::endl; return 1;
  }
}
'''
(A/'files/tools/sycl/int8_prompt_head_split_probe.cpp').write_text(prefix+body+tail)
