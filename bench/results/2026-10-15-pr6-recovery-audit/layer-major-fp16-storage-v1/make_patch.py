#!/usr/bin/env python3
"""Generate a non-applied candidate against current root bytes. No builds/tests."""
from pathlib import Path
import difflib
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
changes = {}
sources = {}

def read(name):
    raw = (ROOT / name).read_bytes()
    sources[name] = hashlib.sha256(raw).hexdigest()
    content = raw.decode("utf-8")
    changes[name] = [content, content]
    return content

def replace(s, before, after):
    if s.count(before) != 1:
        raise ValueError(f"expected one replacement, found {s.count(before)}: {before[:100]}")
    return s.replace(before, after, 1)

name = "include/strata/prefill/layer_major_plan.hpp"
s = read(name)
s = replace(s, "struct LayerMajorSpan {", """// Storage format only: the live residual stays FP32; operator precision is unchanged.
enum class LayerMajorResidualDtype { fp32, fp16 };

inline LayerMajorResidualDtype layer_major_residual_dtype(const char* value) {
    if (!value || std::strcmp(value, "fp32") == 0) return LayerMajorResidualDtype::fp32;
    if (std::strcmp(value, "fp16") == 0) return LayerMajorResidualDtype::fp16;
    throw std::invalid_argument("STRATA_PREFILL_LAYER_MAJOR_R_DTYPE must be fp32 or fp16");
}

inline uint64_t layer_major_storage_bytes(LayerMajorResidualDtype dtype) {
    switch (dtype) {
    case LayerMajorResidualDtype::fp32: return 4;
    case LayerMajorResidualDtype::fp16: return 2;
    }
    throw std::invalid_argument("invalid layer-major residual dtype");
}

inline const char* layer_major_dtype_name(LayerMajorResidualDtype dtype) {
    switch (dtype) {
    case LayerMajorResidualDtype::fp32: return "fp32";
    case LayerMajorResidualDtype::fp16: return "fp16";
    }
    throw std::invalid_argument("invalid layer-major residual dtype");
}

struct LayerMajorSpan {""")
start = s.index("struct LayerMajorResidualPlan {")
end = s.index("    LayerMajorSpan span(", start)
s = s[:start] + """struct LayerMajorResidualPlan {
    int64_t rows = 0, chunk = 0, gpu_rows = 0;
    LayerMajorResidualDtype dtype = LayerMajorResidualDtype::fp32;
    uint64_t row_bytes = 0, total_bytes = 0, gpu_bytes = 0, host_bytes = 0;
    uint64_t staging_bytes = 0;

    // gpu_budget includes the optional host-tail conversion staging allocation.
    // A fully device-resident store needs no staging. Otherwise reserve one
    // FP16 chunk before choosing an automatic whole-chunk device prefix.
    static LayerMajorResidualPlan make(int64_t n, int64_t c, int64_t width,
                                       uint64_t gpu_budget, int64_t requested,
                                       LayerMajorResidualDtype dtype = LayerMajorResidualDtype::fp32) {
        if (n <= 0 || c <= 0 || width <= 0 || requested < -1 || requested > n)
            throw std::invalid_argument("invalid layer-major residual geometry or requested GPU rows");
        LayerMajorResidualPlan p;
        p.rows = n; p.chunk = c; p.dtype = dtype;
        p.row_bytes = layer_major_mul(uint64_t(width), layer_major_storage_bytes(dtype));
        p.total_bytes = layer_major_mul(uint64_t(n), p.row_bytes);
        if (requested < 0) {
            if (p.total_bytes <= gpu_budget) {
                p.gpu_rows = n;
            } else {
                const uint64_t reserve = dtype == LayerMajorResidualDtype::fp16
                    ? layer_major_mul(uint64_t(std::min(c, n)), p.row_bytes) : 0;
                if (reserve > gpu_budget)
                    throw std::invalid_argument("layer-major FP16 staging exceeds free VRAM with headroom");
                const int64_t want = int64_t(std::min<uint64_t>(uint64_t(n),
                    (gpu_budget - reserve) / p.row_bytes));
                p.gpu_rows = (want / c) * c;
            }
        } else {
            p.gpu_rows = requested == n ? n : (requested / c) * c;
        }
        p.gpu_bytes = layer_major_mul(uint64_t(p.gpu_rows), p.row_bytes);
        p.host_bytes = p.total_bytes - p.gpu_bytes;
        if (dtype == LayerMajorResidualDtype::fp16 && p.host_bytes)
            p.staging_bytes = layer_major_mul(uint64_t(std::min(c, n - p.gpu_rows)), p.row_bytes);
        if (layer_major_add(p.gpu_bytes, p.staging_bytes) > gpu_budget)
            throw std::invalid_argument("requested layer-major GPU residual prefix plus staging exceeds free VRAM with headroom");
        return p;
    }

""" + s[end:]
changes[name][1] = s

name = "src/prefill/prefill.cpp"
s = read(name)
s = replace(s, "    const float* layer_major_gpu_input = nullptr;", """    const void* layer_major_gpu_input = nullptr;
    const void* layer_major_host_input = nullptr;
    uint16_t* layer_major_fp16_stage = nullptr;""")
s = replace(s, "        const bool residual_input = hand_in_ != nullptr || m.layer_major_gpu_input != nullptr;", """        const bool residual_input = m.layer_major_active
            ? m.layer_major_host_input != nullptr || m.layer_major_gpu_input != nullptr
            : hand_in_ != nullptr;""")
s = replace(s, """                const auto* base = reinterpret_cast<const uint8_t*>(span.device ? m.layer_major_gpu_input : hand_in_);
                if (!base || cudaMemcpyAsync(m.R, base + size_t(span.offset_bytes), size_t(span.bytes),
                    span.device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, m.cs) != cudaSuccess) {
                    err = "prefill layer-major: residual input copy failed"; return false;
                }""", """                const auto* base = static_cast<const uint8_t*>(span.device
                    ? m.layer_major_gpu_input : m.layer_major_host_input);
                if (!base) { err = "prefill layer-major: missing residual input store"; return false; }
#if defined(STRATA_ENABLE_SYCL)
                if (m.layer_major_residual.dtype == detail::LayerMajorResidualDtype::fp16) {
                    const uint16_t* packed = reinterpret_cast<const uint16_t*>(base + size_t(span.offset_bytes));
                    if (!span.device) {
                        if (!m.layer_major_fp16_stage || span.bytes > m.layer_major_residual.staging_bytes ||
                            cudaMemcpyAsync(m.layer_major_fp16_stage, packed, size_t(span.bytes),
                                            cudaMemcpyHostToDevice, m.cs) != cudaSuccess) {
                            err = "prefill layer-major: FP16 residual staging upload failed"; return false;
                        }
                        packed = m.layer_major_fp16_stage;
                    }
                    // Same in-order queue as H2D and the unchanged FP32 model.
                    // Exactly one restoration, with no intermediate F32 store.
                    from_f16(packed, m.R, int64_t(span.bytes / 2), m.cs);
                } else
#endif
                if (cudaMemcpyAsync(m.R, base + size_t(span.offset_bytes), size_t(span.bytes),
                    span.device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, m.cs) != cudaSuccess) {
                    err = "prefill layer-major: residual input copy failed"; return false;
                }""")
changes[name][1] = s

name = "src/prefill/layer_major.inc"
s = read(name)
s = replace(s, "    const char* mmq = std::getenv(\"STRATA_PREFILL_MMQ\");", """    const auto residual_dtype = detail::layer_major_residual_dtype(
        std::getenv("STRATA_PREFILL_LAYER_MAJOR_R_DTYPE"));
    const char* mmq = std::getenv("STRATA_PREFILL_MMQ");""")
s = replace(s, "        detail::layer_major_requested_rows(std::getenv(\"STRATA_PREFILL_LAYER_MAJOR_R_GPU\")));", "        detail::layer_major_requested_rows(std::getenv(\"STRATA_PREFILL_LAYER_MAJOR_R_GPU\")), residual_dtype);")
s = replace(s, "    std::unique_ptr<float[]> rows(plan.host_bytes ? new float[size_t(plan.host_bytes / sizeof(float))] : nullptr);", "    std::unique_ptr<uint8_t[]> rows(plan.host_bytes ? new uint8_t[size_t(plan.host_bytes)] : nullptr);")
s = replace(s, """    auto free_device = [](float* p) { if (p) (void) cudaFree(p); };
    std::unique_ptr<float, decltype(free_device)> device_rows(nullptr, free_device);""", """    auto free_device = [](uint8_t* p) { if (p) (void) cudaFree(p); };
    std::unique_ptr<uint8_t, decltype(free_device)> device_rows(nullptr, free_device);
    // Allocated only for an FP16 host tail. One in-order queue serializes
    // H2D -> unpack -> model -> pack -> D2H, and the outer lease owns it all.
    std::unique_ptr<uint8_t, decltype(free_device)> fp16_stage(nullptr, free_device);""")
s = replace(s, "        bool active; const float* gpu_input; detail::LayerMajorResidualPlan plan;", """        bool active; const void* gpu_input; const void* host_input; uint16_t* fp16_stage;
        detail::LayerMajorResidualPlan plan;""")
s = replace(s, "            m.layer_major_gpu_input = gpu_input; m.layer_major_residual = plan;", """            m.layer_major_gpu_input = gpu_input; m.layer_major_host_input = host_input;
            m.layer_major_fp16_stage = fp16_stage; m.layer_major_residual = plan;""")
s = replace(s, "              m.layer_major_active, m.layer_major_gpu_input, m.layer_major_residual, on_chunk, on_stage_chunk};", """              m.layer_major_active, m.layer_major_gpu_input, m.layer_major_host_input,
              m.layer_major_fp16_stage, m.layer_major_residual, on_chunk, on_stage_chunk};""")
s = replace(s, "        float* p = nullptr;", "        uint8_t* p = nullptr;")
s = replace(s, "    m.layer_major_active = true;", """    if (plan.staging_bytes) {
        uint8_t* p = nullptr;
        if (cudaMalloc(reinterpret_cast<void**>(&p), size_t(plan.staging_bytes)) != cudaSuccess) {
            err = "prefill layer-major: FP16 residual staging allocation failed"; return false;
        }
        fp16_stage.reset(p);
    }
    m.layer_major_fp16_stage = reinterpret_cast<uint16_t*>(fp16_stage.get());
    m.layer_major_active = true;""")
s = replace(s, """    std::fprintf(stderr, "strata prefill: layer-major candidate, tokens=%lld chunk=%lld residual_total=%llu "
                         "residual_gpu_rows=%lld residual_gpu=%llu residual_host=%llu layer_cache=%llu; "
                         "extra allocations, GPU headroom=536870912, RAM headroom=8589934592\\n",
                 (long long)n, (long long)m.T, (unsigned long long)plan.total_bytes, (long long)plan.gpu_rows,
                 (unsigned long long)plan.gpu_bytes, (unsigned long long)plan.host_bytes,
                 (unsigned long long)cache_bytes);""", """    std::fprintf(stderr, "strata prefill: layer-major candidate, tokens=%lld chunk=%lld residual_dtype=%s "
                         "residual_element_bytes=%llu residual_total=%llu residual_gpu_rows=%lld "
                         "residual_gpu=%llu residual_host=%llu residual_staging=%llu layer_cache=%llu; "
                         "extra allocations, GPU headroom=536870912, RAM headroom=8589934592\\n",
                 (long long)n, (long long)m.T, detail::layer_major_dtype_name(plan.dtype),
                 (unsigned long long)detail::layer_major_storage_bytes(plan.dtype),
                 (unsigned long long)plan.total_bytes, (long long)plan.gpu_rows,
                 (unsigned long long)plan.gpu_bytes, (unsigned long long)plan.host_bytes,
                 (unsigned long long)plan.staging_bytes, (unsigned long long)cache_bytes);
    if (plan.dtype == detail::LayerMajorResidualDtype::fp16)
        std::fprintf(stderr, "strata prefill: FP16 residual storage is experimental/unqualified; "
                             "RNE, finite overflow -> signed Inf, FP16 subnormals preserved, "
                             "underflow -> signed zero, NaN -> quiet NaN; "
                             "one pack/unpack per interlayer boundary, live residual FP32, "
                             "operator/state precision unchanged\\n");""")
s = replace(s, "        hand_in_ = layer == 0 ? nullptr : rows.get();", "        m.layer_major_host_input = layer == 0 ? nullptr : rows.get();")
s = replace(s, """            uint8_t* destination = span.device ? reinterpret_cast<uint8_t*>(device_rows.get())
                                             : reinterpret_cast<uint8_t*>(rows.get());
            const auto status = cudaMemcpyAsync(destination + size_t(span.offset_bytes), residual,
                size_t(span.bytes), span.device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost, m.cs);
            if (status != cudaSuccess || !drain()) {""", """            uint8_t* destination = (span.device ? device_rows.get() : rows.get()) + size_t(span.offset_bytes);
            cudaError_t status = cudaSuccess;
            if (plan.dtype == detail::LayerMajorResidualDtype::fp16) {
                uint16_t* packed = span.device ? reinterpret_cast<uint16_t*>(destination)
                                              : m.layer_major_fp16_stage;
                if (!packed || (!span.device && span.bytes > plan.staging_bytes)) {
                    error = "prefill layer-major: FP16 residual staging bounds exceeded"; return false;
                }
                // to_f16 uses shared RNE bits, without hsat/clamping. Neither
                // placement rounds twice; final-layer callbacks stay FP32.
                to_f16(residual, packed, int64_t(span.bytes / 2), m.cs);
                if (!span.device)
                    status = cudaMemcpyAsync(destination, packed, size_t(span.bytes), cudaMemcpyDeviceToHost, m.cs);
            } else {
                status = cudaMemcpyAsync(destination, residual, size_t(span.bytes),
                    span.device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost, m.cs);
            }
            if (status != cudaSuccess || !drain()) {""")
changes[name][1] = s

name = "include/strata/prefill/kernels.hpp"
s = read(name)
s = replace(s, "void to_f16(const float* x, uint16_t* y, int64_t n, void* stream);", """void to_f16(const float* x, uint16_t* y, int64_t n, void* stream);
#if defined(STRATA_ENABLE_SYCL)
/// Exact FP16 bits -> FP32, including subnormals/Inf/NaN. Layer-major storage restore only.
void from_f16(const uint16_t* x, float* y, int64_t n, void* stream);
#endif""")
changes[name][1] = s

name = "src/prefill/sycl/postops.cpp"
s = read(name)
s = replace(s, "void to_bf16(const float *x, uint16_t *y, int64_t n, void *stream,", """void from_f16(const uint16_t *x, float *y, int64_t n, void *stream) {
  auto z = count(n, 1);
  validate_spans({{y, z * 4}}, {{x, z * 2}});
  for_each(n, stream,
           [=](size_t i) { y[i] = strata::kernels::f32_from_f16(x[i]); });
}
void to_bf16(const float *x, uint16_t *y, int64_t n, void *stream,""")
changes[name][1] = s

name = "src/prefill/layer_major_plan_test.cpp"
s = read(name)
s = replace(s, "    LayerMajorChunkCursor cursor{19, 4};", """    check(layer_major_residual_dtype(nullptr) == LayerMajorResidualDtype::fp32 &&
          layer_major_residual_dtype("fp32") == LayerMajorResidualDtype::fp32 &&
          layer_major_residual_dtype("fp16") == LayerMajorResidualDtype::fp16,
          "dtype defaults to FP32 and accepts only explicit FP32/FP16");
    for (const char* value : {"", "FP16", "bf16", "fp8", " fp16", "fp16 "})
        rejected([&] { layer_major_residual_dtype(value); }, "invalid storage dtype rejected");
    rejected([&] { layer_major_storage_bytes(static_cast<LayerMajorResidualDtype>(99)); },
             "invalid typed format rejected");
    for (const auto dtype : {LayerMajorResidualDtype::fp32, LayerMajorResidualDtype::fp16}) {
        const bool half = dtype == LayerMajorResidualDtype::fp16;
        const uint64_t rb = 10240 * layer_major_storage_bytes(dtype);
        const uint64_t reserve = half ? 8192 * rb : 0;
        const uint64_t full_bytes = half ? 5368709120ull : 10737418240ull;
        const auto host = LayerMajorResidualPlan::make(262144, 8192, 10240, reserve, 0, dtype);
        check(host.total_bytes == full_bytes && host.host_bytes == full_bytes &&
              host.staging_bytes == reserve, "262144 physical rows: 10/5 GiB plus exact staging");
        const auto exact = LayerMajorResidualPlan::make(262144, 8192, 10240, full_bytes, -1, dtype);
        check(exact.gpu_rows == 262144 && !exact.host_bytes && !exact.staging_bytes,
              "exact full-store budget needs no host staging");
        const auto below = LayerMajorResidualPlan::make(262144, 8192, 10240, full_bytes - 1, -1, dtype);
        check(below.gpu_rows == (half ? 245760 : 253952) &&
              below.gpu_bytes + below.staging_bytes <= full_bytes - 1,
              "one byte below physical boundary reserves staging before auto prefix");
        const auto prefix = LayerMajorResidualPlan::make(262144, 8192, 10240,
            2 * 8192 * rb + reserve, -1, dtype);
        check(prefix.gpu_rows == 16384 && prefix.span(16384, 8192).offset_bytes == 0 &&
              prefix.span(253952, 8192).bytes == 8192 * rb,
              "physical boundary mixed store has exact first/last host spans");
        const uint64_t odd_stage = half ? 257 * rb : 0;
        const auto odd = LayerMajorResidualPlan::make(513, 257, 10240,
            257 * rb + odd_stage, 257, dtype);
        check(odd.gpu_rows == 257 && odd.host_bytes == 256 * rb &&
              odd.span(257, 256).offset_bytes == 0 && odd.span(257, 256).bytes == 256 * rb &&
              odd.staging_bytes == (half ? 256 * rb : 0), "257/256 split uses exact partial-tail staging");
        const auto main_fixture = LayerMajorResidualPlan::make(512, 257, 10240,
            512 * rb, 257, dtype);
        check(main_fixture.gpu_rows == 257 && main_fixture.host_bytes == 255 * rb &&
              main_fixture.span(257, 255).offset_bytes == 0 &&
              main_fixture.staging_bytes == (half ? 255 * rb : 0),
              "Main's 512 batched rows use exact 257/255 prefix/tail spans");
        const auto final_gpu = LayerMajorResidualPlan::make(19, 4, 10240, 19 * rb, 19, dtype);
        check(final_gpu.span(16, 3).device && final_gpu.span(16, 3).bytes == 3 * rb &&
              !final_gpu.host_bytes && !final_gpu.staging_bytes,
              "explicit all-GPU odd tail has no staging in either format");
        const uint64_t odd_budget = 4 * rb + (half ? 4 * rb : 0);
        const auto tail = LayerMajorResidualPlan::make(19, 4, 10240, odd_budget, 4, dtype);
        check(!tail.span(16, 3).device && tail.span(16, 3).offset_bytes == 12 * rb &&
              tail.span(16, 3).bytes == 3 * rb, "odd chunk count preserves host-relative final offset");
        rejected([&] { LayerMajorResidualPlan::make(19, 4, 10240, odd_budget - 1, 4, dtype); },
                 "explicit prefix budget includes full staging extent");
        rejected([&] { LayerMajorResidualPlan::make(INT64_MAX, 4, 10240, 0, 0, dtype); },
                 "total multiplication checked for either format");
    }
    rejected([&] { LayerMajorResidualPlan::make(19, 4, 10240, 4 * 10240 * 2 - 1, -1,
                                              LayerMajorResidualDtype::fp16); },
             "host-only FP16 still needs one complete conversion staging chunk");
    const auto partial = LayerMajorResidualPlan::make(19, 4, 10240, 19 * 10240 * 2, 16,
                                                     LayerMajorResidualDtype::fp16);
    check(partial.gpu_rows == 16 && partial.staging_bytes == 3 * 10240 * 2,
          "explicit final partial host chunk fits prefix plus exact small staging");

    LayerMajorChunkCursor cursor{19, 4};""")
changes[name][1] = s

# Existing test target/resolver is reused, so no CMake/extra GPU target is needed.
name = "tests/sycl/prefill_postops.cpp"
s = read(name)
s = replace(s, "int main() {", """// Conversion oracle is the host's IEEE binary16 type, independent of the
// shared bit converter used by the device implementation. This checks storage
// math only, never FP16-vs-FP32 whole-model bit identity or model quality.
void residual_storage_conversion(void* stream) {
  std::vector<uint16_t> codes(65536);
  for (size_t i = 0; i < codes.size(); ++i) codes[i] = uint16_t(i);
  Buffer<uint16_t> packed(codes.size());
  Buffer<float> restored(codes.size());
  packed.set(codes);
  p::from_f16(packed.data(), restored.data(), codes.size(), stream);
  const auto decoded = restored.get();
  for (size_t i = 0; i < codes.size(); ++i) {
    const float expected = float(std::bit_cast<_Float16>(codes[i]));
    need(std::isnan(expected) ? std::isnan(decoded[i])
         : std::bit_cast<uint32_t>(expected) == std::bit_cast<uint32_t>(decoded[i]),
         "exhaustive FP16 decode including signed zeros/subnormals/Inf/NaN");
  }
  std::vector<float> values = {0.f, -0.f, 1.f, -1.f, 65504.f, -65504.f,
      65519.f, 65520.f, -65520.f, 65536.f, -65536.f, 1e30f, -1e30f,
      0x1p-24f, 0x1p-25f, 0x1.8p-24f, 0x1p-14f,
      1.f + 0x1p-11f, 1.f + 3 * 0x1p-11f,
      std::bit_cast<float>(0x7f800000u), std::bit_cast<float>(0xff800000u),
      std::bit_cast<float>(0x7fc12345u)};
  // Odd conversion count and ordinary residual-like values in addition to ties.
  for (int i = 0; i < 491; ++i) values.push_back(std::sin(float(i) * .17f) * 17.f);
  Buffer<float> input(values.size()), output(values.size());
  Buffer<uint16_t> storage(values.size()), stage(values.size());
  input.set(values);
  p::to_f16(input.data(), storage.data(), values.size(), stream);
  p::from_f16(storage.data(), output.data(), values.size(), stream);
  const auto saved = storage.get();
  const auto direct = output.get();
  // Model the host-tail DMA sequence with owners live through checked waits.
  std::vector<uint16_t> host(values.size());
  rt->wait(rt->compute().memcpy(host.data(), storage.data(), host.size() * 2));
  rt->wait(rt->compute().memcpy(stage.data(), host.data(), host.size() * 2));
  p::from_f16(stage.data(), output.data(), values.size(), stream);
  const auto tail = output.get();
  bool saw_rounding = false, saw_overflow = false;
  for (size_t i = 0; i < values.size(); ++i) {
    const auto half = _Float16(values[i]);
    const auto code = std::bit_cast<uint16_t>(half);
    const float expected = float(half);
    if (std::isnan(expected)) {
      need(std::isnan(direct[i]) && std::isnan(tail[i]) &&
           (saved[i] & 0x7c00u) == 0x7c00u && (saved[i] & 0x03ffu), "NaN remains NaN");
    } else {
      need(saved[i] == code && std::bit_cast<uint32_t>(direct[i]) == std::bit_cast<uint32_t>(expected) &&
           std::bit_cast<uint32_t>(tail[i]) == std::bit_cast<uint32_t>(expected),
           "direct and staged storage match independent FP16 round-trip oracle");
      saw_rounding |= std::isfinite(expected) && expected != values[i];
      saw_overflow |= std::isfinite(values[i]) && std::isinf(expected);
    }
  }
  need(saw_rounding && saw_overflow, "FP16 storage changes values and exposes overflow");
}
int main() {""")
s = replace(s, "    auto *s = &rt->compute();", "    auto *s = &rt->compute();\n    residual_storage_conversion(s);")
changes[name][1] = s

base_path = HERE / "source-base-sha256.json"
if base_path.exists() and json.loads(base_path.read_text()) != sources:
    raise RuntimeError("product base changed: review/rebase explicitly; do not overwrite current root")
base_path.write_text(json.dumps(sources, indent=2) + "\n")
dependencies = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in ("cmake/sycl_backend.cmake", "CMakeLists.txt",
                             "include/strata/kernels/f16_bits.hpp", "include/strata/sycl/launch.hpp")}
dependency_path = HERE / "dependency-base-sha256.json"
if dependency_path.exists() and json.loads(dependency_path.read_text()) != dependencies:
    raise RuntimeError("read-only source dependency changed: review explicitly")
dependency_path.write_text(json.dumps(dependencies, indent=2) + "\n")
patch = []
for name, (before, after) in changes.items():
    target = HERE / "files" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(after)
    patch.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                    fromfile="a/" + name, tofile="b/" + name))
(HERE / "source.diff").write_text("".join(patch))
artifacts = {}
for path in sorted((HERE / "files").rglob("*")):
    if path.is_file(): artifacts[str(path.relative_to(HERE))] = hashlib.sha256(path.read_bytes()).hexdigest()
for filename in ("source.diff", "source-base-sha256.json", "dependency-base-sha256.json", "make_patch.py", "implementation.txt"):
    if not (HERE / filename).exists(): continue
    artifacts[filename] = hashlib.sha256((HERE / filename).read_bytes()).hexdigest()
(HERE / "artifact-sha256.json").write_text(json.dumps(artifacts, indent=2) + "\n")
print(json.dumps({"modified_existing_files": len(changes), "patch_sha256": artifacts["source.diff"],
                  "product_writes": 0, "builds_tests_gpu": "not run"}))
