// Source-only CPU wrapper differential harness; root owns all execution.
// Reuse the existing legal blob/Acts generators and reference helpers verbatim.
// Renaming main includes utilities without invoking its parity or benchmark mode.
#define main strata_shared_iq_parity_main
#include "../../src/kernels/cpu/iq_avx2_parity.cpp"
#undef main

#include <array>
#include <cerrno>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace {
constexpr size_t kOutputBudget = 2 * 1024 * 1024;
constexpr double kReferenceTolerance = 1e-5;  // existing iq_avx2_parity rel bound
constexpr int kGuard = 8;
constexpr float kSentinel = -12345.25f;

// Private no-follow creation. Root supplies a fresh filename in an owned 0700
// directory. Successful/failed artifacts remain for root's retention review.
class Output {
    int fd_ = -1;
    size_t bytes_ = 0;
public:
    explicit Output(const std::string& path) {
        namespace fs = std::filesystem;
        const fs::path absolute = fs::absolute(path);
        if (path.empty() || absolute.filename().empty()) throw std::runtime_error("empty/root output path");
        int parent = ::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (parent < 0) throw std::runtime_error("open output root failed");
        try {
            for (const auto& component : absolute.parent_path().relative_path()) {
                const std::string part = component.string();
                if (part == "." || part == "..") throw std::runtime_error("output path must have no dot components");
                const int next = ::openat(parent, part.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
                if (next < 0) throw std::runtime_error("open output parent failed");
                ::close(parent); parent = next;
            }
            struct stat st{};
            if (::fstat(parent, &st) != 0 || st.st_uid != ::geteuid() || (st.st_mode & 0077))
                throw std::runtime_error("output parent must be owned/private");
            fd_ = ::openat(parent, absolute.filename().c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
            if (fd_ < 0) throw std::runtime_error("fresh output creation failed");
            if (::fstat(fd_, &st) != 0 || !S_ISREG(st.st_mode) || st.st_uid != ::geteuid() ||
                (st.st_mode & 0777) != 0600 || st.st_nlink != 1 || st.st_size != 0)
                throw std::runtime_error("output file must be owned 0600 regular single-link empty file");
        } catch (...) {
            ::close(parent);
            if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
            throw;
        }
        ::close(parent);
    }
    ~Output() { if (fd_ >= 0) ::close(fd_); }
    void append(const void* data, size_t n) {
        if (n > kOutputBudget - bytes_) throw std::runtime_error("output byte budget exceeded");
        const auto* p = static_cast<const uint8_t*>(data);
        while (n) {
            const ssize_t written = ::write(fd_, p, n);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) throw std::runtime_error("output write failed");
            p += written; n -= (size_t) written; bytes_ += (size_t) written;
        }
    }
    void word(uint32_t value) {
        const uint8_t b[] = {(uint8_t) value, (uint8_t) (value >> 8), (uint8_t) (value >> 16), (uint8_t) (value >> 24)};
        append(b, sizeof b);
    }
    void floats(const std::vector<float>& values) {
        std::array<uint8_t, 4096> buffer{};
        for (size_t i = 0; i < values.size();) {
            size_t n = std::min(values.size() - i, buffer.size() / 4);
            for (size_t j = 0; j < n; ++j) {
                uint32_t bits; std::memcpy(&bits, &values[i + j], 4);
                for (int k = 0; k < 4; ++k) buffer[4 * j + k] = (uint8_t) (bits >> (8 * k));
            }
            append(buffer.data(), 4 * n); i += n;
        }
    }
    size_t finish() {
        struct stat st{};
        if (::fstat(fd_, &st) != 0 || !S_ISREG(st.st_mode) || st.st_uid != ::geteuid() ||
            (st.st_mode & 0777) != 0600 || st.st_nlink != 1 || (size_t) st.st_size != bytes_)
            throw std::runtime_error("output identity/size changed");
        const int closing = fd_; fd_ = -1;
        if (::close(closing) != 0) throw std::runtime_error("output close failed");
        return bytes_;
    }
};

struct Guarded {
    int rows;
    std::vector<float> storage;
    float* ptr[kMaxT];
    explicit Guarded(int n) : rows(n), storage(kMaxT * (n + 2 * kGuard), kSentinel) {
        for (int t = 0; t < kMaxT; ++t) ptr[t] = storage.data() + t * (n + 2 * kGuard) + kGuard;
    }
    bool untouched(int nt, int r0, int r1) const {
        for (int t = 0; t < kMaxT; ++t)
            for (int r = -kGuard; r < rows + kGuard; ++r)
                if ((t >= nt || r < r0 || r >= r1) && ptr[t][r] != kSentinel) return false;
        return true;
    }
    std::vector<float> active(int nt) const {
        std::vector<float> values;
        values.reserve((size_t) nt * rows);
        for (int t = 0; t < nt; ++t) values.insert(values.end(), ptr[t], ptr[t] + rows);
        return values;
    }
};

bool finite(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); });
}

bool legal_q2(const Acts& a) {
    for (const auto& q : a.q2) {
        if (q.nchunks != kFF / cpu::QKA) return false;
        for (int i = 0; i < kFF; ++i) if (q.q[i] == -128) return false;
        for (int i = 0; i < q.nchunks; ++i)
            if (!std::isfinite(q.scale[i]) || q.scale[i] < 0 || !std::isfinite(q.hx[i])) return false;
    }
    return true;
}

// Independent ggml dots on the exact production-quantized activations. GU
// applies the established SwiGLU formula, without copying dispatch predicates.
std::vector<float> ggml_reference(const cpu::NativeFmt& f, const uint8_t* blob, const Acts& a, bool gu) {
    const int rows = gu ? (int) kFF : (int) kH;
    std::vector<float> result(kMaxT * rows);
    for (int t = 0; t < kMaxT; ++t)
        for (int r = 0; r < rows; ++r) {
            float g = 0, u = 0;
            if (gu) {
                traits_cpu(f.gu_type)->vec_dot((int) kH, &g, 0, blob + r * f.gu_row, 0, a.gup[t], 0, 1);
                traits_cpu(f.gu_type)->vec_dot((int) kH, &u, 0, blob + f.up_off + r * f.gu_row, 0, a.gup[t], 0, 1);
                result[t * rows + r] = (g / (1.f + std::exp(-g))) * u;
            } else {
                traits_cpu(20)->vec_dot((int) kFF, &g, 0, blob + f.down_off + r * f.d_row, 0, a.dnp[t], 0, 1);
                result[t * rows + r] = g;
            }
        }
    return result;
}

std::vector<float> q2_double_reference(const cpu::NativeFmt& f, const uint8_t* blob, const Acts& a) {
    std::vector<float> result(kMaxT * kH);
    const uint8_t* w = blob + f.down_off;
    for (int t = 0; t < kMaxT; ++t)
        for (int r = 0; r < kH; ++r) {
            double sum = 0;
            for (int b = 0; b < kFF / 64; ++b) {
                const uint8_t* block = w + r * f.d_row + b * 18;
                ggml_fp16_t scale; std::memcpy(&scale, block, 2);
                const double d = ggml_fp16_to_fp32(scale);
                for (int i = 0; i < 64; ++i) {
                    const int k = b * 64 + i;
                    const int code = (block[2 + i / 4] >> (2 * (i % 4))) & 3;
                    sum += d * (code - 1) * a.q2[t].scale[k / cpu::QKA] * a.q2[t].q[k];
                }
            }
            result[t * kH + r] = (float) sum;
        }
    return result;
}

int wrapper_cases(Output& output, int type, bool gu) {
    cpu::NativeFmt f;
    std::string error;
    if (!cpu::native_fmt(gu ? type : 18, gu ? 20 : type, kH, kFF, f, error))
        throw std::runtime_error(error);
    std::vector<uint8_t> blob(f.bytes);
    uint64_t seed = 0x776170700000ull + (uint64_t) type + (gu ? 256 : 0);
    fill_blob(blob.data(), f, seed);
    const Acts a(f, 123u + (unsigned) type + (gu ? 256u : 0u));
    if (!legal_q2(a)) throw std::runtime_error("production ActQ violates finite-input legal range");
    const int rows = gu ? (int) kFF : (int) kH;
    const auto reference = !gu && type == 42 ? q2_double_reference(f, blob.data(), a) :
                                             ggml_reference(f, blob.data(), a, gu);
    auto run = [&](int nt, float* const* out, int r0, int r1) {
        if (gu) cpu::native_gu_rows(f, blob.data(), a.gup, nt, out, r0, r1);
        else if (type == 42) cpu::q2_rows_any(blob.data() + f.down_off, f.d_row, (int) kFF / 64,
                                          a.q2p, nt, out, r0, r1);
        else cpu::native_down_rows(f, blob.data(), a.dnp, nt, out, r0, r1);
    };
    int failures = 0;
    for (int nt = 1; nt <= kMaxT; ++nt) {
        Guarded full(rows), partition(rows), range(rows), direct(rows);
        run(nt, full.ptr, 0, rows);
        const int split = rows / 2 + 3;
        run(nt, partition.ptr, 0, 1);
        run(nt, partition.ptr, 1, split);
        run(nt, partition.ptr, split, rows);
        run(nt, range.ptr, 7, rows - 9);
        run(nt, range.ptr, split, split);  // empty interval must change nothing
        if (gu) cpu::iq256_gu_rows_v(0, type, blob.data(), f.gu_row, f.up_off, (int) kH,
                                    a.gup, nt, direct.ptr, 0, rows);
        else if (type == 42) cpu::q2_0_gguf_rows_multi_avx2_v(false, blob.data() + f.down_off,
                                    f.d_row, (int) kFF / 64, a.q2p, nt, direct.ptr, 0, rows);
        else cpu::iq4nl256_down_rows_v(0, blob.data() + f.down_off, f.d_row, (int) kFF,
                                      a.dnp, nt, direct.ptr, 0, rows);
        const auto values = full.active(nt), parts = partition.active(nt), direct_values = direct.active(nt);
        const bool all_finite = finite(values) && finite(parts) && finite(direct_values) && finite(reference);
        const bool padding = full.untouched(nt, 0, rows) && partition.untouched(nt, 0, rows) &&
                             range.untouched(nt, 7, rows - 9) && direct.untouched(nt, 0, rows);
        size_t partition_differ = rows_differ(values.data(), parts.data(), values.size());
        size_t range_differ = 0;
        for (int t = 0; t < nt; ++t)
            range_differ += rows_differ(full.ptr[t] + 7, range.ptr[t] + 7, rows - 16);
        const double rr = rel(values.data(), reference.data(), values.size());
        const double rd = rel(values.data(), direct_values.data(), values.size());
        const bool refs = all_finite && rr <= kReferenceTolerance && rd <= kReferenceTolerance;
        const bool pass = all_finite && padding && refs && !partition_differ && !range_differ;
        failures += !pass;
        std::printf("case phase=%s type=%d nt=%d rows=%d finite=%d legal_actq=1 padding=%d "
                    "partition_differ=%zu range_differ=%zu reference_rel=%.9e direct_rel=%.9e "
                    "reference_gate=%d passed=%d\n", gu ? "gu" : "down", type, nt, rows,
                    all_finite, padding, partition_differ, range_differ, rr, rd, refs, pass);
        // Canonical little-endian record: phase, type, nt, rows, word count,
        // reserved zero; then every active output word in token/row order.
        for (uint32_t word : {gu ? 1u : 2u, (uint32_t) type, (uint32_t) nt, (uint32_t) rows,
                              (uint32_t) values.size(), 0u}) output.word(word);
        output.floats(values);
    }
    return failures;
}
}  // namespace

int main(int argc, char** argv) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc != 3 || std::string(argv[1]) != "--out") {
        std::fprintf(stderr, "usage: native_wrapper_parity --out /private/fresh-output.bin\n");
        return 2;
    }
    try {
        ggml_cpu_init();
        std::printf("native_wrapper_parity cpu=%s avx2=%d avx512=%d avxvnni=%d H=%lld FF=%lld "
                    "max_nt=8 tolerance=1e-5\n", cpu::cpu_name().c_str(), cpu::cpu_avx2_ok(),
                    cpu::cpu_avx512_ok(), cpu::cpu_avxvnni_ok(), (long long) kH, (long long) kFF);
        for (const char* key : {"STRATA_IQ_MT_MIN", "STRATA_NO_IQ256", "STRATA_NO_IQ512",
                "STRATA_NO_IQ4NL", "STRATA_IQ256_GATHER", "STRATA_IQ3S_MT1", "STRATA_NO_AVXVNNI",
                "STRATA_KQ256", "STRATA_FORCE_ISA", "STRATA_Q2_BITPLANE", "STRATA_NO_Q8K_AVX2",
                "STRATA_NATIVE_DISPATCH_HISTOGRAM"}) {
            const char* value = std::getenv(key);
            std::printf("env %s=%s\n", key, value ? value : "<unset>");
        }
        if (!cpu::cpu_avx2_ok()) throw std::runtime_error("AVX2 required; no fake CPU feature execution");
        std::printf("effective gather_fast=%d gather_setting=%d iq256_variant=%d gu18_min=%d gu21_min=%d\n",
                    cpu::cpu_gather_fast(), cpu::iq256_gather_setting(), cpu::iq256_variant(),
                    cpu::native_gu_mt_min(18), cpu::native_gu_mt_min(21));
        Output output(argv[2]);
        for (uint32_t word : {0x53575031u, 1u, 48u, (uint32_t) kH, (uint32_t) kFF, 8u, 0u, 0u}) output.word(word);
        int failures = 0;
        for (int type : {18, 21, 22, 23}) failures += wrapper_cases(output, type, true);
        for (int type : {20, 42}) failures += wrapper_cases(output, type, false);
        const size_t bytes = output.finish();
        if (bytes != 1107104) throw std::runtime_error("complete differential output length mismatch");
        std::printf("summary cases=48 failures=%d bytes=%zu local_gates_passed=%d "
                    "T_H_bitwise_compared=0 full_model_shape_qualified=0 performance_eligible=0 adopted=0\n",
                    failures, bytes, failures == 0);
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "native_wrapper_parity failure: %s\n", error.what());
        return 1;
    }
}
