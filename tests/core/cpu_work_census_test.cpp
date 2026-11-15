#include "strata/core/cpu_work_census.hpp"

#include <climits>
#include <cstdlib>
#include <string>

namespace {
using Census = strata::core::CpuWorkCensus;
void require(bool value) { if (!value) std::abort(); }
int cases = 0;
template<class F> void test(const char* name, F&& f) {
    f(); ++cases; std::printf("cpu_work_census case=%s status=PASS\n", name);
}
struct Job { int nt; };
struct Fixture {
    Census::Format formats[2]{};
    int32_t slots[8] = {-1, 0, -1, -1, -1, 1, -1, -1};
    int32_t ids[8] = {0, 1, 0, 2, 0, 3, 2, 1};
    int32_t kind[8] = {-1, 0, -1, -1, -1, -1, -1, 0};
    int16_t job_of[4] = {0, -1, 1, 2};
    Job jobs[3] = {{3}, {2}, {1}};
    Fixture() {
        for (auto& f : formats) {
            f.gu_type = 18; f.d_type = 20; f.gu_act = 15; f.d_act = 8;
            f.n_embd = 2560; f.n_ff = 640; f.bytes = 1510400;
        }
    }
    void record(Census& c, int layer) {
        c.record(layer, 4, 2, 4, slots, formats[layer], ids, kind, job_of, jobs, 3, 100);
    }
    void window(Census& c) {
        c.begin_window(c.windows() + 1, (int64_t)c.windows() + 10, 4);
        record(c, 0); record(c, 1); c.end_window();
    }
};
struct Output {
    FILE* file = std::tmpfile();
    Output() { require(file != nullptr); }
    ~Output() { std::fclose(file); }
    std::string text() {
        std::rewind(file); std::string text; char buf[1024]; size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, file))) text.append(buf, n);
        return text;
    }
};
}

int main() {
    test("actual_job_nt_not_window_nt", [] {
        Fixture f; Census c(1, 2, 4, f.slots, f.formats); f.window(c);
        require(c.ok() && c.windows() == 1 && c.cpu_jobs() == 6 && c.cpu_entries() == 12);
        require(c.bins_used() == 1 && c.bins()[0].jobs[0] == 2 && c.bins()[0].jobs[1] == 2 &&
            c.bins()[0].jobs[2] == 2 && c.bins()[0].jobs[3] == 0 && c.bins()[0].pool_ns == 200);
        Output out; c.write_begin(out.file, 33214); c.finish(out.file, 1, 4, 6, 12, 4, 0);
        require(c.ok() && c.closed());
        const auto text = out.text();
        require(text.find("phase=decode valid=1 windows=1 rows=4 callbacks=2 k=2 cpu_jobs=6") != std::string::npos);
        require(text.find("resident_bits=0000000000000002") != std::string::npos);
        require(text.find("jobs_NT=2,2,2,0,0,0,0,0") != std::string::npos);
    });
    test("multiple_windows_and_format_pairs", [] {
        Fixture f; f.formats[1].d_type = 42; Census c(4, 2, 4, f.slots, f.formats);
        f.window(c); f.window(c);
        Output out; c.finish(out.file, 2, 8, 12, 24, 8, 0);
        require(c.ok() && c.bins_used() == 2 && c.bins()[0].calls == 2 && c.bins()[1].calls == 2);
    });
    test("zero_CPU_window_is_explicit", [] {
        Fixture f; const int32_t ids[] = {1}, kind[] = {0};
        Census c(2, 2, 4, f.slots, f.formats); c.begin_window(1, 0, 1);
        for (int l = 0; l < 2; ++l)
            c.record(l, 1, 1, 4, f.slots, f.formats[l], ids, kind, f.job_of, (const Job*)nullptr, 0, 0);
        c.end_window(); Output out; c.finish(out.file, 1, 1, 0, 0, 2, 0);
        require(c.ok() && c.bins()[0].calls == 2);
    });
    test("PCIe_entries_and_unique_jobs_differ", [] {
        Fixture f; const int32_t ids[] = {0, 0}, kind[] = {1, 1};
        Census c(2, 2, 4, f.slots, f.formats); c.begin_window(1, 0, 2);
        for (int l = 0; l < 2; ++l)
            c.record(l, 2, 1, 4, f.slots, f.formats[l], ids, kind, f.job_of, (const Job*)nullptr, 0, 0);
        c.end_window(); Output out; c.finish(out.file, 1, 2, 0, 0, 0, 2);
        require(c.ok() && c.bins()[0].entries[2] == 4);
    });
    test("partial_or_reordered_layer_rejected", [] {
        for (bool reordered : {false, true}) {
            Fixture f; Census c(1, 2, 4, f.slots, f.formats); c.begin_window(1, 10, 4);
            f.record(c, reordered ? 1 : 0); c.end_window(); require(!c.ok());
        }
    });
    test("window_identity_and_bounds", [] {
        for (int variant = 0; variant < 5; ++variant) {
            Fixture f; Census c(1, 2, 4, f.slots, f.formats);
            if (variant == 0) c.begin_window(2, 0, 4);
            if (variant == 1) c.begin_window(1, -1, 4);
            if (variant == 2) c.begin_window(1, 0, 9);
            if (variant == 3) { c.begin_window(1, 0, 4); c.begin_window(1, 0, 4); }
            if (variant == 4) { f.window(c); c.begin_window(2, 10, 4); }
            require(!c.ok());
        }
    });
    test("changed_cache_or_format_rejected", [] {
        for (int variant = 0; variant < 3; ++variant) {
            Fixture f; Census c(1, 2, 4, f.slots, f.formats); c.begin_window(1, 10, 4);
            if (variant == 0) f.slots[1] = 3; // same membership, different live slot
            if (variant == 1) f.slots[0] = 2;
            if (variant == 2) ++f.formats[0].act_bytes;
            f.record(c, 0); require(!c.ok());
        }
    });
    test("unknown_kind_invalid_id_and_inconsistent_kind", [] {
        for (int variant = 0; variant < 4; ++variant) {
            Fixture f; Census c(1, 2, 4, f.slots, f.formats); c.begin_window(1, 10, 4);
            if (variant == 0) f.kind[0] = INT_MAX;
            if (variant == 1) f.ids[0] = 4;
            if (variant == 2) f.kind[2] = 1;
            if (variant == 3) f.kind[1] = -1;
            f.record(c, 0); require(!c.ok());
        }
    });
    test("actual_job_map_and_nt_are_required", [] {
        for (int variant = 0; variant < 4; ++variant) {
            Fixture f; Census c(1, 2, 4, f.slots, f.formats); c.begin_window(1, 10, 4);
            if (variant == 0) f.job_of[2] = 0;
            if (variant == 1) f.job_of[2] = -1;
            if (variant == 2) f.jobs[0].nt = 4;
            if (variant == 3) f.jobs[2].nt = 0;
            f.record(c, 0); require(!c.ok());
        }
    });
    test("duplicate_ID_in_one_token_rejected", [] {
        Fixture f; Census c(1, 2, 4, f.slots, f.formats); c.begin_window(1, 10, 4);
        f.ids[1] = f.ids[0]; f.kind[1] = -1; f.record(c, 0); require(!c.ok());
    });
    test("request_footer_conservation", [] {
        for (int variant = 0; variant < 6; ++variant) {
            Fixture f; Census c(1, 2, 4, f.slots, f.formats); f.window(c);
            int64_t values[] = {1, 4, 6, 12, 4, 0}; ++values[variant];
            Output out; c.finish(out.file, values[0], values[1], values[2], values[3], values[4], values[5]);
            require(!c.ok() && c.closed());
        }
    });
    test("thread_identity_is_not_TLS", [] {
        Fixture f; Census c(1, 2, 4, f.slots, f.formats); c.begin_window(1, 10, 4);
        std::thread wrong([&] { f.record(c, 0); }); wrong.join();
        require(!c.ok() && c.cpu_jobs() == 0); Output out; c.abort(out.file);
        require(out.text().find("error=wrong_thread") != std::string::npos);
    });
    test("aborted_request_is_not_a_partial_pass", [] {
        Fixture f; Census c(1, 2, 4, f.slots, f.formats); c.begin_window(1, 10, 4);
        f.record(c, 0); Output out; c.abort(out.file);
        require(!c.ok() && c.closed() && out.text().find("valid=0") != std::string::npos);
    });
    test("request_storage_is_independent", [] {
        Fixture f; Census a(1, 2, 4, f.slots, f.formats), b(2, 2, 4, f.slots, f.formats);
        f.window(a); require(a.cpu_jobs() == 6 && b.cpu_jobs() == 0 && b.windows() == 0 && b.ok());
    });
    test("checked_uint64_and_invalid_construction", [] {
        uint64_t n = UINT64_MAX - 1; require(Census::checked_add(n, 1));
        require(!Census::checked_add(n, 1) && n == UINT64_MAX);
        Fixture f; Census c(0, 49, 513, f.slots, f.formats);
        Output out; c.write_begin(out.file, 1); c.abort(out.file); require(!c.ok());
    });
    std::printf("cpu_work_census_fixture cases=%d status=PASS\n", cases);
}
