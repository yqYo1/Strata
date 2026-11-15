// Main-authored; repository LICENSE applies. CPU metadata contract only.
#include "strata/prefill/moe_upload_census.hpp"
#include <string>

int main() {
    using strata::prefill::detail::MoeUploadCensus;
    MoeUploadCensus census;
    census.record(640, true);
    census.record(1280, false);
    census.record(8192, true);
    for (int rows : {-1, 0, 8193}) {
        bool refused = false;
        try { census.record(rows, true); }
        catch (const std::invalid_argument&) { refused = true; }
        if (!refused) return 1;
    }
    if (census.upload_count != 2 || census.hit_count != 1) return 2;
    FILE* file = std::tmpfile();
    if (!file) return 3;
    if (!census.emit_after_checked_drain(file, 7, 0, 32768, 22, 20, 1971200)) return 4;
    std::rewind(file);
    char data[4096]{};
    const auto bytes = std::fread(data, 1, sizeof(data) - 1, file);
    std::fclose(file);
    const std::string text(data, bytes);
    for (const char* expected : {"\"accepted_uploads\":2", "\"cache_hits\":1",
            "\"requested_upload_bytes\":3942400", "\"accepted_upload_rows_histogram\":[[640,1],[8192,1]]",
            "\"cache_hit_rows_histogram\":[[1280,1]]", "\"gu_type\":22,\"down_type\":20"})
        if (text.find(expected) == std::string::npos) return 5;
    MoeUploadCensus empty;
    if (empty.emit_after_checked_drain(stdout, 0, 0, 32768, 22, 20, 1971200)) return 6;
    for (int i = 0; i < 512; ++i) empty.record(1, i % 2 == 0);
    bool refused = false;
    try { empty.record(1, true); }
    catch (const std::invalid_argument&) { refused = true; }
    if (!refused) return 7;
    std::puts("MoE census host controls passed: literal upload/hit/byte/format bins;invalid rows;empty frame;512expert bound. No GPU qualification.");
    return 0;
}
