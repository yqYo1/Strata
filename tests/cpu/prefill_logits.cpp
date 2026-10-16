#include "strata/program/prefill_logits.hpp"
#include "strata/prefill/draft_uploads.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace pl = strata::program::prefill_logits;
namespace fs = std::filesystem;

static void require(bool ok) { if (!ok) throw std::runtime_error("contract check failed"); }
static std::string read(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

// Main supplies a new, empty, owned directory. No GPU/toolkit needed. This test
// intentionally keeps its tiny output fixtures for Main's ordinary retention review.
int main(int argc, char** argv) try {
    require(argc == 2 && fs::is_directory(argv[1]) && fs::is_empty(argv[1]));
    const fs::path dir(argv[1]);
    std::string err;
    std::vector<int64_t> p;
    require(pl::parse_positions("0,3,8", 9, p, err) && p == std::vector<int64_t>({0,3,8}));
    for (const char* bad : {"", "-1", "+1", " 1", "1 ", "1x", ",1", "1,", "1,,2",
                            "0,0", "2,1", "9", "9223372036854775808"}) {
        require(!pl::parse_positions(bad, 9, p, err) && p.empty());
    }
    require(!pl::parse_positions("0", 0, p, err));
    std::string many;
    for (int i = 0; i < 64; ++i) { if (i) many += ','; many += std::to_string(i); }
    require(pl::parse_positions(many, 64, p, err) && p.size() == 64);
    require(!pl::parse_positions(many + ",64", 65, p, err));
    {
        pl::Writer w;
        require(!w.begin((dir/"oversized").string(), {0,1}, 2, INT32_MAX, false, -1, err));
        require(!fs::exists(dir/"oversized"));
    }
    {
        pl::Writer w;
        require(w.begin((dir/"valid").string(), {0,3}, 4, 3, true, 13, err));
        require(w.row_bytes() == 12 && w.next_position() == 0);
        require(w.chunk(0, 2, err));
        std::fill(w.row(), w.row() + 3, 1.f);
        require(w.append(0, 2, err));
        require(w.chunk(2, 2, err));
        std::fill(w.row(), w.row() + 3, 2.f);
        require(w.append(3, 1, err) && w.next_position() == -1);
        require(!fs::exists(dir/"valid.meta"));
        require(w.finish(err) && fs::file_size(dir/"valid") == 32);
        const std::string meta = read(dir/"valid.meta");
        require(meta.find("head=native\nhead_ggml_type=13\n") != std::string::npos);
        require(meta.find("0\t0\t2\n1\t3\t1\ncomplete\t2\t32\n") != std::string::npos);
        std::ifstream f(dir/"valid", std::ios::binary);
        int32_t dims[2]{};
        float rows[6]{};
        f.read(reinterpret_cast<char*>(dims), sizeof(dims));
        f.read(reinterpret_cast<char*>(rows), sizeof(rows));
        require(f.good() && dims[0] == 3 && dims[1] == 2 && rows[0] == 1.f && rows[5] == 2.f);
    }
    {
        pl::Writer w;
        require(w.begin((dir/"missing").string(), {1}, 2, 3, false, -1, err));
        require(w.chunk(0, 1, err));
        require(!w.finish(err) && !fs::exists(dir/"missing.meta"));
    }
    {
        pl::Writer w;
        require(w.begin((dir/"out-of-order").string(), {1}, 2, 3, false, -1, err));
        require(!w.chunk(1, 1, err) && !w.finish(err));
        require(!fs::exists(dir/"out-of-order.meta"));
    }
    {
        pl::Writer w;
        require(w.begin((dir/"nan").string(), {0}, 1, 3, false, -1, err));
        require(w.chunk(0, 1, err));
        w.row()[0] = std::numeric_limits<float>::quiet_NaN();
        require(!w.append(0, 1, err) && !w.finish(err));
        require(!fs::exists(dir/"nan.meta"));
    }
    {
        pl::Writer w;
        require(!w.begin((dir/"valid").string(), {0}, 1, 3, false, -1, err));
        require(fs::file_size(dir/"valid") == 32); // no overwrite of a completed fixture
    }
    {
        std::ofstream(dir/"raw-existing") << "sentinel";
        pl::Writer w;
        require(!w.begin((dir/"raw-existing").string(), {0}, 1, 3, false, -1, err));
        require(read(dir/"raw-existing") == "sentinel");
    }
    {
        pl::Writer w;
        require(w.begin((dir/"publish-race").string(), {0}, 1, 3, false, -1, err));
        require(w.chunk(0, 1, err));
        require(w.append(0, 1, err));
        std::ofstream(dir/"publish-race.meta") << "sentinel";
        require(!w.finish(err) && read(dir/"publish-race.meta") == "sentinel");
    }
    {
        bool owner = true, waited = false;
        {
            strata::prefill::detail::DraftUploadGuard lease(
                [&] { require(owner); waited = true; return true; }, [] { std::abort(); });
            lease.arm(); // emulate an early return/exception after a partial enqueue
        }
        require(waited);
        owner = false;
    }
    return 0;
} catch (const std::exception& exc) {
    std::fprintf(stderr, "prefill logits contract: %s\n", exc.what());
    return 1;
}
