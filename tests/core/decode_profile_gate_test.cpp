// Main-authored host fixture; repository LICENSE. No SYCL or GPU calls.
#include "strata/sycl/decode_profile_gate.hpp"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
using strata::sycl_backend::decode_profile_gate::Config;
using strata::sycl_backend::decode_profile_gate::Gate;
using strata::sycl_backend::decode_profile_gate::configuration;
namespace fs = std::filesystem;
unsigned cases = 0;

void require(bool value, const char* message) {
    if (!value) {
        std::fprintf(stderr, "decode gate fixture: %s\n", message);
        std::exit(1);
    }
}

struct Directory {
    fs::path path;
    Directory() {
        const char* temporary = std::getenv("TMPDIR");
        require(temporary && *temporary == '/', "private controller TMPDIR required");
        std::string name = std::string(temporary) + "/decode-gate-XXXXXX";
        const char* created = ::mkdtemp(name.data());
        require(created != nullptr, "temporary directory creation");
        path = created;
    }
    ~Directory() {
        std::error_code error;
        fs::remove_all(path, error);
        require(!error, "fixture directory cleanup");
    }
    Config config() const { return {path.string(), 7, 1500, true}; }
};

void store(const fs::path& path, const std::string& value) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    require(fd >= 0, "peer file creation");
    require(::write(fd, value.data(), value.size()) == static_cast<ssize_t>(value.size()), "peer file write");
    require(::close(fd) == 0, "peer file close");
}

// The peer observes an atomically published event and replies through rename.
// It never reads or changes the process environment after fixture startup.
bool exchange(Gate& gate, Directory& directory, uint64_t round, unsigned sequence,
              unsigned corrupt, std::atomic<unsigned>& drains, std::string& error) {
    bool replied = false;
    const auto expected_drains = drains.load() + 1;
    std::thread peer([&] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (std::chrono::steady_clock::now() < deadline) {
            std::ifstream input(directory.path / "event");
            std::string version, extra;
            uint64_t request = 0, observed_sequence = 0, observed_round = 0, raw = 0;
            if (input >> version >> request >> observed_sequence >> observed_round >> raw) {
                require(!(input >> extra), "event trailing bytes");
                if (observed_sequence == sequence) {
                    require(version == "v1" && request == 7 && observed_round == round && raw > 0,
                            "event identity/extent/raw clock");
                    require(drains.load() == expected_drains, "event preceded checked completion");
                    std::string answer = "v1 " + std::to_string(corrupt == 1 ? 8 : 7) + " " +
                        std::to_string(corrupt == 2 ? sequence + 1 : sequence) + " " +
                        std::to_string(corrupt == 3 ? round + 1 : round) + "\n";
                    if (corrupt == 4) answer += "trailing\n";
                    if (corrupt == 5) answer.assign(200, 'x');
                    const auto temporary = directory.path / "peer.tmp";
                    if (corrupt == 6) {
                        store(directory.path / "target", answer);
                        require(::symlink("target", temporary.c_str()) == 0, "peer symlink creation");
                    } else if (corrupt == 9) {
                        require(::mkdir(temporary.c_str(), 0700) == 0, "peer non-file creation");
                    } else if (corrupt == 10) {
                        require(::mkfifo(temporary.c_str(), 0600) == 0, "peer FIFO creation");
                    } else {
                        store(temporary, answer);
                        if (corrupt == 7) require(::chmod(temporary.c_str(), 0644) == 0, "peer mode change");
                        if (corrupt == 8)
                            require(::link(temporary.c_str(), (directory.path / "second-link").c_str()) == 0,
                                    "peer hardlink creation");
                    }
                    require(::rename(temporary.c_str(), (directory.path / "ack").c_str()) == 0,
                            "peer atomic publication");
                    replied = true;
                    return;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    const bool result = gate.checkpoint(round, [&] { ++drains; return true; }, error);
    peer.join();
    require(replied, "peer did not observe the expected event");
    return result;
}

void parsing() {
    const auto disabled = configuration(nullptr, nullptr);
    require(disabled.valid && disabled.directory.empty() && disabled.request == 0, "unset startup pair");
    ++cases;
    for (const char* request : {"7", "18446744073709551615"}) {
        const auto config = configuration("/private", request);
        require(config.valid && config.request > 0 && config.directory == "/private", "valid startup pair");
        ++cases;
    }
    const std::vector<std::pair<const char*, const char*>> invalid{
        {nullptr, "7"}, {"/private", nullptr}, {"", "7"}, {"relative", "7"},
        {"/private", ""}, {"/private", "0"}, {"/private", "-1"}, {"/private", "+1"},
        {"/private", "7x"}, {"/private", "18446744073709551616"}, {"/private", " 7"}};
    for (const auto& pair : invalid) {
        require(!configuration(pair.first, pair.second).valid, "malformed startup pair accepted");
        ++cases;
    }
    const std::string too_long = "/" + std::string(4096, 'x');
    require(!configuration(too_long.c_str(), "7").valid, "oversized startup directory accepted");
    ++cases;
}

void inactive_and_invalid() {
    Directory directory;
    for (const auto& config : std::vector<Config>{{}, {(directory.path / "absent").string(), 8, 30, true}}) {
        Gate gate(config, 7);
        unsigned drains = 0;
        std::string error;
        require(!gate.enabled() && gate.complete() &&
            gate.checkpoint(0, [&] { ++drains; return false; }, error) &&
            gate.checkpoint(100, [&] { ++drains; return false; }, error) &&
            drains == 0 && error.empty() && fs::is_empty(directory.path), "inactive gate performed work");
        ++cases;
    }
    const std::vector<Config> invalid{{"/private", 7, 30, false}, {"relative", 7, 30, true},
        {"/private", 0, 30, true}, {"", 7, 30, true}, {"/private", 7, 0, true},
        {"/private", 7, 30001, true}};
    for (const auto& config : invalid) {
        Gate gate(config, 7);
        unsigned drains = 0;
        std::string error;
        require(!gate.complete() && !gate.checkpoint(0, [&] { ++drains; return true; }, error) &&
            drains == 0 && !error.empty() && !gate.complete(), "invalid direct configuration passed");
        ++cases;
    }
}

void success_and_ordering() {
    {
        Directory directory;
        Gate gate(directory.config(), 7);
        std::atomic<unsigned> drains{0};
        std::string error;
        require(gate.enabled() && !gate.complete(), "selected gate prematurely complete");
        unsigned sequence = 0;
        for (uint64_t round = 0; round <= 32; ++round) {
            if (round == 0 || round == 8 || round == 16 || round == 24) {
                require(exchange(gate, directory, round, ++sequence, 0, drains, error), "valid handshake failed");
                require(!fs::exists(directory.path / "ack"), "consumed acknowledgement remains");
            } else {
                const auto count = drains.load();
                require(gate.checkpoint(round, [&] { ++drains; return false; }, error) && drains.load() == count,
                        "non-boundary called completion");
            }
            require(gate.complete() == (round >= 24), "gate completion extent differs");
        }
        require(error.empty() && drains == 4, "successful gate lost a boundary");
        ++cases;
    }
    for (unsigned throwing = 0; throwing < 2; ++throwing) {
        Directory directory;
        Gate gate(directory.config(), 7);
        unsigned drains = 0;
        std::string error;
        require(!gate.checkpoint(0, [&]() -> bool {
            ++drains;
            if (throwing) throw std::runtime_error("completion failure");
            return false;
        }, error), "failed/throwing completion passed");
        require(!gate.complete() && !error.empty() && fs::is_empty(directory.path) &&
            !gate.checkpoint(0, [&] { ++drains; return true; }, error) && drains == 1,
            "completion failure published/retried a gate");
        ++cases;
    }
    for (unsigned mode = 0; mode < 3; ++mode) {
        Directory directory;
        Gate gate(directory.config(), 7);
        std::atomic<unsigned> drains{0};
        std::string error;
        if (mode) require(exchange(gate, directory, 0, 1, 0, drains, error), "ordering setup");
        const auto before = drains.load();
        require(!gate.checkpoint(mode == 0 ? 1 : mode == 1 ? 0 : 2,
            [&] { ++drains; return true; }, error) && !gate.complete() && drains.load() == before,
            "start/duplicate/skipped round accepted");
        ++cases;
    }
    {
        Directory directory;
        Gate gate(directory.config(), 7);
        std::atomic<unsigned> drains{0};
        std::string error;
        require(exchange(gate, directory, 0, 1, 0, drains, error), "partial setup");
        for (uint64_t round = 1; round < 8; ++round)
            require(gate.checkpoint(round, [] { return false; }, error), "partial non-boundary");
        require(!gate.complete() && drains == 1, "partial sequence qualified");
        ++cases;
    }
}

void rejected_peers() {
    for (unsigned mode = 1; mode <= 10; ++mode) {
        Directory directory;
        Gate gate(directory.config(), 7);
        std::atomic<unsigned> drains{0};
        std::string error;
        require(!exchange(gate, directory, 0, 1, mode, drains, error) && !gate.complete() && !error.empty(),
                "wrong identity/extent/type/mode/links accepted");
        require(!gate.checkpoint(1, [&] { ++drains; return true; }, error) && drains == 1,
                "peer failure did not remain closed");
        ++cases;
    }
    for (unsigned mode = 0; mode < 5; ++mode) {
        Directory directory;
        auto config = directory.config();
        config.timeout_ms = 30;
        if (mode == 1) store(directory.path / "ack", "stale\n");
        if (mode == 2) require(::chmod(directory.path.c_str(), 0770) == 0, "directory mode setup");
        if (mode == 3) {
            require(::symlink(directory.path.c_str(), (directory.path / "link").c_str()) == 0,
                    "directory symlink setup");
            config.directory = (directory.path / "link").string();
        }
        if (mode == 4) store(directory.path / "event.tmp", "stale\n");
        Gate gate(config, 7);
        unsigned drains = 0;
        std::string error;
        const auto start = std::chrono::steady_clock::now();
        require(!gate.checkpoint(0, [&] { ++drains; return true; }, error) && drains == 1 &&
            !gate.complete() && !error.empty(), "timeout/stale/private-directory failure accepted");
        require(std::chrono::steady_clock::now() - start < std::chrono::seconds(1), "bounded failure exceeded deadline");
        if (mode != 0) require(!fs::exists(directory.path / "event"), "failed setup published an event");
        ++cases;
    }
}
} // namespace

int main() {
    parsing();
    inactive_and_invalid();
    success_and_ordering();
    rejected_peers();
    require(cases == 45, "declared case extent differs");
    std::printf("decode_profile_gate cases=%u status=PASS\n", cases);
}
