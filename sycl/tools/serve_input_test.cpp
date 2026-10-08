// Real pipes and production reader, without any GPU or runtime dependency.
#include "strata/serve_input.hpp"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <fcntl.h>
#include <string>
#include <thread>

namespace {
void write_all(int fd, const std::string& text) {
    size_t at = 0;
    while (at != text.size()) {
        const ssize_t count = ::write(fd, text.data() + at, text.size() - at);
        if (count < 0 && errno == EINTR) continue;
        assert(count > 0);
        at += static_cast<size_t>(count);
    }
}
struct Pipe {
    int fd[2];
    Pipe() { assert(::pipe(fd) == 0); }
    ~Pipe() { for (int p : fd) if (p >= 0) ::close(p); }
    void end() { ::close(fd[1]); fd[1] = -1; }
};
}
int main() {
    // An open writer must not strand the engine's reader on normal/error exit.
    for (int i = 0; i < 16; ++i) {
        Pipe pipe;
        const auto start = std::chrono::steady_clock::now();
        {
            strata::ServeInput input(pipe.fd[0]);
            if (i & 1) { input.close(); input.close(); }
        }
        assert(std::chrono::steady_clock::now() - start < std::chrono::seconds(2));
        assert(::fcntl(pipe.fd[0], F_GETFD) >= 0); // caller still owns descriptor
    }
    std::puts("PASS open-pipe shutdown, repeat close and caller-owned fd");
    {
        Pipe pipe;
        strata::ServeInput input(pipe.fd[0]);
        write_all(pipe.fd[1], "GEN 1\r\nST");
        std::string line;
        assert(input.next_line(line) && line == "GEN 1");
        write_all(pipe.fd[1], "OP\r\nQUIT\npartial");
        pipe.end();
        assert(input.next_line(line) && line == "QUIT");
        assert(input.next_line(line) && line == "partial");
        assert(!input.next_line(line));
        assert(input.stop_request.load());
    }
    std::puts("PASS STOP, split lines, CRLF and final EOF fragment");
    {
        Pipe pipe;
        std::string request = "GEN 1 ";
        for (int i = 0; i < 262144; ++i) request += "248045,";
        strata::ServeInput input(pipe.fd[0]);
        std::jthread writer([&] { write_all(pipe.fd[1], request + "\nQUIT\n"); pipe.end(); });
        std::string line;
        assert(input.next_line(line) && line == request);
        assert(input.next_line(line) && line == "QUIT");
        assert(!input.next_line(line));
    }
    std::puts("PASS complete 262144-token protocol line (CPU input only)");
    {
        Pipe pipe;
        strata::ServeInput input(pipe.fd[0]);
        std::jthread waiter([&] { std::string line; assert(!input.next_line(line)); });
        input.close();
    }
    std::puts("PASS close wakes protocol waiter");
}
