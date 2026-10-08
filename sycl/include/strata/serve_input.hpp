#pragma once
// Linux stdin reader whose lifetime ends before the engine's GPU objects.
#if !defined(_WIN32)
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <poll.h>
#include <string>
#include <thread>
#include <unistd.h>

namespace strata {
class ServeInput {
    int fd_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::string> lines_;
    bool eof_ = false;
public:
    std::atomic<bool> stop_request{false};
private:
    // Last member: everything used by the reader exists before it starts.
    std::jthread reader_;
    void read(std::stop_token stopping) {
        std::string buffer;
        char chunk[4096];
        auto next = [&](std::string& line) {
            while (!stopping.stop_requested()) {
                const size_t newline = buffer.find('\n');
                if (newline != std::string::npos) {
                    line.assign(buffer, 0, newline);
                    buffer.erase(0, newline + 1);
                    return true;
                }
                pollfd input{fd_, POLLIN, 0};
                const int available = ::poll(&input, 1, 100);
                if (available == 0 || (available < 0 && errno == EINTR)) continue;
                if (available < 0 || (input.revents & POLLNVAL)) return false;
                const ssize_t n = ::read(fd_, chunk, sizeof chunk);
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) {
                    if (buffer.empty()) return false;
                    line.swap(buffer);
                    buffer.clear();
                    return true;
                }
                buffer.append(chunk, static_cast<size_t>(n));
            }
            return false;
        };
        std::string line;
        while (next(line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line == "STOP") { stop_request.store(true); continue; }
            std::lock_guard lock(mutex_);
            lines_.push_back(std::move(line));
            ready_.notify_one();
        }
        std::lock_guard lock(mutex_);
        eof_ = true;
        ready_.notify_all();
    }
public:
    explicit ServeInput(int fd = STDIN_FILENO)
        : fd_(fd), reader_([this](std::stop_token stopping) { read(stopping); }) {}
    ~ServeInput() { close(); }
    void close() {
        reader_.request_stop();
        if (reader_.joinable()) reader_.join();
    }
    bool try_next_line(std::string& line) {
        std::lock_guard lock(mutex_);
        if (lines_.empty()) return false;
        line = std::move(lines_.front());
        lines_.pop_front();
        return true;
    }
    template<class Consume>
    void consume_control_lines(Consume consume) {
        std::lock_guard lock(mutex_);
        for (auto it = lines_.begin(); it != lines_.end();) {
            if (consume(*it)) it = lines_.erase(it);
            else ++it;
        }
    }
    bool next_line(std::string& line) {
        std::unique_lock lock(mutex_);
        ready_.wait(lock, [&] { return !lines_.empty() || eof_; });
        if (lines_.empty()) return false;
        line = std::move(lines_.front());
        lines_.pop_front();
        return true;
    }
};
} // namespace strata
#endif
