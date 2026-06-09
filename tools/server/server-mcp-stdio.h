#pragma once

#include "server-common.h"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <atomic>
#include <deque>

struct mcp_stdio_config {
    std::string server_id;
    std::string command;
    std::vector<std::string> args;
    std::string cwd;
    std::map<std::string, std::string> env;
};

struct mcp_stdio_session {
    std::string session_id;
    std::string server_id;

    std::atomic<bool> process_started{false};
    std::atomic<bool> process_exited{false};
    std::atomic<bool> process_reaped{false};
    std::atomic<int> exit_code{0};
    std::string last_error;

    mutable std::mutex stderr_mutex;
    std::deque<std::string> stderr_tail;
    std::atomic<bool> bytes_truncated{false};

    mutable std::mutex callback_mutex;

#ifdef _WIN32
    void * hProcess    = nullptr;
    void * hJob        = nullptr;
    void * hStdinWrite = nullptr;
    void * hStdoutRead = nullptr;
    void * hStderrRead = nullptr;
#else
    int pid       = -1;
    int fd_stdin  = -1;
    int fd_stdout = -1;
    int fd_stderr = -1;
#endif

    std::thread stdout_thread;
    std::thread stderr_thread;

    struct ws_state {
        void * ws = nullptr;
        std::atomic<bool> is_alive{true};
        std::function<void(void *, const std::string &)> write_fn;
        std::function<void(void *)> close_fn;
    };

    std::shared_ptr<ws_state> websocket_state;

    mcp_stdio_session();
    ~mcp_stdio_session();

    bool start(const mcp_stdio_config & config);
    void terminate();
    bool write_stdin(const std::string & data);
    json get_diagnostics() const;

private:
    void capture_stdout();
    void capture_stderr();
};

class mcp_stdio_manager {
public:
    mcp_stdio_manager();
    ~mcp_stdio_manager();

    std::shared_ptr<mcp_stdio_session> create_session(const mcp_stdio_config & config);
    std::shared_ptr<mcp_stdio_session> get_session(const std::string & session_id);
    void delete_session(const std::string & session_id);
    json get_all_sessions();

private:
    std::mutex mutex;
    std::map<std::string, std::shared_ptr<mcp_stdio_session>> sessions;
};
