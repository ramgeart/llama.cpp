#include "server-mcp-stdio.h"
#include "log.h"
#include <sstream>
#include <iostream>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>
#include <spawn.h>
#if defined(__linux__)
#include <sys/prctl.h>
#endif
extern char **environ;
#endif

mcp_stdio_session::mcp_stdio_session() {}

mcp_stdio_session::~mcp_stdio_session() {
    terminate();
    if (stdout_thread.joinable()) stdout_thread.join();
    if (stderr_thread.joinable()) stderr_thread.join();
}

#ifdef _WIN32
static std::wstring utf8_to_wide(const std::string & str) {
    if (str.empty()) return L"";
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstrTo[0], size_needed);
    return wstrTo;
}

static std::wstring windows_argv_to_command_line(const std::wstring & command, const std::vector<std::wstring> & args) {
    auto quote_arg = [](const std::wstring & arg) {
        if (arg.empty()) return std::wstring(L"\"\"");
        if (arg.find_first_of(L" \t\n\v\"") == std::wstring::npos) return arg;

        std::wstring res = L"\"";
        for (size_t i = 0; i < arg.size(); ++i) {
            size_t backslashes = 0;
            while (i < arg.size() && arg[i] == L'\\') {
                i++;
                backslashes++;
            }

            if (i == arg.size()) {
                res.append(backslashes * 2, L'\\');
                break;
            } else if (arg[i] == L'\"') {
                res.append(backslashes * 2 + 1, L'\\');
                res.append(1, L'\"');
            } else {
                res.append(backslashes, L'\\');
                res.append(1, arg[i]);
            }
        }
        res.append(L"\"");
        return res;
    };

    std::wstring cmd_line = quote_arg(command);
    for (const auto & arg : args) {
        cmd_line += L" " + quote_arg(arg);
    }
    return cmd_line;
}
#endif

bool mcp_stdio_session::start(const mcp_stdio_config & config) {
    server_id = config.server_id;

#ifdef _WIN32
    // Create Pipes
    SECURITY_ATTRIBUTES saAttr;
    saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
    saAttr.bInheritHandle = TRUE;
    saAttr.lpSecurityDescriptor = NULL;

    HANDLE hStdinRead = NULL;
    HANDLE hStdoutWrite = NULL;
    HANDLE hStderrWrite = NULL;

    auto cleanup_win_pipes = [&]() {
        if (hStdinRead)    { CloseHandle(hStdinRead);    hStdinRead    = NULL; }
        if (hStdinWrite)   { CloseHandle((HANDLE)hStdinWrite);   hStdinWrite   = nullptr; }
        if (hStdoutRead)   { CloseHandle((HANDLE)hStdoutRead);   hStdoutRead   = nullptr; }
        if (hStdoutWrite)  { CloseHandle(hStdoutWrite);  hStdoutWrite  = NULL; }
        if (hStderrRead)   { CloseHandle((HANDLE)hStderrRead);   hStderrRead   = nullptr; }
        if (hStderrWrite)  { CloseHandle(hStderrWrite);  hStderrWrite  = NULL; }
    };

    if (!CreatePipe(&hStdinRead, (PHANDLE)&hStdinWrite, &saAttr, 0)) {
        last_error = "CreatePipe(stdin) failed: " + std::to_string(GetLastError());
        return false;
    }
    if (!SetHandleInformation((HANDLE)hStdinWrite, HANDLE_FLAG_INHERIT, 0)) {
        last_error = "SetHandleInformation(stdin) failed: " + std::to_string(GetLastError());
        cleanup_win_pipes();
        return false;
    }

    if (!CreatePipe((PHANDLE)&hStdoutRead, &hStdoutWrite, &saAttr, 0)) {
        last_error = "CreatePipe(stdout) failed: " + std::to_string(GetLastError());
        cleanup_win_pipes();
        return false;
    }
    if (!SetHandleInformation((HANDLE)hStdoutRead, HANDLE_FLAG_INHERIT, 0)) {
        last_error = "SetHandleInformation(stdout) failed: " + std::to_string(GetLastError());
        cleanup_win_pipes();
        return false;
    }

    if (!CreatePipe((PHANDLE)&hStderrRead, &hStderrWrite, &saAttr, 0)) {
        last_error = "CreatePipe(stderr) failed: " + std::to_string(GetLastError());
        cleanup_win_pipes();
        return false;
    }
    if (!SetHandleInformation((HANDLE)hStderrRead, HANDLE_FLAG_INHERIT, 0)) {
        last_error = "SetHandleInformation(stderr) failed: " + std::to_string(GetLastError());
        cleanup_win_pipes();
        return false;
    }

    // Job Object
    hJob = CreateJobObjectW(NULL, NULL);
    if (hJob) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION jeli = {0};
        jeli.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(hJob, JobObjectExtendedLimitInformation, &jeli, sizeof(jeli));
    }

    // Prepare Command Line
    std::wstring wcommand = utf8_to_wide(config.command);
    std::vector<std::wstring> wargs;
    for (const auto & arg : config.args) {
        wargs.push_back(utf8_to_wide(arg));
    }
    std::wstring wfull_cmd = windows_argv_to_command_line(wcommand, wargs);

    // CreateProcessW requires a mutable buffer for the command line
    std::vector<wchar_t> wcmd_buffer(wfull_cmd.begin(), wfull_cmd.end());
    wcmd_buffer.push_back(L'\0');

    STARTUPINFOW siStartInfo;
    ZeroMemory(&siStartInfo, sizeof(STARTUPINFOW));
    siStartInfo.cb = sizeof(STARTUPINFOW);
    siStartInfo.hStdError = hStderrWrite;
    siStartInfo.hStdOutput = hStdoutWrite;
    siStartInfo.hStdInput = hStdinRead;
    siStartInfo.dwFlags |= STARTF_USESTDHANDLES;

    PROCESS_INFORMATION piProcInfo;
    ZeroMemory(&piProcInfo, sizeof(PROCESS_INFORMATION));

    std::wstring wcwd = utf8_to_wide(config.cwd);
    const wchar_t * lpCurrentDirectory = wcwd.empty() ? NULL : wcwd.c_str();

    // Environment
    std::vector<wchar_t> env_block;
    if (!config.env.empty()) {
        LPWCH current_env = GetEnvironmentStringsW();
        if (current_env) {
            std::map<std::wstring, std::wstring> env_map;
            for (LPWCH e = current_env; *e; e += wcslen(e) + 1) {
                std::wstring s(e);
                size_t pos = s.find(L'=');
                if (pos != std::wstring::npos) {
                    env_map[s.substr(0, pos)] = s.substr(pos + 1);
                }
            }
            FreeEnvironmentStringsW(current_env);

            for (const auto & pair : config.env) {
                env_map[utf8_to_wide(pair.first)] = utf8_to_wide(pair.second);
            }

            for (const auto & pair : env_map) {
                std::wstring line = pair.first + L"=" + pair.second + L'\0';
                env_block.insert(env_block.end(), line.begin(), line.end());
            }
            env_block.push_back(L'\0');
        }
    }

    LPVOID lpEnvironment = env_block.empty() ? NULL : env_block.data();

    if (!CreateProcessW(NULL, wcmd_buffer.data(), NULL, NULL, TRUE, CREATE_NO_WINDOW | CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT, lpEnvironment, lpCurrentDirectory, &siStartInfo, &piProcInfo)) {
        last_error = "CreateProcessW failed: " + std::to_string(GetLastError());
        cleanup_win_pipes();
        return false;
    }

    hProcess = piProcInfo.hProcess;
    if (hJob) {
        if (!AssignProcessToJobObject(hJob, hProcess)) {
            last_error = "AssignProcessToJobObject failed: " + std::to_string(GetLastError());
            TerminateProcess(hProcess, 1);
            CloseHandle(hProcess);
            CloseHandle(piProcInfo.hThread);
            hProcess = nullptr;
            cleanup_win_pipes();
            return false;
        }
    }
    ResumeThread(piProcInfo.hThread);
    CloseHandle(piProcInfo.hThread);

    CloseHandle(hStdinRead);
    CloseHandle(hStdoutWrite);
    CloseHandle(hStderrWrite);

#else
    int pipe_stdin[2]  = {-1, -1};
    int pipe_stdout[2] = {-1, -1};
    int pipe_stderr[2] = {-1, -1};

    auto cleanup_posix_pipes = [&]() {
        if (pipe_stdin[0]  != -1) { close(pipe_stdin[0]);  pipe_stdin[0]  = -1; }
        if (pipe_stdin[1]  != -1) { close(pipe_stdin[1]);  pipe_stdin[1]  = -1; }
        if (pipe_stdout[0] != -1) { close(pipe_stdout[0]); pipe_stdout[0] = -1; }
        if (pipe_stdout[1] != -1) { close(pipe_stdout[1]); pipe_stdout[1] = -1; }
        if (pipe_stderr[0] != -1) { close(pipe_stderr[0]); pipe_stderr[0] = -1; }
        if (pipe_stderr[1] != -1) { close(pipe_stderr[1]); pipe_stderr[1] = -1; }
    };

    if (pipe(pipe_stdin) != 0) {
        last_error = "pipe(stdin) failed: errno " + std::to_string(errno);
        return false;
    }
    if (pipe(pipe_stdout) != 0) {
        last_error = "pipe(stdout) failed: errno " + std::to_string(errno);
        cleanup_posix_pipes();
        return false;
    }
    if (pipe(pipe_stderr) != 0) {
        last_error = "pipe(stderr) failed: errno " + std::to_string(errno);
        cleanup_posix_pipes();
        return false;
    }

    pid = fork();
    if (pid == 0) {
        // Child
        setsid();
#if defined(__linux__)
        prctl(PR_SET_PDEATHSIG, SIGTERM);
#endif
        dup2(pipe_stdin[0], STDIN_FILENO);
        dup2(pipe_stdout[1], STDOUT_FILENO);
        dup2(pipe_stderr[1], STDERR_FILENO);

        close(pipe_stdin[0]); close(pipe_stdin[1]);
        close(pipe_stdout[0]); close(pipe_stdout[1]);
        close(pipe_stderr[0]); close(pipe_stderr[1]);

        if (!config.cwd.empty()) {
            if (chdir(config.cwd.c_str()) != 0) {
                perror("chdir");
                _exit(1);
            }
        }

        for (const auto & pair : config.env) {
            setenv(pair.first.c_str(), pair.second.c_str(), 1);
        }

        std::vector<char *> argv;
        argv.push_back(const_cast<char *>(config.command.c_str()));
        for (const auto & arg : config.args) {
            argv.push_back(const_cast<char *>(arg.c_str()));
        }
        argv.push_back(nullptr);

        execvp(config.command.c_str(), argv.data());
        perror("execvp");
        _exit(1);
    } else if (pid > 0) {
        // Parent
        fd_stdin = pipe_stdin[1];
        fd_stdout = pipe_stdout[0];
        fd_stderr = pipe_stderr[0];
        close(pipe_stdin[0]);
        close(pipe_stdout[1]);
        close(pipe_stderr[1]);
    } else {
        last_error = "fork() failed: errno " + std::to_string(errno);
        cleanup_posix_pipes();
        return false;
    }
#endif

    process_started = true;
    stdout_thread = std::thread(&mcp_stdio_session::capture_stdout, this);
    stderr_thread = std::thread(&mcp_stdio_session::capture_stderr, this);
    return true;
}

void mcp_stdio_session::terminate() {
#ifdef _WIN32
    if (hJob) {
        CloseHandle(hJob);
        hJob = nullptr;
    }
    if (hProcess) {
        TerminateProcess(hProcess, 1);
        CloseHandle(hProcess);
        hProcess = nullptr;
    }
    if (hStdinWrite) { CloseHandle(hStdinWrite); hStdinWrite = nullptr; }
    if (hStdoutRead) { CloseHandle(hStdoutRead); hStdoutRead = nullptr; }
    if (hStderrRead) { CloseHandle(hStderrRead); hStderrRead = nullptr; }
#else
    if (pid > 0) {
        // Send SIGTERM to the entire process group
        kill(-pid, SIGTERM);

        // Wait a bit for graceful exit
        int status;
        for (int i = 0; i < 5; ++i) {
            int res = waitpid(pid, &status, WNOHANG);
            if (res > 0) {
                process_reaped = true;
                if (WIFEXITED(status)) {
                    exit_code = WEXITSTATUS(status);
                } else {
                    exit_code = 1;
                }
                break;
            } else if (res == -1) {
                if (errno == ECHILD) {
                    process_reaped = true; // Already reaped by something else
                }
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // If still alive, SIGKILL
        if (!process_reaped.exchange(true)) {
            kill(-pid, SIGKILL);
            waitpid(pid, &status, 0);
            exit_code = 1;
        }
        pid = -1;
    }
    if (fd_stdin != -1) { close(fd_stdin); fd_stdin = -1; }
    if (fd_stdout != -1) { close(fd_stdout); fd_stdout = -1; }
    if (fd_stderr != -1) { close(fd_stderr); fd_stderr = -1; }
#endif
}

bool mcp_stdio_session::write_stdin(const std::string & data) {
    if (!process_started || process_exited) return false;
    std::string line = data + "\n";
#ifdef _WIN32
    DWORD bytesWritten = 0;
    if (WriteFile(hStdinWrite, line.c_str(), (DWORD)line.size(), &bytesWritten, NULL)) {
        return bytesWritten == line.size();
    }
    return false;
#else
    size_t total_written = 0;
    while (total_written < line.size()) {
        ssize_t res = write(fd_stdin, line.c_str() + total_written, line.size() - total_written);
        if (res > 0) {
            total_written += res;
        } else if (res == -1) {
            if (errno == EINTR) continue;
            return false; // EPIPE or other errors
        } else {
            return false;
        }
    }
    return true;
#endif
}

void mcp_stdio_session::capture_stdout() {
    char buffer[4096];
    std::string leftover;
    const size_t max_leftover = 1024 * 1024; // 1MB limit for very long lines
    while (true) {
#ifdef _WIN32
        DWORD bytesRead;
        if (!ReadFile(hStdoutRead, buffer, sizeof(buffer), &bytesRead, NULL) || bytesRead == 0) break;
        std::string chunk(buffer, bytesRead);
#else
        ssize_t bytesRead = read(fd_stdout, buffer, sizeof(buffer));
        if (bytesRead <= 0) break;
        std::string chunk(buffer, bytesRead);
#endif
        if (leftover.size() + chunk.size() > max_leftover) {
            leftover.clear(); // Drop very long lines that exceed the buffer
            bytes_truncated = true;
        }
        leftover += chunk;
        size_t pos;
        while ((pos = leftover.find('\n')) != std::string::npos) {
            std::string line = leftover.substr(0, pos);
            if (!line.empty() && line.back() == '\r') line.pop_back();

            std::shared_ptr<ws_state> state;
            {
                std::lock_guard<std::mutex> lock(callback_mutex);
                state = websocket_state;
            }

            if (state) {
                state->write(line);
            }

            leftover.erase(0, pos + 1);
        }
    }
    process_exited = true;

#ifndef _WIN32
    if (pid > 0 && !process_reaped.exchange(true)) {
        int status;
        int res = waitpid(pid, &status, 0);
        if (res > 0) {
            if (WIFEXITED(status)) {
                exit_code = WEXITSTATUS(status);
            } else {
                exit_code = 1;
            }
        }
    }
#endif

    std::shared_ptr<ws_state> state;
    {
        std::lock_guard<std::mutex> lock(callback_mutex);
        state = websocket_state;
    }
    if (state) {
        state->close();
    }
}

void mcp_stdio_session::capture_stderr() {
    char buffer[4096];
    std::string leftover;
    const size_t max_leftover = 1024 * 1024; // 1MB limit
    while (true) {
#ifdef _WIN32
        DWORD bytesRead;
        if (!ReadFile(hStderrRead, buffer, sizeof(buffer), &bytesRead, NULL) || bytesRead == 0) break;
        std::string chunk(buffer, bytesRead);
#else
        ssize_t bytesRead = read(fd_stderr, buffer, sizeof(buffer));
        if (bytesRead <= 0) break;
        std::string chunk(buffer, bytesRead);
#endif
        if (leftover.size() + chunk.size() > max_leftover) {
            leftover.clear();
            bytes_truncated = true;
        }
        leftover += chunk;
        size_t pos;
        while ((pos = leftover.find('\n')) != std::string::npos) {
            std::string line = leftover.substr(0, pos);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            {
                std::lock_guard<std::mutex> lock(stderr_mutex);
                stderr_tail.push_back(line);
                if (stderr_tail.size() > 100) stderr_tail.pop_front();
            }
            leftover.erase(0, pos + 1);
        }
    }
}

json mcp_stdio_session::get_diagnostics() const {
    std::lock_guard<std::mutex> lock(stderr_mutex);
    json res;
    res["session_id"] = session_id;
    res["server_id"] = server_id;
    res["process_started"] = (bool)process_started;
    res["process_exited"] = (bool)process_exited;
    res["exit_code"] = (bool)process_exited ? json(exit_code.load()) : json(nullptr);
    res["last_error"] = last_error.empty() ? json(nullptr) : json(last_error);
    res["stderr_tail"] = stderr_tail;
    res["bytes_truncated"] = (bool)bytes_truncated;
    return res;
}

mcp_stdio_manager::mcp_stdio_manager() {}
mcp_stdio_manager::~mcp_stdio_manager() {
    std::lock_guard<std::mutex> lock(mutex);
    sessions.clear();
}

std::shared_ptr<mcp_stdio_session> mcp_stdio_manager::create_session(const mcp_stdio_config & config) {
    auto session = std::make_shared<mcp_stdio_session>();
    session->session_id = random_string();
    if (session->start(config)) {
        std::lock_guard<std::mutex> lock(mutex);
        sessions[session->session_id] = session;
        return session;
    }
    return nullptr;
}

std::shared_ptr<mcp_stdio_session> mcp_stdio_manager::get_session(const std::string & session_id) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = sessions.find(session_id);
    if (it != sessions.end()) return it->second;
    return nullptr;
}

void mcp_stdio_manager::delete_session(const std::string & session_id) {
    std::shared_ptr<mcp_stdio_session> session;
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = sessions.find(session_id);
        if (it != sessions.end()) {
            session = it->second;
            sessions.erase(it);
        }
    }
    if (session) {
        session->terminate();
    }
}

json mcp_stdio_manager::get_all_sessions() {
    std::lock_guard<std::mutex> lock(mutex);
    json res = json::array();
    for (const auto & pair : sessions) {
        res.push_back({
            {"session_id", pair.first},
            {"server_id", pair.second->server_id}
        });
    }
    return res;
}
