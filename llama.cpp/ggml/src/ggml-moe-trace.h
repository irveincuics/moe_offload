#pragma once

#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>

// Only included by ggml-backend.cpp. All timestamps use the same host clock.
struct ggml_moe_trace {
    FILE * file = nullptr;
    std::mutex mutex;
    std::string request;
    int64_t eval = -1, token = -1, n_tokens = 0;
    uint64_t serial = 0;

    static std::string quote(const char * text) {
        std::string out = "\"";
        for (const unsigned char * p = (const unsigned char *) text; *p; ++p) {
            if (*p == '"' || *p == '\\') { out += '\\'; out += *p; }
            else if (*p < 32) {
                char escape[7];
                snprintf(escape, sizeof(escape), "\\u%04x", *p);
                out += escape;
            } else { out += *p; }
        }
        return out + '"';
    }

    void emit(const char * event, const char * tensor = "", int expert = -1,
              uint64_t slot = 0, uint64_t bytes = 0, const char * detail = "", const char * extra = "") {
        if (!file) { return; }
        std::lock_guard<std::mutex> lock(mutex);
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        if (fprintf(file,
                "{\"seq\":%llu,\"host_ns\":%lld,\"request\":%s,\"eval\":%lld,\"token\":%lld,\"n_tokens\":%lld,"
                "\"event\":%s,\"tensor\":%s,\"expert\":%d,\"slot\":%llu,\"bytes\":%llu,\"detail\":%s%s}\n",
                (unsigned long long) ++serial, (long long) ns, quote(request.c_str()).c_str(),
                (long long) eval, (long long) token, (long long) n_tokens, quote(event).c_str(),
                quote(tensor).c_str(), expert, (unsigned long long) slot, (unsigned long long) bytes,
                quote(detail).c_str(), extra) < 0) {
            GGML_ABORT("M0 trace write failed");
        }
    }
};

static ggml_moe_trace moe_trace;

bool ggml_backend_moe_trace_active(void) { return moe_trace.file != nullptr; }

void ggml_backend_moe_trace_cuda_event(const char * event, const char * tensor, size_t offset, size_t bytes, double elapsed_ms) {
    char extra[160];
    snprintf(extra, sizeof(extra), ",\"offset\":%llu,\"device_elapsed_ms\":%.9f", (unsigned long long) offset, elapsed_ms);
    moe_trace.emit(event, tensor, -1, 0, bytes, "cuda_event; correlate_by_eval_tensor_offset; separate_clock", extra);
}

bool ggml_backend_moe_trace_open(const char * path, const char * request) {
    if (moe_trace.file) { return false; }
    moe_trace.file = fopen(path, "wx");
    if (!moe_trace.file) { return false; }
    moe_trace.request = request;
    moe_trace.serial = 0;
    moe_trace.eval = moe_trace.token = -1;
    moe_trace.n_tokens = 0;
    moe_trace.emit("request_begin", "", -1, 0, 0, "host_steady_clock; synchronized_diagnostic; no_prediction");
    return true;
}

void ggml_backend_moe_trace_step(int64_t eval, int64_t token, int64_t n_tokens) {
    std::lock_guard<std::mutex> lock(moe_trace.mutex);
    moe_trace.eval = eval;
    moe_trace.token = token;
    moe_trace.n_tokens = n_tokens;
}

void ggml_backend_moe_trace_close(void) {
    if (!moe_trace.file) { return; }
    moe_trace.emit("request_end");
    if (fclose(moe_trace.file) != 0) { GGML_ABORT("M0 trace close failed"); }
    moe_trace.file = nullptr;
}
