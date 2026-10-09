#include "llama.h"
#include "ggml-backend.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__linux__)
#include <sys/resource.h>
#endif

using json = nlohmann::ordered_json;
using clock_type = std::chrono::steady_clock;
namespace fs = std::filesystem;

static json read_json(const fs::path & path) {
    std::ifstream in(path);
    if (!in) { throw std::runtime_error("cannot read " + path.string()); }
    json result; in >> result; return result;
}

static void write_json(const fs::path & path, const json & value) {
    std::ofstream out(path);
    out << value.dump(2) << '\n';
    if (!out) { throw std::runtime_error("cannot write " + path.string()); }
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int n = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
    if (n <= 0) { throw std::runtime_error("empty prompt"); }
    std::vector<llama_token> result(n);
    n = llama_tokenize(vocab, text.c_str(), text.size(), result.data(), n, true, true);
    if (n < 0) { throw std::runtime_error("tokenization failed"); }
    result.resize(n); return result;
}

static std::string render(const json & request, size_t body_bytes) {
    const std::string body = request.value("body", "");
    // Do not cut a UTF-8 code point. Questions/instructions are never truncated.
    while (body_bytes < body.size() && body_bytes > 0 && ((unsigned char) body[body_bytes] & 0xc0) == 0x80) {
        --body_bytes;
    }
    return "<|im_start|>user\n" + request.at("prefix").get<std::string>() + body.substr(0, body_bytes) +
        request.at("suffix").get<std::string>() + "\n/no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
}

static void env_flag(const char * name, bool value) {
#if defined(_WIN32)
    _putenv_s(name, value ? "1" : "0");
#else
    setenv(name, value ? "1" : "0", 1);
#endif
}

static void quiet_log(enum ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR ||
            std::string(text).find("ssd_moe_cache: final_evals=") == 0) { fputs(text, stderr); }
}

static int run(const json & config) {
    const fs::path output = fs::weakly_canonical(config.at("output").get<std::string>());
    const std::string out = output.string() + "/";
    if (out.find("/home/cuichengshang/moe_ssd/test/") != 0 && out.find("/ssd/test/") != 0) {
        throw std::runtime_error("output must be under moe_ssd/test (or its /ssd/test container mount)");
    }
    if (fs::exists(output)) { throw std::runtime_error("refusing to overwrite output directory"); }
    fs::create_directories(output);
    const auto baseline = config.value("baseline", "resident");
    const auto action = config.value("action", "replay");
    if (baseline != "resident" && baseline != "mmap" && baseline != "v2" && baseline != "async") {
        throw std::runtime_error("invalid baseline");
    }
    if (action != "freeze" && action != "replay" && action != "generate") { throw std::runtime_error("invalid action"); }
    const bool diagnostic = config.value("diagnostic", false);
    const bool verify = config.value("verify_weights", false);
    if (verify && !diagnostic) { throw std::runtime_error("weight verification requires diagnostic mode"); }
    env_flag("GGML_MOE_ASYNC_DEMAND", baseline == "async");
    env_flag("GGML_MOE_MMAP_ONLY", baseline == "mmap");
    env_flag("GGML_MOE_VERIFY_WEIGHTS", verify);
    env_flag("GGML_MOE_FORCE_GPU", config.value("gpu_experts", true));

    if (!diagnostic) {
        llama_log_set(quiet_log, nullptr);
        ggml_log_set(quiet_log, nullptr);
    }

    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = config.value("gpu_layers", 99);
    mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    mp.ssd_moe_offload = baseline != "resident";
    mp.ssd_moe_cache_size = config.value("cache_mib", size_t(8192)) * 1024 * 1024;
    mp.ssd_moe_cache_workers = config.value("workers", 4u);
    mp.ssd_moe_stats = config.value("ssd_stats", false);
    mp.lazy_mode = mp.ssd_moe_offload ? LLAMA_LAZY_MODE_ON : LLAMA_LAZY_MODE_OFF;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    if (!mp.ssd_moe_cache_size || !mp.ssd_moe_cache_workers) { throw std::runtime_error("cache/workers must be positive"); }
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(config.at("model").get<std::string>().c_str(), mp), llama_model_free);
    if (!model) { throw std::runtime_error("model load failed"); }
    char arch[128] = {};
    llama_model_meta_val_str(model.get(), "general.architecture", arch, sizeof(arch));
    if (std::string(arch) != "qwen3moe") { throw std::runtime_error("M0 template and trace audit target Qwen3 MoE"); }
    const auto * vocab = llama_model_get_vocab(model.get());
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_predict = config.value("n_predict", 128);
    const int batch_size = config.value("batch", 32);
    if (n_predict <= 0 || batch_size <= 0) { throw std::runtime_error("n_predict/batch must be positive"); }
    json requests = read_json(config.at("input").get<std::string>()).at("requests");
    json frozen = json::array(), summaries = json::array();
    for (auto request : requests) {
        const std::string id = request.at("id");
        if (id.empty() || id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") != std::string::npos) {
            throw std::runtime_error("unsafe request id");
        }
        const fs::path dir = output / id;
        fs::create_directory(dir);
        std::cerr << "M0 request " << id << " baseline=" << baseline << " action=" << action << '\n';
        std::vector<llama_token> prompt, continuation;
        if (request.contains("prompt_tokens")) {
            prompt = request.at("prompt_tokens").get<std::vector<llama_token>>();
        } else {
            const std::string body = request.value("body", "");
            const size_t limit = config.value("prompt_tokens", size_t(256));
            size_t lo = 0, hi = body.size();
            while (lo < hi) {
                const size_t mid = (lo + hi + 1) / 2;
                if (tokenize(vocab, render(request, mid)).size() <= limit) { lo = mid; } else { hi = mid - 1; }
            }
            request["rendered_prompt"] = render(request, lo);
            prompt = tokenize(vocab, request["rendered_prompt"]);
        }
        if (action == "replay") { continuation = request.at("continuation_tokens").get<std::vector<llama_token>>(); }
        const int steps = action == "replay" ? (int) continuation.size() : n_predict;
        if (prompt.empty() || steps <= 0) { throw std::runtime_error("empty frozen sequence"); }
        for (const auto * sequence : { &prompt, &continuation }) {
            for (auto token : *sequence) {
                if (token < 0 || token >= n_vocab) { throw std::runtime_error("token outside vocabulary"); }
            }
        }
        auto cp = llama_context_default_params();
        cp.n_ctx = prompt.size() + steps + 32;
        cp.n_batch = batch_size;
        cp.n_ubatch = batch_size;
        cp.n_threads = config.value("threads", 4);
        cp.n_threads_batch = cp.n_threads;
        cp.no_perf = false;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx) { throw std::runtime_error("context init failed"); }
        if (diagnostic && !ggml_backend_moe_trace_open((dir / "trace.jsonl").c_str(), id.c_str())) {
            throw std::runtime_error("trace open failed");
        }
        const bool save_logits = config.value("save_logits", diagnostic || action == "freeze");
        std::ofstream logits_out;
        if (save_logits) { logits_out.open(dir / "logits.f32", std::ios::binary); }
        std::vector<double> decode_ms;
        std::vector<double> token_ms;
        std::vector<llama_token> generated;
        int64_t eval = 0;
        auto decode = [&](llama_token * tokens, int count, int position) {
            ggml_backend_moe_trace_step(eval++, position, count);
            const auto start = clock_type::now();
            if (llama_decode(ctx.get(), llama_batch_get_one(tokens, count)) != 0) { throw std::runtime_error("decode failed"); }
            llama_synchronize(ctx.get());
            return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
        };
        const auto begin = clock_type::now();
        for (size_t p = 0; p < prompt.size(); p += batch_size) {
            decode(prompt.data() + p, std::min<size_t>(batch_size, prompt.size() - p), p);
        }
        const double prefill_ms = std::chrono::duration<double, std::milli>(clock_type::now() - begin).count();
        double ttft_ms = 0;
        auto last_ready = begin;
        for (int step = 0; step < steps; ++step) {
            const float * logits = llama_get_logits_ith(ctx.get(), -1);
            for (int v = 0; v < n_vocab; ++v) {
                if (!std::isfinite(logits[v])) { throw std::runtime_error("non-finite logits"); }
            }
            const llama_token greedy = std::max_element(logits, logits + n_vocab) - logits;
            generated.push_back(greedy);
            const auto ready = clock_type::now();
            if (step == 0) { ttft_ms = std::chrono::duration<double, std::milli>(ready - begin).count(); }
            else { token_ms.push_back(std::chrono::duration<double, std::milli>(ready - last_ready).count()); }
            last_ready = ready;
            if (save_logits) {
                logits_out.write((const char *) logits, n_vocab * sizeof(float));
                if (!logits_out) { throw std::runtime_error("logits write failed"); }
            }
            llama_token next = action == "replay" ? continuation[step] : greedy;
            if (step + 1 == steps || (action != "replay" && llama_vocab_is_eog(vocab, next))) { break; }
            decode_ms.push_back(decode(&next, 1, prompt.size() + step));
        }
        const double wall_ms = std::chrono::duration<double, std::milli>(clock_type::now() - begin).count();
        ctx.reset();
        if (diagnostic) { ggml_backend_moe_trace_close(); }
        request["prompt_tokens"] = prompt;
        if (action != "replay") { request["continuation_tokens"] = generated; }
        frozen.push_back(request);
        long peak_rss_kib = 0;
#if defined(__linux__)
        struct rusage usage = {};
        if (getrusage(RUSAGE_SELF, &usage) == 0) { peak_rss_kib = usage.ru_maxrss; }
#endif
        json summary = {{"id", id}, {"baseline", baseline}, {"action", action}, {"diagnostic", diagnostic},
            {"verify_weights", verify}, {"prompt_tokens", prompt.size()}, {"generated_tokens", generated.size()},
            {"greedy_tokens", generated}, {"n_vocab", n_vocab}, {"layers", llama_model_n_layer(model.get())},
            {"prefill_ms", prefill_ms}, {"ttft_ms", ttft_ms}, {"decode_ms", decode_ms}, {"token_ms", token_ms}, {"wall_ms", wall_ms},
            {"peak_process_rss_kib", peak_rss_kib}, {"app_cache", "new_context_each_request"},
            {"os_page_cache", "uncontrolled_no_drop_caches"}, {"logits_saved", save_logits},
            {"device_timeline", diagnostic ? "CUDA events correlated by eval/tensor/offset; synchronized diagnostic" : "not collected"}};
        write_json(dir / "metrics.json", summary);
        summaries.push_back(summary);
    }
    write_json(output / "sequences.json", {{"template", "qwen3_chatml_no_think_v1"}, {"requests", frozen}});
    write_json(output / "metrics.json", summaries);
    write_json(output / "config.json", config);
    model.reset();
    llama_backend_free();
    return 0;
}

int main(int argc, char ** argv) {
    if (argc != 2 || std::string(argv[1]) == "--help") {
        std::cout << "Usage: llama-moe-m0 CONFIG.json\nSee docs/ssd-moe-m0.md for the experiment driver.\n";
        return argc == 2 ? 0 : 1;
    }
    try { return run(read_json(argv[1])); }
    catch (const std::exception & e) { std::cerr << "M0 failed: " << e.what() << '\n'; return 1; }
}
