#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace moe_history {

struct route {
    uint64_t lo = 0, hi = 0;
    void add(int id) {
        if (id < 0 || id >= 128) { throw std::runtime_error("expert outside route mask"); }
        (id < 64 ? lo : hi) |= uint64_t(1) << (id % 64);
    }
    bool has(int id) const { return ((id < 64 ? lo : hi) >> (id % 64)) & 1; }
    int size() const { return __builtin_popcountll(lo) + __builtin_popcountll(hi); }
    double similarity(const route & b) const {
        const int total = size() + b.size();
        return total ? 2.0 * (__builtin_popcountll(lo & b.lo) + __builtin_popcountll(hi & b.hi)) / total : 0;
    }
};

using path = std::vector<route>;
struct request { std::string id; std::vector<path> tokens; };
struct bank {
    int layers = 0, experts = 0, top_k = 0;
    std::vector<request> requests;
    size_t storage_bytes() const {
        size_t bytes = requests.capacity() * sizeof(request);
        for (const auto & req : requests) {
            bytes += req.id.capacity() + req.tokens.capacity() * sizeof(path);
            for (const auto & p : req.tokens) { bytes += p.capacity() * sizeof(route); }
        }
        return bytes;
    }
    explicit bank(const std::string & filename) {
        std::ifstream in(filename);
        std::string magic;
        size_t count = 0;
        if (!(in >> magic >> layers >> experts >> top_k >> count) || magic != "MOE_ROUTE_BANK_V1" ||
                layers < 1 || layers > 128 || experts != 128 || top_k < 1 || top_k > experts || count > 10000) {
            throw std::runtime_error("invalid route bank header: " + filename);
        }
        size_t total = 0;
        for (size_t i = 0; i < count; ++i) {
            request r;
            size_t steps = 0;
            if (!(in >> r.id >> steps) || steps > 100000 || total + steps > 100000) {
                throw std::runtime_error("invalid route bank length");
            }
            total += steps;
            r.tokens.resize(steps, path(layers));
            for (auto & p : r.tokens) {
                for (auto & value : p) {
                    if (!(in >> value.lo >> value.hi) || value.size() != top_k) {
                        throw std::runtime_error("invalid route bank entry");
                    }
                }
            }
            requests.push_back(std::move(r));
        }
        std::string extra;
        if (in >> extra) { throw std::runtime_error("trailing route bank data"); }
    }
};

struct config {
    std::string mode = "ac";
    int distance = 4, span = 4, order = 3, neighbors = 8;
    double alpha = 0.5;
};

class predictor {
    struct neighbor { double score; route target; size_t serial; };
    std::shared_ptr<const bank> data;
    config cfg;
    path current;
    std::vector<std::deque<route>> history;
    int last_layer = -1;

    std::vector<double> vote(std::vector<neighbor> & matches) const {
        std::vector<double> scores(data->experts, 0);
        const size_t n = std::min(matches.size(), size_t(cfg.neighbors));
        std::partial_sort(matches.begin(), matches.begin() + n, matches.end(), [](const neighbor & a, const neighbor & b) {
            return a.score != b.score ? a.score > b.score : a.serial < b.serial;
        });
        double weight = 0;
        for (size_t i = 0; i < n; ++i) {
            const double w = matches[i].score * matches[i].score;
            weight += w;
            for (int e = 0; e < data->experts; ++e) {
                if (matches[i].target.has(e)) { scores[e] += w; }
            }
        }
        if (weight > 0) { for (double & value : scores) { value /= weight; } }
        return scores;
    }

public:
    predictor(std::shared_ptr<const bank> source, config options) : data(std::move(source)), cfg(std::move(options)),
            current(data->layers), history(data->layers) {
        if (cfg.distance < 1 || cfg.distance > data->layers || cfg.span < 1 || cfg.order < 1 ||
                cfg.neighbors < 1 || cfg.alpha < 0 || cfg.alpha > 1 ||
                (cfg.mode != "last" && cfg.mode != "a" && cfg.mode != "c" && cfg.mode != "ac")) {
            throw std::runtime_error("invalid history predictor configuration");
        }
    }

    void observe(int layer, route actual) {
        if (layer == 0) {
            if (last_layer != -1 && last_layer != data->layers - 1) { throw std::runtime_error("incomplete route token"); }
            last_layer = -1;
        }
        if (layer != last_layer + 1 || actual.size() != data->top_k) {
            throw std::runtime_error("non-sequential route observation");
        }
        current[layer] = actual;
        history[layer].push_back(actual);
        if (history[layer].size() > size_t(cfg.order)) { history[layer].pop_front(); }
        last_layer = layer;
    }

    int target_layer() const { return (last_layer + cfg.distance) % data->layers; }
    bool next_token() const { return last_layer + cfg.distance >= data->layers; }

    std::vector<double> predict() const {
        if (last_layer < 0) { throw std::runtime_error("prediction before observation"); }
        const int target = target_layer();
        std::vector<double> spatial(data->experts, 0), temporal(data->experts, 0);
        if (cfg.mode == "last") {
            if (!history[target].empty()) {
                for (int e = 0; e < data->experts; ++e) { temporal[e] = history[target].back().has(e); }
            }
            return temporal;
        }
        if (cfg.mode == "a" || cfg.mode == "ac") {
            std::vector<neighbor> matches;
            size_t serial = 0;
            const int first = std::max(0, last_layer - cfg.span + 1);
            for (const auto & req : data->requests) {
                for (size_t t = 0; t + size_t(next_token()) < req.tokens.size(); ++t) {
                    double similarity = 0;
                    for (int l = first; l <= last_layer; ++l) { similarity += current[l].similarity(req.tokens[t][l]); }
                    similarity /= last_layer - first + 1;
                    matches.push_back({similarity, req.tokens[t + size_t(next_token())][target], serial++});
                }
            }
            spatial = vote(matches);
        }
        const auto & seq = history[target];
        if ((cfg.mode == "c" || cfg.mode == "ac") && !seq.empty()) {
            std::vector<neighbor> matches;
            size_t serial = 0;
            for (const auto & req : data->requests) {
                for (size_t t = seq.size() - 1; t + 1 < req.tokens.size(); ++t) {
                    double similarity = 0, weight = 0;
                    for (size_t lag = 0; lag < seq.size(); ++lag) {
                        const double w = double(seq.size() - lag);
                        similarity += w * seq[seq.size() - lag - 1].similarity(req.tokens[t - lag][target]);
                        weight += w;
                    }
                    matches.push_back({similarity / weight, req.tokens[t + 1][target], serial++});
                }
            }
            temporal = vote(matches);
        }
        if (cfg.mode == "a") { return spatial; }
        if (cfg.mode == "c") { return temporal; }
        const bool has_temporal = std::any_of(temporal.begin(), temporal.end(), [](double x) { return x > 0; });
        for (int e = 0; e < data->experts; ++e) {
            spatial[e] = has_temporal ? cfg.alpha * spatial[e] + (1 - cfg.alpha) * temporal[e] : spatial[e];
        }
        return spatial;
    }

    static std::vector<int> rank(const std::vector<double> & scores) {
        std::vector<int> ids;
        for (size_t e = 0; e < scores.size(); ++e) { if (scores[e] > 0) { ids.push_back(int(e)); } }
        std::sort(ids.begin(), ids.end(), [&](int a, int b) { return scores[a] != scores[b] ? scores[a] > scores[b] : a < b; });
        return ids;
    }
};
} // namespace moe_history
