// src/models/sensenova_loop/meta.hpp - what a sensenova_loop pack's metadata says about the model: free of the runtime, so a
// host test can check it without a GPU.
//
// SenseNova Loop is the qwen35 dense hybrid (GDN and gated attention layers, dense SwiGLU) with a loop: its stored layers
// [loop.start, loop.end) run loop.count times in a row, each visit with its own cells and recurrent state, and two-stream
// hyper-connections (mhc.*) around the layers [mhc.start, mhc.end), opened at the first of them and closed at the last on
// every visit.
#pragma once

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/dense/model.hpp"
#include "basalt/dense/pack.hpp"
#include "basalt/model/adapter.hpp"

#include <string>
#include <vector>

namespace basalt::models::sensenova_loop {

constexpr const char* kArch = "sensenova_loop";

inline const MetaValue* key(const GgufFile& g, const char* suffix) { return g.get(std::string(kArch) + "." + suffix); }
inline uint64_t key_u(const GgufFile& g, const char* suffix) {
    const MetaValue* v = key(g, suffix);
    return v ? v->u : 0;
}

inline std::string validate_meta(const GgufFile& g) {
    const MetaValue* arch = g.get("general.architecture");
    if (!arch) return "missing general.architecture";
    if (arch->s != kArch) return "architecture is '" + arch->s + "', this adapter requires '" + kArch + "'";
    for (const char* k : {"block_count", "embedding_length", "feed_forward_length", "attention.head_count",
                          "attention.head_count_kv", "attention.key_length", "rope.dimension_count", "ssm.state_size",
                          "ssm.group_count", "ssm.time_step_rank", "ssm.conv_kernel", "loop.end", "loop.count", "mhc.end",
                          "mhc.streams", "mhc.iterations"})
        if (key_u(g, k) == 0) return std::string("missing or zero ") + kArch + "." + k;
    if (!g.get("tokenizer.ggml.tokens")) return "missing tokenizer.ggml.tokens";
    return {};
}

inline model::ModelDesc describe_meta(const GgufFile& g) {
    model::ModelDesc d;
    d.arch = kArch;
    d.n_layers = (int64_t) key_u(g, "block_count");
    d.n_embd = (int64_t) key_u(g, "embedding_length");
    d.context_length = (int64_t) key_u(g, "context_length");
    if (const MetaValue* t = g.get("tokenizer.ggml.tokens")) d.n_vocab = (int64_t) t->count;
    return d;
}

/// What the pack's loop and hyper-connection keys say, before they are laid over the stored layers.
struct Loop {
    int start = 0, end = 0, count = 1;       ///< the stored layers [start, end) run count times
    int hc_start = 0, hc_end = 0;            ///< the hyper-connections span the stored layers [hc_start, hc_end)
    std::vector<dense::Mixer> stored;        ///< each stored layer's kind
};

/// The dense stack's geometry: the stored layers' (dense::pack_geometry), then the loop as the visits' plan and the
/// hyper-connections; `loop` gets what the keys say.  Empty `err` when it is whole.
inline dense::Geometry geometry(const GgufFile& g, Loop& loop, std::string& err) {
    dense::Geometry o = dense::pack_geometry(g, kArch, err);
    if (!err.empty()) return o;
    const int n = o.n_layers;
    loop.start = (int) key_u(g, "loop.start");
    loop.end = (int) key_u(g, "loop.end");
    loop.count = (int) key_u(g, "loop.count");
    loop.hc_start = (int) key_u(g, "mhc.start");
    loop.hc_end = (int) key_u(g, "mhc.end");
    loop.stored = o.kinds;
    if (!(loop.start < loop.end && loop.end <= n && loop.count >= 1 && loop.count <= 8)) {
        err = "the loop's layers or pass count are out of range";
        return o;
    }
    // the hyper-connections inside one phase of the loop (before it, in it, after it), so each visit opens and closes them
    const int phases[4] = {0, loop.start, loop.end, n};
    bool inside = false;
    for (int p = 0; p < 3; ++p) inside |= phases[p] <= loop.hc_start && loop.hc_start < loop.hc_end && loop.hc_end <= phases[p + 1];
    if (!inside) {
        err = "the hyper-connections cross a phase of the loop";
        return o;
    }
    if (key_u(g, "mhc.streams") != 2) {
        err = "hyper-connections of " + std::to_string((unsigned long long) key_u(g, "mhc.streams")) + " streams: two only";
        return o;
    }
    if (const MetaValue* f = key(g, "mhc.fixed_identity"); f && f->u != 0) {
        err = "hyper-connections with a fixed identity mix are not run";
        return o;
    }
    o.hc_streams = 2;
    o.hc_iters = (int) key_u(g, "mhc.iterations");
    if (const MetaValue* s = key(g, "mhc.pre_scale")) o.hc_pre_scale = (float) s->num();
    o.physical.clear();
    for (int il = 0; il < loop.start; ++il) o.physical.push_back(il);
    for (int c = 0; c < loop.count; ++c)
        for (int il = loop.start; il < loop.end; ++il) o.physical.push_back(il);
    for (int il = loop.end; il < n; ++il) o.physical.push_back(il);
    o.n_layers = (int) o.physical.size();
    o.kinds.clear();
    for (int p : o.physical) o.kinds.push_back(loop.stored[(size_t) p]);
    err = o.check();
    return o;
}

}  // namespace basalt::models::sensenova_loop
