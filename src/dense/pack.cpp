// src/dense/pack.cpp - a dense pack's tensors into dense::Weights (see basalt/dense/pack.hpp).
#include "basalt/dense/pack.hpp"

#include "basalt/kernels/mhc.hpp"
#include "basalt/kernels/nvfp4_planes.hpp"

#include <cuda_runtime.h>

#include <utility>

namespace basalt::dense {

std::string linear_type_error(const std::string& area, const std::string& base, uint32_t type, int64_t rows, int64_t cols) {
    if (type == kPackNvfp4)
        return area + ": " + base + ".weight is NVFP4 in ggml blocks, from a pack older than NVFP4_PLANES: re-convert the pack "
               "(tools/convert/convert.py --checkpoint, docs/model-file.md)";
    if (type != kPackQ8_0 && type != kPackNvfp4Planes && type != kPackF8)
        return area + ": " + base + ".weight is " + ggml_type_name(type) + "; the dense kernels read Q8_0, NVFP4_PLANES and F8_E4M3";
    if (type == kPackNvfp4Planes && !kernels::nvfp4p::fits(rows, cols))
        return area + ": " + base + ".weight: NVFP4_PLANES of " + std::to_string(rows) + " x " + std::to_string(cols) +
               " (planes need rows in 128s and 64-value groups)";
    if (type == kPackF8 && cols % 256 != 0)
        return area + ": " + base + ".weight: F8_E4M3 rows of " + std::to_string(cols) + " values (the kernels read multiples of 256)";
    return {};
}

Geometry pack_geometry(const GgufFile& g, const std::string& prefix, std::string& err) {
    const auto key = [&](const char* k) { return g.get(prefix + "." + k); };
    const auto u = [&](const char* k) -> uint64_t { const MetaValue* v = key(k); return v ? v->u : 0; };
    Geometry o;
    o.n_layers = (int) u("block_count");
    o.n_embd = (int) u("embedding_length");
    o.n_ff = (int) u("feed_forward_length");
    if (const MetaValue* t = g.get("tokenizer.ggml.tokens")) o.n_vocab = (int) t->count;
    o.n_head = (int) u("attention.head_count");
    o.n_kv = (int) u("attention.head_count_kv");
    o.head_dim = (int) u("attention.key_length");
    o.context_length = (int64_t) u("context_length");
    if (const MetaValue* e = key("attention.layer_norm_rms_epsilon")) o.eps = (float) e->num();
    o.rope.rot = (int) u("rope.dimension_count");
    if (const MetaValue* f = key("rope.freq_base")) o.rope.theta = (float) f->num();
    // interleaved M-RoPE: pair i takes h when i % 3 == 1 and i < 3 sections[1], w when i % 3 == 2 and i < 3 sections[2]
    int sec[3] = {0, 0, 0};
    if (const MetaValue* s = key("rope.dimension_sections"))
        for (size_t i = 0; i < 3 && i < s->items.size(); ++i) sec[i] = (int) s->items[i].u;
    if (sec[0] + sec[1] + sec[2] != o.rope.rot / 2) err = "rope.dimension_sections do not cover the rotary width";
    for (int i = 0; i < o.rope.rot / 2 && i < 64; ++i)
        o.rope.axis[i] = (int8_t) (i % 3 == 1 && i < 3 * sec[1] ? 1 : i % 3 == 2 && i < 3 * sec[2] ? 2 : 0);
    o.gdn_state = (int) u("ssm.state_size");
    o.gdn_k_heads = (int) u("ssm.group_count");
    o.gdn_v_heads = (int) u("ssm.time_step_rank");
    o.gdn_conv = (int) u("ssm.conv_kernel");
    o.gdn_gate = kernels::GdnGate::Silu;
    const int interval = (int) u("full_attention_interval");
    const MetaValue* rec = key("attention.recurrent_layers");
    if (!rec && interval <= 0 && err.empty()) err = "neither attention.recurrent_layers nor full_attention_interval";
    for (int il = 0; il < o.n_layers; ++il) {
        const bool recurrent = rec && rec->items.size() == (size_t) o.n_layers ? rec->items[(size_t) il].u != 0
                                                                                : interval > 0 && (il + 1) % interval != 0;
        o.kinds.push_back(recurrent ? Mixer::Gdn : Mixer::Attention);
    }
    if (err.empty()) err = o.check();
    return o;
}

PackReader::PackReader(const GgufFile& gg, std::string area, core::LoadPlan& plan, int file)
    : gg_(gg), area_(std::move(area)), plan_(plan), file_(file) {
    for (const TensorInfo& t : gg.tensors()) tensors_[t.name] = t;
}

PackReader::~PackReader() {
    for (auto& [dev, b] : blocks_) cudaFree(b.base);
    if (host_) cudaFreeHost(host_);
}

bool PackReader::reserve(const std::function<bool(const std::string&)>& here, std::string& err) {
    uint64_t bytes = 0;
    for (const TensorInfo& t : gg_.tensors())
        if (here(t.name)) bytes += (tensor_payload_bytes(t) + 255) & ~(uint64_t) 255;
    int dev = 0;
    cudaGetDevice(&dev);
    Block& b = blocks_[dev];
    if (b.base || cudaMalloc((void**) &b.base, bytes) != cudaSuccess) {
        (void) cudaGetLastError();
        if (!b.base) blocks_.erase(dev);
        err = area_ + ": out of device memory for " + std::to_string((unsigned long long) (bytes >> 20)) +
              " MiB of weights on CUDA" + std::to_string(dev);
        return false;
    }
    b.size = bytes;
    return true;
}

const TensorInfo* PackReader::need(const std::string& name, int type, int64_t elems, std::string& err) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        err = area_ + ": the pack has no " + name;
        return nullptr;
    }
    if ((int) it->second.type != type || (int64_t) it->second.elements() != elems) {
        err = area_ + ": " + name + " is not the expected type and size";
        return nullptr;
    }
    return &it->second;
}

void* PackReader::take(uint64_t bytes, const std::string& name, std::string& err) {
    int dev = 0;
    cudaGetDevice(&dev);
    const auto it = blocks_.find(dev);
    const uint64_t need = (bytes + 255) & ~(uint64_t) 255;
    if (it == blocks_.end() || it->second.used + need > it->second.size) {
        err = area_ + ": " + name + " does not fit the weights' block on CUDA" + std::to_string(dev);
        return nullptr;
    }
    void* d = it->second.base + it->second.used;
    it->second.used += need;
    ++n_tensors_;
    return d;
}

void PackReader::queue(const TensorInfo& t, void* dst) {
    core::LoadItem it;
    it.file = file_;
    it.off = gg_.data_start() + t.offset;
    it.bytes = tensor_payload_bytes(t);
    it.dev = dst;
    cudaGetDevice(&it.device);
    plan_.add(it);
}

void* PackReader::upload(const TensorInfo& t, std::string& err) {
    void* d = take(tensor_payload_bytes(t), t.name, err);
    if (!d) return nullptr;
    queue(t, d);
    dev_bytes_ += tensor_payload_bytes(t);
    return d;
}

const uint16_t* PackReader::host_embedding(const std::string& name, int64_t rows, int64_t cols, std::string& err) {
    const TensorInfo* t = need(name, kPackBf16, rows * cols, err);
    if (!t) return nullptr;
    if (host_ || cudaHostAlloc(&host_, tensor_payload_bytes(*t), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        err = area_ + ": out of page-locked memory for the token embedding";
        return nullptr;
    }
    core::LoadItem it;
    it.file = file_;
    it.off = gg_.data_start() + t->offset;
    it.bytes = tensor_payload_bytes(*t);
    it.host = (uint8_t*) host_;
    plan_.add(it);
    host_bytes_ = tensor_payload_bytes(*t);
    void* dev = nullptr;
    cudaHostGetDevicePointer(&dev, host_, 0);
    return (const uint16_t*) dev;
}

const float* PackReader::floats(const std::string& name, int64_t elems, std::string& err) {
    const TensorInfo* t = need(name, kPackF32, elems, err);
    return t ? (const float*) upload(*t, err) : nullptr;
}

const uint16_t* PackReader::bf16(const std::string& name, int64_t elems, std::string& err) {
    const TensorInfo* t = need(name, kPackBf16, elems, err);
    return t ? (const uint16_t*) upload(*t, err) : nullptr;
}

const uint16_t* PackReader::bf16_pair(const std::string& a, const std::string& b, int64_t elems, std::string& err) {
    const TensorInfo* ta = need(a, kPackBf16, elems, err);
    const TensorInfo* tb = ta ? need(b, kPackBf16, elems, err) : nullptr;
    if (!tb) return nullptr;
    void* d = take(tensor_payload_bytes(*ta) + tensor_payload_bytes(*tb), a, err);
    if (!d) return nullptr;
    uint64_t at = 0;
    for (const TensorInfo* t : {ta, tb}) {
        queue(*t, (char*) d + at);
        at += tensor_payload_bytes(*t);
    }
    dev_bytes_ += at;
    return (const uint16_t*) d;
}

bool PackReader::linear(const std::string& base, int64_t rows, int64_t cols, Linear& l, std::string& err) {
    const auto it = tensors_.find(base + ".weight");
    if (it == tensors_.end()) {
        err = area_ + ": the pack has no " + base + ".weight";
        return false;
    }
    const int type = (int) it->second.type;
    if (err = linear_type_error(area_, base, it->second.type, rows, cols); !err.empty()) return false;
    const TensorInfo* t = need(base + ".weight", type, rows * cols, err);
    if (!t || (int64_t) t->shape[0] != cols) {
        if (err.empty()) err = area_ + ": " + base + ".weight has the wrong shape";
        return false;
    }
    l.type = type;
    l.rows = rows;
    l.cols = cols;
    l.w = upload(*t, err);
    if (!l.w) return false;
    if (type == kPackNvfp4Planes || type == kPackF8) {
        const auto sc = tensors_.find(base + ".scale");
        if (type == kPackF8 && base == "output" && sc != tensors_.end() && (int64_t) sc->second.elements() == rows && rows > 1) {
            l.row_scales = floats(base + ".scale", rows, err);
            return l.row_scales != nullptr;
        }
        l.scale = floats(base + ".scale", 1, err);
        if (!l.scale) return false;
    }
    return true;
}

bool PackReader::layer(const Geometry& g, int il, Mixer kind, Layer& L, std::string& err) {
    const int H = g.n_embd;
    const int C = g.conv_channels(), VD = g.value_width(), HV = g.gdn_v_heads, AQ = g.n_head * g.head_dim;
    const int KV = g.n_kv * g.head_dim;
    const std::string p = "blk." + std::to_string(il) + ".";
    L.kind = kind;
    L.attn_norm = floats(p + "attn_norm.weight", H, err);
    L.ffn_norm = floats(p + "post_attention_norm.weight", H, err);
    if (!L.attn_norm || !L.ffn_norm) return false;
    if (kind == Mixer::Gdn) {
        if (!linear(p + "attn_qkv", C, H, L.qkv, err) || !linear(p + "attn_gate", VD, H, L.z, err) ||
            !linear(p + "ssm_out", H, VD, L.out, err))
            return false;
        L.w_ab = bf16_pair(p + "ssm_alpha.weight", p + "ssm_beta.weight", (int64_t) HV * H, err);
        L.dt = floats(p + "ssm_dt.bias", HV, err);
        L.ssm_a = floats(p + "ssm_a", HV, err);
        L.conv = floats(p + "ssm_conv1d.weight", (int64_t) C * g.gdn_conv, err);
        L.gamma = floats(p + "ssm_norm.weight", g.gdn_state, err);
        if (!L.w_ab || !L.dt || !L.ssm_a || !L.conv || !L.gamma) return false;
    } else {
        if (!linear(p + "attn_q", 2 * AQ, H, L.q, err) || !linear(p + "attn_k", KV, H, L.k, err) ||
            !linear(p + "attn_v", KV, H, L.v, err) || !linear(p + "attn_output", H, AQ, L.o, err))
            return false;
        L.q_norm = floats(p + "attn_q_norm.weight", g.head_dim, err);
        L.k_norm = floats(p + "attn_k_norm.weight", g.head_dim, err);
        if (!L.q_norm || !L.k_norm) return false;
    }
    if (!linear(p + "ffn_gate", g.n_ff, H, L.gate, err) || !linear(p + "ffn_up", g.n_ff, H, L.up, err) ||
        !linear(p + "ffn_down", H, g.n_ff, L.down, err))
        return false;
    if (has(p + "hc_attn_fn.weight") || has(p + "hc_ffn_fn.weight")) {   // the router of each sublayer: [8][2 H] and [8]
        const int64_t n = (int64_t) kernels::kHc2Coef * 2 * H;
        L.hc.attn_w = floats(p + "hc_attn_fn.weight", n, err);
        L.hc.attn_b = L.hc.attn_w ? floats(p + "hc_attn_fn.bias", kernels::kHc2Coef, err) : nullptr;
        L.hc.ffn_w = L.hc.attn_b ? floats(p + "hc_ffn_fn.weight", n, err) : nullptr;
        L.hc.ffn_b = L.hc.ffn_w ? floats(p + "hc_ffn_fn.bias", kernels::kHc2Coef, err) : nullptr;
        if (!L.hc.ffn_b) return false;
    }
    return true;
}

bool PackReader::mtp(const Geometry& g, Mtp& d, std::string& err) {
    const int H = g.n_embd, AQ = g.n_head * g.head_dim, KV = g.n_kv * g.head_dim;
    Layer& L = d.layer;
    L.kind = Mixer::Attention;
    d.norm_e = floats("mtp.pre_fc_norm_embedding.weight", H, err);
    d.norm_h = floats("mtp.pre_fc_norm_hidden.weight", H, err);
    d.norm = floats("mtp.norm.weight", H, err);
    L.attn_norm = floats("mtp.attn_norm.weight", H, err);
    L.ffn_norm = floats("mtp.post_attention_norm.weight", H, err);
    L.q_norm = floats("mtp.attn_q_norm.weight", g.head_dim, err);
    L.k_norm = floats("mtp.attn_k_norm.weight", g.head_dim, err);
    if (!d.norm_e || !d.norm_h || !d.norm || !L.attn_norm || !L.ffn_norm || !L.q_norm || !L.k_norm ||
        !linear("mtp.fc", H, 2 * H, d.fc, err) || !linear("mtp.attn_q", 2 * AQ, H, L.q, err) ||
        !linear("mtp.attn_k", KV, H, L.k, err) || !linear("mtp.attn_v", KV, H, L.v, err) ||
        !linear("mtp.attn_output", H, AQ, L.o, err) || !linear("mtp.ffn_gate", g.n_ff, H, L.gate, err) ||
        !linear("mtp.ffn_up", g.n_ff, H, L.up, err) || !linear("mtp.ffn_down", H, g.n_ff, L.down, err))
        return false;
    d.present = true;
    return true;
}

}  // namespace basalt::dense
