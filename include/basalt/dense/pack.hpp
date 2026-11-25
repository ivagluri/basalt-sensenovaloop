// include/basalt/dense/pack.hpp - a dense pack's tensors read into dense::Weights on one card, for a dense model's adapter:
// its geometry from the pack's metadata (`<prefix>.*`, the qwen35 keys), each card's weights in one device allocation, the
// token embedding in page-locked RAM, every range read in the start's one pass (core::LoadPlan).
//
// The tensor names are the dense packs' GGUF names: blk.N.{attn_norm, post_attention_norm, attn_qkv, attn_gate, ssm_out,
// ssm_alpha, ssm_beta, ssm_dt.bias, ssm_a, ssm_conv1d, ssm_norm, attn_q, attn_k, attn_v, attn_output, attn_q_norm,
// attn_k_norm, ffn_gate, ffn_up, ffn_down, hc_attn_fn, hc_ffn_fn}, output, output_norm, token_embd, mtp.*.
#pragma once

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/core/loader.hpp"
#include "basalt/dense/model.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <string>

namespace basalt::dense {

constexpr int kPackF32 = 0, kPackQ8_0 = 8, kPackBf16 = 30, kPackNvfp4 = 40, kPackF8 = 44, kPackNvfp4Planes = 45;

/// Why `<base>.weight`, of tensor type `type` and rows x cols, is not a matrix the dense kernels read; empty when it is.
/// `area` names the model in the message.
std::string linear_type_error(const std::string& area, const std::string& base, uint32_t type, int64_t rows, int64_t cols);

/// The dense stack's geometry from `<prefix>.*` metadata (block_count stored layers, their kinds from
/// attention.recurrent_layers or full_attention_interval, interleaved M-RoPE sections); empty `err` when it is whole.  A
/// loop or hyper-connections are the model's to add.
Geometry pack_geometry(const GgufFile& g, const std::string& prefix, std::string& err);

class PackReader {
public:
    /// `area`: the model's name in messages.  Reads are queued in `plan` against its file `file`.
    PackReader(const GgufFile& gg, std::string area, core::LoadPlan& plan, int file);
    ~PackReader();
    PackReader(const PackReader&) = delete;
    PackReader& operator=(const PackReader&) = delete;

    /// One allocation on the current device for every tensor `here` names (256-byte aligned each), carved by the reads.
    bool reserve(const std::function<bool(const std::string&)>& here, std::string& err);
    /// The token embedding (BF16 [rows][cols]) in page-locked, device-mapped RAM, read in the pass: its device pointer.
    const uint16_t* host_embedding(const std::string& name, int64_t rows, int64_t cols, std::string& err);
    const float* floats(const std::string& name, int64_t elems, std::string& err);
    const uint16_t* bf16(const std::string& name, int64_t elems, std::string& err);
    /// Two BF16 tensors in one buffer, the first then the second.
    const uint16_t* bf16_pair(const std::string& a, const std::string& b, int64_t elems, std::string& err);
    /// `<base>.weight` (Q8_0, NVFP4_PLANES or F8_E4M3, rows x cols) and for the last two its tensor scale `<base>.scale`
    /// (F32 [1]; the output head's F8 may carry a scale per row).
    bool linear(const std::string& base, int64_t rows, int64_t cols, Linear& l, std::string& err);
    /// Stored layer `il` of kind `kind`, with its hyper-connection routers when the pack has them (hc enter / leave are
    /// the model's to set).
    bool layer(const Geometry& g, int il, Mixer kind, Layer& L, std::string& err);
    /// The MTP drafter layer (mtp.*).
    bool mtp(const Geometry& g, Mtp& d, std::string& err);
    bool has(const std::string& name) const { return tensors_.count(name) != 0; }
    uint64_t device_bytes() const { return dev_bytes_; }
    uint64_t host_bytes() const { return host_bytes_; }
    size_t count() const { return n_tensors_; }

private:
    const TensorInfo* need(const std::string& name, int type, int64_t elems, std::string& err) const;
    void* take(uint64_t bytes, const std::string& name, std::string& err);
    void* upload(const TensorInfo& t, std::string& err);
    void queue(const TensorInfo& t, void* dst);

    const GgufFile& gg_;
    std::string area_;
    core::LoadPlan& plan_;
    int file_ = 0;
    std::map<std::string, TensorInfo> tensors_;
    struct Block {
        uint8_t* base = nullptr;
        uint64_t size = 0, used = 0;
    };
    std::map<int, Block> blocks_;   // per CUDA device
    void* host_ = nullptr;          // the page-locked embedding
    uint64_t dev_bytes_ = 0, host_bytes_ = 0;
    size_t n_tensors_ = 0;
};

}  // namespace basalt::dense
