// src/kernels/cuda/gdn_prompt.cu - the prompt path's GDN (include/basalt/kernels/gdn_prompt.hpp): gates, conv, and the
// gated delta rule in chunks of 16 tokens on the tensor cores.
// Per chunk and value head, gdn_prepare_kernel builds the L2-normalized BF16 q/k packet, the within-chunk solve
// (I + L)^-1 diag(beta), the decayed q.k products and the cumulative decays; gdn_chunk_kernel then carries the FP32
// state across the chunks: prediction K S and history Q S (TF32 / BF16 MMA), the residual through the solve, the
// output, and S = decay S + K^T diag(suffix) Delta.
// Adapted from Infernix (Ian Ranson and contributors), src/ops/linear_attention/gated_delta_net/chunked/,
// Apache-2.0: see THIRD_PARTY_LICENSES.
#include "basalt/kernels/gdn_prompt.hpp"
#include "gdn_prompt_heads.hpp"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "basalt/log.hpp"

namespace basalt::kernels {

namespace gdn_hv48 {
constexpr int HV = 48;
#include "gdn_prompt_body.cuh"
}  // namespace gdn_hv48

bool gdn_prompt_supported(int state, int k_heads, int v_heads, int conv_taps) {
    return state == 128 && k_heads == 16 && (v_heads == 48 || v_heads == 32) && conv_taps == 4;
}

// Each entry point runs the kernels of the model's value-head count (48: Qwen3.8 27B and Flash-Next; 32: SenseNova Loop's 9B).
void gdn_gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T, void* stream, int v_heads) {
    if (v_heads == 32) gdn_hv32::gates(ab, dt, ssm_a, gate, beta, T, stream);
    else gdn_hv48::gates(ab, dt, ssm_a, gate, beta, T, stream);
}
void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, void* stream, int v_heads) {
    if (v_heads == 32) gdn_hv32::conv(history, qkv, conv_w, h, T, stream);
    else gdn_hv48::conv(history, qkv, conv_w, h, T, stream);
}
void gdn_conv(float* history, const uint16_t* qkv, const float* conv_w, uint16_t* h, int64_t T, void* stream, int v_heads) {
    if (v_heads == 32) gdn_hv32::conv(history, qkv, conv_w, h, T, stream);
    else gdn_hv48::conv(history, qkv, conv_w, h, T, stream);
}
uint64_t gdn_work_bytes(int64_t T, int v_heads) { return v_heads == 32 ? gdn_hv32::work_bytes(T) : gdn_hv48::work_bytes(T); }
void gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, void* work, float* y, uint16_t* y16, int64_t T, void* stream,
                    GdnGate act, float* yn, uint8_t* y8, float* y8_scale, int v_heads) {
    if (v_heads == 32) gdn_hv32::recurrence(state, h, gate, beta, z, gamma, eps, work, y, y16, T, stream, act, yn, y8, y8_scale);
    else gdn_hv48::recurrence(state, h, gate, beta, z, gamma, eps, work, y, y16, T, stream, act, yn, y8, y8_scale);
}
void gdn_recurrence_bf16(float* state, const uint16_t* h, const float* gate, const float* beta, const uint16_t* z,
                         const float* gamma, float eps, void* work, float* y, int64_t T, void* stream, GdnGate act,
                         uint8_t* y8, float* y8_scale, uint16_t* yn16, float* rowmax, int v_heads) {
    if (v_heads == 32)
        gdn_hv32::recurrence_bf16(state, h, gate, beta, z, gamma, eps, work, y, T, stream, act, y8, y8_scale, yn16, rowmax);
    else gdn_hv48::recurrence_bf16(state, h, gate, beta, z, gamma, eps, work, y, T, stream, act, y8, y8_scale, yn16, rowmax);
}

}  // namespace basalt::kernels
