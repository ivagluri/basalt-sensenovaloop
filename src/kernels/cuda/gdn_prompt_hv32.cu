// src/kernels/cuda/gdn_prompt_hv32.cu - (32 value heads) the prompt path's GDN (include/basalt/kernels/gdn_prompt.hpp): gates, conv, and the
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

namespace basalt::kernels::gdn_hv32 {
constexpr int HV = 32;
#include "gdn_prompt_body.cuh"
}  // namespace basalt::kernels::gdn_hv32
