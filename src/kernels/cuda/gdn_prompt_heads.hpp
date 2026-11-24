// src/kernels/cuda/gdn_prompt_heads.hpp - what gdn_prompt_body.cuh defines in each value-head count's namespace
// (gdn_prompt.cu: gdn_hv48, gdn_prompt_hv32.cu: gdn_hv32); the public entry points (gdn_prompt.hpp) pick one.
#pragma once

#include "basalt/kernels/gdn_prompt.hpp"

#include <cstdint>

namespace basalt::kernels {
namespace gdn_hv48 {
void gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T, void* stream);
void conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, void* stream);
void conv(float* history, const uint16_t* qkv, const float* conv_w, uint16_t* h, int64_t T, void* stream);
uint64_t work_bytes(int64_t T);
void recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z, const float* gamma,
                float eps, void* work, float* y, uint16_t* y16, int64_t T, void* stream, GdnGate act, float* yn, uint8_t* y8,
                float* y8_scale);
void recurrence_bf16(float* state, const uint16_t* h, const float* gate, const float* beta, const uint16_t* z,
                     const float* gamma, float eps, void* work, float* y, int64_t T, void* stream, GdnGate act, uint8_t* y8,
                     float* y8_scale, uint16_t* yn16, float* rowmax);
}  // namespace gdn_hv48
namespace gdn_hv32 {
void gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T, void* stream);
void conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, void* stream);
void conv(float* history, const uint16_t* qkv, const float* conv_w, uint16_t* h, int64_t T, void* stream);
uint64_t work_bytes(int64_t T);
void recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z, const float* gamma,
                float eps, void* work, float* y, uint16_t* y16, int64_t T, void* stream, GdnGate act, float* yn, uint8_t* y8,
                float* y8_scale);
void recurrence_bf16(float* state, const uint16_t* h, const float* gate, const float* beta, const uint16_t* z,
                     const float* gamma, float eps, void* work, float* y, int64_t T, void* stream, GdnGate act, uint8_t* y8,
                     float* y8_scale, uint16_t* yn16, float* rowmax);
}  // namespace gdn_hv32
}  // namespace basalt::kernels
