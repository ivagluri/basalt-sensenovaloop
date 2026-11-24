// src/kernels/cuda/mhc2.cu - two-stream hyper-connections with a log-space Sinkhorn (include/basalt/kernels/mhc.hpp,
// hc2_pre and hc2_merge): the router of a sublayer, the branch it reads and the merge of its output into the streams.
//
// hc2_pre: a block per token.  One pass over the token's 2 n_embd stream values sums their squares and the router's
// eight dot products in fp32 (each thread a fixed set of values, the block's partials summed in warp order, so a token's
// result does not depend on T); one thread then forms the coefficients (sigmoids and the 2x2 Sinkhorn in fp32, expf and
// logf); a second pass writes the branch.  hc2_merge: a thread per eight values of a token.
#include "basalt/kernels/mhc.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

constexpr int kThreads = 256, kWarps = kThreads / 32, kSums = kHc2Coef + 1;   // eight dot products and the sum of squares

__device__ __forceinline__ float bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ void unpack8(const uint4 u, float (&v)[8]) {
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[2 * i] = __uint_as_float(w[i] << 16);
        v[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
}
__device__ __forceinline__ uint4 pack8(const float (&v)[8]) {
    uint32_t w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t lo = __bfloat16_as_ushort(__float2bfloat16_rn(v[2 * i]));
        const uint32_t hi = __bfloat16_as_ushort(__float2bfloat16_rn(v[2 * i + 1]));
        w[i] = lo | (hi << 16);
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
}
__device__ __forceinline__ float lse2(float a, float b) {
    const float m = fmaxf(a, b);
    return m + logf(expf(a - m) + expf(b - m));
}

__global__ void __launch_bounds__(kThreads) hc2_pre_kernel(uint16_t* xs, const uint16_t* x, int enter, const float* __restrict__ w,
                                                           const float* __restrict__ bias, int H, int iters, float eps,
                                                           float pre_scale, float* __restrict__ coef, uint16_t* branch) {
    __shared__ float part[kWarps][kSums];
    __shared__ float cf[kHc2Coef];
    const int64_t t = blockIdx.x;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n8 = 2 * H / 8, h8 = H / 8;
    uint16_t* srow = xs + t * 2 * H;
    const uint16_t* xrow = x + t * H;
    float acc[kSums] = {};
    for (int c = tid; c < n8; c += kThreads) {
        const int e0 = c * 8;
        float v[8];
        unpack8(*reinterpret_cast<const uint4*>(enter ? xrow + (c % h8) * 8 : srow + e0), v);
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[kHc2Coef] = fmaf(v[i], v[i], acc[kHc2Coef]);
#pragma unroll
        for (int j = 0; j < kHc2Coef; ++j) {
            const float4 a = *reinterpret_cast<const float4*>(w + (size_t) j * 2 * H + e0);
            const float4 b = *reinterpret_cast<const float4*>(w + (size_t) j * 2 * H + e0 + 4);
            float d = acc[j];
            d = fmaf(a.x, v[0], d); d = fmaf(a.y, v[1], d); d = fmaf(a.z, v[2], d); d = fmaf(a.w, v[3], d);
            d = fmaf(b.x, v[4], d); d = fmaf(b.y, v[5], d); d = fmaf(b.z, v[6], d); d = fmaf(b.w, v[7], d);
            acc[j] = d;
        }
    }
#pragma unroll
    for (int j = 0; j < kSums; ++j) {
        float v = acc[j];
#pragma unroll
        for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
        if (lane == 0) part[warp][j] = v;
    }
    __syncthreads();
    if (tid == 0) {
        float s[kSums];
#pragma unroll
        for (int j = 0; j < kSums; ++j) {
            s[j] = part[0][j];
            for (int q = 1; q < kWarps; ++q) s[j] += part[q][j];
        }
        const float inv = rsqrtf(s[kHc2Coef] / (float) (2 * H) + eps);
        float l[kHc2Coef];
#pragma unroll
        for (int j = 0; j < kHc2Coef; ++j) l[j] = s[j] * inv + bias[j];
        float m[2][2] = {{l[4], l[5]}, {l[6], l[7]}};
        for (int it = 0; it < iters; ++it) {
            for (int i = 0; i < 2; ++i) {   // rows (over src), then columns (over dst)
                const float r = lse2(m[i][0], m[i][1]);
                m[i][0] -= r; m[i][1] -= r;
            }
            for (int j = 0; j < 2; ++j) {
                const float r = lse2(m[0][j], m[1][j]);
                m[0][j] -= r; m[1][j] -= r;
            }
        }
        float* out = coef + t * kHc2Coef;
        out[0] = 1.0f / (1.0f + expf(-l[0]));
        out[1] = 1.0f / (1.0f + expf(-l[1]));
        out[2] = 2.0f / (1.0f + expf(-l[2]));
        out[3] = 2.0f / (1.0f + expf(-l[3]));
        out[4] = expf(m[0][0]); out[5] = expf(m[0][1]); out[6] = expf(m[1][0]); out[7] = expf(m[1][1]);
#pragma unroll
        for (int j = 0; j < kHc2Coef; ++j) cf[j] = out[j];
    }
    __syncthreads();
    const float p0 = bfr(cf[0]), p1 = bfr(cf[1]);
    for (int c = tid; c < h8; c += kThreads) {
        float r0[8], r1[8], b[8];
        if (enter) {
            const uint4 u = *reinterpret_cast<const uint4*>(xrow + c * 8);
            unpack8(u, r0);
            unpack8(u, r1);
            *reinterpret_cast<uint4*>(srow + c * 8) = u;
            *reinterpret_cast<uint4*>(srow + H + c * 8) = u;
        } else {
            unpack8(*reinterpret_cast<const uint4*>(srow + c * 8), r0);
            unpack8(*reinterpret_cast<const uint4*>(srow + H + c * 8), r1);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            b[i] = fmaf(p1, r1[i], p0 * r0[i]);
            if (pre_scale != 1.0f) b[i] = bfr(b[i]) * pre_scale;
        }
        *reinterpret_cast<uint4*>(branch + t * H + c * 8) = pack8(b);
    }
}

__global__ void __launch_bounds__(kThreads) hc2_merge_kernel(uint16_t* xs, const uint16_t* out, const float* __restrict__ coef,
                                                             int H, int leave, uint16_t* x_mean) {
    const int64_t t = blockIdx.y;
    const int c = blockIdx.x * kThreads + threadIdx.x;
    if (c >= H / 8) return;
    const float* k = coef + t * kHc2Coef;
    const float q0 = bfr(k[2]), q1 = bfr(k[3]);
    const float m00 = bfr(k[4]), m01 = bfr(k[5]), m10 = bfr(k[6]), m11 = bfr(k[7]);
    uint16_t* srow = xs + t * 2 * H;
    float r0[8], r1[8], o[8], n0[8], n1[8];
    unpack8(*reinterpret_cast<const uint4*>(srow + c * 8), r0);
    unpack8(*reinterpret_cast<const uint4*>(srow + H + c * 8), r1);
    unpack8(*reinterpret_cast<const uint4*>(out + t * H + c * 8), o);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        n0[i] = bfr(bfr(fmaf(m01, r1[i], m00 * r0[i])) + bfr(q0 * o[i]));
        n1[i] = bfr(bfr(fmaf(m11, r1[i], m10 * r0[i])) + bfr(q1 * o[i]));
    }
    if (leave) {
        float mean[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) mean[i] = (n0[i] + n1[i]) * 0.5f;
        *reinterpret_cast<uint4*>(x_mean + t * H + c * 8) = pack8(mean);
        return;
    }
    *reinterpret_cast<uint4*>(srow + c * 8) = pack8(n0);
    *reinterpret_cast<uint4*>(srow + H + c * 8) = pack8(n1);
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "%s: %s", what, cudaGetErrorString(e));
        std::abort();
    }
}

}  // namespace

void hc2_pre(uint16_t* xs, const uint16_t* x, bool enter, const float* w, const float* bias, int n_tokens, const Hc2Args& a,
             float* coef, uint16_t* branch, void* stream) {
    if (n_tokens <= 0) return;
    if (a.n_embd <= 0 || a.n_embd % 8 || a.iters < 1) {
        basalt::log::error("basalt", "hc2_pre: n_embd %d, iters %d", a.n_embd, a.iters);
        std::abort();
    }
    hc2_pre_kernel<<<(unsigned) n_tokens, kThreads, 0, (cudaStream_t) stream>>>(xs, x, enter ? 1 : 0, w, bias, a.n_embd, a.iters,
                                                                                a.rms_eps, a.pre_scale, coef, branch);
    check("hc2_pre");
}

void hc2_merge(uint16_t* xs, const uint16_t* out, const float* coef, int n_tokens, int n_embd, bool leave, uint16_t* x_mean,
               void* stream) {
    if (n_tokens <= 0) return;
    if (n_embd <= 0 || n_embd % 8) {
        basalt::log::error("basalt", "hc2_merge: n_embd %d", n_embd);
        std::abort();
    }
    const dim3 grid((unsigned) ((n_embd / 8 + kThreads - 1) / kThreads), (unsigned) n_tokens);
    hc2_merge_kernel<<<grid, kThreads, 0, (cudaStream_t) stream>>>(xs, out, coef, n_embd, leave ? 1 : 0, x_mean);
    check("hc2_merge");
}

}  // namespace basalt::kernels
