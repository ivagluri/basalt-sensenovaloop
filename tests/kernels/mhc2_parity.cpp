// tests/kernels/mhc2_parity.cpp - the two-stream hyper-connection kernels (hc2_pre, hc2_merge; kernels/mhc.hpp) against a
// CPU port of the reference's router (SenseNova Loop's MHCRouter: fp32 router of the RMS-normalised streams, log-space
// Sinkhorn, BF16 branch and merge), with and without `enter` / `leave`.  The coefficients agree to fp32 rounding (the
// device's expf / logf against the host's); the BF16 rows may differ by one BF16 step where a coefficient's last bits
// decide a rounding, and only rarely.  Also: a token's rows do not depend on T (the first token alone gives the same bits).
#include "basalt/kernels/mhc.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = basalt::kernels;

namespace {

uint16_t to_bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float from_bf(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float bfr(float v) { return from_bf(to_bf(v)); }
template <class T> T* dev(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template <class T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}
float lse2(float a, float b) {
    const float m = std::fmax(a, b);
    return m + std::log(std::exp(a - m) + std::exp(b - m));
}

// the reference router for one token: coefficients and the branch
void cpu_pre(const float* r /*[2H]*/, const std::vector<float>& w, const std::vector<float>& bias, int H, int iters,
             float* cf, float* branch) {
    double ss = 0;
    for (int e = 0; e < 2 * H; ++e) ss += (double) r[e] * r[e];
    const double inv = 1.0 / std::sqrt(ss / (2.0 * H) + 1e-6);
    float l[8];
    for (int j = 0; j < 8; ++j) {
        double d = 0;
        for (int e = 0; e < 2 * H; ++e) d += (double) w[(size_t) j * 2 * H + e] * r[e];
        l[j] = (float) (d * inv) + bias[j];
    }
    float m[2][2] = {{l[4], l[5]}, {l[6], l[7]}};
    for (int it = 0; it < iters; ++it) {
        for (int i = 0; i < 2; ++i) { const float s = lse2(m[i][0], m[i][1]); m[i][0] -= s; m[i][1] -= s; }
        for (int j = 0; j < 2; ++j) { const float s = lse2(m[0][j], m[1][j]); m[0][j] -= s; m[1][j] -= s; }
    }
    cf[0] = 1.0f / (1.0f + std::exp(-l[0]));
    cf[1] = 1.0f / (1.0f + std::exp(-l[1]));
    cf[2] = 2.0f / (1.0f + std::exp(-l[2]));
    cf[3] = 2.0f / (1.0f + std::exp(-l[3]));
    cf[4] = std::exp(m[0][0]); cf[5] = std::exp(m[0][1]); cf[6] = std::exp(m[1][0]); cf[7] = std::exp(m[1][1]);
    const float p0 = bfr(cf[0]), p1 = bfr(cf[1]);
    for (int i = 0; i < H; ++i) branch[i] = bfr(std::fma(p1, r[H + i], p0 * r[i]));
}
void cpu_merge(float* r /*[2H], in place*/, const float* out, const float* cf, int H, bool leave, float* mean) {
    const float q0 = bfr(cf[2]), q1 = bfr(cf[3]), m00 = bfr(cf[4]), m01 = bfr(cf[5]), m10 = bfr(cf[6]), m11 = bfr(cf[7]);
    for (int i = 0; i < H; ++i) {
        const float a = bfr(bfr(std::fma(m01, r[H + i], m00 * r[i])) + bfr(q0 * out[i]));
        const float b = bfr(bfr(std::fma(m11, r[H + i], m10 * r[i])) + bfr(q1 * out[i]));
        if (leave) mean[i] = bfr((a + b) * 0.5f);
        else { r[i] = a; r[H + i] = b; }
    }
}

struct Diff {
    size_t n = 0, off1 = 0, more = 0;   // values, one BF16 step apart, further apart
};
Diff bf_diff(const std::vector<uint16_t>& got, const std::vector<float>& want) {
    Diff d;
    for (size_t i = 0; i < want.size(); ++i) {
        ++d.n;
        const uint16_t w = to_bf(want[i]);
        if (got[i] == w) continue;
        if (std::abs((int) got[i] - (int) w) == 1) ++d.off1;
        else ++d.more;
    }
    return d;
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    const int H = 4096, T = 7, iters = 20;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> w((size_t) 8 * 2 * H), bias(8);
    for (auto& v : w) v = 0.01f * nd(rng);
    for (auto& v : bias) v = nd(rng);
    // a stream pair per token (differing streams), a residual row, a sublayer output
    std::vector<uint16_t> xs((size_t) T * 2 * H), x((size_t) T * H), out((size_t) T * H);
    for (auto& v : xs) v = to_bf(3.0f * nd(rng));
    for (auto& v : x) v = to_bf(3.0f * nd(rng));
    for (auto& v : out) v = to_bf(nd(rng));
    float *dw = dev(w), *db = dev(bias), *dcf = dev(std::vector<float>((size_t) T * 8));
    bool ok = true;
    k::Hc2Args a;
    a.n_embd = H;
    a.iters = iters;
    for (int enter = 0; enter < 2; ++enter) {
        uint16_t *dxs = dev(xs), *dx = dev(x);
        k::hc2_pre(dxs, dx, enter != 0, dw, db, T, a, dcf, dx, nullptr);   // branch into x (it may alias)
        const auto cf = down(dcf, (size_t) T * 8);
        const auto br = down(dx, (size_t) T * H), sx = down(dxs, (size_t) T * 2 * H);
        double worst = 0;
        std::vector<float> want_br((size_t) T * H), cfr((size_t) T * 8);
        for (int t = 0; t < T; ++t) {
            std::vector<float> r(2 * H);
            for (int e = 0; e < 2 * H; ++e) r[e] = from_bf(enter ? x[(size_t) t * H + e % H] : xs[(size_t) t * 2 * H + e]);
            cpu_pre(r.data(), w, bias, H, iters, &cfr[(size_t) t * 8], &want_br[(size_t) t * H]);
            for (int j = 0; j < 8; ++j)
                worst = std::fmax(worst, std::fabs(cf[t * 8 + j] - cfr[(size_t) t * 8 + j]) / std::fmax(1e-3, std::fabs(cfr[(size_t) t * 8 + j])));
        }
        const Diff d = bf_diff(br, want_br);
        bool streams = true;
        if (enter)
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < H; ++i)
                    streams &= sx[(size_t) t * 2 * H + i] == x[(size_t) t * H + i] && sx[(size_t) t * 2 * H + H + i] == x[(size_t) t * H + i];
        const bool p = worst < 1e-4 && d.more == 0 && d.off1 * 1000 < d.n && streams;
        ok &= p;
        std::printf("hc2_pre%s: coefficients worst rel %.2e; branch %zu values, %zu one BF16 step off, %zu further%s: %s\n",
                    enter ? " (enter)" : "", worst, d.n, d.off1, d.more, enter ? ", streams copied" : "", p ? "ok" : "FAIL");
        // the first token alone: the same bits
        uint16_t *dxs1 = dev(xs), *dx1 = dev(x);
        float* dcf1 = dev(std::vector<float>(8));
        k::hc2_pre(dxs1, dx1, enter != 0, dw, db, 1, a, dcf1, dx1, nullptr);
        const auto b1 = down(dx1, (size_t) H);
        const auto c1 = down(dcf1, 8);
        const bool same = std::memcmp(b1.data(), br.data(), (size_t) H * 2) == 0 && std::memcmp(c1.data(), cf.data(), 32) == 0;
        ok &= same;
        std::printf("hc2_pre%s: token 0 alone gives the same bits: %s\n", enter ? " (enter)" : "", same ? "ok" : "FAIL");
        cudaFree(dxs1); cudaFree(dx1); cudaFree(dcf1);
        cudaFree(dxs); cudaFree(dx);
    }
    for (int leave = 0; leave < 2; ++leave) {
        // coefficients from the device's pre (merge is checked on the same coefficients)
        uint16_t *dxs = dev(xs), *dx = dev(x), *dout = dev(out);
        k::hc2_pre(dxs, dx, false, dw, db, T, a, dcf, dx, nullptr);
        const auto cf = down(dcf, (size_t) T * 8);
        k::hc2_merge(dxs, dout, dcf, T, H, leave != 0, dout, nullptr);   // the mean into out's rows (it may alias)
        std::vector<float> want((size_t) T * (leave ? H : 2 * H));
        for (int t = 0; t < T; ++t) {
            std::vector<float> r(2 * H), o(H), mean(H);
            for (int e = 0; e < 2 * H; ++e) r[e] = from_bf(xs[(size_t) t * 2 * H + e]);
            for (int i = 0; i < H; ++i) o[i] = from_bf(out[(size_t) t * H + i]);
            cpu_merge(r.data(), o.data(), &cf[(size_t) t * 8], H, leave != 0, mean.data());
            if (leave) std::copy(mean.begin(), mean.end(), want.begin() + (size_t) t * H);
            else std::copy(r.begin(), r.end(), want.begin() + (size_t) t * 2 * H);
        }
        const auto got = leave ? down(dout, (size_t) T * H) : down(dxs, (size_t) T * 2 * H);
        const Diff d = bf_diff(got, want);
        const bool p = d.more == 0 && d.off1 * 1000 < d.n;
        ok &= p;
        std::printf("hc2_merge%s: %zu values, %zu one BF16 step off, %zu further: %s\n", leave ? " (leave)" : "", d.n, d.off1,
                    d.more, p ? "ok" : "FAIL");
        cudaFree(dxs); cudaFree(dx); cudaFree(dout);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::printf("CUDA error: %s\n", cudaGetErrorString(e));
        return 1;
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
