// tests/kernels/gdn_heads_parity.cpp - the prompt path's GDN kernels (gdn_prompt.hpp) at each value-head count they are
// built for (48: Qwen3.8 27B and Flash-Next; 32: SenseNova Loop 9B) against a double-precision CPU reference: the gates,
// the conv (history in and out) and the chunked recurrence (output before the norm and the state after the chunk) for a
// few value heads, among them heads past the key heads (value head h meets key head h % 16).  The gating projection
// (gdn_gating_proj) is checked against the same gates of a CPU product.
// Bars as gdn_chunk_parity's: the chunked recurrence rounds q/k and the update's Delta to BF16.
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/gdn_prompt.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = basalt::kernels;

namespace {

constexpr int S = 128, HK = 16;
constexpr float EPS = 1e-6f;
constexpr double kRms = 1e-2, kWorst = 1.5e-2;

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

struct Err {
    double rms, worst;
};
Err compare(const std::vector<float>& x, const std::vector<double>& r) {
    double d2 = 0, r2 = 0, dm = 0, rm = 0;
    for (size_t i = 0; i < r.size(); ++i) {
        if (!std::isfinite(x[i])) return {NAN, NAN};
        const double d = (double) x[i] - r[i];
        d2 += d * d; r2 += r[i] * r[i];
        dm = std::max(dm, std::fabs(d)); rm = std::max(rm, std::fabs(r[i]));
    }
    return {r2 > 0 ? std::sqrt(d2 / r2) : std::sqrt(d2), rm > 0 ? dm / rm : dm};
}
bool pass(const Err& e) { return std::isfinite(e.rms) && e.rms <= kRms && e.worst <= kWorst; }

// one value head's recurrence in double (gdn_chunk_parity's): q/k normalized, S = g S + k (beta (v - g S^T k))^T,
// o = S^T q / sqrt(S)
void cpu_head(int HV, const std::vector<float>& h, const std::vector<float>& gate, const std::vector<float>& beta,
              const std::vector<float>& st0, int head, int64_t T, std::vector<double>& out, std::vector<double>& st) {
    const int C = 2 * HK * S + HV * S, qh = head % HK;
    st.assign((size_t) S * S, 0.0);
    for (int r = 0; r < S; ++r)
        for (int c = 0; c < S; ++c) st[(size_t) r * S + c] = st0[(size_t) r * HV * S + (size_t) head * S + c];
    out.assign((size_t) T * S, 0.0);
    std::vector<double> q(S), kk(S), kv(S);
    for (int64_t t = 0; t < T; ++t) {
        const float* ht = h.data() + (size_t) t * C;
        double qs = 0, ks = 0;
        for (int i = 0; i < S; ++i) {
            q[i] = ht[qh * S + i]; kk[i] = ht[HK * S + qh * S + i];
            qs += q[i] * q[i]; ks += kk[i] * kk[i];
        }
        const double qi = 1.0 / std::sqrt(qs + EPS), ki = 1.0 / std::sqrt(ks + EPS);
        for (int i = 0; i < S; ++i) { q[i] *= qi; kk[i] *= ki; }
        const double g = std::exp((double) gate[(size_t) t * HV + head]), b = beta[(size_t) t * HV + head];
        for (int c = 0; c < S; ++c) {
            double a = 0;
            for (int r = 0; r < S; ++r) a += st[(size_t) r * S + c] * kk[r];
            kv[c] = a;
        }
        for (int c = 0; c < S; ++c) {
            const double d = (ht[2 * HK * S + head * S + c] - g * kv[c]) * b;
            double o = 0;
            for (int r = 0; r < S; ++r) {
                double& s = st[(size_t) r * S + c];
                s = g * s + kk[r] * d;
                o += s * q[r];
            }
            out[(size_t) t * S + c] = o / std::sqrt((double) S);
        }
    }
}

bool check(int HV, int64_t T, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    const int C = 2 * HK * S + HV * S, H = 4096;
    bool ok = true;
    auto line = [&](const char* what, const Err& e) {
        const bool p = pass(e);
        ok &= p;
        std::printf("HV=%d T=%lld %-34s rel RMS %.2e worst %.2e %s\n", HV, (long long) T, what, e.rms, e.worst, p ? "ok" : "FAIL");
    };

    // gates: from [T][2 HV] a/b rows, and through the fused gating projection of [T][H] rows
    std::vector<float> ab((size_t) T * 2 * HV), dt(HV), sa(HV);
    for (auto& v : ab) v = nd(rng);
    for (int i = 0; i < HV; ++i) { dt[i] = 0.2f * nd(rng); sa[i] = -0.1f - u(rng); }
    std::vector<double> rg((size_t) T * HV), rb((size_t) T * HV);
    for (int64_t t = 0; t < T; ++t)
        for (int i = 0; i < HV; ++i) {
            const double v = (double) ab[(size_t) t * 2 * HV + i] + dt[i];
            rg[(size_t) t * HV + i] = (v > 20.0 ? v : std::log1p(std::exp(v))) * sa[i];
            rb[(size_t) t * HV + i] = 1.0 / (1.0 + std::exp(-(double) ab[(size_t) t * 2 * HV + HV + i]));
        }
    float *dab = dev(ab), *ddt = dev(dt), *dsa = dev(sa);
    float *dg = dev(std::vector<float>((size_t) T * HV)), *dbt = dev(std::vector<float>((size_t) T * HV));
    k::gdn_gates(dab, ddt, dsa, dg, dbt, T, nullptr, HV);
    line("gdn_gates gate", compare(down(dg, (size_t) T * HV), rg));
    line("gdn_gates beta", compare(down(dbt, (size_t) T * HV), rb));
    {
        std::vector<float> x((size_t) T * H), wabf((size_t) 2 * HV * H);
        std::vector<uint16_t> wab(wabf.size());
        for (auto& v : x) v = nd(rng);
        for (size_t i = 0; i < wab.size(); ++i) { wab[i] = to_bf(0.02f * nd(rng)); wabf[i] = from_bf(wab[i]); }
        std::vector<double> pg((size_t) T * HV), pb((size_t) T * HV);
        for (int64_t t = 0; t < T; ++t)
            for (int i = 0; i < HV; ++i) {
                double a = 0, b = 0;
                for (int c = 0; c < H; ++c) {
                    a += (double) x[(size_t) t * H + c] * wabf[(size_t) i * H + c];
                    b += (double) x[(size_t) t * H + c] * wabf[(size_t) (HV + i) * H + c];
                }
                const double v = a + dt[i];
                pg[(size_t) t * HV + i] = (v > 20.0 ? v : std::log1p(std::exp(v))) * sa[i];
                pb[(size_t) t * HV + i] = 1.0 / (1.0 + std::exp(-b));
            }
        k::gdn_gating_proj(dev(x), dev(wab), ddt, dsa, dg, dbt, T, HV, H, nullptr);
        line("gdn_gating_proj gate", compare(down(dg, (size_t) T * HV), pg));
        line("gdn_gating_proj beta", compare(down(dbt, (size_t) T * HV), pb));
    }

    // conv: 4 taps + SiLU over the chunk, the history its last three inputs
    std::vector<float> qkv((size_t) T * C), cw((size_t) C * 4), hist((size_t) C * 3);
    for (auto& v : qkv) v = nd(rng);
    for (auto& v : cw) v = 0.3f * nd(rng);
    for (auto& v : hist) v = nd(rng);
    std::vector<double> rh((size_t) T * C), rhist((size_t) C * 3);
    for (int c = 0; c < C; ++c) {
        auto in = [&](int64_t t) -> double { return t >= 0 ? qkv[(size_t) t * C + c] : hist[(size_t) c * 3 + (size_t) (t + 3)]; };
        for (int64_t t = 0; t < T; ++t) {
            const double s = in(t - 3) * cw[c * 4] + in(t - 2) * cw[c * 4 + 1] + in(t - 1) * cw[c * 4 + 2] + in(t) * cw[c * 4 + 3];
            rh[(size_t) t * C + c] = s / (1.0 + std::exp(-s));
        }
        for (int j = 0; j < 3; ++j) rhist[(size_t) c * 3 + j] = in(T - 3 + j);
    }
    float *dh = dev(hist), *dhc = dev(std::vector<float>((size_t) T * C));
    k::gdn_conv(dh, dev(qkv), dev(cw), dhc, T, nullptr, HV);
    line("gdn_conv output", compare(down(dhc, (size_t) T * C), rh));
    line("gdn_conv history", compare(down(dh, (size_t) C * 3), rhist));

    // the chunked recurrence on the conv's output (q/k not yet normalized), a non-zero state
    const auto hc = down(dhc, (size_t) T * C);
    std::vector<float> gate((size_t) T * HV), beta((size_t) T * HV), z((size_t) T * HV * S), st0((size_t) S * HV * S);
    for (size_t i = 0; i < gate.size(); ++i) { gate[i] = -0.02f - 0.8f * u(rng); beta[i] = u(rng); }
    for (auto& v : z) v = nd(rng);
    for (auto& v : st0) v = 0.05f * nd(rng);
    float *dst = dev(st0), *dy = dev(std::vector<float>((size_t) T * HV * S));
    void* work = nullptr;
    cudaMalloc(&work, k::gdn_work_bytes(T, HV));
    k::gdn_recurrence(dst, dhc, dev(gate), dev(beta), dev(z), dev(std::vector<float>(S, 1.0f)), EPS, work, dy, nullptr, T,
                      nullptr, k::GdnGate::Silu, nullptr, nullptr, nullptr, HV);
    const auto y = down(dy, (size_t) T * HV * S), st = down(dst, (size_t) S * HV * S);
    for (int head : {0, 17, HV - 1}) {
        std::vector<double> ro, rs;
        cpu_head(HV, hc, gate, beta, st0, head, T, ro, rs);
        std::vector<float> yh((size_t) T * S), sh((size_t) S * S);
        for (int64_t t = 0; t < T; ++t)
            for (int c = 0; c < S; ++c) yh[(size_t) t * S + c] = y[((size_t) t * HV + head) * S + c];
        for (int r = 0; r < S; ++r)
            for (int c = 0; c < S; ++c) sh[(size_t) r * S + c] = st[(size_t) r * HV * S + (size_t) head * S + c];
        char what[64];
        std::snprintf(what, sizeof what, "recurrence head %d output", head);
        line(what, compare(yh, ro));
        std::snprintf(what, sizeof what, "recurrence head %d state", head);
        line(what, compare(sh, rs));
    }
    cudaFree(work);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::printf("CUDA error: %s\n", cudaGetErrorString(e));
        return false;
    }
    return ok;
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    bool ok = true;
    for (int hv : {48, 32}) {
        if (!k::gdn_prompt_supported(S, HK, hv, 4)) {
            std::printf("HV=%d not supported\n", hv);
            return 1;
        }
        ok &= check(hv, 100, 7u + (unsigned) hv);   // not a multiple of the 16-token chunk
        ok &= check(hv, 16, 3u + (unsigned) hv);
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
