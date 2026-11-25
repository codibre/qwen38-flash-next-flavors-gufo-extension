#include "gsq/kernels/nan_probe.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace gufo::ext::gsq {
namespace {

struct NanBox {
  int n;
  float v[8];
  unsigned long long idx[8];
  int tag[16];
  int layer[16];
  unsigned long long ntok[16];
  unsigned long long pass[16];
};

NanBox* g_box = nullptr;
std::atomic<unsigned long long> g_pass{0};

__global__ void ScanF32Kernel(const float* d, unsigned long long n,
                              NanBox* box, int tagc, int layer,
                              unsigned long long ntok,
                              unsigned long long pass, float huge) {
  const unsigned long long stride =
      static_cast<unsigned long long>(gridDim.x) * blockDim.x;
  for (unsigned long long i = static_cast<unsigned long long>(blockIdx.x) *
                                  blockDim.x +
                              threadIdx.x;
       i < n; i += stride) {
    const float v = d[i];
    if (!__builtin_isfinite(v) || (huge > 0.0f && fabsf(v) > huge)) {
      const int k = atomicAdd(&box->n, 1);
      if (k < 8) {
        box->v[k] = v;
        box->idx[k] = i;
        box->layer[k] = layer;
        box->ntok[k] = ntok;
        box->tag[k] = tagc;
        box->pass[k] = pass;
      }
    }
  }
}

__global__ void ScanF16Kernel(const __half* d, unsigned long long n,
                              NanBox* box, int tagc, int layer,
                              unsigned long long ntok,
                              unsigned long long pass, float huge) {
  const unsigned long long stride =
      static_cast<unsigned long long>(gridDim.x) * blockDim.x;
  for (unsigned long long i = static_cast<unsigned long long>(blockIdx.x) *
                                  blockDim.x +
                              threadIdx.x;
       i < n; i += stride) {
    const float v = __half2float(d[i]);
    if (!__builtin_isfinite(v) || (huge > 0.0f && fabsf(v) > huge)) {
      const int k = atomicAdd(&box->n, 1);
      if (k < 8) {
        box->v[k] = v;
        box->idx[k] = i;
        box->layer[k] = layer;
        box->ntok[k] = ntok;
        box->tag[k] = tagc;
        box->pass[k] = pass;
      }
    }
  }
}

void EnsureBox() {
  static std::atomic<bool> started{false};
  if (started.exchange(true)) return;
  if (hipHostMalloc(&g_box, sizeof(NanBox), 0) != hipSuccess) {
    g_box = nullptr;
    return;
  }
  std::memset(g_box, 0, sizeof(NanBox));
  std::thread([] {
    for (;;) {
      const int n = __atomic_exchange_n(&g_box->n, 0, __ATOMIC_ACQ_REL);
      if (n > 0) {
        const int m = n < 8 ? n : 8;
        for (int k = 0; k < m; ++k) {
          std::fprintf(stderr,
                       "[gsq-nan] %c layer=%d ntok=%llu pass=%llu hit %d: %g "
                       "@ elem %llu\n",
                       g_box->tag[k] ? g_box->tag[k] : '?', g_box->layer[k],
                       g_box->ntok[k], g_box->pass[k], k, g_box->v[k],
                       g_box->idx[k]);
        }
        std::fflush(stderr);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
  }).detach();
}

}  // namespace

bool NanDbgEnabled() {
  static const bool on = std::getenv("GSQ_NAN") != nullptr;
  return on;
}

// Optional magnitude threshold (GSQ_NAN_HUGE): values with |v| over it are
// reported like non-finites, to trace overflow drift layer by layer.
static float NanHugeThreshold() {
  static const float t = [] {
    const char* e = std::getenv("GSQ_NAN_HUGE");
    return e != nullptr ? std::strtof(e, nullptr) : 0.0f;
  }();
  return t;
}

void NanPassEnter() {
  if (NanDbgEnabled()) g_pass.fetch_add(1);
}

void NanScanF32(const float* d, std::size_t n, const char* what, int layer,
                std::uint32_t ntok, hipStream_t stream) {
  if (!NanDbgEnabled() || d == nullptr || n == 0) return;
  EnsureBox();
  if (g_box == nullptr) return;
  ScanF32Kernel<<<2048, 256, 0, stream>>>(
      d, n, g_box, what != nullptr ? what[0] : '?', layer, ntok,
      g_pass.load(), NanHugeThreshold());
}

void NanScanF16(const __half* d, std::size_t n, const char* what, int layer,
                std::uint32_t ntok, hipStream_t stream) {
  if (!NanDbgEnabled() || d == nullptr || n == 0) return;
  EnsureBox();
  if (g_box == nullptr) return;
  ScanF16Kernel<<<2048, 256, 0, stream>>>(
      d, n, g_box, what != nullptr ? what[0] : '?', layer, ntok,
      g_pass.load(), NanHugeThreshold());
}

void NanLaunchCheck(const char* site, int layer) {
  if (!NanDbgEnabled()) return;
  const hipError_t e = hipGetLastError();
  if (e == hipSuccess) return;
  static std::atomic<long> reports{0};
  const long r = reports.fetch_add(1);
  if (r < 24) {
    std::fprintf(stderr, "[gsq-launch] %s layer=%d pass=%llu: %s\n", site,
                 layer, g_pass.load(), hipGetErrorString(e));
    std::fflush(stderr);
  }
}

void NanScanHostReport(const char* what, int layer, std::uint32_t row,
                       const float* v, std::size_t n) {
  static std::atomic<bool> on{false};
  static std::atomic<long> reports{0};
  static std::atomic<long> total{0};
  if (!NanDbgEnabled()) return;
  on.store(true);
  int bad = 0, nonfinite = 0, huge = 0;
  float first = 0.0f;
  std::size_t where = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const float x = v[i];
    if (!__builtin_isfinite(x)) { ++nonfinite; if (!bad) { first = x; where = i; } ++bad; }
    else if (std::fabs(x) > 1e3f) { ++huge; if (!bad) { first = x; where = i; } ++bad; }
  }
  if (bad == 0) return;
  const long t = total.fetch_add(bad);
  const long r = reports.fetch_add(1);
  if (r < 12) {
    std::fprintf(stderr,
                 "[gsq-nan] %s layer=%d row=%u bad=%d (nonfinite=%d huge=%d) "
                 "first=%g @ %zu (cum=%ld)\n",
                 what, layer, row, bad, nonfinite, huge, first, where, t);
    std::fflush(stderr);
  }
}

}  // namespace gufo::ext::gsq
