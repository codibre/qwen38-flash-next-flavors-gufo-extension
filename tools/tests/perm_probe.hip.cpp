// Standalone probe for the v_perm byte order used by the Q2_0 register table.
//
//   hipcc -O2 --offload-arch=gfx1151 tools/tests/perm_probe.hip.cpp -o /tmp/perm_probe
//   /tmp/perm_probe
//
// No gufo objects, no model: this only asks what __builtin_amdgcn_perm(a, b,
// sel) actually returns per byte, and whether the Q2_0 table built the way
// kernels.hip.cpp builds it yields {-d, +-0, d, 2d}. The routed kernel turned
// every output into one constant NaN, which is a byte-order / operand-order
// question, so it is answered here rather than by guessing at the ISA.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#define CK(x)                                                                \
  do {                                                                       \
    hipError_t e_ = (x);                                                     \
    if (e_ != hipSuccess) {                                                  \
      std::fprintf(stderr, "hip error %s at %d\n", hipGetErrorString(e_), __LINE__); \
      return 1;                                                              \
    }                                                                        \
  } while (0)

// Selector for a pair of table lookups: result byte j takes the byte named by
// selector byte j of the 8-byte source {second operand, first operand}.
__host__ __device__ constexpr std::uint32_t SelFor(int q0, int q1) {
  return static_cast<std::uint32_t>(2 * q0) |
         (static_cast<std::uint32_t>(2 * q0 + 1) << 8) |
         (static_cast<std::uint32_t>(2 * q1) << 16) |
         (static_cast<std::uint32_t>(2 * q1 + 1) << 24);
}

__global__ void ByteOrderProbe(std::uint32_t* out) {
  // Byte values equal their index: sources are b = 0x03020100 and
  // a = 0x07060504, so every result byte spells out the index it came from.
  const std::uint32_t a = 0x07060504U, b = 0x03020100U;
  for (int s = 0; s < 8; ++s)
    out[s] = static_cast<std::uint32_t>(
        __builtin_amdgcn_perm(static_cast<int>(a), static_cast<int>(b), s));
  // The 16 selectors the Q2_0 table actually uses. gfx1151's v_perm_b32 takes
  // ONE selector byte per result byte (a value >= 8 fills the lane with 0xff),
  // and indexes the 8-byte source with the second operand's bytes first:
  // indices 0..3 come from `b`, 4..7 from `a`.
  for (int q0 = 0; q0 < 4; ++q0)
    for (int q1 = 0; q1 < 4; ++q1)
      out[16 + q0 * 4 + q1] = static_cast<std::uint32_t>(__builtin_amdgcn_perm(
          static_cast<int>(0x07060504U), static_cast<int>(0x03020100U),
          static_cast<int>(SelFor(q0, q1))));
}

// Exactly the table construction from kernels.hip.cpp's kQ20 branch.
__global__ void TableProbe(const std::uint16_t* ds_in, std::uint32_t* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const std::uint32_t ds = ds_in[i];
  const std::uint32_t dbl = __half_as_ushort(__hmul(
      __builtin_bit_cast(__half, static_cast<std::uint16_t>(ds)),
      __builtin_bit_cast(__half, static_cast<std::uint16_t>(0x4000))));
  const std::uint32_t tab0 = (ds ^ 0x8000U) | ((ds & 0x8000U) << 16);
  const std::uint32_t tab1 = ds | (dbl << 16);
  // Entry q as a pair: with the corrected operand order, {tab0, tab1} is laid
  // out as indices 0..3 = tab0 (q = 0, 1) and 4..7 = tab1 (q = 2, 3).
  for (int q = 0; q < 4; ++q) {
    const std::uint32_t r = static_cast<std::uint32_t>(__builtin_amdgcn_perm(
        static_cast<int>(tab1), static_cast<int>(tab0),
        static_cast<int>(SelFor(q, q))));
    out[i * 4 + q] = r;
  }
}

namespace {
float HalfToF(std::uint16_t h) { return __half2float(__builtin_bit_cast(__half, h)); }

std::uint16_t Expect(std::uint16_t ds, int q) {
  // The reference the float path produced: (float(q) - 1.0f) * float(d),
  // rounded once into F16.
  const float d = HalfToF(ds);
  const float v = (static_cast<float>(q) - 1.0F) * d;
  return __half_as_ushort(__float2half_rn(v));
}
}  // namespace

int main() {
  std::uint32_t* bo = nullptr;
  CK(hipMalloc(&bo, 32 * sizeof(std::uint32_t)));
  ByteOrderProbe<<<1, 1>>>(bo);
  CK(hipDeviceSynchronize());
  std::vector<std::uint32_t> h(32);
  CK(hipMemcpy(h.data(), bo, 32 * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
  std::printf("perm(a=0x07060504, b=0x03020100, sel), bytes spell the index used:\n");
  for (int s = 0; s < 8; ++s) std::printf("  sel %d -> 0x%08x\n", s, h[s]);
  std::printf("selectors used for (q0, q1):\n");
  for (int q0 = 0; q0 < 4; ++q0)
    for (int q1 = 0; q1 < 4; ++q1) {
      const std::uint32_t r = h[16 + q0 * 4 + q1];
      // A correct pair lookup wants bytes (2q0, 2q0+1, 2q1, 2q1+1) in lanes
      // 0..3, i.e. the value 0x<(2q1+1)><2q1><<(2q0+1)><2q0>.
      const std::uint32_t want = (2U * q1 + 1U) << 24 | (2U * q1) << 16 |
                                 ((2U * q0 + 1U) << 8) | (2U * q0);
      std::printf("  q0=%d q1=%d -> 0x%08x %s\n", q0, q1, r,
                  r == want ? "ok" : "WRONG");
    }

  // Scales: zeros, denormals, the 2*d saturation band, max finite, inf, NaN,
  // both signs.
  std::vector<std::uint16_t> ds;
  for (std::uint32_t v : {0x0000U, 0x8000U, 0x0001U, 0x0002U, 0x01FFU, 0x0200U,
                          0x03FFU, 0x3800U, 0x3BFFU, 0x3C00U, 0x5B7EU, 0x5BBFU,
                          0x5BFFU, 0x5C00U, 0x7BFFU, 0x7C00U, 0x7C01U, 0x7FFFU,
                          0xBC00U, 0xFBFFU, 0xFC00U, 0xFFFFU}) {
    ds.push_back(static_cast<std::uint16_t>(v));
  }
  // Every normal scale too, both signs: the exhaustive check.
  for (std::uint32_t v = 0; v < 0x10000U; ++v) ds.push_back(static_cast<std::uint16_t>(v));
  const int n = static_cast<int>(ds.size());

  std::uint16_t* di = nullptr;
  std::uint32_t* doi = nullptr;
  CK(hipMalloc(&di, sizeof(std::uint16_t) * n));
  CK(hipMalloc(&doi, sizeof(std::uint32_t) * n * 4));
  CK(hipMemcpy(di, ds.data(), sizeof(std::uint16_t) * n, hipMemcpyHostToDevice));
  TableProbe<<<(n + 63) / 64, 64>>>(di, doi, n);
  CK(hipDeviceSynchronize());
  std::vector<std::uint32_t> res(n * 4);
  CK(hipMemcpy(res.data(), doi, sizeof(std::uint32_t) * n * 4, hipMemcpyDeviceToHost));

  int shown = 0, bad = 0, bad_finite = 0;
  for (int i = 0; i < n; ++i) {
    for (int q = 0; q < 4; ++q) {
      const std::uint32_t r = res[i * 4 + q];
      const std::uint16_t got_lo = static_cast<std::uint16_t>(r & 0xFFFFU);
      const std::uint16_t got_hi = static_cast<std::uint16_t>(r >> 16);
      const std::uint16_t want = Expect(ds[i], q);
      if (got_lo != want || got_hi != want) {
        ++bad;
        const bool d_fin = (ds[i] & 0x7C00U) != 0x7C00U;
        const float w = HalfToF(want);
        const bool want_fin = (want & 0x7C00U) != 0x7C00U;
        if (d_fin && want_fin) ++bad_finite;
        if (shown < 20) {
          std::printf("  MISMATCH ds=0x%04x q=%d got lo/hi 0x%04x/0x%04x want 0x%04x (%g)\n",
                      ds[i], q, got_lo, got_hi, want, w);
          ++shown;
        }
      }
    }
  }
  std::printf("table check: %d scales x 4 codes, %d mismatches (%d with finite d AND finite value)\n",
              n, bad, bad_finite);
  return bad_finite ? 1 : 0;
}
