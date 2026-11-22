// Bitwise gate for the routed Q2_0 dequant, without the gufo dequant header.
//
//   KERNELS_O=<candidate kernels.hip.cpp.o> bash tools/tests/bitexact_q20.sh
//
// Trick: with a one-hot activation x = e_j the routed GEMM degenerates to a
// row copy, out[i] = w[i][j], so every dequantized weight element can be read
// back and compared BITWISE against the host reference
//
//     h = f16_rn((float(q) - 1.0f) * float(d))      // the path the kernel had
//                                                   // before the perm table
//
// No tolerance band, no oracle library: 640 columns x 640 rows x N experts =
// ~1.2M elements per run, and the scales include the boundary cases (0, -0,
// min denormal, 65504, and the d where 2d lands between max-finite and inf).
// Non-finite scales are excluded on purpose: the perm table intentionally
// differs there (see the kernel comment), and a real tensor cannot have them.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                \
  do {                                                                       \
    hipError_t e_ = (x);                                                     \
    if (e_ != hipSuccess) {                                                  \
      std::fprintf(stderr, "hip error %s at %s:%d\n", hipGetErrorString(e_), \
                   __FILE__, __LINE__);                                      \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

namespace gufo::models::qwen38_flash_next::rocm {
enum class WeightType : std::uint32_t {
  kF32 = 0, kF16 = 1, kQ5_1 = 7, kQ8_0 = 8, kQ4_K = 12, kQ5_K = 13,
  kQ6_K = 14, kIQ3_XXS = 18, kIQ4_NL = 20, kIQ3_S = 21, kIQ2_S = 22,
  kIQ4_XS = 23, kBF16 = 30, kQ2_0 = 42,
};
std::size_t RoutedCompactRows(std::size_t slots, std::size_t n_experts);
void RoutedCompact(const std::int32_t* ids, const std::uint32_t* counts,
                   std::int32_t* pad_bounds, std::int32_t* cursors,
                   std::int32_t* rows_token, std::int32_t* rows_slot,
                   std::uint32_t n_tokens, std::uint32_t k,
                   std::uint32_t n_experts, hipStream_t stream);
void RoutedTiles(const std::uint32_t* counts, std::int32_t* tiles,
                 std::uint32_t n_experts, std::uint32_t tile_rows,
                 std::uint32_t capacity, hipStream_t stream);
bool RoutedF16Gemm(const void* w, WeightType type, const __half* x,
                   const std::int32_t* tiles, std::uint32_t n_tiles,
                   std::uint32_t tile_rows, const std::int32_t* pad_bounds,
                   const std::int32_t* rows_in, const std::int32_t* rows_out,
                   const float* swiglu_gate, float* out, __half* out_half,
                   std::size_t m, std::size_t k, hipStream_t stream);
}  // namespace gufo::models::qwen38_flash_next::rocm

using gufo::models::qwen38_flash_next::rocm::RoutedCompact;
using gufo::models::qwen38_flash_next::rocm::RoutedCompactRows;
using gufo::models::qwen38_flash_next::rocm::RoutedF16Gemm;
using gufo::models::qwen38_flash_next::rocm::RoutedTiles;
using gufo::models::qwen38_flash_next::rocm::WeightType;

namespace {

std::uint64_t Splitmix(std::uint64_t* s) {
  std::uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// f16 bit patterns the reference and the table must agree on, including the
// 2*d overflow boundary: for d in (32752, 36848) the true 2d lands below the
// round-to-inf midpoint, so a naive exponent bump would be wrong here.
std::uint16_t ScalePattern(int row) {
  static const std::uint16_t kEdge[] = {
      0x0000,             // +0
      0x8000,             // -0
      0x0001,             // min denormal
      0x01FF,             // max denormal
      0x0400,             // 2^-14 (first normal)
      0x3C00,             // 1.0
      0xBC00,             // -1.0
      0x7BFF,             // 65504 (max finite)
      0xFBFF,             // -65504
      0x4000,             // 2.0
      0x5B7F,             // ~32752: 2d == max finite
      0x5B80,             // just above: 2d overflows into the rounding band
      0x5C00,             // ~40000: 2d rounds to inf
      0x3800,             // 0.5
      0x2000,             // 2^-9, small normal
      0xBBFF,             // -65504 mirrored again for a live negative row
  };
  return kEdge[row % (sizeof(kEdge) / sizeof(kEdge[0]))];
}

struct Q20Expert {
  std::vector<std::uint8_t> bytes;    // [m][k/64] * 18
  std::vector<std::uint32_t> expect;  // [m][k] f32 bits the device must return
};

Q20Expert MakeExpert(std::size_t m, std::size_t k, std::uint64_t seed) {
  Q20Expert w;
  w.bytes.assign(m * (k / 64) * 18, 0);
  w.expect.assign(m * k, 0);
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t b = 0; b < k / 64; ++b) {
      std::uint8_t* blk = w.bytes.data() + (i * (k / 64) + b) * 18;
      const std::uint16_t ds =
          (i < 16) ? ScalePattern(static_cast<int>(i))
                   : static_cast<std::uint16_t>(
                         Splitmix(&seed) & 0x7BFFU);  // finite, any sign
      blk[0] = static_cast<std::uint8_t>(ds & 0xFF);
      blk[1] = static_cast<std::uint8_t>(ds >> 8);
      for (int q = 0; q < 16; ++q) blk[2 + q] = static_cast<std::uint8_t>(Splitmix(&seed));
      const float d = __half2float(__builtin_bit_cast(
          __half, ds));
      for (int e = 0; e < 64; ++e) {
        const int byte = 2 + (e >> 2);
        const std::uint32_t q = (blk[byte] >> (2 * (e & 3))) & 0x3U;
        const float v = (static_cast<float>(q) - 1.0F) * d;
        // What the kernel holds is the f16 fragment; the GEMM adds it into an
        // f32 accumulator, so the device must return float(f16_rn(v)) exactly.
        const __half h = __float2half_rn(v);
        w.expect[i * k + b * 64 + e] =
            __builtin_bit_cast(std::uint32_t, __half2float(h));
      }
    }
  }
  return w;
}

}  // namespace

int main(int argc, char** argv) {
  std::size_t m = 640, k = 640;
  int experts = 3;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--m") && i + 1 < argc) m = std::atoll(argv[++i]);
    else if (!std::strcmp(argv[i], "--k") && i + 1 < argc) k = std::atoll(argv[++i]);
    else if (!std::strcmp(argv[i], "--experts") && i + 1 < argc) experts = std::atoi(argv[++i]);
    else {
      std::fprintf(stderr, "usage: %s [--m 640] [--k 640] [--experts 3]\n", argv[0]);
      return 2;
    }
  }
  if (m % 64 || k % 64) {
    std::fprintf(stderr, "m and k must be multiples of 64 (Q2_0 block)\n");
    return 2;
  }

  const std::size_t slots = 1;
  const std::size_t compact = RoutedCompactRows(slots, static_cast<std::size_t>(experts));
  const std::size_t capacity =
      (slots + static_cast<std::size_t>(experts) * 15) / 16 +
      static_cast<std::size_t>(experts) + 1;

  std::int32_t *ids_d = nullptr, *tiles_d = nullptr, *bounds_d = nullptr,
               *curs_d = nullptr, *rt_d = nullptr, *rs_d = nullptr;
  std::uint32_t* counts_d = nullptr;
  __half *x_d = nullptr, *zeros_d = nullptr;
  float* out_d = nullptr;
  CK(hipMalloc(&ids_d, sizeof(std::int32_t)));
  CK(hipMalloc(&counts_d, sizeof(std::uint32_t) * experts));
  CK(hipMalloc(&tiles_d, sizeof(std::int32_t) * capacity));
  CK(hipMalloc(&bounds_d, sizeof(std::int32_t) * (experts + 1)));
  CK(hipMalloc(&curs_d, sizeof(std::int32_t) * (compact + experts + 1)));
  CK(hipMalloc(&rt_d, sizeof(std::int32_t) * compact));
  CK(hipMalloc(&rs_d, sizeof(std::int32_t) * compact));
  CK(hipMalloc(&x_d, sizeof(__half) * k));
  CK(hipMalloc(&zeros_d, sizeof(__half) * k));
  CK(hipMalloc(&out_d, sizeof(float) * m));
  CK(hipMemset(zeros_d, 0, sizeof(__half) * k));

  std::vector<float> host(m);
  long checked = 0, bad = 0;
  int first_bad_expert = -1;

  for (int e = 0; e < experts; ++e) {
    Q20Expert w = MakeExpert(m, k, 0x51ED1234ABCD0000ULL + 7919ULL * e);
    void* w_d = nullptr;
    CK(hipMalloc(&w_d, w.bytes.size()));
    CK(hipMemcpy(w_d, w.bytes.data(), w.bytes.size(), hipMemcpyHostToDevice));

    const std::int32_t id = e;
    CK(hipMemcpy(ids_d, &id, sizeof(id), hipMemcpyHostToDevice));
    std::vector<std::uint32_t> counts(experts, 0);
    counts[e] = 1;
    CK(hipMemcpy(counts_d, counts.data(), sizeof(std::uint32_t) * experts,
                 hipMemcpyHostToDevice));
    RoutedCompact(ids_d, counts_d, bounds_d, curs_d, rt_d, rs_d, 1, 1,
                  static_cast<std::uint32_t>(experts), nullptr);
    RoutedTiles(counts_d, tiles_d, static_cast<std::uint32_t>(experts), 16,
                static_cast<std::uint32_t>(capacity), nullptr);
    CK(hipDeviceSynchronize());

    for (std::size_t j = 0; j < k; ++j) {
      CK(hipMemcpy(x_d, zeros_d, sizeof(__half) * k, hipMemcpyDeviceToDevice));
      const __half one = __float2half_rn(1.0F);
      CK(hipMemcpy(x_d + j, &one, sizeof(one), hipMemcpyHostToDevice));
      if (!RoutedF16Gemm(w_d, WeightType::kQ2_0, x_d, tiles_d,
                         static_cast<std::uint32_t>(capacity), 16, bounds_d,
                         rt_d, rs_d, nullptr, out_d, nullptr, m, k, nullptr)) {
        std::fprintf(stderr, "expert %d col %zu: RoutedF16Gemm refused\n", e, j);
        return 3;
      }
      CK(hipMemcpy(host.data(), out_d, sizeof(float) * m, hipMemcpyDeviceToHost));
      for (std::size_t i = 0; i < m; ++i) {
        const std::uint32_t got = __builtin_bit_cast(std::uint32_t, host[i]);
        const std::uint32_t want = w.expect[i * k + j];
        ++checked;
        if (got != want) {
          ++bad;
          if (first_bad_expert < 0) {
            first_bad_expert = e;
            const std::size_t off = (i * (k / 64) + (j >> 6)) * 18;
            const unsigned ds = w.bytes[off] | (w.bytes[off + 1] << 8);
            std::printf(
                "  MISMATCH e=%d row=%zu col=%zu q=%u got=0x%08x want=0x%08x "
                "d=0x%04x\n",
                e, i, j,
                (w.bytes[off + 2 + ((j & 63) >> 2)] >> (2 * (j & 3))) & 3U, got,
                want, ds);
          }
        }
      }
    }
    CK(hipFree(w_d));
    std::printf("  expert %d checked (%zu elements so far, %ld mismatches)\n", e,
                checked, bad);
  }

  std::printf("\nQ2_0 dequant bitwise: %ld elements checked, %ld mismatches -> %s\n",
              checked, bad, bad == 0 ? "PASS" : "FAIL");
  if (bad) std::printf("first failing expert: %d\n", first_bad_expert);
  return bad == 0 ? 0 : 1;
}
