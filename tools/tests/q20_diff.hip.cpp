// Differential bitwise gate for the routed Q2_0 dequant.
//
// The setup is the decode benchmark (tools/bench/qfn_routed_decode_bench.hip.cpp)
// copied verbatim up to the launch point -- real dimensions, real routed
// metadata, real production launch path -- so the only thing under test is the
// arithmetic. Weights get a deterministic Q2_0 payload whose scales sweep the
// whole finite f16 range, then one routed FFN layer (gate, up with swiglu,
// down) runs per activation pattern and every output buffer is appended to the
// dump.
//
// Build it twice, once against the shipped kernels object and once against a
// candidate object, and `cmp` the two dumps. Byte-identical means the rewrite
// moved nothing observable. There is deliberately no reference implementation
// and no tolerance: the kernel's own summation order is part of its behaviour,
// and a rewrite is only allowed to be indistinguishable from it.
//
//   KERNELS_O=<kernels.o> bash tools/tests/q20_diff.sh
//   ./q20_diff --emit old.bin && ./q20_diff_new --emit new.bin && cmp old.bin new.bin
//
// It allocates weights on the device and launches kernels; it never opens a
// GGUF, never touches the module store, the gufo binary or llama-swap.
// Q2_0 is the format under test -- pass no --type.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include <cmath>
#include <random>

#define CK(x)                                                                \
  do {                                                                       \
    hipError_t e_ = (x);                                                     \
    if (e_ != hipSuccess) {                                                  \
      std::fprintf(stderr, "hip error %s at %s:%d\n", hipGetErrorString(e_), \
                   __FILE__, __LINE__);                                      \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

// The extension's own ABI (src/model/kernels/rocm/kernels.hpp). Redeclared
// here so the bench links against kernels.o without pulling the SDK headers;
// the mangled names must match, which build.sh verifies with nm.
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

using gufo::models::qwen38_flash_next::rocm::WeightType;

namespace {

constexpr std::size_t kTail = 64;  // DeviceModel::kTailMargin equivalent.

std::size_t BlockElems(WeightType t) {
  switch (t) {
    case WeightType::kQ4_K:
    case WeightType::kQ5_K: return 256;
    case WeightType::kIQ4_NL: return 32;
    default: return 64;  // Q2_0, Q5_1, Q8_0
  }
}

// Bytes per GGML block, matching the storages the WMMA family reads.
std::size_t BlockBytes(WeightType t) {
  switch (t) {
    case WeightType::kQ4_K: return 144;
    case WeightType::kQ5_K: return 176;
    case WeightType::kQ8_0: return 34;
    case WeightType::kQ5_1: return 18;
    case WeightType::kIQ4_NL: return 18;
    case WeightType::kQ2_0: return 18;
    default: std::fprintf(stderr, "unsupported type in bench\n"), std::exit(2);
  }
}

std::size_t RowBytes(std::size_t k_elems, WeightType t) {
  return (k_elems / BlockElems(t)) * BlockBytes(t);
}

void* AllocWeights(std::size_t experts, std::size_t rows, std::size_t k_elems,
                   WeightType t, std::size_t* total) {
  const std::size_t per = RowBytes(k_elems, t);
  const std::size_t bytes = experts * rows * per + kTail;
  void* p = nullptr;
  CK(hipMalloc(&p, bytes));
  CK(hipMemset(p, 0x49, bytes));  // any fixed pattern: values are irrelevant
  CK(hipMemset(static_cast<std::uint8_t*>(p) + bytes - kTail, 0, kTail));
  *total = experts * rows * per;
  return p;
}

struct Ctx {
  WeightType type;
  std::size_t k, m, experts, topk;
  std::uint32_t n_tokens = 1, tile_rows = 16;
  int rotate = 8;  // distinct expert sets per timing loop (DRAM-cold experts)
  std::size_t slots = 0, compact = 0, capacity = 0;
  std::int32_t *tiles = nullptr, *bounds = nullptr, *cursors = nullptr;
  std::int32_t *rows_token = nullptr, *rows_slot = nullptr;
  __half *x = nullptr, *glu = nullptr;
  float *gate_out = nullptr, *down_out = nullptr;
  void *w_gate = nullptr, *w_up = nullptr, *w_down = nullptr;
  std::size_t gate_bytes = 0, up_bytes = 0, down_bytes = 0;
};

float TimeRun(hipEvent_t a, hipEvent_t b, const std::function<void()>& fn,
              int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) fn();
  CK(hipDeviceSynchronize());
  CK(hipEventRecord(a));
  for (int i = 0; i < iters; ++i) fn();
  CK(hipEventRecord(b));
  CK(hipEventSynchronize(b));
  float ms = 0.0f;
  CK(hipEventElapsedTime(&ms, a, b));
  return ms / iters;
}

}  // namespace

#include <functional>

int main(int argc, char** argv) {
  using namespace gufo::models::qwen38_flash_next::rocm;
  Ctx c{};
  std::string type_name = "q2_0";
  int iters = 20;
  int reps = 5;
  const char* emit_path = nullptr;
  int patterns = 24;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--k") && i + 1 < argc) c.k = std::atoll(argv[++i]);
    else if (!std::strcmp(argv[i], "--m") && i + 1 < argc) c.m = std::atoll(argv[++i]);
    else if (!std::strcmp(argv[i], "--experts") && i + 1 < argc) c.experts = std::atoll(argv[++i]);
    else if (!std::strcmp(argv[i], "--topk") && i + 1 < argc) c.topk = std::atoll(argv[++i]);
    else if (!std::strcmp(argv[i], "--type") && i + 1 < argc) type_name = argv[++i];
    else if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) c.n_tokens = static_cast<std::uint32_t>(std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--rotate") && i + 1 < argc) c.rotate = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--emit") && i + 1 < argc) emit_path = argv[++i];
    else if (!std::strcmp(argv[i], "--patterns") && i + 1 < argc) patterns = std::atoi(argv[++i]);
    else {
      std::fprintf(stderr,
                   "usage: %s [--k 4096] [--m 2048] [--experts 24] [--topk 8] "
                   "[--type q2_0|q4_k|q8_0] [--tokens «redacted-secret» [--rotate 1] "
                   "[--iters 20] [--reps 5]\n",
                   argv[0]);
      return 2;
    }
  }
  c.type = type_name == "q4_k"   ? WeightType::kQ4_K
           : type_name == "q8_0" ? WeightType::kQ8_0
           : type_name == "q5_1" ? WeightType::kQ5_1
                                 : WeightType::kQ2_0;
  const std::size_t be = BlockElems(c.type);
  if (c.k % be != 0 || c.m % be != 0) {
    std::fprintf(stderr, "k and m must be multiples of %zu for %s\n", be,
                 type_name.c_str());
    return 2;
  }
  if (c.topk > c.experts) c.topk = c.experts;
  c.slots = static_cast<std::size_t>(c.n_tokens) * c.topk;
  c.compact = RoutedCompactRows(c.slots, c.experts);
  c.capacity = (c.slots + c.experts * 15) / 16 + c.experts + 1;

  std::printf(
      "type=%s k=%zu m=%zu experts=%zu topk=%zu tokens=%u tile_rows=%u\n"
      "  compact_rows=%zu tiles_capacity=%zu  bytes/row(gate/up)=%zu\n",
      type_name.c_str(), c.k, c.m, c.experts, c.topk, c.n_tokens, c.tile_rows,
      c.compact, c.capacity, RowBytes(c.k, c.type));

  // Weights: gate/up are [expert][m][k]; the down projection is
  // [expert][k][m] (its contraction extent is the expert width).
  c.w_gate = AllocWeights(c.experts, c.m, c.k, c.type, &c.gate_bytes);
  c.w_up = AllocWeights(c.experts, c.m, c.k, c.type, &c.up_bytes);
  c.w_down = AllocWeights(c.experts, c.k, c.m, c.type, &c.down_bytes);
  std::printf("  device weights: %.1f MiB\n",
              (c.gate_bytes + c.up_bytes + c.down_bytes) / 1048576.0);

  // Routing. Each of `rotate` sets picks a disjoint expert window, so a
  // timing loop walks DRAM-cold weights the way real MoE decode does; with
  // one set the whole working set sits in the 32 MiB LLC and the number is
  // flattering (measured: LLC serves 872 GB/s vs 241 from DRAM).
  const int R = std::max(1, c.rotate);
  std::vector<std::int32_t> ids_h(R * c.topk);
  std::vector<std::uint32_t> counts_h(R * c.experts);
  for (int r = 0; r < R; ++r) {
    for (std::size_t j = 0; j < c.topk; ++j) {
      std::uint32_t e = static_cast<std::uint32_t>((r * c.topk + j) % c.experts);
      ids_h[r * c.topk + j] = static_cast<std::int32_t>(e);
      counts_h[r * c.experts + e] = c.n_tokens;
    }
  }

  std::int32_t* ids_d = nullptr;
  std::uint32_t* counts_d = nullptr;
  CK(hipMalloc(&ids_d, sizeof(std::int32_t) * R * c.topk));
  CK(hipMalloc(&counts_d, sizeof(std::uint32_t) * R * c.experts));
  CK(hipMemcpy(ids_d, ids_h.data(), sizeof(std::int32_t) * R * c.topk,
               hipMemcpyHostToDevice));
  CK(hipMemcpy(counts_d, counts_h.data(), sizeof(std::uint32_t) * R * c.experts,
               hipMemcpyHostToDevice));

  std::vector<std::int32_t> tiles_h(R * c.capacity, -1);
  std::vector<std::int32_t> bounds_h(R * (c.experts + 1));
  std::vector<std::int32_t> cursors_h(R * (c.compact + c.experts + 1));
  std::vector<std::int32_t> rows_t_h(R * c.compact), rows_s_h(R * c.compact);
  std::vector<std::int32_t*> tiles_p(R), bounds_p(R), curs_p(R), rt_p(R),
      rs_p(R);
  auto setp = [&](std::vector<std::int32_t*>& p, std::size_t per) {
    for (int r = 0; r < R; ++r) {
      void* q = nullptr;
      CK(hipMalloc(&q, sizeof(std::int32_t) * per));
      p[r] = static_cast<std::int32_t*>(q);
    }
  };
  setp(tiles_p, c.capacity);
  setp(bounds_p, c.experts + 1);
  setp(curs_p, c.compact + c.experts + 1);
  setp(rt_p, c.compact);
  setp(rs_p, c.compact);
  CK(hipMalloc(&c.x, sizeof(__half) * c.k * c.n_tokens));
  CK(hipMalloc(&c.glu, sizeof(__half) * c.m * c.slots));
  CK(hipMalloc(&c.gate_out, sizeof(float) * c.m * c.slots));
  CK(hipMalloc(&c.down_out, sizeof(float) * c.k * c.slots));
  CK(hipMemset(c.x, 0x30, sizeof(__half) * c.k * c.n_tokens));

  // Build every set's routing metadata up front so the timed loop only ever
  // launches the GEMM.
  for (int r = 0; r < R; ++r) {
    RoutedCompact(ids_d + r * c.topk, counts_d + r * c.experts, bounds_p[r],
                  curs_p[r], rt_p[r], rs_p[r], c.n_tokens,
                  static_cast<std::uint32_t>(c.topk),
                  static_cast<std::uint32_t>(c.experts), nullptr);
    RoutedTiles(counts_d + r * c.experts, tiles_p[r],
                static_cast<std::uint32_t>(c.experts), c.tile_rows,
                static_cast<std::uint32_t>(c.capacity), nullptr);
  }
  CK(hipDeviceSynchronize());

  // ---- differential dump mode ----------------------------------------------
  // Everything above this line is the decode benchmark, unchanged: real
  // dimensions, real routed metadata, the real production launch path. From
  // here we fill the weights with a deterministic Q2_0 payload, run one routed
  // FFN layer per activation pattern, and write out every buffer the layer
  // produced.
  //
  // Two builds of this file -- one against the shipped kernels object, one
  // against a candidate object -- must dump byte-identical bytes for a dequant
  // rewrite to be acceptable. No oracle, no tolerance band: the question is
  // whether anything observable moved.

  auto fill_q20 = [&](void* dev, std::size_t rows, std::size_t k_elems,
                      std::uint64_t seed) {
    // block_q2_0 as this kernel reads it: 18 bytes per 64 elements — an f16
    // scale at bytes 0..1, then 16 code bytes holding four 2-bit codes each.
    const std::size_t blocks = k_elems / 64;
    const std::size_t bytes = c.experts * rows * blocks * 18;
    std::vector<std::uint8_t> h(bytes);
    std::mt19937_64 rng(seed);
    for (std::size_t i = 0; i + 18 <= bytes; i += 18) {
      // Scales across the whole finite f16 range -- denormal, normal, and the
      // band where 2*d saturates to the finite max -- but never inf/NaN: the
      // model never produces those and no rewrite is expected to agree there.
      std::uint16_t ds = static_cast<std::uint16_t>(rng() & 0x73FFU);
      if (ds == 0) ds = 0x0001;
      h[i] = static_cast<std::uint8_t>(ds & 0xFFU);
      h[i + 1] = static_cast<std::uint8_t>(ds >> 8);
      const std::uint64_t codes = rng();
      for (int j = 0; j < 16; ++j)
        h[i + 2 + j] = static_cast<std::uint8_t>(codes >> (8 * j));
    }
    CK(hipMemcpy(dev, h.data(), bytes, hipMemcpyHostToDevice));
  };
  fill_q20(c.w_gate, c.m, c.k, 0xC0FFEE11ULL);
  fill_q20(c.w_up, c.m, c.k, 0xC0FFEE22ULL);
  fill_q20(c.w_down, c.k, c.m, 0xC0FFEE33ULL);
  CK(hipDeviceSynchronize());

  std::vector<__half> xh(c.k * c.n_tokens);
  std::vector<float> gth(c.m * c.slots);
  std::vector<__half> glh(c.m * c.slots);
  std::vector<float> dnh(c.k * c.slots);

  FILE* out = nullptr;
  if (emit_path != nullptr) {
    out = std::fopen(emit_path, "wb");
    if (out == nullptr) {
      std::fprintf(stderr, "cannot open %s for output\n", emit_path);
      return 4;
    }
    const std::uint32_t hdr[6] = {
        static_cast<std::uint32_t>(c.m), static_cast<std::uint32_t>(c.k),
        static_cast<std::uint32_t>(c.experts), static_cast<std::uint32_t>(c.topk),
        static_cast<std::uint32_t>(patterns), 1u};
    std::fwrite(hdr, sizeof(hdr[0]), 6, out);
  }

  long launches = 0, nonfinite = 0;
  for (int p = 0; p < patterns; ++p) {
    for (std::size_t i = 0; i < xh.size(); ++i) {
      float v;
      switch (p % 6) {
        case 0: v = 1.0f; break;
        case 1: v = -1.0f; break;
        case 2: v = (i % 2) ? 0.5f : -0.25f; break;
        case 3: v = 65504.0f; break;   // f16 max: exercises accumulation range
        case 4: v = 6.0e-8f; break;    // f16 denormal activation
        default:
          v = static_cast<float>((static_cast<int>(i * 37u + p * 11u) % 401) - 200) /
              64.0f;
      }
      xh[i] = __float2half_rn(v);
    }
    CK(hipMemcpy(c.x, xh.data(), sizeof(__half) * xh.size(),
                 hipMemcpyHostToDevice));

    // Zero the outputs so the dump is "what the kernels wrote", not whatever
    // hipMalloc handed back: untouched padding then reads 0.0 instead of
    // uninitialized bytes that could differ between two builds for reasons
    // that have nothing to do with the kernel.
    CK(hipMemset(c.gate_out, 0, sizeof(float) * c.m * c.slots));
    CK(hipMemset(c.glu, 0, sizeof(__half) * c.m * c.slots));
    CK(hipMemset(c.down_out, 0, sizeof(float) * c.k * c.slots));

    const std::size_t r = static_cast<std::size_t>(p) % R;
    const std::uint32_t n_tiles = static_cast<std::uint32_t>(c.capacity);
    const bool g =
        RoutedF16Gemm(c.w_gate, c.type, c.x, tiles_p[r], n_tiles, c.tile_rows,
                      bounds_p[r], rt_p[r], rs_p[r], nullptr, c.gate_out,
                      nullptr, c.m, c.k, nullptr);
    CK(hipDeviceSynchronize());
    const bool u = RoutedF16Gemm(c.w_up, c.type, c.x, tiles_p[r], n_tiles,
                                 c.tile_rows, bounds_p[r], rt_p[r], rs_p[r],
                                 c.gate_out, nullptr, c.glu, c.m, c.k, nullptr);
    CK(hipDeviceSynchronize());
    const bool d = RoutedF16Gemm(c.w_down, c.type,
                                 reinterpret_cast<const __half*>(c.glu),
                                 tiles_p[r], n_tiles, c.tile_rows, bounds_p[r],
                                 rt_p[r], rs_p[r], nullptr, c.down_out, nullptr,
                                 c.k, c.m, nullptr);
    CK(hipDeviceSynchronize());
    if (!g || !u || !d) {
      std::fprintf(stderr, "RoutedF16Gemm refused the shape (pattern %d)\n", p);
      return 3;
    }
    CK(hipMemcpy(gth.data(), c.gate_out, sizeof(float) * gth.size(),
                 hipMemcpyDeviceToHost));
    CK(hipMemcpy(glh.data(), c.glu, sizeof(__half) * glh.size(),
                 hipMemcpyDeviceToHost));
    CK(hipMemcpy(dnh.data(), c.down_out, sizeof(float) * dnh.size(),
                 hipMemcpyDeviceToHost));
    for (float v : gth) nonfinite += std::isfinite(v) ? 0 : 1;
    for (float v : dnh) nonfinite += std::isfinite(v) ? 0 : 1;
    if (out != nullptr) {
      std::fwrite(gth.data(), sizeof(float), gth.size(), out);
      std::fwrite(glh.data(), sizeof(__half), glh.size(), out);
      std::fwrite(dnh.data(), sizeof(float), dnh.size(), out);
    }
    ++launches;
  }
  if (out != nullptr) std::fclose(out);

  std::printf("differential dump: %ld layer runs, %ld non-finite outputs\n",
              launches, nonfinite);
  return 0;
}
