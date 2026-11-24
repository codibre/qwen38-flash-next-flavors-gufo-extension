#include "gsq/kernels/routed_wmma.hpp"

#include <hip/hip_bfloat16.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "gsq/kernels/iq_codebooks.hpp"

namespace quant = gufo::models::qwen38_flash_next::iqref;

namespace gufo::ext::gsq::wmma {
constexpr unsigned kThreads = 256;


__device__ __forceinline__ float SigmoidF(float x) {
  return 1.0f / (1.0f + __expf(-x));
}
__device__ __forceinline__ float SoftplusF(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}
__device__ __forceinline__ float SiluF(float x) {
  return x * SigmoidF(x);
}
/// One block_q8_1 (mmq's decode activation format): d and the row sum, then
/// the 32 codes.
struct Q8_1Block {
  __half2 ds;
  std::int8_t qs[32];
};
static_assert(sizeof(Q8_1Block) == 36);

/// Keeps fast-math from folding a stored value's arithmetic into what
/// follows: the quantization must see exactly the float that was stored.
__device__ __forceinline__ float Materialize(float v) {
  asm volatile("" : "+v"(v));
  return v;
}

/// Lane i receives lane (i ^ offset)'s value, as __shfl_xor(v, offset) does
/// across a wave32. On gfx1151 it is a DPP row_xmask (offsets below 16) or
/// v_permlanex16 (16): a register move instead of an LDS crossbar round trip
/// (ds_bpermute). Only the transport differs, so reductions stay bitwise;
/// every lane must be active, as the shuffle's callers already require.
__device__ __forceinline__ int XorLane(int v, int offset) {
#if defined(__gfx1151__)
  switch (offset) {
    case 1:
      return __builtin_amdgcn_update_dpp(0, v, 0x161, 0xF, 0xF, false);
    case 2:
      return __builtin_amdgcn_update_dpp(0, v, 0x162, 0xF, 0xF, false);
    case 4:
      return __builtin_amdgcn_update_dpp(0, v, 0x164, 0xF, 0xF, false);
    case 8:
      return __builtin_amdgcn_update_dpp(0, v, 0x168, 0xF, 0xF, false);
    case 16:
      return __builtin_amdgcn_permlanex16(v, v, 0x76543210, 0xfedcba98, false,
                                          false);
  }
#endif
  return __shfl_xor(v, offset);
}

__device__ __forceinline__ unsigned XorLane(unsigned v, int offset) {
  return static_cast<unsigned>(XorLane(static_cast<int>(v), offset));
}

__device__ __forceinline__ float XorLane(float v, int offset) {
  return __builtin_bit_cast(float, XorLane(__builtin_bit_cast(int, v), offset));
}

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// BF16 fragments carry the same sixteen 16-bit lanes.
using v16bf = __attribute__((ext_vector_type(16))) __bf16;

__device__ __forceinline__ v8f WmmaBf16(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
      __builtin_bit_cast(v16bf, a), __builtin_bit_cast(v16bf, b), c);
}

// Keep indexer queries in F32: narrowing them before ranking can swap blocks
// at the selection boundary. A thread scores one key, reusing it across all

//                  [b1: 16 tokens x 16 bytes]   offset 256
//                  [16 fp32 token scales    ]   offset 512
//
// with tt = token / 16 and kb the 32-element K block, 576 bytes per tile.
struct Q8_0Block {
  __half d;
  std::int8_t qs[32];
};
static_assert(sizeof(Q8_0Block) == 34, "block_q8_0 must be 34 bytes");
/// W8A8TailKernel limits: tokens past the last full tile and K width.
constexpr std::size_t kW8A8TailMaxTokens = 16;
constexpr std::size_t kW8A8TailMaxK = 10240;

using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

// bucket padded to a 16-row tile (`pad_bounds`), so a token tile never
// straddles experts; `rows_out` maps a compact row to its (token, slot)
// output row or -1 for padding.
constexpr std::size_t kRoutedTileTokens = 16;

struct Q4KBlock {
  __half d;
  __half dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
static_assert(sizeof(Q4KBlock) == 144, "block_q4_K must be 144 bytes");

struct Q5_1Block {
  __half d;
  __half m;
  std::uint32_t qh;
  std::uint8_t qs[16];
};
static_assert(sizeof(Q5_1Block) == 24, "block_q5_1 must be 24 bytes");

// Routed F16 WMMA expert GEMM. The int8 kernel above pays a float epilogue
// and an activation-sum correction every K block because the per-32 scales
// of both operands sit outside the integer dot product; here the weights are
// dequantized to F16 right after the LDS read (each wave decodes only its own
// sixteen rows) and the activations are F16 rows, so the matrix core
// accumulates the whole K extent in F32 with no per-block work. The codes
// stay packed in LDS (4 bits for Q4_K, 4 + 1 for Q5_1), which keeps a
// two-K-block stage at 11-12 KB and five blocks resident per WGP.
//
// A code becomes a half through a byte permute into the mantissa of 1024.0
// (0x6400 | q = 1024 + q exactly for q < 32, the half's unit being 1 there),
// a packed subtract of 1024 (exact), then one packed FMA:
//
//     w = q * scale + bias
//
// with (scale, bias) = (d * sc, -dmin * mn) for Q4_K and (d, m) for Q5_1,
// staged per (row, K block) as a half2.
//
// grid (m / BM, tiles): `tiles[y]` packs the expert in the low 16 bits and
// the token macro tile index in the high 16, so no block is launched for an
// empty tile; the row blocks of one tile are consecutive in dispatch order so
// they share the tile's gathered activations through L2. Block (x, y)
// computes rows x*BM.. of the expert against its
// compact rows [pad_bounds[e] + j*BN, +BN) and scatters them to
// out[rows_out[c]][row] (F32, or F16 with the SwiGLU applied when `out_half`
// is given: the up projection then writes the down projection's input).
constexpr std::uint32_t kHalfMagic = 0x64646464U;  // 1024.0 high bytes

/// block_q5_K: the Q4_K header, 32 high-bit bytes (bit s of byte j is the
/// fifth bit of element j of K block s), then the Q4_K nibble layout.
constexpr std::size_t kQ5KBlockBytes = 176;

/// block_iq4_nl: a half d and 16 nibble bytes, where the low nibble of byte
/// j is element j and the high nibble is element j + 16, both mapped through
/// the nonlinear kValuesIq4Nl codebook.
constexpr std::size_t kIq4NlBlockBytes = 18;
constexpr std::size_t kIq3xxsBlockBytes = 98;
constexpr std::size_t kIq3sBlockBytes = 110;
constexpr std::size_t kIq2sBlockBytes = 82;
constexpr std::size_t kIq4XsBlockBytes = 136;
constexpr std::size_t kQ20BlockBytes = 18;
constexpr std::size_t kQ6kBlockBytes = 210;

template<WeightType kType>
__device__ __forceinline__ std::size_t RoutedF16RowBytes(std::size_t k) {
  return kType == WeightType::kQ4_K      ? (k / 256) * sizeof(Q4KBlock)
         : kType == WeightType::kQ5_K    ? (k / 256) * kQ5KBlockBytes
         : kType == WeightType::kQ5_1    ? (k / 32) * sizeof(Q5_1Block)
         : kType == WeightType::kIQ4_NL  ? (k / 32) * kIq4NlBlockBytes
         : kType == WeightType::kIQ3_XXS ? (k / 256) * kIq3xxsBlockBytes
         : kType == WeightType::kIQ3_S   ? (k / 256) * kIq3sBlockBytes
         : kType == WeightType::kIQ2_S   ? (k / 256) * kIq2sBlockBytes
         : kType == WeightType::kIQ4_XS  ? (k / 256) * kIq4XsBlockBytes
         : kType == WeightType::kQ2_0    ? (k / 64) * kQ20BlockBytes
         : kType == WeightType::kQ6_K    ? (k / 256) * kQ6kBlockBytes
                                         : (k / 32) * sizeof(Q8_0Block);
}

/// Bit `s` of each of the four bytes of `w`, packed into bits 0-3.
__device__ __forceinline__ std::uint32_t GatherBit(std::uint32_t w, int s) {
  // 0x01020408 moves byte b's bit to bit 24 + b.
  return (((w >> s) & 0x01010101U) * 0x01020408U) >> 24U;
}

/// Four packed 5-bit codes: `nib` holds the 4-bit parts one per byte, `bits`
/// bits j..j+3 of the Q5_1 high-bit word, spread to bit 4 of each byte.
__device__ __forceinline__ std::uint32_t SpreadHighBits(std::uint32_t bits) {
  // 0x00204081 = 1 + 2^7 + 2^14 + 2^21: bit b of `bits` lands at 8b, every
  // cross term falls off the 0x01010101 mask.
  return (__umul24(bits, 0x00204081U) & 0x01010101U) << 4U;
}

/// Lane i receives lane i ^ 16's value (XorLane(v, 16)) without tying the
/// result to v: every lane is active, so the instruction's old operand is
/// never read and needs no copy of v.
__device__ __forceinline__ std::uint32_t SwapHalves(std::uint32_t v) {
  std::uint32_t old;
  asm("" : "=v"(old));
  return static_cast<std::uint32_t>(
      __builtin_amdgcn_permlanex16(static_cast<int>(old), static_cast<int>(v),
                                   0x76543210, 0xfedcba98, false, false));
}

/// Four halves from four code bytes: 1024 + q as F16, minus `magic` (1024,
/// or 1152 for a signed byte carried as q + 128), then the affine.
__device__ __forceinline__ void CodesToHalves(std::uint32_t codes,
                                              __half2 magic, __half2 scale2,
                                              __half2 bias2, __half2& lo,
                                              __half2& hi) {
  const std::uint32_t p0 =
      __builtin_amdgcn_perm(codes, kHalfMagic, 0x01050004U);
  const std::uint32_t p1 =
      __builtin_amdgcn_perm(codes, kHalfMagic, 0x03070206U);
  lo = __hfma2(__hadd2(__builtin_bit_cast(__half2, p0), magic), scale2, bias2);
  hi = __hfma2(__hadd2(__builtin_bit_cast(__half2, p1), magic), scale2, bias2);
}

template<WeightType kType, int BM, int BN, int BK>
__launch_bounds__(256) __global__
    void RoutedF16GEMMKernel(const void* __restrict__ w,
                             const __half* __restrict__ x,
                             const std::int32_t* __restrict__ tiles,
                             const std::int32_t* __restrict__ pad_bounds,
                             const std::int32_t* __restrict__ x_rows,
                             const std::int32_t* __restrict__ rows_token,
                             const std::int32_t* __restrict__ rows_slot,
                             std::uint32_t used,
                             const float* __restrict__ swiglu_gate,
                             float* __restrict__ out,
                             __half* __restrict__ out_half, std::size_t m,
                             std::size_t k) {
  static_assert(BM == 128 || BM == 256, "eight waves, 16-row tiles");
  static_assert(BN % 16 == 0 && BN / 16 <= 8);
  static_assert(BK == 2, "one stage is one 32-byte Q4_K nibble group");
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = BM / 128;  // 16-row tiles per wave
  constexpr bool kQ5 = kType == WeightType::kQ5_1;
  constexpr bool kQ5K = kType == WeightType::kQ5_K;
  constexpr bool kQ8 = kType == WeightType::kQ8_0;
  constexpr bool kIQ4 = kType == WeightType::kIQ4_NL;
  constexpr bool kIQ3X = kType == WeightType::kIQ3_XXS;
  constexpr bool kIQ3S = kType == WeightType::kIQ3_S;
  constexpr bool kIQ2 = kType == WeightType::kIQ2_S;
  constexpr bool kI4XS = kType == WeightType::kIQ4_XS;
  constexpr bool kQ20 = kType == WeightType::kQ2_0;
  constexpr bool kQ6K = kType == WeightType::kQ6_K;
  constexpr bool kKQuant = kType == WeightType::kQ4_K || kQ5K;
  // Codebook formats stage one assembled uint4 per 32-element sub-block.
  constexpr bool kNativePacked =
      kIQ4 || kI4XS || kIQ3X || kIQ3S || kIQ2 || kQ20;
  // 16-byte code chunks per row and stage: Q4_K's nibble pair and Q5_1's
  // two nibble blocks are two, Q8_0's two byte blocks are four, and a Q6_K
  // stage carries two 32-element segments of {ql half, qh half} each.
  constexpr int kChunks = kQ6K ? 4 * BK : kQ8 ? 2 * BK : BK;

  // Codes and affine parameters are staged in LDS, with nearby rows
  // permuted across the bank groups. The activation plane is
  // [kb][16-element quarter][token][16 B], so a fragment read is 256
  // contiguous bytes; the epilogue reuses it all.
  // Q5_1 rows stage their blocks as stored: a row's BK blocks are 16-byte
  // aligned, so whole 16-byte chunks load coalesced and each wave reads its
  // rows' scale, high bits and codes from LDS.
  constexpr int kQ5Bytes = BK * 24;  // one row's blocks per stage
  constexpr int kCodeBytes = kQ5 ? BM * kQ5Bytes : BM * kChunks * 16;
  constexpr int kHighBytes = (kQ5K || kQ6K) ? BK * BM * 4 : 0;
  constexpr int kScaleBytes = kQ5 ? 0 : BK * BM * 4;
  // One slot of padding per activation quarter plane: the eight chunks of
  // a token then land on eight bank groups when they are written.
  constexpr int kActStride = BN + 1;
  constexpr int kActBytes = BK * 4 * kActStride * 16;
  // The epilogue transposes one 16x16 tile per wave through the same
  // bytes (8 KB), which the narrow tile's stages do not reach.
  constexpr int kStageBytes = kCodeBytes + kHighBytes + kScaleBytes + kActBytes;
  constexpr int kEpilogueBytes = 8 * 1024;
  constexpr int kLdsBytes =
      kStageBytes > kEpilogueBytes ? kStageBytes : kEpilogueBytes;
  // The codebook formats resolve magnitudes through tables that live past
  // the shared stage/epilogue region, so tile reuse never touches them.
  constexpr std::size_t kLutBytes = ((kIQ4 || kI4XS) ? 1024 : 0) +
                                    (kIQ3X ? 1024 : 0) + (kIQ3S ? 2048 : 0) +
                                    (kIQ2 ? 8192 : 0);
  __shared__
      __attribute__((aligned(16))) std::uint8_t lds[kLdsBytes + kLutBytes];
  auto* s_codes = reinterpret_cast<uint4*>(lds);
  auto* s_high = reinterpret_cast<std::uint32_t*>(lds + kCodeBytes);
  auto* s_scale =
      reinterpret_cast<std::uint32_t*>(lds + kCodeBytes + kHighBytes);
  auto* s_act =
      reinterpret_cast<uint4*>(lds + kCodeBytes + kHighBytes + kScaleBytes);
  auto* s_lut = reinterpret_cast<std::uint32_t*>(lds + kLdsBytes);
  auto* s_lut64 = reinterpret_cast<std::uint64_t*>(lds + kLdsBytes);

  const std::int32_t tile = tiles[blockIdx.y];
  if (tile < 0) {
    // Sentinel from the device-built decode tile map: this launch slot is
    // past the real tiles of the current routing.
    return;
  }
  const int expert = tile & 0xFFFF;
  const int t_local = (tile >> 16) * BN;
  const int bucket_begin = pad_bounds[expert];
  const int bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  const int live_tok_tiles =
      std::min(kTokTiles, (bucket_rows - t_local + 15) / 16);
  const int num_kb = static_cast<int>(k / 32);
  const int m_i = static_cast<int>(m);
  const std::size_t row_bytes = RoutedF16RowBytes<kType>(k);
  const auto* w_expert = static_cast<const std::uint8_t*>(w) +
                         static_cast<std::size_t>(expert) * m * row_bytes;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int r_block = static_cast<int>(blockIdx.x) * BM;

  // Thread tid fetches 16-byte chunk tid % 2 of row tid / 2 + 128u.
  const int f_c = tid & 1;
  const std::uint8_t* f_ptr[kWaveRowTiles];
  bool f_live[kWaveRowTiles];
  uint4 f_header[kWaveRowTiles];
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
    const int r = r_block + (tid >> 1) + (u * 128);
    f_live[u] = r < m_i;
    f_ptr[u] = w_expert +
               static_cast<std::size_t>(f_live[u] ? r : (m_i - 1)) * row_bytes;
    f_header[u] = make_uint4(0u, 0u, 0u, 0u);
  }
  // The next stage's weights and activations, fetched one stage ahead. The
  // (scale, bias) pair is derived from the raw header word only when the
  // stage is committed, so nothing waits on the loads before the compute.
  uint4 f_codes[kWaveRowTiles];
  uint4 f_codes_hi[kWaveRowTiles];  ///< Q8_0: the block's second 16 codes
  uint4 f_qh[kWaveRowTiles][2];     ///< Q5_K: the superblock's high bits
  std::uint32_t f_high[kWaveRowTiles];
  std::uint32_t f_q6sc[kWaveRowTiles];  ///< Q6_K: half2 (d*sA, d*sB)
  std::uint32_t f_dm[kWaveRowTiles];    ///< Q8_0 / IQ4_NL: d
  int f_sb32[kWaveRowTiles];            ///< Q4_K: the K block in its superblock
  constexpr int kActFetch = BN <= 64 ? 2 : 4;
  uint4 a_data[kActFetch];
  // Q5_1 chunk c is part c % kQ5Parts of row c / kQ5Parts.
  constexpr int kQ5Parts = kQ5Bytes / 16;
  constexpr int kQ5Chunks = BM * kQ5Parts;
  constexpr int kQ5Fetch = kQ5 ? (kQ5Chunks + 255) / 256 : 1;
  static_assert(!kQ5 || kQ5Bytes % 16 == 0);
  const std::uint8_t* q5_src[kQ5Fetch];
  uint4 f_q5[kQ5Fetch];
  if constexpr (kQ5) {
#pragma unroll
    for (int i = 0; i < kQ5Fetch; ++i) {
      const int chunk = tid + (i * 256);
      const int r = r_block + (chunk / kQ5Parts);
      q5_src[i] = chunk < kQ5Chunks
                      ? w_expert +
                            (static_cast<std::size_t>(r < m_i ? r : m_i - 1) *
                             row_bytes) +
                            ((chunk % kQ5Parts) * 16)
                      : nullptr;
    }
  }

  // Activation fetch: BN tokens x (BK * 64) bytes per stage in 16-byte
  // chunks, eight per token; each thread fetches consecutive 256-chunk
  // strides, up to four for a 128-token tile.
  constexpr int kActChunks = BN * BK * 4;
  static_assert(kActChunks <= kActFetch * 256);
  const __half* a_src[kActFetch];
  int a_slot[kActFetch];
#pragma unroll
  for (int i = 0; i < kActFetch; ++i) {
    const int chunk = tid + (i * 256);
    const int t = chunk / (BK * 4);
    const int sub = chunk % (BK * 4);
    const int c_row = t_local + t;
    const std::int32_t src = (chunk < kActChunks && c_row < bucket_rows)
                                 ? x_rows[bucket_begin + c_row]
                                 : -1;
    a_src[i] = src >= 0 ? x + (static_cast<std::size_t>(src) * k) + (sub * 8)
                        : nullptr;
    // s_act[(kb * 4 + quarter) * kActStride + t]
    a_slot[i] = chunk < kActChunks ? (sub * kActStride) + t : -1;
  }

  const auto swizzle = [](int row, int c) {
    return (row * kChunks) +
           (c ^ (kChunks == 4 ? ((row >> 1) & 3) : ((row >> 2) & 1)));
  };

  const auto fetch_stage = [&](int kb0) {
    if constexpr (kQ5) {
#pragma unroll
      for (int i = 0; i < kQ5Fetch; ++i) {
        f_q5[i] = q5_src[i] != nullptr
                      ? *reinterpret_cast<const uint4*>(q5_src[i] + (kb0 * 24))
                      : make_uint4(0u, 0u, 0u, 0u);
      }
    }
#pragma unroll
    for (int u = 0; u < (kQ5 ? 0 : kWaveRowTiles); ++u) {
      if constexpr (kIQ4) {
        // block_iq4_nl is 18 bytes: d followed by the 16 code bytes.
        const auto* blk = f_ptr[u] + ((kb0 + f_c) * kIq4NlBlockBytes);
        f_dm[u] = *reinterpret_cast<const std::uint16_t*>(blk);
        f_codes[u] = *reinterpret_cast<const uint4*>(blk + 2);
      } else if constexpr (kI4XS) {
        // block_iq4_xs is 136 bytes / 256 elements: the same 32-element
        // sub-block layout as IQ4_NL (the golden's lane/nibble mapping),
        // but with a 6-bit scale per sub-block stored as
        // [d | scales_h | scales_l[4] | qs[128]]. The effective sub-block
        // scale d * (ls - 32) is pre-multiplied into f_dm so the IQ4_NL
        // compute arm serves it unchanged.
        const auto* blk = f_ptr[u] + (((kb0 + f_c) >> 3) * kIq4XsBlockBytes);
        const int ib32 = (kb0 + f_c) & 7;
        const float d = __half2float(__builtin_bit_cast(
            __half, *reinterpret_cast<const std::uint16_t*>(blk)));
        const std::uint32_t sh =
            *reinterpret_cast<const std::uint16_t*>(blk + 2);
        const std::uint32_t ls =
            ((blk[4 + (ib32 >> 1)] >> (4 * (ib32 & 1))) & 0xFU) |
            (((sh >> (2 * ib32)) & 0x3U) << 4U);
        f_dm[u] = static_cast<std::uint32_t>(__half_as_ushort(__float2half_rn(
            d * static_cast<float>(static_cast<int>(ls) - 32))));
        // The 136-byte row stride leaves codes 8-byte aligned at worst;
        // stage through memcpy like the IQ3 arm rather than a uint4 load.
        __builtin_memcpy(&f_codes[u], blk + 8 + (ib32 * 16), 16);
      } else if constexpr (kIQ3X || kIQ3S) {
        // IQ3_XXS/IQ3_S stage one 32-element sub-block per chunk: the
        // eight group indices (x, y), the aux word (XXS) or sign bytes
        // (S) in z, and d packed with the sub-block scale bits (and the
        // qh byte for IQ3_S) in w.
        const std::size_t bytes = kIQ3X ? kIq3xxsBlockBytes : kIq3sBlockBytes;
        const auto* blk = f_ptr[u] + (((kb0 + f_c) >> 3) * bytes);
        const int ib32 = (kb0 + f_c) & 7;
        const std::uint32_t dh16 = *reinterpret_cast<const std::uint16_t*>(blk);
        const __half dh =
            __builtin_bit_cast(__half, static_cast<std::uint16_t>(dh16));
        uint2 q32{};
        __builtin_memcpy(&q32, blk + 2 + 8 * ib32, 8);
        if constexpr (kIQ3X) {
          std::uint32_t aux = 0;
          __builtin_memcpy(&aux, blk + 66 + 4 * ib32, 4);
          const std::uint16_t dbits = __half_as_ushort(
              __float2half_rn(__half2float(dh) *
                              ((0.5F + static_cast<float>(aux >> 28)) * 0.5F)));
          f_codes[u] = make_uint4(
              q32.x, q32.y, aux,
              f_live[u] ? dh16 | (static_cast<std::uint32_t>(dbits) << 16)
                        : 0U);
        } else {
          std::uint32_t signs = 0;
          __builtin_memcpy(&signs, blk + 74 + 4 * ib32, 4);
          const std::uint32_t qh = blk[66 + ib32];
          const std::uint32_t nib1 =
              1U + 2U * ((static_cast<std::uint32_t>(blk[106 + (ib32 >> 1)]) >>
                          (4 * (ib32 & 1))) &
                         0xFU);
          f_codes[u] =
              make_uint4(q32.x, q32.y, signs,
                         f_live[u] ? dh16 | (nib1 << 16) | (qh << 24) : 0U);
        }
      } else if constexpr (kIQ2) {
        // block_iq2_s is 82 bytes. Stage one 32-element sub-block per
        // chunk: the four grid-index bytes (x), the four sign bytes that
        // physically overlap qs (y), the qh high-bit byte (z), and the
        // two sub-block scales (low nibble for groups 0-1, high nibble
        // for 2-3) pre-multiplied into d as an half2 pair (w).
        const auto* blk = f_ptr[u] + (((kb0 + f_c) >> 3) * kIq2sBlockBytes);
        const int ib32 = (kb0 + f_c) & 7;
        const std::uint32_t dh16 = *reinterpret_cast<const std::uint16_t*>(blk);
        const float d = __half2float(
            __builtin_bit_cast(__half, static_cast<std::uint16_t>(dh16)));
        std::uint32_t qs4 = 0;
        std::uint32_t sn4 = 0;
        __builtin_memcpy(&qs4, blk + 2 + 4 * ib32, 4);
        __builtin_memcpy(&sn4, blk + 34 + 4 * ib32, 4);
        const std::uint32_t sc = blk[74 + ib32];
        const std::uint32_t qh = blk[66 + ib32];
        const __half db_lo = __float2half_rn(d * (0.5F + (sc & 0xFU)) * 0.25F);
        const __half db_hi = __float2half_rn(d * (0.5F + (sc >> 4)) * 0.25F);
        f_codes[u] = make_uint4(
            qs4, sn4, qh,
            f_live[u] ? static_cast<std::uint32_t>(__half_as_ushort(db_lo)) |
                            (static_cast<std::uint32_t>(__half_as_ushort(db_hi))
                             << 16)
                      : 0U);
      } else if constexpr (kQ20) {
        // block_q2_0 is 18 bytes / 64 elements: two consecutive chunks
        // share one block and its f16 scale. The chunk's 8 code bytes
        // (four elements each, value d * (q - 1)) ride x/y and the
        // block's d is staged in w.
        const auto* blk = f_ptr[u] + (((kb0 + f_c) >> 1) * kQ20BlockBytes);
        const int sub = (kb0 + f_c) & 1;
        std::uint32_t q0 = 0;
        std::uint32_t q1 = 0;
        __builtin_memcpy(&q0, blk + 2 + 8 * sub, 4);
        __builtin_memcpy(&q1, blk + 6 + 8 * sub, 4);
        const std::uint32_t dh16 = *reinterpret_cast<const std::uint16_t*>(blk);
        f_codes[u] = make_uint4(q0, q1, 0U, f_live[u] ? dh16 : 0U);
      } else if constexpr (kQ6K) {
        // block_q6_K is 210 bytes / 256 elements (6.5625 bpw, the
        // {ql, qh, scales, d} order): the golden's Q6Value
        // lane mapping. One 32-element segment per chunk reads its code
        // bytes from one ql half (byte e for even segments, byte 32 + e
        // for odd; low nibbles for segments 0/1, high nibbles for 2/3)
        // and its qh bytes supply the two-bit high parts at shift
        // 2 * segment. The per-16-element signed scales are
        // pre-multiplied with the block d into a half2 (like IQ4_XS);
        // the segment index rides the s_high slot.
        const auto* blk = f_ptr[u] + (((kb0 + f_c) >> 3) * kQ6kBlockBytes);
        const int ib32 = (kb0 + f_c) & 7;
        const int seg = ib32 & 3;
        const int hl = ib32 >> 2;
        const std::uint8_t* ql = blk + (hl * 64) + ((seg & 1) ? 32 : 0);
        __builtin_memcpy(&f_codes[u], ql, 16);
        __builtin_memcpy(&f_codes_hi[u], ql + 16, 16);
        __builtin_memcpy(&f_qh[u][0], blk + 128 + (hl * 32), 16);
        __builtin_memcpy(&f_qh[u][1], blk + 128 + (hl * 32) + 16, 16);
        f_high[u] = static_cast<std::uint32_t>(seg);
        const float d = __half2float(__builtin_bit_cast(
            __half, *reinterpret_cast<const std::uint16_t*>(blk + 208)));
        const std::int8_t* sc =
            reinterpret_cast<const std::int8_t*>(blk + 192) + (hl * 8) +
            (seg * 2);
        const __half2 ds = __floats2half2_rn(d * static_cast<float>(sc[0]),
                                             d * static_cast<float>(sc[1]));
        f_q6sc[u] = f_live[u] ? __builtin_bit_cast(std::uint32_t, ds) : 0U;
      } else if constexpr (kQ8) {
        // block_q8_0 is 34 bytes, so the code loads are 2-byte aligned.
        const auto* blk = f_ptr[u] + ((kb0 + f_c) * 34);
        f_dm[u] = *reinterpret_cast<const std::uint16_t*>(blk);
        __builtin_memcpy(&f_codes[u], blk + 2, 16);
        __builtin_memcpy(&f_codes_hi[u], blk + 18, 16);
      } else {
        constexpr int kBlockChunks = kQ5K ? 11 : 9;
        constexpr int kCodeChunk = kQ5K ? 3 : 1;
        const int block = kb0 / 8;
        const auto* blk =
            reinterpret_cast<const uint4*>(f_ptr[u]) + (block * kBlockChunks);
        // The K sweep enters a new superblock every eight Q8-sized blocks.
        if (kb0 % 8 == 0) {
          f_header[u] = blk[0];
          if constexpr (kQ5K) {
            f_qh[u][0] = blk[1];
            f_qh[u][1] = blk[2];
          }
        }
        const int sb32 = (kb0 % 8) + f_c;
        f_codes[u] = blk[kCodeChunk + (sb32 / 2) * 2 + f_c];
        f_sb32[u] = sb32;
        if constexpr (kQ5K) {
          // Bit sb32 of the 32 high-bit bytes, packed as the Q5_1 word.
          const std::uint32_t qh[8] = {f_qh[u][0].x, f_qh[u][0].y, f_qh[u][0].z,
                                       f_qh[u][0].w, f_qh[u][1].x, f_qh[u][1].y,
                                       f_qh[u][1].z, f_qh[u][1].w};
          std::uint32_t high = 0;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            high |= GatherBit(qh[i], sb32) << (4 * i);
          }
          f_high[u] = high;
        }
      }
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      a_data[i] = a_src[i] != nullptr
                      ? *reinterpret_cast<const uint4*>(a_src[i] + (kb0 * 32))
                      : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  const auto commit_stage = [&]() {
    if constexpr (kQ5) {
#pragma unroll
      for (int i = 0; i < kQ5Fetch; ++i) {
        if (q5_src[i] != nullptr)
          s_codes[tid + (i * 256)] = f_q5[i];
      }
    }
#pragma unroll
    for (int u = 0; u < (kQ5 ? 0 : kWaveRowTiles); ++u) {
      const int row = (tid >> 1) + (u * 128);
      std::uint32_t scale_bias = 0;
      if constexpr (kQ8) {
        s_codes[swizzle(row, 2 * f_c)] = f_codes[u];
        s_codes[swizzle(row, (2 * f_c) + 1)] = f_codes_hi[u];
        scale_bias = f_live[u] ? f_dm[u] : 0U;  // half2 (d, 0)
      } else if constexpr (kQ6K) {
        s_codes[swizzle(row, 4 * f_c)] = f_codes[u];
        s_codes[swizzle(row, (4 * f_c) + 1)] = f_codes_hi[u];
        s_codes[swizzle(row, (4 * f_c) + 2)] = f_qh[u][0];
        s_codes[swizzle(row, (4 * f_c) + 3)] = f_qh[u][1];
      } else {
        s_codes[swizzle(row, f_c)] = f_codes[u];
      }
      if constexpr (kQ5K || kQ6K) {
        s_high[(f_c * BM) + row] = f_high[u];
      }
      if constexpr (kIQ4 || kI4XS) {
        // half2 (d, d): the codebook values carry the code itself, so only
        // the block scale remains; the affine bias is unused. IQ4_XS rides
        // this path with the per-sub-block scale pre-multiplied at fetch.
        scale_bias = f_live[u] ? (f_dm[u] | (f_dm[u] << 16)) : 0U;
      } else if constexpr (kIQ3X || kIQ3S || kIQ2 || kQ20) {
        // The sub-block scale travels inside the staged word itself
        // (w), so the shared scale slot stays zero.
        scale_bias = 0U;
      } else if constexpr (kQ6K) {
        scale_bias = f_q6sc[u];
      } else if constexpr (!kQ8) {
        const int sb32 = f_sb32[u];
        // The first four scale/min bytes live in y/z. Their remaining
        // low nibbles share w; the high two bits reuse those same y/z bytes.
        // Decode the fixed words directly, avoiding header-member selection.
        const uint4 h = f_header[u];
        const unsigned shift = 8U * (static_cast<unsigned>(sb32) & 3U);
        const std::uint32_t scale_byte = h.y >> shift;
        const std::uint32_t min_byte = h.z >> shift;
        std::uint32_t sc;
        std::uint32_t mn;
        if (sb32 < 4) {
          sc = scale_byte & 0x3FU;
          mn = min_byte & 0x3FU;
        } else {
          const std::uint32_t packed = h.w >> shift;
          sc = (packed & 0x0FU) | ((scale_byte & 0xC0U) >> 2U);
          mn = ((packed >> 4U) & 0x0FU) | ((min_byte & 0xC0U) >> 2U);
        }
        const __half2 dm = __builtin_bit_cast(__half2, f_header[u].x);
        const float scale =
            f_live[u] ? __low2float(dm) * static_cast<float>(sc) : 0.0F;
        const float offset =
            f_live[u] ? __high2float(dm) * static_cast<float>(mn) : 0.0F;
        scale_bias = __builtin_bit_cast(std::uint32_t,
                                        __floats2half2_rn(scale, -offset));
      }
      s_scale[(f_c * BM) + row] = scale_bias;
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      if (a_slot[i] >= 0) {
        s_act[a_slot[i]] = a_data[i];
      }
    }
  };

  v8f acc[kWaveRowTiles][kTokTiles];
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
      acc[u][j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

  const __half2 magic =
      __floats2half2_rn(kQ8 ? -1152.0F : -1024.0F, kQ8 ? -1152.0F : -1024.0F);
  const auto compute_stage = [&]() {
    uint4 raw[kWaveRowTiles][BK];
    if constexpr (!kQ8 && !kQ5) {
#pragma unroll
      for (int u = 0; u < kWaveRowTiles; ++u) {
        const int row = (wave_id * 16) + (u * 128) + sub_lane;
#pragma unroll
        for (int c = 0; c < BK; ++c) {
          raw[u][c] = s_codes[swizzle(row, c)];
        }
      }
    }
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      v16h a_lo[kWaveRowTiles];
      v16h a_hi[kWaveRowTiles];
#pragma unroll
      for (int u = 0; u < kWaveRowTiles; ++u) {
        const int row = (wave_id * 16) + (u * 128) + sub_lane;
        // Q5_1 block kb of the row: d | m, the high-bit word, 16 codes.
        uint2 q5_head{};
        uint2 q5_lo{};
        uint2 q5_hi{};
        std::uint32_t scale_bias = 0U;
        if constexpr (kQ5) {
          const std::uint8_t* blk = lds + (row * kQ5Bytes) + (kb * 24);
          q5_head = *reinterpret_cast<const uint2*>(blk);
          q5_lo = *reinterpret_cast<const uint2*>(blk + 8);
          q5_hi = *reinterpret_cast<const uint2*>(blk + 16);
          scale_bias = r_block + row < m_i ? q5_head.x : 0U;
        } else if constexpr (!kIQ3X && !kIQ3S && !kIQ2 && !kQ20) {
          scale_bias = s_scale[(kb * BM) + row];
        }
        const __half2 sb = __builtin_bit_cast(__half2, scale_bias);
        const __half2 scale2 = __low2half2(sb);
        const __half2 bias2 = __high2half2(sb);
        if constexpr (kIQ4 || kI4XS) {
          // IQ4_NL/IQ4_XS: byte j of a block holds element j in its low nibble
          // and element j + 16 in its high nibble. The code halves are
          // built with the same permutes as the affine formats — so the
          // fragment placement matches the proven paths exactly — then
          // translated through the nonlinear codebook and scaled by the
          // block's d (no bias: the codebook carries the sign).
          __half2 h[16];
          const std::uint32_t words[4] = {raw[u][kb].x, raw[u][kb].y,
                                          raw[u][kb].z, raw[u][kb].w};
          const auto* lut16 = reinterpret_cast<const __half*>(s_lut);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const std::uint32_t lo_codes = words[i] & 0x0F0F0F0FU;
            const std::uint32_t hi_codes = (words[i] >> 4U) & 0x0F0F0F0FU;
            const std::uint32_t lo0 =
                __builtin_amdgcn_perm(lo_codes, kHalfMagic, 0x01050004U);
            const std::uint32_t lo1 =
                __builtin_amdgcn_perm(lo_codes, kHalfMagic, 0x03070206U);
            const std::uint32_t hi0 =
                __builtin_amdgcn_perm(hi_codes, kHalfMagic, 0x01050004U);
            const std::uint32_t hi1 =
                __builtin_amdgcn_perm(hi_codes, kHalfMagic, 0x03070206U);
            h[2 * i] = __hmul2(
                __halves2half2(lut16[lo0 & 0xFU], lut16[(lo0 >> 16) & 0xFU]),
                scale2);
            h[2 * i + 1] = __hmul2(
                __halves2half2(lut16[lo1 & 0xFU], lut16[(lo1 >> 16) & 0xFU]),
                scale2);
            h[8 + 2 * i] = __hmul2(
                __halves2half2(lut16[hi0 & 0xFU], lut16[(hi0 >> 16) & 0xFU]),
                scale2);
            h[8 + 2 * i + 1] = __hmul2(
                __halves2half2(lut16[hi1 & 0xFU], lut16[(hi1 >> 16) & 0xFU]),
                scale2);
          }
          __builtin_memcpy(&a_lo[u], &h[0], 32);
          __builtin_memcpy(&a_hi[u], &h[8], 32);
        } else if constexpr (kIQ3X || kIQ3S) {
          // IQ3_XXS/IQ3_S: two grid indices per eight elements and four
          // magnitudes per grid entry. z carries the aux word (XXS) or
          // the four sign bytes (S); w carries the packed scale (and
          // qh for IQ3_S). Values mirror the golden reference's float
          // arithmetic before rounding into F16 fragments.
          const std::uint32_t qw[2] = {raw[u][kb].x, raw[u][kb].y};
          const std::uint32_t z = raw[u][kb].z;
          const std::uint32_t w = raw[u][kb].w;
          float db;
          if constexpr (kIQ3X) {
            const std::uint16_t db16 = static_cast<std::uint16_t>(w >> 16);
            db = __half2float(__builtin_bit_cast(__half, db16));
          } else {
            const std::uint16_t d16 = static_cast<std::uint16_t>(w & 0xFFFFU);
            db = __half2float(__builtin_bit_cast(__half, d16)) *
                 static_cast<float>((w >> 16) & 0xFFU);
          }
          const std::uint32_t qh = (w >> 24) & 0xFFU;
          float v[32];
#pragma unroll
          for (int e = 0; e < 32; ++e) {
            const int l = (e >> 3) & 3;
            const int j = e & 7;
            const int qb = 2 * l + (j >> 2);  // qs byte within the chunk
            const std::uint32_t gi = (qw[qb >> 2] >> (8 * (qb & 3))) & 0xFFU;
            std::uint32_t g = gi;
            if constexpr (kIQ3S)
              g |= ((qh >> qb) & 1U) << 8;
            const float mag =
                static_cast<float>((s_lut[g] >> (8 * (j & 3))) & 0xFFU);
            const bool neg =
                kIQ3X ? ((quant::kSignsIq2xs[(z >> (7 * l)) & 127U] >> j) &
                         1U) != 0U
                      : (((z >> (8 * l)) >> j) & 1U) != 0U;
            v[e] = neg ? -db * mag : db * mag;
          }
          __half2 h[16];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            h[2 * i] = __floats2half2_rn(v[4 * i], v[4 * i + 1]);
            h[2 * i + 1] = __floats2half2_rn(v[4 * i + 2], v[4 * i + 3]);
            h[8 + 2 * i] = __floats2half2_rn(v[16 + 4 * i], v[16 + 4 * i + 1]);
            h[8 + 2 * i + 1] =
                __floats2half2_rn(v[16 + 4 * i + 2], v[16 + 4 * i + 3]);
          }
          __builtin_memcpy(&a_lo[u], &h[0], 32);
          __builtin_memcpy(&a_hi[u], &h[8], 32);
        } else if constexpr (kIQ2) {
          // IQ2_S: ten-bit grid index (qs byte plus two qh high bits per
          // group) selects eight byte magnitudes; signs are one bit per
          // element in the overlapping y word; the sub-block scale was
          // pre-multiplied into d at fetch time.
          const std::uint32_t x = raw[u][kb].x;
          const std::uint32_t sn = raw[u][kb].y;
          const std::uint32_t qh = raw[u][kb].z;
          const __half2 dbs = __builtin_bit_cast(__half2, raw[u][kb].w);
          float v[32];
#pragma unroll
          for (int e = 0; e < 32; ++e) {
            const int l = (e >> 3) & 3;
            const int j = e & 7;
            const std::uint32_t gi = (x >> (8 * l)) & 0xFFU;
            const std::uint32_t idx = gi | (((qh << (8 - 2 * l)) & 0x300U));
            const float mag = static_cast<float>(
                (s_lut64[idx] >> (8 * static_cast<std::size_t>(j))) & 0xFFU);
            const float db =
                __half2float(l < 2 ? __low2half(dbs) : __high2half(dbs));
            v[e] = (((sn >> (8 * l)) >> j) & 1U) != 0U ? -db * mag : db * mag;
          }
          __half2 h[16];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            h[2 * i] = __floats2half2_rn(v[4 * i], v[4 * i + 1]);
            h[2 * i + 1] = __floats2half2_rn(v[4 * i + 2], v[4 * i + 3]);
            h[8 + 2 * i] = __floats2half2_rn(v[16 + 4 * i], v[16 + 4 * i + 1]);
            h[8 + 2 * i + 1] =
                __floats2half2_rn(v[16 + 4 * i + 2], v[16 + 4 * i + 3]);
          }
          __builtin_memcpy(&a_lo[u], &h[0], 32);
          __builtin_memcpy(&a_hi[u], &h[8], 32);
        } else if constexpr (kQ20) {
          // Q2_0: four elements per byte, value = d * (q - 1); the
          // block's d was staged in w at fetch time. Values go through
          // the same float arithmetic as the golden reference before
          // rounding into F16 fragments.
          const std::uint32_t qw[2] = {raw[u][kb].x, raw[u][kb].y};
          const float d = __half2float(__builtin_bit_cast(
              __half, static_cast<std::uint16_t>(raw[u][kb].w)));
          float v[32];
#pragma unroll
          for (int e = 0; e < 32; ++e) {
            // Element e lives at bit 2*e of the staged word (x holds code
            // bytes 0-3 = elements 0-15, y bytes 4-7 = elements 16-31):
            // byte e/4 at bit offset 2*(e%4) within it, as the golden.
            const std::uint32_t q = (qw[e >> 4] >> (2 * (e & 15))) & 0x3U;
            v[e] = (static_cast<float>(q) - 1.0F) * d;
          }
          __half2 h[16];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            h[2 * i] = __floats2half2_rn(v[4 * i], v[4 * i + 1]);
            h[2 * i + 1] = __floats2half2_rn(v[4 * i + 2], v[4 * i + 3]);
            h[8 + 2 * i] = __floats2half2_rn(v[16 + 4 * i], v[16 + 4 * i + 1]);
            h[8 + 2 * i + 1] =
                __floats2half2_rn(v[16 + 4 * i + 2], v[16 + 4 * i + 3]);
          }
          __builtin_memcpy(&a_lo[u], &h[0], 32);
          __builtin_memcpy(&a_hi[u], &h[8], 32);
        } else if constexpr (kQ6K) {
          // Q6_K: the staged code half supplies 32 nibbles (low or high
          // by segment), and qh byte e adds the two-bit high part at
          // shift 2 * segment. The block d is already folded into the
          // per-16 scale half2; values stay in float before the single
          // F16 fragment rounding, matching the golden's ordering up to
          // that rounding.
          const uint4 c0 = s_codes[swizzle(row, 4 * kb)];
          const uint4 c1 = s_codes[swizzle(row, (4 * kb) + 1)];
          const uint4 hb0 = s_codes[swizzle(row, (4 * kb) + 2)];
          const uint4 hb1 = s_codes[swizzle(row, (4 * kb) + 3)];
          const std::uint32_t seg = s_high[(kb * BM) + row] & 3U;
          const std::uint32_t sbs = s_scale[(kb * BM) + row];
          const std::uint32_t cw[8] = {c0.x, c0.y, c0.z, c0.w,
                                       c1.x, c1.y, c1.z, c1.w};
          const std::uint32_t hw[8] = {hb0.x, hb0.y, hb0.z, hb0.w,
                                       hb1.x, hb1.y, hb1.z, hb1.w};
          float v[32];
#pragma unroll
          for (int e = 0; e < 32; ++e) {
            const std::uint32_t lo =
                (cw[e >> 2] >> ((e & 3) * 8 + ((seg & 2U) != 0U ? 4U : 0U))) &
                0xFU;
            const std::uint32_t hi =
                (hw[e >> 2] >> ((e & 3) * 8 + 2U * seg)) & 3U;
            const float ds = __half2float(__builtin_bit_cast(
                __half, static_cast<std::uint16_t>(e < 16 ? (sbs & 0xFFFFU)
                                                          : (sbs >> 16))));
            v[e] = ds * static_cast<float>((static_cast<int>(hi) << 4) +
                                           static_cast<int>(lo) - 32);
          }
          __half2 h[16];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            h[2 * i] = __floats2half2_rn(v[4 * i], v[4 * i + 1]);
            h[2 * i + 1] = __floats2half2_rn(v[4 * i + 2], v[4 * i + 3]);
            h[8 + 2 * i] = __floats2half2_rn(v[16 + 4 * i], v[16 + 4 * i + 1]);
            h[8 + 2 * i + 1] =
                __floats2half2_rn(v[16 + 4 * i + 2], v[16 + 4 * i + 3]);
          }
          __builtin_memcpy(&a_lo[u], &h[0], 32);
          __builtin_memcpy(&a_hi[u], &h[8], 32);
        } else {
          std::uint32_t nib[8];
          if constexpr (kQ8) {
            // Q8_0: the block's 32 signed bytes are chunks 2 kb and 2 kb + 1;
            // flipping the sign bit carries q + 128, which the 1152 magic
            // takes back out.
            const uint4 c0 = s_codes[swizzle(row, 2 * kb)];
            const uint4 c1 = s_codes[swizzle(row, (2 * kb) + 1)];
            const std::uint32_t words[8] = {c0.x, c0.y, c0.z, c0.w,
                                            c1.x, c1.y, c1.z, c1.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              nib[i] = words[i] ^ 0x80808080U;
            }
          } else if constexpr (kQ5) {
            // Q5_1: elements 0-15 take the low nibbles of the 16 code
            // bytes, 16-31 the high, plus bit j of the high-bit word.
            const std::uint32_t high = q5_head.y;
            const std::uint32_t words[4] = {q5_lo.x, q5_lo.y, q5_hi.x, q5_hi.y};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              nib[i] = (words[i] & 0x0F0F0F0FU) |
                       SpreadHighBits((high >> (4 * i)) & 0xFU);
              nib[4 + i] = ((words[i] >> 4U) & 0x0F0F0F0FU) |
                           SpreadHighBits((high >> (16 + 4 * i)) & 0xFU);
            }
          } else {
            // Q4_K / Q5_K: elements 0-15 of K block kb0 + kb are the low
            // (kb = 0) or high (kb = 1) nibbles of chunk 0, elements 16-31
            // of chunk 1; Q5_K adds bit j of the staged high-bit word.
            const unsigned shift = 4U * static_cast<unsigned>(kb);
            const std::uint32_t words[8] = {
                raw[u][0].x, raw[u][0].y, raw[u][0].z, raw[u][0].w,
                raw[u][1].x, raw[u][1].y, raw[u][1].z, raw[u][1].w};
            const std::uint32_t high = kQ5K ? s_high[(kb * BM) + row] : 0U;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              nib[i] = (words[i] >> shift) & 0x0F0F0F0FU;
              if constexpr (kQ5K) {
                nib[i] |= SpreadHighBits((high >> (4 * i)) & 0xFU);
              }
            }
          }
          __half2 h[16];
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            CodesToHalves(nib[i], magic, scale2, bias2, h[2 * i], h[2 * i + 1]);
          }
          __builtin_memcpy(&a_lo[u], &h[0], 32);
          __builtin_memcpy(&a_hi[u], &h[8], 32);
        }
      }

#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
        if constexpr ((kQ5 || kQ8) && BN >= 48) {
          // Keep one token tile's LDS fragments live at a time. Hoisting
          // all eight tiles spills registers and defeats the wider tile's
          // reuse of each weight decode. This is a compiler barrier only.
          asm volatile("" ::: "memory");
        }
        // A short expert bucket has no output in the remaining token
        // tiles, so omit their WMMA work.
        if constexpr ((kQ5 || kQ8) && BN >= 48 && kTokTiles > 1) {
          if (j >= live_tok_tiles)
            continue;
        }
        const uint4* frag =
            s_act + ((kb * 4) * kActStride) + (j * 16) + sub_lane;
        uint4 b[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          b[q] = frag[q * kActStride];
        }
        v16h b_lo;
        v16h b_hi;
        __builtin_memcpy(&b_lo, &b[0], 32);
        __builtin_memcpy(&b_hi, &b[2], 32);
#pragma unroll
        for (int u = 0; u < kWaveRowTiles; ++u) {
          acc[u][j] = Wmma(a_lo[u], b_lo, acc[u][j]);
          acc[u][j] = Wmma(a_hi[u], b_hi, acc[u][j]);
        }
      }
    }
  };

  const auto run_stage = [&](int kb0) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }
    compute_stage();
    __syncthreads();
  };
  // IQ4_NL/IQ4_XS: the 16 codebook values as F16. One table entry per code
  // index; the decode permutes fix the fragment placement, so the values
  // only need an index. The first stage's barrier publishes this table.
  if constexpr (kIQ4 || kI4XS) {
    if (tid < 16) {
      reinterpret_cast<__half*>(s_lut)[tid] =
          __float2half_rn(static_cast<float>(quant::kValuesIq4Nl[tid]));
    }
  }
  // IQ3_XXS/IQ3_S: the signed magnitude grids (256/512 entries; IQ3_S
  // indexes with the qh high bit). Signs come from the public constant.
  if constexpr (kIQ3X) {
    if (tid < 256)
      s_lut[tid] = quant::kIq3xxsGrid[tid];
  }
  if constexpr (kIQ3S) {
    for (int i = tid; i < 512; i += 256)
      s_lut[i] = quant::kIq3sGrid[i];
  }
  // IQ2_S: the 1024-entry uint64 grid; each entry supplies the eight
  // byte magnitudes of one four-bit-plus-two-high-bits index group.
  if constexpr (kIQ2) {
    for (int i = tid; i < 1024; i += 256)
      s_lut64[i] = quant::kIq2sGrid[i];
  }
  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    run_stage(kb0);
  }

  // One wave writes a complete 128-byte line of F16 output. Padding the
  // shared row by two floats also makes the accumulator scatter conflict-free.
  // Narrow buckets keep the lighter wave-local epilogue below.
  if constexpr (BN >= 48) {
    static_assert(kLdsBytes >= 16 * (BM + 2) * sizeof(float));
    if (out_half != nullptr) {
      constexpr unsigned stride = BM + 2;
      float* scratch = reinterpret_cast<float*>(lds);
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
        for (int u = 0; u < kWaveRowTiles; ++u) {
#pragma unroll
          for (int l = 0; l < 8; ++l)
            scratch[sub_lane * stride + wave_id * 16 + u * 128 + 2 * l +
                    half_id] = acc[u][j][l];
        }
        __syncthreads();
#pragma unroll
        for (int round = 0; round < 16 * BM / (256 * 2); ++round) {
          const unsigned flat = (round * 256 + tid) * 2;
          const unsigned tr = flat / BM, row = flat % BM;
          const unsigned t = t_local + j * 16 + tr, r = r_block + row;
          if (t < unsigned(bucket_rows) && r < m) {
              const std::int32_t dst = rows_slot[bucket_begin + t];
            if (dst >= 0) {
              float2 v =
                  *reinterpret_cast<const float2*>(scratch + tr * stride + row);
              const size_t o = size_t(dst) * m + r;
              if (swiglu_gate != nullptr) {
                v.x *= SiluF(swiglu_gate[o]);
                if (r + 1 < m)
                  v.y *= SiluF(swiglu_gate[o + 1]);
              }
              if (m % 2 == 0 && r + 1 < m)
                *reinterpret_cast<__half2*>(out_half + o) =
                    __floats2half2_rn(v.x, v.y);
              else {
                out_half[o] = __float2half(v.x);
                if (r + 1 < m)
                  out_half[o + 1] = __float2half(v.y);
              }
            }
          }
        }
        __syncthreads();
      }
      return;
    }
  }
  // Transpose each 16x16 tile through LDS, then scatter the 16 rows of each
  // token to its output row.
  float* tile_scratch = reinterpret_cast<float*>(lds) + (wave_id * 256);
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
    const int r0 = r_block + (wave_id * 16) + (u * 128);
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile_scratch[(sub_lane * 16) + (2 * l) + half_id] = acc[u][j][l];
      }
      __builtin_amdgcn_wave_barrier();
      const int t0 = t_local + (j * 16);
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int flat = (s * 32) + lane_id;
        const int t = t0 + (flat >> 4);
        const int r = r0 + (flat & 15);
        if (t < bucket_rows && r < m_i) {
            const std::int32_t dst = rows_slot[bucket_begin + t];
          if (dst >= 0) {
            const std::size_t o = (static_cast<std::size_t>(dst) * m) +
                                  static_cast<std::size_t>(r);
            float v = tile_scratch[flat];
            if (out_half != nullptr) {
              out_half[o] = __float2half(
                  swiglu_gate != nullptr ? v * SiluF(swiglu_gate[o]) : v);
            } else {
              out[o] = v;
            }
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

// Wide routed down projection (Q5_1 / Q8_0 weights, F16 rows in
// and out) over 128-token tiles: RoutedF16GEMMKernel's arithmetic, laid out
// like the paired gate/up waves. Each wave owns sixteen rows and keeps their
// codes in registers; activations are the only LDS stage.
//
// A gfx1151 WMMA reads the A operand of row r from one lane only: lane r for
// even rows, lane r + 16 for odd ones (the other copy is ignored). So each
// lane decodes one K16 half of its row's K32 block, the half the WMMA
// reading it needs first (lo in a reading lane, hi in its partner), and one
// v_permlanex16 per dword hands every reading lane its other half. Every
// weight is decoded once per wave instead of twice, with no selects.
//
// Every output keeps RoutedF16GEMMKernel's values exactly: the same F16
// weights (code + magic, then one packed FMA with the same scale and bias),
// the same F16 activations, the same chain of two K16 WMMAs per K32 block in
// K order from zero, and one round-to-nearest F16 store. Rows therefore do
// not depend on the tile height or on the other rows in the pass.
//
// grid (ceil(m / 128), tiles): `tiles[y]` is expert | (128-token tile << 16).
template<WeightType kType>
__launch_bounds__(256) __global__
    void RoutedDownF16Kernel(const void* __restrict__ w,
                             const __half* __restrict__ x,
                             const std::int32_t* __restrict__ tiles,
                             const std::int32_t* __restrict__ pad_bounds,
                             const std::int32_t* __restrict__ x_rows,
                             const std::int32_t* __restrict__ rows_token,
                             const std::int32_t* __restrict__ rows_slot,
                             std::uint32_t used,
                             __half* __restrict__ out, std::size_t m,
                             std::size_t k) {
  constexpr int BN = 128;
  constexpr int BK = 2;  // K32 blocks per stage
  constexpr int kTokTiles = BN / 16;
  constexpr bool kQ5 = kType == WeightType::kQ5_1;
  constexpr int kBlockBytes = kQ5 ? 24 : 34;
  // Activation stage: [kb][16-element quarter][token][16 B], as in
  // RoutedF16GEMMKernel, one padding slot per quarter plane.
  constexpr int kActStride = BN + 1;
  constexpr int kActBytes = BK * 4 * kActStride * 16;
  // Epilogue: half the tile's F16 results at a time, rows padded to keep
  // 16-byte alignment and spread the 2-byte writes across banks.
  constexpr int kOutStride = 128 + 8;
  constexpr int kEpiTokens = 64;
  constexpr int kEpiBytes = kEpiTokens * kOutStride * 2;
  constexpr int kLdsBytes = kActBytes > kEpiBytes ? kActBytes : kEpiBytes;
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[kLdsBytes];
  auto* s_act = reinterpret_cast<uint4*>(lds);

  const std::int32_t tile = tiles[blockIdx.y];
  const int expert = tile & 0xFFFF;
  const int t_local = (tile >> 16) * BN;
  const int bucket_begin = pad_bounds[expert];
  const int bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  const int live_tok_tiles =
      std::min(kTokTiles, (bucket_rows - t_local + 15) / 16);
  const int num_kb = static_cast<int>(k / 32);
  const int m_i = static_cast<int>(m);
  const std::size_t row_bytes = (k / 32) * kBlockBytes;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int r_block = static_cast<int>(blockIdx.x) * 128;
  // The K16 half this lane decodes: lo where the WMMA reads A, hi in the
  // partner lane.
  const int f_c = half_id ^ (sub_lane & 1);
  const int row = r_block + (wave_id * 16) + sub_lane;
  const bool row_live = row < m_i;
  const std::uint8_t* w_row =
      static_cast<const std::uint8_t*>(w) +
      (static_cast<std::size_t>(expert) * m +
       static_cast<std::size_t>(row_live ? row : m_i - 1)) *
          row_bytes;

  // Activation fetch: BN tokens x (BK * 64) bytes per stage in 16-byte
  // chunks, eight per token; four per thread.
  constexpr int kActFetch = BN * BK * 4 / 256;
  const __half* a_src[kActFetch];
  int a_slot[kActFetch];
#pragma unroll
  for (int i = 0; i < kActFetch; ++i) {
    const int chunk = tid + (i * 256);
    const int t = chunk / (BK * 4);
    const int sub = chunk % (BK * 4);
    const int c_row = t_local + t;
    const std::int32_t src =
        c_row < bucket_rows ? x_rows[bucket_begin + c_row] : -1;
    a_src[i] = src >= 0 ? x + (static_cast<std::size_t>(src) * k) + (sub * 8)
                        : nullptr;
    a_slot[i] = (sub * kActStride) + t;
  }

  // Raw weights (scale word and 16 code bytes per K32 block) in two sets:
  // a stage computes from one while the next stage loads into the other.
  // Activations pass through registers into LDS before each stage.
  uint4 w_codes[2][BK];
  std::uint32_t w_head[2][BK];  ///< Q8_0: d; Q5_1: d | m
  std::uint32_t w_high[2][BK];  ///< Q5_1: the block's high-bit word
  // Padding tokens keep whatever their slots held: their columns are never
  // stored, and WMMA columns do not mix.
  uint4 a_data[kActFetch];
#pragma unroll
  for (int i = 0; i < kActFetch; ++i)
    a_data[i] = make_uint4(0u, 0u, 0u, 0u);
  const auto fetch_stage = [&]<int S>(int kb0) {
#pragma unroll
    for (int b = 0; b < BK; ++b) {
      const std::uint8_t* blk = w_row + ((kb0 + b) * kBlockBytes);
      if constexpr (kQ5) {
        const uint2 head = *reinterpret_cast<const uint2*>(blk);
        w_head[S][b] = head.x;
        w_high[S][b] = head.y;
        w_codes[S][b] = *reinterpret_cast<const uint4*>(blk + 8);
      } else {
        w_head[S][b] = *reinterpret_cast<const std::uint16_t*>(blk);
        __builtin_memcpy(&w_codes[S][b], blk + 2 + (16 * f_c), 16);
      }
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      if (a_src[i] != nullptr)
        a_data[i] = *reinterpret_cast<const uint4*>(a_src[i] + (kb0 * 32));
    }
  };

  v8f acc[kTokTiles];
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j)
    acc[j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};

  constexpr float kMagic = kQ5 ? -1024.0F : -1152.0F;
  const __half2 magic = __floats2half2_rn(kMagic, kMagic);
  const auto compute_stage = [&]<int S>() {
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      // This lane's K16 half of the row's block, as RoutedF16GEMMKernel
      // forms it: (scale, bias) = (d, 0) for Q8_0, (d, m) for Q5_1.
      const std::uint32_t scale_bias =
          kQ5 ? w_head[S][kb] : (row_live ? w_head[S][kb] : 0U);
      const __half2 sb = __builtin_bit_cast(__half2, scale_bias);
      const __half2 scale2 = __low2half2(sb);
      const __half2 bias2 = __high2half2(sb);
      const uint4 c = w_codes[S][kb];
      const std::uint32_t words[4] = {c.x, c.y, c.z, c.w};
      __half2 h[8];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        std::uint32_t nib;
        if constexpr (kQ5) {
          nib = ((words[i] >> (4U * f_c)) & 0x0F0F0F0FU) |
                SpreadHighBits((w_high[S][kb] >> (16 * f_c + 4 * i)) & 0xFU);
        } else {
          nib = words[i] ^ 0x80808080U;
        }
        CodesToHalves(nib, magic, scale2, bias2, h[2 * i], h[2 * i + 1]);
      }
      using v8u = __attribute__((__vector_size__(32))) std::uint32_t;
      v8u own;
      __builtin_memcpy(&own, &h[0], 32);
      v8u peer;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        peer[i] = SwapHalves(own[i]);
      const v16h a_lo = __builtin_bit_cast(v16h, own);
      const v16h a_hi = __builtin_bit_cast(v16h, peer);
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
        // One token tile's LDS fragments live at a time (a compiler
        // barrier only): hoisting all eight spills registers.
        asm volatile("" ::: "memory");
        if (j >= live_tok_tiles)
          continue;
        const uint4* frag =
            s_act + ((kb * 4) * kActStride) + (j * 16) + sub_lane;
        uint4 b[4];
#pragma unroll
        for (int q = 0; q < 4; ++q)
          b[q] = frag[q * kActStride];
        v16h b_lo;
        v16h b_hi;
        __builtin_memcpy(&b_lo, &b[0], 32);
        __builtin_memcpy(&b_hi, &b[2], 32);
        acc[j] = Wmma(a_lo, b_lo, acc[j]);
        acc[j] = Wmma(a_hi, b_hi, acc[j]);
      }
    }
  };

  const auto run_stage = [&]<int S>(int kb0) {
#pragma unroll
    for (int i = 0; i < kActFetch; ++i)
      s_act[a_slot[i]] = a_data[i];
    __syncthreads();
    if (kb0 + BK < num_kb)
      fetch_stage.template operator()<1 - S>(kb0 + BK);
    compute_stage.template operator()<S>();
    __syncthreads();
  };
  fetch_stage.template operator()<0>(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += 2 * BK) {
    run_stage.template operator()<0>(kb0);
    if (kb0 + BK < num_kb)
      run_stage.template operator()<1>(kb0 + BK);
  }

  // Lane (token n, rows 2i + half) holds acc[j][i]: stage F16 values per
  // half tile, then store each token's 128 rows as 16-byte chunks.
  __half* scratch = reinterpret_cast<__half*>(lds);
#pragma unroll
  for (int part = 0; part < BN / kEpiTokens; ++part) {
    if (part * (kEpiTokens / 16) >= live_tok_tiles)
      break;
    if (part > 0)
      __syncthreads();
#pragma unroll
    for (int jj = 0; jj < kEpiTokens / 16; ++jj) {
      const int j = part * (kEpiTokens / 16) + jj;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        scratch[((jj * 16) + sub_lane) * kOutStride + (wave_id * 16) + (2 * i) +
                half_id] = __float2half(acc[j][i]);
    }
    __syncthreads();
#pragma unroll
    for (int unit = 0; unit < kEpiTokens * 16 / 256; ++unit) {
      const int flat = (unit * 256) + tid;
      const int tr = flat / 16;
      const int chunk = flat % 16;
      const int t = t_local + (part * kEpiTokens) + tr;
      const int r = r_block + (chunk * 8);
      if (t >= bucket_rows || r >= m_i)
        continue;
        const std::int32_t dst = rows_slot[bucket_begin + t];
      if (dst < 0)
        continue;
      const __half* src = scratch + (tr * kOutStride) + (chunk * 8);
      __half* o = out + (static_cast<std::size_t>(dst) * m) + r;
      if (m % 8 == 0) {
        *reinterpret_cast<uint4*>(o) = *reinterpret_cast<const uint4*>(src);
      } else {
        for (int e = 0; e < 8 && r + e < m_i; ++e)
          o[e] = src[e];
      }
    }
  }
}

// Paired K-quant gate/up projection with SwiGLU over 128-token tiles (F16
// rows in and out). Each wave owns eight gate rows and their matching up
// rows as one 16-row WMMA tile, with the codes in registers; activations
// are the only LDS stage. A stage is one 32-byte nibble group: K32 blocks
// 2g (low nibbles) and 2g + 1 (high), whose 16-byte halves each hold one
// K16 fragment of both blocks.
//
// As in RoutedDownF16Kernel, each lane decodes the K16 fragment its WMMA
// reads first (lo in the lane the hardware reads, hi in its partner) and
// one v_permlanex16 per dword supplies the other. A superblock's codes and
// header are loaded raw one stage ahead and decoded when their stage is
// committed, so no stage waits on the loads it has just issued.
//
// Every output keeps the previous paired route's arithmetic: the same F16
// weights (code * (d * sc) - dmin * mn, rounded once to F16 pairs), the same
// chain of two K16 WMMAs per K32 block from zero in K order, and the same
// SwiGLU rounding, so a row matches the separate gate and up projections at
// every tile height.
//
// grid (ceil(m / 64), tiles): `tiles[y]` is expert | (128-token tile << 16).
template<WeightType kType>
__launch_bounds__(256) __global__
    void RoutedPairF16Kernel(const void* __restrict__ w_gate,
                             const void* __restrict__ w_up,
                             const __half* __restrict__ x,
                             const std::int32_t* __restrict__ tiles,
                             const std::int32_t* __restrict__ pad_bounds,
                             const std::int32_t* __restrict__ x_rows,
                             const std::int32_t* __restrict__ rows_token,
                             const std::int32_t* __restrict__ rows_slot,
                             std::uint32_t used,
                             __half* __restrict__ out, std::size_t m,
                             std::size_t k) {
  constexpr int BN = 128;
  constexpr int BK = 2;  // K32 blocks per stage: one nibble group
  constexpr int kTokTiles = BN / 16;
  constexpr int kRows = 64;  // output rows (gate/up pairs) per block
  constexpr bool kQ5K = kType == WeightType::kQ5_K;
  constexpr std::size_t kBlockBytes = kQ5K ? kQ5KBlockBytes : sizeof(Q4KBlock);
  constexpr int kActStride = BN + 1;
  constexpr int kActBytes = BK * 4 * kActStride * 16;
  constexpr int kPairStride = kRows + 2;
  constexpr int kEpilogueBytes = BN * kPairStride * 2;
  constexpr int kLdsBytes =
      kActBytes > kEpilogueBytes ? kActBytes : kEpilogueBytes;
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[kLdsBytes];
  auto* s_act = reinterpret_cast<uint4*>(lds);

  const std::int32_t tile = tiles[blockIdx.y];
  const int expert = tile & 0xFFFF;
  const int t_local = (tile >> 16) * BN;
  const int bucket_begin = pad_bounds[expert];
  const int bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  const int live_tok_tiles =
      std::min(kTokTiles, (bucket_rows - t_local + 15) / 16);
  const int num_sb = static_cast<int>(k / 256);
  const int m_i = static_cast<int>(m);
  const std::size_t row_bytes = (k / 256) * kBlockBytes;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int r_block = static_cast<int>(blockIdx.x) * kRows;
  // Tile rows 0-7 are gate rows, 8-15 the matching up rows.
  const int row = r_block + (wave_id * 8) + (sub_lane % 8);
  const bool row_live = row < m_i;
  const int f_c = half_id ^ (sub_lane & 1);
  const std::uint8_t* w_row =
      static_cast<const std::uint8_t*>(sub_lane < 8 ? w_gate : w_up) +
      (static_cast<std::size_t>(expert) * m +
       static_cast<std::size_t>(row_live ? row : m_i - 1)) *
          row_bytes;

  constexpr int kActFetch = BN * BK * 4 / 256;
  const __half* a_src[kActFetch];
  int a_slot[kActFetch];
#pragma unroll
  for (int i = 0; i < kActFetch; ++i) {
    const int chunk = tid + (i * 256);
    const int t = chunk / (BK * 4);
    const int sub = chunk % (BK * 4);
    const int c_row = t_local + t;
    const std::int32_t src =
        c_row < bucket_rows ? x_rows[bucket_begin + c_row] : -1;
    a_src[i] = src >= 0 ? x + (static_cast<std::size_t>(src) * k) + (sub * 8)
                        : nullptr;
    a_slot[i] = (sub * kActStride) + t;
  }

  // A superblock's raw header, high bits and this lane's code chunks,
  // loaded with its first stage's fetch and decoded at that stage's commit.
  uint4 raw_header;
  uint4 raw_codes[4];  ///< chunk f_c of groups 0-3
  uint4 raw_qh[2];     ///< Q5_K: the 32 high-bit bytes
  // The committed superblock: header, nibble-packed codes per group.
  uint4 header = make_uint4(0u, 0u, 0u, 0u);
  uint4 codes[4];
  uint4 qh[2];
  uint4 a_data[kActFetch];
#pragma unroll
  for (int i = 0; i < kActFetch; ++i)
    a_data[i] = make_uint4(0u, 0u, 0u, 0u);

  const auto fetch_stage = [&]<int G>(int sb) {
    if constexpr (G == 0) {
      const std::uint8_t* blk = w_row + (sb * kBlockBytes);
      const auto* chunks = reinterpret_cast<const uint4*>(blk);
      constexpr int kCodeChunk = kQ5K ? 3 : 1;
      raw_header = chunks[0];
      if constexpr (kQ5K) {
        raw_qh[0] = chunks[1];
        raw_qh[1] = chunks[2];
      }
#pragma unroll
      for (int g = 0; g < 4; ++g)
        raw_codes[g] = chunks[kCodeChunk + (g * 2) + f_c];
    }
    const int kb0 = (sb * 8) + (G * 2);
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      if (a_src[i] != nullptr)
        a_data[i] = *reinterpret_cast<const uint4*>(a_src[i] + (kb0 * 32));
    }
  };

  // The committed stage's (scale, bias) for K32 block kb of the group, and
  // Q5_K's high-bit word.
  std::uint32_t own_scale = 0;  ///< block 2G + f_c
  std::uint32_t own_high = 0;   ///< Q5_K: block 2G + f_c
  const auto commit_stage = [&]<int G>() {
    if constexpr (G == 0) {
      header = raw_header;
#pragma unroll
      for (int g = 0; g < 4; ++g)
        codes[g] = raw_codes[g];
      if constexpr (kQ5K) {
        qh[0] = raw_qh[0];
        qh[1] = raw_qh[1];
      }
    }
    const int sb32 = (2 * G) + f_c;
    // The first four scale/min bytes live in y/z; the others' low nibbles
    // share w, their high two bits reuse those y/z bytes.
    const unsigned shift = 8U * (static_cast<unsigned>(sb32) & 3U);
    const std::uint32_t scale_byte = header.y >> shift;
    const std::uint32_t min_byte = header.z >> shift;
    std::uint32_t sc;
    std::uint32_t mn;
    if (sb32 < 4) {
      sc = scale_byte & 0x3FU;
      mn = min_byte & 0x3FU;
    } else {
      const std::uint32_t packed = header.w >> shift;
      sc = (packed & 0x0FU) | ((scale_byte & 0xC0U) >> 2U);
      mn = ((packed >> 4U) & 0x0FU) | ((min_byte & 0xC0U) >> 2U);
    }
    const __half2 dm = __builtin_bit_cast(__half2, header.x);
    const float scale =
        row_live ? __low2float(dm) * static_cast<float>(sc) : 0.0F;
    const float offset =
        row_live ? __high2float(dm) * static_cast<float>(mn) : 0.0F;
    own_scale =
        __builtin_bit_cast(std::uint32_t, __floats2half2_rn(scale, -offset));
    if constexpr (kQ5K) {
      // Bit sb32 of the 32 high-bit bytes, packed as the Q5_1 word.
      const std::uint32_t q[8] = {qh[0].x, qh[0].y, qh[0].z, qh[0].w,
                                  qh[1].x, qh[1].y, qh[1].z, qh[1].w};
      std::uint32_t high = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        high |= GatherBit(q[i], sb32) << (4 * i);
      own_high = high;
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i)
      s_act[a_slot[i]] = a_data[i];
  };

  v8f acc[kTokTiles];
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j)
    acc[j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};

  const __half2 magic = __floats2half2_rn(-1024.0F, -1024.0F);
  const auto compute_stage = [&]<int G>() {
    const uint4 c = codes[G];
    const std::uint32_t words[4] = {c.x, c.y, c.z, c.w};
    const std::uint32_t peer_scale = SwapHalves(own_scale);
    const std::uint32_t peer_high = kQ5K ? SwapHalves(own_high) : 0U;
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      const __half2 sb =
          __builtin_bit_cast(__half2, f_c == kb ? own_scale : peer_scale);
      const __half2 scale2 = __low2half2(sb);
      const __half2 bias2 = __high2half2(sb);
      const std::uint32_t high =
          kQ5K ? ((f_c == kb ? own_high : peer_high) >> (16U * f_c)) : 0U;
      __half2 h[8];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        std::uint32_t nib = (words[i] >> (4U * kb)) & 0x0F0F0F0FU;
        if constexpr (kQ5K)
          nib |= SpreadHighBits((high >> (4 * i)) & 0xFU);
        CodesToHalves(nib, magic, scale2, bias2, h[2 * i], h[2 * i + 1]);
      }
      using v8u = __attribute__((__vector_size__(32))) std::uint32_t;
      v8u own;
      __builtin_memcpy(&own, &h[0], 32);
      v8u peer;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        peer[i] = SwapHalves(own[i]);
      const v16h a_lo = __builtin_bit_cast(v16h, own);
      const v16h a_hi = __builtin_bit_cast(v16h, peer);
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
        // One token tile's LDS fragments live at a time (a compiler
        // barrier only): hoisting all eight spills registers.
        asm volatile("" ::: "memory");
        if (j >= live_tok_tiles)
          continue;
        const uint4* frag =
            s_act + ((kb * 4) * kActStride) + (j * 16) + sub_lane;
        uint4 b[4];
#pragma unroll
        for (int q = 0; q < 4; ++q)
          b[q] = frag[q * kActStride];
        v16h b_lo;
        v16h b_hi;
        __builtin_memcpy(&b_lo, &b[0], 32);
        __builtin_memcpy(&b_hi, &b[2], 32);
        acc[j] = Wmma(a_lo, b_lo, acc[j]);
        acc[j] = Wmma(a_hi, b_hi, acc[j]);
      }
    }
  };

  const auto run_stage = [&]<int G>(int sb) {
    commit_stage.template operator()<G>();
    __syncthreads();
    if constexpr (G < 3)
      fetch_stage.template operator()<G + 1>(sb);
    else if (sb + 1 < num_sb)
      fetch_stage.template operator()<0>(sb + 1);
    compute_stage.template operator()<G>();
    __syncthreads();
  };
  fetch_stage.template operator()<0>(0);
  for (int sb = 0; sb < num_sb; ++sb) {
    run_stage.template operator()<0>(sb);
    run_stage.template operator()<1>(sb);
    run_stage.template operator()<2>(sb);
    run_stage.template operator()<3>(sb);
  }

  // Lane (token n, rows 2l + half) holds gate rows in acc[j][l] and the
  // matching up rows in acc[j][l + 4]. SwiGLU in registers, then all F16
  // results are staged once so stores cover the full 64-column output row.
  __half* scratch = reinterpret_cast<__half*>(lds);
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j) {
    if (j >= live_tok_tiles)
      continue;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      float product = acc[j][l + 4] * acc[j][l];
      asm volatile("" : "+v"(product));
      float value = product * SigmoidF(acc[j][l]);
      asm volatile("" : "+v"(value));
      scratch[(j * 16 + sub_lane) * kPairStride + wave_id * 8 + 2 * l +
              half_id] = __float2half(value);
    }
  }
  __syncthreads();
#pragma unroll
  for (int unit = 0; unit < BN * kRows / (2 * 256); ++unit) {
    const int flat = (unit * 256 + tid) * 2;
    const int t = t_local + flat / kRows;
    const int r = r_block + flat % kRows;
    if (t < bucket_rows && r < m_i) {
        const std::int32_t dst = rows_slot[bucket_begin + t];
      if (dst >= 0) {
        const auto offset = static_cast<std::size_t>(dst) * m + r;
        const int src = (flat / kRows) * kPairStride + flat % kRows;
        if (m % 2 == 0 && r + 1 < m_i) {
          *reinterpret_cast<__half2*>(out + offset) =
              *reinterpret_cast<const __half2*>(scratch + src);
        } else {
          out[offset] = scratch[src];
          if (r + 1 < m_i)
            out[offset + 1] = scratch[src + 1];
        }
      }
    }
  }
}

/// Compacts the routed assignments by expert with 16-row padded buckets.
/// One block: exclusive scan of the padded counts into pad_bounds[0..E].
/// Upper bound on experts the device-side tile-map builder scans in one
/// block (both shipped checkpoints fit: 256 pruned, 512 original).
constexpr std::uint32_t kMaxRoutedTilesExperts = 1024;

/// The (expert, column tile) launch map for a batch whose routing the host
/// cannot read back: decode replays a captured graph, so the map is built
/// device-side from the routing histogram as `e | (j << 16)` entries, then
/// padded to the launch bound with -1 sentinels the GEMM exits on. The
/// serial prefix over expert tile counts is fine at these expert counts.
__global__ void RoutedTilesKernel(const std::uint32_t* __restrict__ counts,
                                  std::int32_t* __restrict__ tiles,
                                  std::uint32_t n_experts,
                                  std::uint32_t tile_rows,
                                  std::uint32_t capacity) {
  __shared__ std::uint32_t off[kMaxRoutedTilesExperts + 1];
  if (threadIdx.x == 0) {
    std::uint32_t run = 0;
    for (std::uint32_t e = 0; e < n_experts; ++e) {
      off[e] = run;
      run += (counts[e] + tile_rows - 1) / tile_rows;
    }
    off[n_experts] = run;
  }
  __syncthreads();
  if (threadIdx.x < n_experts) {
    const std::uint32_t base = off[threadIdx.x];
    const std::uint32_t n = off[threadIdx.x + 1] - base;
    for (std::uint32_t j = 0; j < n; ++j) {
      tiles[base + j] = static_cast<std::int32_t>(threadIdx.x) |
                        static_cast<std::int32_t>(j << 16);
    }
  }
  __syncthreads();
  for (std::uint32_t i = off[n_experts] + threadIdx.x; i < capacity;
       i += blockDim.x) {
    tiles[i] = -1;
  }
}

// Tile map for a native-quantized dense projection: the degenerate single-
// expert view of the routed GEMM's contract -- every token row in one
// bucket, identity scatter. Built on the device so a captured graph
// rebuilds it from the stream order rather than a stale host snapshot.

void RoutedTiles(const std::uint32_t* counts, std::int32_t* tiles,
                 std::uint32_t n_experts, std::uint32_t tile_rows,
                 std::uint32_t capacity, hipStream_t stream) {
  hipLaunchKernelGGL(RoutedTilesKernel, dim3(1), dim3(1024), 0, stream, counts,
                     tiles, n_experts, tile_rows, capacity);
}

template<int BN>
bool LaunchRoutedF16(const void* w, WeightType type, const __half* x,
                     const std::int32_t* tiles, std::uint32_t n_tiles,
                     const std::int32_t* pad_bounds,
                     const std::int32_t* x_rows, const std::int32_t* rows_token,
                     const std::int32_t* rows_slot, std::uint32_t used,
                     const float* swiglu_gate, float* out, __half* out_half,
                     std::size_t m, std::size_t k, hipStream_t stream) {
  constexpr int kBM = 128;
  constexpr int kBK = 2;
  const dim3 grid(static_cast<unsigned int>((m + kBM - 1) / kBM), n_tiles);
  switch (type) {
    case WeightType::kQ4_K:
      hipLaunchKernelGGL((RoutedF16GEMMKernel<WeightType::kQ4_K, kBM, BN, kBK>),
                         grid, dim3(kThreads), 0, stream, w, x, tiles,
                         pad_bounds, x_rows, rows_token, rows_slot, used, swiglu_gate, out,
                         out_half, m, k);
      return true;
    case WeightType::kQ5_1:
      hipLaunchKernelGGL((RoutedF16GEMMKernel<WeightType::kQ5_1, kBM, BN, kBK>),
                         grid, dim3(kThreads), 0, stream, w, x, tiles,
                         pad_bounds, x_rows, rows_token, rows_slot, used, swiglu_gate, out,
                         out_half, m, k);
      return true;
    case WeightType::kQ8_0:
      hipLaunchKernelGGL((RoutedF16GEMMKernel<WeightType::kQ8_0, kBM, BN, kBK>),
                         grid, dim3(kThreads), 0, stream, w, x, tiles,
                         pad_bounds, x_rows, rows_token, rows_slot, used, swiglu_gate, out,
                         out_half, m, k);
      return true;
    case WeightType::kQ5_K:
      hipLaunchKernelGGL((RoutedF16GEMMKernel<WeightType::kQ5_K, kBM, BN, kBK>),
                         grid, dim3(kThreads), 0, stream, w, x, tiles,
                         pad_bounds, x_rows, rows_token, rows_slot, used, swiglu_gate, out,
                         out_half, m, k);
      return true;
    case WeightType::kIQ4_NL:
      hipLaunchKernelGGL(
          (RoutedF16GEMMKernel<WeightType::kIQ4_NL, kBM, BN, kBK>), grid,
          dim3(kThreads), 0, stream, w, x, tiles, pad_bounds, x_rows, rows_token, rows_slot, used,
          swiglu_gate, out, out_half, m, k);
      return true;
    case WeightType::kIQ3_XXS:
      hipLaunchKernelGGL(
          (RoutedF16GEMMKernel<WeightType::kIQ3_XXS, kBM, BN, kBK>), grid,
          dim3(kThreads), 0, stream, w, x, tiles, pad_bounds, x_rows, rows_token, rows_slot, used,
          swiglu_gate, out, out_half, m, k);
      return true;
    case WeightType::kIQ3_S:
      hipLaunchKernelGGL(
          (RoutedF16GEMMKernel<WeightType::kIQ3_S, kBM, BN, kBK>), grid,
          dim3(kThreads), 0, stream, w, x, tiles, pad_bounds, x_rows, rows_token, rows_slot, used,
          swiglu_gate, out, out_half, m, k);
      return true;
    case WeightType::kIQ2_S:
      hipLaunchKernelGGL(
          (RoutedF16GEMMKernel<WeightType::kIQ2_S, kBM, BN, kBK>), grid,
          dim3(kThreads), 0, stream, w, x, tiles, pad_bounds, x_rows, rows_token, rows_slot, used,
          swiglu_gate, out, out_half, m, k);
      return true;
    case WeightType::kIQ4_XS:
      hipLaunchKernelGGL(
          (RoutedF16GEMMKernel<WeightType::kIQ4_XS, kBM, BN, kBK>), grid,
          dim3(kThreads), 0, stream, w, x, tiles, pad_bounds, x_rows, rows_token, rows_slot, used,
          swiglu_gate, out, out_half, m, k);
      return true;
    case WeightType::kQ2_0:
      hipLaunchKernelGGL((RoutedF16GEMMKernel<WeightType::kQ2_0, kBM, BN, kBK>),
                         grid, dim3(kThreads), 0, stream, w, x, tiles,
                         pad_bounds, x_rows, rows_token, rows_slot, used, swiglu_gate, out,
                         out_half, m, k);
      return true;
    case WeightType::kQ6_K:
      hipLaunchKernelGGL((RoutedF16GEMMKernel<WeightType::kQ6_K, kBM, BN, kBK>),
                         grid, dim3(kThreads), 0, stream, w, x, tiles,
                         pad_bounds, x_rows, rows_token, rows_slot, used, swiglu_gate, out,
                         out_half, m, k);
      return true;
    default:
      return false;
  }
}

bool RoutedF16Gemm(const void* w, WeightType type, const __half* x,
                   const std::int32_t* tiles, std::uint32_t n_tiles,
                   std::uint32_t tile_rows, const std::int32_t* pad_bounds,
                   const std::int32_t* x_rows, const std::int32_t* rows_token,
                     const std::int32_t* rows_slot, std::uint32_t used,
                   const float* swiglu_gate, float* out, __half* out_half,
                   std::size_t m, std::size_t k, hipStream_t stream) {
  const std::size_t block_elems =
      (type == WeightType::kQ4_K || type == WeightType::kQ5_K ||
       type == WeightType::kIQ3_XXS || type == WeightType::kIQ3_S ||
       type == WeightType::kIQ2_S || type == WeightType::kIQ4_XS ||
       type == WeightType::kQ6_K)
          ? 256
      : type == WeightType::kIQ4_NL ? 32
                                    : 64;
  if (m == 0 || k == 0 || k % block_elems != 0 || n_tiles == 0 ||
      (out_half == nullptr) == (out == nullptr)) {
    return false;
  }
  switch (tile_rows) {
    case 16:
      return LaunchRoutedF16<16>(w, type, x, tiles, n_tiles, pad_bounds,
                                 x_rows, rows_token, rows_slot, used, swiglu_gate, out, out_half,
                                 m, k, stream);
    case 48:
      return LaunchRoutedF16<48>(w, type, x, tiles, n_tiles, pad_bounds,
                                 x_rows, rows_token, rows_slot, used, swiglu_gate, out, out_half,
                                 m, k, stream);
    case 64:
      return LaunchRoutedF16<64>(w, type, x, tiles, n_tiles, pad_bounds,
                                 x_rows, rows_token, rows_slot, used, swiglu_gate, out, out_half,
                                 m, k, stream);
    case 128: {
      if (out_half == nullptr || swiglu_gate != nullptr)
        return false;
      const dim3 grid(static_cast<unsigned int>((m + 127) / 128), n_tiles);
      const auto kernel =
          type == WeightType::kQ5_1   ? RoutedDownF16Kernel<WeightType::kQ5_1>
          : type == WeightType::kQ8_0 ? RoutedDownF16Kernel<WeightType::kQ8_0>
                                      : nullptr;
      if (kernel == nullptr)
        return false;
      hipLaunchKernelGGL(kernel, grid, dim3(kThreads), 0, stream, w, x, tiles,
                         pad_bounds, x_rows, rows_token, rows_slot, used, out_half, m, k);
      return true;
    }
    default:
      return false;
  }
}

bool RoutedGatedF16Gemm(const void* gate, const void* up, WeightType type,
                        const __half* x, const std::int32_t* tiles,
                        std::uint32_t n_tiles, const std::int32_t* pad_bounds,
                        const std::int32_t* x_rows, const std::int32_t* rows_token, const std::int32_t* rows_slot,
                        std::uint32_t used, __half* out,
                        std::size_t m, std::size_t k, hipStream_t stream) {
  // 128-token tiles decode each expert's weights once per wide bucket; in
  // prefill they also beat narrower tiles on short buckets.
  if (m == 0 || k == 0 || k % 256 != 0 || n_tiles == 0 || out == nullptr) {
    return false;
  }
  const dim3 grid(static_cast<unsigned int>((m + 63) / 64), n_tiles);
  const auto kernel =
      type == WeightType::kQ4_K   ? RoutedPairF16Kernel<WeightType::kQ4_K>
      : type == WeightType::kQ5_K ? RoutedPairF16Kernel<WeightType::kQ5_K>
                                  : nullptr;
  if (kernel == nullptr)
    return false;
  hipLaunchKernelGGL(kernel, grid, dim3(kThreads), 0, stream, gate, up, x,
                     tiles, pad_bounds, x_rows, rows_token, rows_slot, used, out, m, k);
  return true;
}

constexpr unsigned kSsmProjectionTileTokens = 32;


}  // namespace gufo::ext::gsq::wmma
