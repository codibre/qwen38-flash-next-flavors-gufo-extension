#ifndef GUFO_EXT_GSQ_KERNELS_ROUTED_WMMA_HPP_
#define GUFO_EXT_GSQ_KERNELS_ROUTED_WMMA_HPP_

// The routed F16 WMMA GEMMs over the flavor codebook encodings, ported
// kernel for kernel from the standalone Flash-Next fork: the weights stay
// in their disk encoding (block_iq3_s / block_iq2_s / IQ4_NL blocks) and
// dequantize to F16 after the LDS read, accumulating the whole K extent in
// F32 — every row bitwise equal to the fork's outputs.

#include <hip/hip_fp16.h>

#include <cstddef>
#include <cstdint>

namespace gufo::ext::gsq::wmma {

/// The weight encodings these GEMMs run. The values are the core GgmlType
/// numbers, so a FlavorState type casts straight in.
enum class WeightType : std::uint32_t {
  kF32 = 0,
  kF16 = 1,
  kQ5_1 = 7,
  kQ8_0 = 8,
  kQ4_K = 12,
  kQ5_K = 13,
  kQ6_K = 14,
  kIQ3_XXS = 18,
  kIQ4_NL = 20,
  kIQ3_S = 21,
  kIQ2_S = 22,
  kIQ4_XS = 23,
  kBF16 = 30,
  kQ2_0 = 42,
};

/// Tile heights: narrow passes take 16 token rows per tile, wide prompt
/// passes 48 (decoding each expert's weights fewer times per bucket); any
/// tile height gives every row the same accumulation order, so outputs are
/// bitwise identical.
constexpr std::uint32_t kRoutedTileRowsNarrow = 16;
constexpr std::uint32_t kRoutedTileRowsWide = 48;

/// Rows per tile for `slots` routed rows over `num_experts`: wide once the
/// slots could fill every expert's first tile, narrow below.
[[nodiscard]] constexpr std::uint32_t RoutedTileRows(std::uint32_t slots,
                                                     std::uint32_t num_experts) {
  return slots >= 16U * num_experts ? kRoutedTileRowsWide
                                    : kRoutedTileRowsNarrow;
}

/// Worst-case tile-map entries for `rows` slots over `num_experts` at
/// `tile_rows`: every expert's rows padded to a tile.
[[nodiscard]] constexpr std::size_t RoutedTileCapacity(std::uint32_t rows,
                                                       std::uint32_t num_experts,
                                                       std::uint32_t tile_rows) {
  return (static_cast<std::size_t>(rows) +
          static_cast<std::size_t>(num_experts) * (tile_rows - 1)) /
             tile_rows +
         num_experts;
}

/// Writes the tile map (expert | tile << 16, `capacity` entries with -1
/// sentinels the GEMMs exit on) from the per-expert device `counts`, for
/// passes whose routing the host cannot read back (captured graphs).
void RoutedTiles(const std::uint32_t* counts, std::int32_t* tiles,
                 std::uint32_t n_experts, std::uint32_t tile_rows,
                 std::uint32_t capacity, hipStream_t stream);

/// Routed expert GEMM on the F16 WMMA cores over F16 activation rows `x`;
/// exactly one of `out` (F32) and `out_half` (F16) receives the result; a
/// non-null `swiglu_gate` (the gate projection's F32 rows) turns it into
/// silu(gate) * result. `tiles`/`n_tiles`/`tile_rows` come from RoutedTiles
/// above. `pad_bounds`, `rows_token` and `rows_slot` are the executor's
/// compacted routing (MoeScratch): results land at row
/// `rows_token * used + rows_slot` (-1 rows are padding). `x_rows[r]` names
/// the activation row a compact row reads — `rows_token` for the gate and
/// up projections (token rows of `x`), and the gate/up output map
/// (BuildDownRows) for the down projection.
bool RoutedF16Gemm(const void* w, WeightType type, const __half* x,
                   const std::int32_t* tiles, std::uint32_t n_tiles,
                   std::uint32_t tile_rows, const std::int32_t* pad_bounds,
                   const std::int32_t* x_rows,
                   const std::int32_t* rows_token,
                   const std::int32_t* rows_slot, std::uint32_t used,
                   const float* swiglu_gate, float* out, __half* out_half,
                   std::size_t m, std::size_t k, hipStream_t stream);

}  // namespace gufo::ext::gsq::wmma

#endif  // GUFO_EXT_GSQ_KERNELS_ROUTED_WMMA_HPP_
