#ifndef GUFO_EXT_GSQ_FORMAT_HPP_
#define GUFO_EXT_GSQ_FORMAT_HPP_

// Storage geometry of the pruned ("flavor") quantizations. The Flash-Next
// runtime sizes and binds the formats of its own checkpoints (weights.hpp);
// these are the codebook formats of the pruned GGUFs, which only this
// extension's binding, upload and routed WMMA kernels read. They match
// ggml's block_iq*_s layouts byte for byte.

#include <cstddef>
#include <cstdint>

#include <gufo/ext/gguf_reader.hpp>

namespace gufo::ext::gsq {

struct FlavorFormat {
  std::uint32_t block;  ///< Elements per block.
  std::uint32_t bytes;  ///< Encoded bytes per block.
};

/// Geometry of the flavor quantizations; a zero block for every type this
/// module does not read.
[[nodiscard]] constexpr FlavorFormat FlavorFormatOf(
    core::GgmlType type) noexcept {
  switch (type) {
    case core::GgmlType::kIQ3_XXS:
      return {256, 98};
    case core::GgmlType::kIQ3_S:
      return {256, 110};
    case core::GgmlType::kIQ2_S:
      return {256, 82};
    case core::GgmlType::kIQ4_XS:
      return {256, 136};
    default:
      return {0, 0};
  }
}

/// A routed expert projection encoded in a flavor format.
[[nodiscard]] constexpr bool IsFlavorEncoding(core::GgmlType t) noexcept {
  return FlavorFormatOf(t).block != 0;
}

/// Encoded bytes of a `cols`-element row in a flavor format; 0 when `cols`
/// is not a whole number of blocks or the type is not a flavor format.
[[nodiscard]] constexpr std::size_t FlavorRowBytes(core::GgmlType type,
                                                   std::uint64_t cols) {
  const FlavorFormat f = FlavorFormatOf(type);
  if (f.block == 0 || cols % f.block != 0) {
    return 0;
  }
  return static_cast<std::size_t>(cols / f.block) * f.bytes;
}

/// The `format_state` the binding attaches to every routed expert projection
/// of a flavor checkpoint (all three of gate/up/down, so the MoE stage sees
/// one route for the expert FFN at every batch width), and the upload keeps
/// alive beside the device copy. The MoE stage reads the stored GgmlType and
/// launches the matching WMMA kernel.
struct FlavorState {
  core::GgmlType type;
};

/// The stored type of an uploaded flavor tensor; kF32 when absent.
[[nodiscard]] inline core::GgmlType FlavorType(
    const void* format_state) noexcept {
  return format_state == nullptr
             ? core::GgmlType::kF32
             : static_cast<const FlavorState*>(format_state)->type;
}

}  // namespace gufo::ext::gsq

#endif  // GUFO_EXT_GSQ_FORMAT_HPP_
