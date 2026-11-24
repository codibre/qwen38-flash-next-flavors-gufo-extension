#ifndef GUFO_EXT_GSQ_KERNELS_ROWS_HPP_
#define GUFO_EXT_GSQ_KERNELS_ROWS_HPP_

// Route compaction for the flavor MoE path.

#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::ext::gsq::wmma {

/// Decode/verify batches (n_tokens <= 16) never fill expert_counts: the
/// host keeps those passes on its vector route and skips the histogram.
/// The flavor WMMA kernels need the bucketed maps anyway, so build them
/// here — same 16-row padded-bucket contract as the host's own
/// RoutedPadBounds + RoutedScatter, fed straight from `ids`.
void DecodeRoutes(const std::int32_t* ids, std::uint32_t* counts,
                  std::int32_t* pad_bounds, std::int32_t* cursors,
                  std::int32_t* rows_token, std::int32_t* rows_slot,
                  std::uint32_t slots, std::uint32_t k,
                  std::uint32_t n_experts, hipStream_t stream);

}  // namespace gufo::ext::gsq::wmma

#endif  // GUFO_EXT_GSQ_KERNELS_ROWS_HPP_
