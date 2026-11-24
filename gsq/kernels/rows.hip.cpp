#include "gsq/kernels/rows.hpp"

namespace gufo::ext::gsq::wmma {

namespace {

/// One block: histogram `ids` into `counts`, scan the 16-row padded bucket
/// bounds, scatter the (token, slot) assignments, then blank the padding so
/// a later narrower pass cannot read stale assignments through it. Mirrors
/// the host kernels RoutedPadBoundsKernel and RoutedScatterKernel exactly
/// (cursor reset included); `k` is the slots per token (routing used).
__global__ void DecodeRoutesKernel(const std::int32_t* __restrict__ ids,
                                   std::uint32_t* __restrict__ counts,
                                   std::int32_t* __restrict__ pad_bounds,
                                   std::int32_t* __restrict__ cursors,
                                   std::int32_t* __restrict__ rows_token,
                                   std::int32_t* __restrict__ rows_slot,
                                   std::uint32_t slots, std::uint32_t k,
                                   std::uint32_t n_experts) {
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    counts[e] = 0u;
    cursors[e] = 0;
  }
  __syncthreads();
  for (std::uint32_t s = threadIdx.x; s < slots; s += blockDim.x) {
    if (ids[s] >= 0) {
      atomicAdd(&counts[ids[s]], 1u);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    std::int32_t running = 0;
    for (std::uint32_t e = 0; e < n_experts; ++e) {
      pad_bounds[e] = running;
      running += (counts[e] + 15u) / 16u * 16u;
    }
    pad_bounds[n_experts] = running;
  }
  __syncthreads();
  for (std::uint32_t s = threadIdx.x; s < slots; s += blockDim.x) {
    const std::int32_t e = ids[s];
    if (e < 0) {
      continue;
    }
    const std::int32_t at = pad_bounds[e] + atomicAdd(&cursors[e], 1);
    rows_token[at] = static_cast<std::int32_t>(s / k);
    rows_slot[at] = static_cast<std::int32_t>(s);
  }
  __syncthreads();
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    const std::uint32_t padded = (counts[e] + 15u) / 16u * 16u;
    for (std::uint32_t row = counts[e]; row < padded; ++row) {
      const std::int32_t at = pad_bounds[e] + row;
      rows_token[at] = -1;
      rows_slot[at] = -1;
    }
  }
}

}  // namespace

void DecodeRoutes(const std::int32_t* ids, std::uint32_t* counts,
                  std::int32_t* pad_bounds, std::int32_t* cursors,
                  std::int32_t* rows_token, std::int32_t* rows_slot,
                  std::uint32_t slots, std::uint32_t k,
                  std::uint32_t n_experts, hipStream_t stream) {
  hipLaunchKernelGGL(DecodeRoutesKernel, dim3(1), dim3(1024), 0, stream, ids,
                     counts, pad_bounds, cursors, rows_token, rows_slot, slots,
                     k, n_experts);
}

}  // namespace gufo::ext::gsq::wmma
