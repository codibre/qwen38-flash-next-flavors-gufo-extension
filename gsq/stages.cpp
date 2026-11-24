#include "gsq/stages.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "gsq/format.hpp"
#include "gsq/kernels/rows.hpp"
#include "gsq/kernels/routed_wmma.hpp"

namespace fn = ::gufo::ext::qwen::flash_next;

namespace gufo::ext::gsq {
#include <cstdio>
#include <cstdlib>

namespace {

std::size_t Align4(std::size_t n) { return (n + 3U) & ~std::size_t{3}; }

// Scratch layout: the device tile map, then the down projection's
// activation-row map. Sizes are the worst case over passes (every pass is
// narrower than `max_batch`, and a routed pair never exceeds one per
// expert slot).
struct Sizes {
  std::size_t expert_max = 1;
  std::size_t tiles_entries = 0;
  std::size_t down_rows_entries = 0;
};

Sizes ComputeSizes(const fn::Limits& limits) {
  Sizes s;
  for (const fn::Layer& l : limits.layers) {
    const fn::Tensor router = l.tensor(fn::TensorId::kRouter);
    if (!router.empty()) {
      s.expert_max = std::max(s.expert_max, static_cast<std::size_t>(router.rows()));
    }
  }
  const std::size_t rows = static_cast<std::size_t>(limits.max_batch) * s.expert_max;
  // 16-row tiles maximize the entry count, so the narrow bound covers wide
  // passes too.
  s.tiles_entries =
      wmma::RoutedTileCapacity(static_cast<std::uint32_t>(rows),
                               static_cast<std::uint32_t>(s.expert_max),
                               wmma::kRoutedTileRowsNarrow);
  s.down_rows_entries =
      rows + s.expert_max * (wmma::kRoutedTileRowsWide - 1);
  return s;
}

class GsqStages final : public fn::Stages {
public:
  [[nodiscard]] std::size_t ScratchBytes(const fn::Limits& limits) const override {
    const Sizes s = ComputeSizes(limits);
    return Align4(s.tiles_entries * sizeof(std::int32_t)) +
           s.down_rows_entries * sizeof(std::int32_t);
  }

  bool Attach(const fn::Attachment& attachment, std::string* error) override {
    sizes_ = ComputeSizes(attachment.limits);
    tiles_ = static_cast<std::int32_t*>(attachment.scratch);
    down_x_ = tiles_ + Align4(sizes_.tiles_entries * sizeof(std::int32_t)) /
                            sizeof(std::int32_t);
    (void)error;
    return true;
  }

  bool MoeFfn(const fn::StageContext& ctx, const fn::MoeRouting& routing,
              const float* x, float* out) override {
    const fn::Tensor gate = ctx.layer.tensor(fn::TensorId::kFfnGateExps);
    if (gate.format_state() == nullptr) {
      // GGUF-encoded experts (or none): the runtime's own route.
      return ctx.defaults.MoeFfn(ctx, routing, x, out);
    }
    const fn::Tensor up = ctx.layer.tensor(fn::TensorId::kFfnUpExps);
    const fn::Tensor down = ctx.layer.tensor(fn::TensorId::kFfnDownExps);
    if (up.format_state() == nullptr || down.format_state() == nullptr) {
      if (ctx.error->empty()) {
        *ctx.error =
            "gsq: flavor expert trio partially carries flavor state in layer " +
            std::to_string(ctx.layer.index());
      }
      return false;
    }
    if (!ctx.defaults.SharedExpert(ctx, routing, x, routing.overlap)) {
      return false;
    }
    if (ctx.n_tokens == 0) {
      if (routing.overlap && !ctx.defaults.JoinSide(ctx)) return false;
      ctx.defaults.MoeEpilogue(ctx, routing, out, /*half_down=*/true);
      return true;
    }
    ctx.defaults.CompactRoutes(ctx, routing);
    const auto* hx = static_cast<const __half*>(
        ctx.defaults.HalfRows(ctx, x, gate.cols()));
    if (hx == nullptr) {
      if (ctx.error->empty()) {
        *ctx.error = "gsq: no F16 activation rows for the flavor experts";
      }
      return false;
    }

    const std::uint32_t experts = routing.num_experts;
    const std::uint32_t used = routing.used;
    const std::uint32_t rows = ctx.n_tokens * used;
    const std::uint32_t tile_rows = wmma::RoutedTileRows(rows, experts);
    const std::uint32_t capacity = static_cast<std::uint32_t>(
        wmma::RoutedTileCapacity(rows, experts, tile_rows));
    // Vector-sized passes (decode and draft verify) never fill
    // expert_counts: the host keeps them on its own vector route. The
    // flavor kernels are WMMA-only and need the bucketed maps, so build
    // them from the ids with the host's padded-bucket contract.
    if (ctx.n_tokens <= 16) {
      wmma::DecodeRoutes(routing.ids, routing.scratch.expert_counts,
                         routing.scratch.routed_bounds,
                         routing.scratch.routed_cursors,
                         routing.scratch.rows_token,
                         routing.scratch.rows_slot, rows, used, experts,
                         ctx.stream);
    }
    wmma::RoutedTiles(routing.scratch.expert_counts, tiles_, experts,
                      tile_rows, capacity, ctx.stream);

    const auto gate_type =
        static_cast<wmma::WeightType>(FlavorType(gate.format_state()));
    const auto up_type =
        static_cast<wmma::WeightType>(FlavorType(up.format_state()));
    const auto down_type =
        static_cast<wmma::WeightType>(FlavorType(down.format_state()));

    if (!wmma::RoutedF16Gemm(gate.data(), gate_type, hx, tiles_, capacity,
                             tile_rows, routing.scratch.routed_bounds,
                             routing.scratch.rows_token,
                             routing.scratch.rows_token,
                             routing.scratch.rows_slot, used, nullptr,
                             routing.scratch.gate, nullptr, gate.rows(),
                             gate.cols(), ctx.stream)) {
      if (ctx.error->empty()) {
        *ctx.error = "gsq: flavor gate projection failed";
      }
      return false;
    }
    if (!wmma::RoutedF16Gemm(up.data(), up_type, hx, tiles_, capacity,
                             tile_rows, routing.scratch.routed_bounds,
                             routing.scratch.rows_token,
                             routing.scratch.rows_token,
                             routing.scratch.rows_slot, used,
                             routing.scratch.gate, nullptr,
                             reinterpret_cast<__half*>(routing.scratch.up),
                             up.rows(), up.cols(), ctx.stream)) {
      if (ctx.error->empty()) {
        *ctx.error = "gsq: flavor up projection failed";
      }
      return false;
    }
    // The down GEMM reads the SwiGLU output the up projection staged at the
    // global routing-slot rows; the kernel indexes activations by slot
    // (rows_slot doubles as the input map, matching the host's own route).
    if (!wmma::RoutedF16Gemm(down.data(), down_type,
                             reinterpret_cast<const __half*>(routing.scratch.up),
                             tiles_, capacity, tile_rows,
                             routing.scratch.routed_bounds,
                             routing.scratch.rows_slot,
                             routing.scratch.rows_token,
                             routing.scratch.rows_slot, used, nullptr, nullptr,
                             reinterpret_cast<__half*>(routing.scratch.down),
                             down.rows(), down.cols(), ctx.stream)) {
      if (ctx.error->empty()) {
        *ctx.error = "gsq: flavor down projection failed";
      }
      return false;
    }
    if (routing.overlap && !ctx.defaults.JoinSide(ctx)) return false;
    ctx.defaults.MoeEpilogue(ctx, routing, out, /*half_down=*/true);
    return true;
  }

private:
  Sizes sizes_;
  std::int32_t* tiles_{nullptr};
  std::int32_t* down_x_{nullptr};
};

}  // namespace

std::unique_ptr<fn::Stages> CreateGsqStages(std::string* error) {
  (void)error;
  return std::make_unique<GsqStages>();
}

}  // namespace gufo::ext::gsq
