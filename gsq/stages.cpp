#include "gsq/stages.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "gsq/format.hpp"
#include "gsq/kernels/nan_probe.hpp"
#include "gsq/kernels/rows.hpp"
#include "gsq/kernels/routed_wmma.hpp"

namespace fn = ::gufo::ext::qwen::flash_next;

namespace gufo::ext::gsq {
#include <cstdio>
#include <cstdlib>

namespace {

std::size_t Align4(std::size_t n) { return (n + 3U) & ~std::size_t{3}; }

// Scratch layout: the device tile map, then the route-compaction maps the
// Linear/LinearGroup route needs (it has no MoeScratch to borrow). Sizes
// are the worst case over passes (every pass is narrower than
// `max_batch`, and a routed pair never exceeds one row per expert slot).
struct Sizes {
  std::size_t expert_max = 1;
  std::size_t tiles_entries = 0;
  std::size_t down_rows_entries = 0;
  std::size_t slot_rows_max = 0;
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
  s.slot_rows_max = rows;
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
           Align4(s.expert_max * sizeof(std::uint32_t)) +
           Align4((s.expert_max + 1) * sizeof(std::int32_t)) +
           Align4(s.expert_max * sizeof(std::int32_t)) +
           s.down_rows_entries * 2 * sizeof(std::int32_t);
  }

  bool Attach(const fn::Attachment& attachment, std::string* error) override {
    sizes_ = ComputeSizes(attachment.limits);
    std::uint8_t* p = static_cast<std::uint8_t*>(attachment.scratch);
    tiles_ = reinterpret_cast<std::int32_t*>(p);
    p += Align4(sizes_.tiles_entries * sizeof(std::int32_t));
    counts_ = reinterpret_cast<std::uint32_t*>(p);
    p += Align4(sizes_.expert_max * sizeof(std::uint32_t));
    bounds_ = reinterpret_cast<std::int32_t*>(p);
    p += Align4((sizes_.expert_max + 1) * sizeof(std::int32_t));
    cursors_ = reinterpret_cast<std::int32_t*>(p);
    p += Align4(sizes_.expert_max * sizeof(std::int32_t));
    rows_token_lin_ = reinterpret_cast<std::int32_t*>(p);
    p += Align4(sizes_.down_rows_entries * sizeof(std::int32_t));
    rows_slot_lin_ = reinterpret_cast<std::int32_t*>(p);
    (void)error;
    return true;
  }

  // One projection of a flavor tensor against a LinearInput: expert ids
  // are compacted into the 16-row bucket maps, then the WMMA GEMM decodes
  // the codebook after the LDS read. Output row r reads activation row
  // r / in.slots, exactly the contract's addressing.
  bool BuildRoute(const fn::StageContext& ctx, const fn::Tensor& w,
                  const fn::LinearInput& in) {
    const std::uint32_t k = std::max(in.slots, 1U);
    const std::uint32_t slot_rows = in.rows * in.slots;
    if (in.ids == nullptr) {
      if (ctx.error != nullptr && ctx.error->empty()) {
        *ctx.error = "gsq: flavor projection without expert ids";
      }
      return false;
    }
    if (slot_rows == 0) {
      route_vacuous_ = true;
      return true;
    }
    if (slot_rows > sizes_.slot_rows_max) {
      if (ctx.error != nullptr && ctx.error->empty()) {
        *ctx.error = "gsq: flavor route exceeds its scratch";
      }
      return false;
    }
    route_vacuous_ = false;
    wmma::DecodeRoutes(in.ids, counts_, bounds_, cursors_, rows_token_lin_,
                       rows_slot_lin_, slot_rows, k, w.experts(), ctx.stream);
    route_tile_rows_ = wmma::RoutedTileRows(slot_rows, w.experts());
    route_capacity_ = wmma::RoutedTileCapacity(slot_rows, w.experts(),
                                               route_tile_rows_);
    wmma::RoutedTiles(counts_, tiles_, w.experts(), route_tile_rows_,
                      route_capacity_, ctx.stream);
    if (hipGetLastError() != hipSuccess) {
      if (ctx.error != nullptr && ctx.error->empty()) {
        *ctx.error = "gsq: flavor tile map launch failed";
      }
      return false;
    }
    return true;
  }

  bool RoutedProjection(const fn::StageContext& ctx, const fn::Tensor& w,
                        const fn::LinearInput& in, float* out) {
    if (route_vacuous_) {
      return true;
    }
    if (NanDbgEnabled()) {
      static long lin_calls = 0;
      if (++lin_calls <= 12) {
        std::fprintf(stderr,
                     "[gsq-lin] pass rows=%u slots=%u w=%ux%u e=%u ntok=%u "
                     "layer=%d\n",
                     in.rows, in.slots, w.rows(), w.cols(), w.experts(),
                     ctx.n_tokens, ctx.layer.index());
      }
      NanScanF32(in.x, static_cast<std::size_t>(in.rows) * w.cols(),
                 in.slots == 1 ? "L" : "l", ctx.layer.index(), ctx.n_tokens,
                 ctx.stream);
    }
    // DefaultStages::HalfRows sizes the F16 conversion by ctx.n_tokens, but
    // these activations are in.rows wide: the down projection consumes the
    // per-slot SwiGLU buffer (n_tokens * slots rows) while the context still
    // counts tokens. Converting only n_tokens rows leaves the GEMM reading
    // stale rows of a previous activation — the garbage that surfaced as
    // 'no finite values' in the next attention block. Carry the true row
    // count in a context copy.
    fn::StageContext act = ctx;
    act.n_tokens = in.rows;
    const auto hx = static_cast<const __half*>(
        ctx.defaults.HalfRows(act, in.x, w.cols()));
    if (hx == nullptr) {
      if (ctx.error != nullptr && ctx.error->empty()) {
        *ctx.error = "gsq: missing F16 activation rows for the flavor linear";
      }
      return false;
    }
    const auto weight_type =
        static_cast<wmma::WeightType>(FlavorType(w.format_state()));
    // slots == 1 means the activations are already per slot (the down
    // projection after SwiGLU); wider inputs are per token.
    const std::int32_t* x_rows = in.slots == 1 ? rows_slot_lin_ : rows_token_lin_;
    if (!wmma::RoutedF16Gemm(w.data(), weight_type, hx, tiles_,
                             route_capacity_, route_tile_rows_, bounds_,
                             x_rows, rows_token_lin_, rows_slot_lin_,
                             std::max(in.slots, 1U), nullptr, out, nullptr,
                             w.rows(), w.cols(), ctx.stream)) {
      if (ctx.error != nullptr && ctx.error->empty()) {
        *ctx.error = "gsq: flavor GEMM launch failed";
      }
      return false;
    }
    NanLaunchCheck("lin_gemm", ctx.layer.index());
    if (NanDbgEnabled())
      NanScanF32(out, static_cast<std::size_t>(in.rows) * in.slots * w.rows(),
                 in.slots == 1 ? "M" : "m", ctx.layer.index(), ctx.n_tokens,
                 ctx.stream);
    return true;
  }

  // A projection of a tensor with format_state(). The default rejects
  // these tensors; the contract routes them here.
  bool Linear(const fn::StageContext& ctx, const fn::Tensor& w,
              const fn::LinearInput& in, float* out) override {
    if (w.format_state() == nullptr) {
      return ctx.defaults.Linear(ctx, w, in, out);
    }
    if (NanDbgEnabled()) {
      NanPassEnter();
    }
    if (!BuildRoute(ctx, w, in)) {
      return false;
    }
    return RoutedProjection(ctx, w, in, out);
  }

  // The routed gate + up pair (and the dense groups, which never carry a
  // flavor tensor). One compaction serves the whole group; the executor
  // runs SwiGLU over the two outputs afterwards.
  bool LinearGroup(const fn::StageContext& ctx,
                   std::span<const fn::LinearJob> jobs,
                   const fn::LinearInput& in) override {
    bool any_flavor = false;
    for (const fn::LinearJob& job : jobs) {
      any_flavor = any_flavor || job.w.format_state() != nullptr;
    }
    if (!any_flavor) {
      for (const fn::LinearJob& job : jobs) {
        if (!ctx.defaults.Linear(ctx, job.w, in, job.out)) {
          return false;
        }
      }
      return true;
    }
    if (!BuildRoute(ctx, jobs[0].w, in)) {
      return false;
    }
    for (const fn::LinearJob& job : jobs) {
      if (job.w.format_state() == nullptr) {
        if (!ctx.defaults.Linear(ctx, job.w, in, job.out)) {
          return false;
        }
      } else if (!RoutedProjection(ctx, job.w, in, job.out)) {
        return false;
      }
    }
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
    if (routing.last_only && ctx.n_tokens > 0) {
      // Component-cache catch-up: only the last row's output is consumed,
      // and the earlier rows carry route choices nobody committed to.
      // The runtime's own route cannot read flavor experts either, so run
      // one flavor pass over the last row alone.
      const std::size_t last = ctx.n_tokens - 1;
      fn::StageContext one = ctx;
      one.n_tokens = 1;
      fn::MoeRouting tail = routing;
      tail.ids += last * routing.used;
      tail.weights += last * routing.used;
      tail.router += static_cast<std::size_t>(last) * routing.router_stride;
      tail.last_only = false;
      return MoeFfn(one, tail,
                    x + last * static_cast<std::size_t>(gate.cols()),
                    out + last * static_cast<std::size_t>(down.rows()));
    }
    if (NanDbgEnabled()) NanPassEnter();
    if (!ctx.defaults.SharedExpert(ctx, routing, x, routing.overlap)) {
      return false;
    }
    NanLaunchCheck("shared_expert", ctx.layer.index());
    if (ctx.n_tokens == 0) {
      if (routing.overlap && !ctx.defaults.JoinSide(ctx)) return false;
      ctx.defaults.MoeEpilogue(ctx, routing, out, /*half_down=*/true);
      return true;
    }
    if (NanDbgEnabled())
      NanScanF32(x, static_cast<std::size_t>(ctx.n_tokens) * gate.cols(), "x",
                 ctx.layer.index(), ctx.n_tokens, ctx.stream);
    ctx.defaults.CompactRoutes(ctx, routing);
    NanLaunchCheck("compact_routes", ctx.layer.index());
    const auto* hx = static_cast<const __half*>(
        ctx.defaults.HalfRows(ctx, x, gate.cols()));
    NanLaunchCheck("half_rows", ctx.layer.index());
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
      NanLaunchCheck("decode_routes", ctx.layer.index());
    }
    wmma::RoutedTiles(routing.scratch.expert_counts, tiles_, experts,
                      tile_rows, capacity, ctx.stream);
    NanLaunchCheck("routed_tiles", ctx.layer.index());

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
    NanLaunchCheck("gate_gemm", ctx.layer.index());
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
    NanLaunchCheck("up_gemm", ctx.layer.index());
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
    NanLaunchCheck("down_gemm", ctx.layer.index());
    if (NanDbgEnabled()) {
      NanScanF32(routing.scratch.gate,
                 static_cast<std::size_t>(rows) * gate.rows(), "gate",
                 ctx.layer.index(), ctx.n_tokens, ctx.stream);
      NanScanF16(reinterpret_cast<const __half*>(routing.scratch.up),
                 static_cast<std::size_t>(rows) * up.rows(), "up",
                 ctx.layer.index(), ctx.n_tokens, ctx.stream);
      NanScanF16(reinterpret_cast<const __half*>(routing.scratch.down),
                 static_cast<std::size_t>(rows) * down.rows(), "down",
                 ctx.layer.index(), ctx.n_tokens, ctx.stream);
    }
    if (routing.overlap && !ctx.defaults.JoinSide(ctx)) return false;
    ctx.defaults.MoeEpilogue(ctx, routing, out, /*half_down=*/true);
    NanLaunchCheck("epilogue", ctx.layer.index());
    if (NanDbgEnabled())
      NanScanF32(out, static_cast<std::size_t>(ctx.n_tokens) * down.rows(),
                 "out", ctx.layer.index(), ctx.n_tokens, ctx.stream);
    return true;
  }

private:
 Sizes sizes_;
 std::int32_t* tiles_{nullptr};
 std::uint32_t* counts_{nullptr};
 std::int32_t* bounds_{nullptr};
 std::int32_t* cursors_{nullptr};
 std::int32_t* rows_token_lin_{nullptr};
 std::int32_t* rows_slot_lin_{nullptr};
 std::uint32_t route_tile_rows_{16};
 std::uint32_t route_capacity_{0};
 bool route_vacuous_{false};
};

}  // namespace

std::unique_ptr<fn::Stages> CreateGsqStages(std::string* error) {
  (void)error;
  return std::make_unique<GsqStages>();
}

}  // namespace gufo::ext::gsq
