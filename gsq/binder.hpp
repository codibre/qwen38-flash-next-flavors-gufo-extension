#ifndef GUFO_EXT_GSQ_BINDER_HPP_
#define GUFO_EXT_GSQ_BINDER_HPP_

// The flavor binding: Flash-Next's trunk/draft descriptors bound from a
// pruned ("GSQ") GGUF. It is the runtime's own binder (ModelWeights::Bind)
// with exactly two extensions: routed expert projections may also arrive in
// the codebook quantizations this extension reads — in which case all three
// projections of the expert FFN carry a gsq::FlavorState as
// TensorRef::format_state — and the PLE table may be Q5_1. Everything else,
// including validation and the shard location of each payload, is the
// runtime's.

#include <memory>
#include <string>

#include <gufo/ext/families/qwen/models/flash_next/weights.hpp>

#include "gsq/format.hpp"

namespace gufo::ext::gsq {

using ::gufo::models::qwen38_flash_next::Config;
using ::gufo::models::qwen38_flash_next::ModelWeights;
using ::gufo::models::qwen38_flash_next::MtpWeights;

/// The FlavorState objects a binding attached, alive as long as the
/// descriptors that point at them.
class FlavorStates {
public:
  FlavorStates();
  ~FlavorStates();
  FlavorStates(FlavorStates&&) noexcept;
  FlavorStates& operator=(FlavorStates&&) noexcept = default;
  /// Interns one flavor type and returns its stable address.
  [[nodiscard]] const FlavorState* Add(core::GgmlType type);
  /// Owns an F32 buffer synthesized at bind time (an F16 slot the runtime
  /// consumes as F32) and returns its stable address for kHostShard refs.
  [[nodiscard]] std::uint8_t* KeepHost(std::vector<float> values);
  void Clear();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Binds and validates the flavor trunk. On success `states` owns every
/// format_state the descriptors reference.
[[nodiscard]] bool BindGsqTrunk(const core::GgufReader& reader,
                                ModelWeights* weights, FlavorStates* states,
                                std::string* error);

/// Binds the MTP draft block from a `shared-*` sidecar (or the artifact's
/// own block), flavor experts included.
[[nodiscard]] bool BindGsqMtp(const core::GgufReader& reader,
                              const Config& trunk, MtpWeights* mtp,
                              FlavorStates* states, std::string* error);

}  // namespace gufo::ext::gsq

#endif  // GUFO_EXT_GSQ_BINDER_HPP_
