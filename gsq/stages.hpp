#ifndef GUFO_EXT_GSQ_STAGES_HPP_
#define GUFO_EXT_GSQ_STAGES_HPP_

// The flavor stages: the host's Flash-Next pipeline with the overrides
// that keep layers whose routed experts carry a gsq::FlavorState on the
// codebook path. The MoE FFN routes through the WMMA GEMMs in
// gsq/kernels, and so do the bare Linear/LinearGroup calls — the MTP
// verify and cache catch-up passes reach the experts as single
// projections, where the runtime's own stages reject a custom format.
// Everything else — dense and attention projections, the shared expert,
// routing, the epilogue, the draft head — is the runtime's own.

#include <memory>
#include <string>

#include <gufo/ext/families/qwen/models/flash_next/stages.hpp>

namespace gufo::ext::gsq {

/// Stages for a pruned ("flavor") artifact: defaults everywhere, the WMMA
/// flavor MoE on the layers that need it.
[[nodiscard]] std::unique_ptr<::gufo::ext::qwen::flash_next::Stages>
CreateGsqStages(std::string* error);

}  // namespace gufo::ext::gsq

#endif  // GUFO_EXT_GSQ_STAGES_HPP_
