#ifndef GUFO_EXT_GSQ_STAGES_HPP_
#define GUFO_EXT_GSQ_STAGES_HPP_

// The flavor stages: the host's Flash-Next pipeline with one override —
// the MoE FFN of layers whose routed experts carry a gsq::FlavorState.
// Those three projections stay in their codebook encoding on device, where
// no GGUF kernel reads them; the WMMA GEMMs in gsq/kernels decode them
// after the LDS read, bitwise equal to the standalone reader. Everything
// else — dense and attention projections, the shared expert, routing, the
// epilogue, the draft head — is the runtime's own.

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
