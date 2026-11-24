// The flavor provider: serves pruned ("GSQ") Flash-Next checkpoints through
// the host runner with the flavor weight source and stages. The GGUF probe
// recognizes flavor codebook expert tensors and Q5_1 PLE tables — storage
// core cannot size, so the provider declares accepts_unsized_tensors and
// opens those files itself.
#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <gufo/ext/families/qwen/models/flash_next/runner.hpp>
#include <gufo/ext/generated/qwen38_flash_next_flavors_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "gsq/stages.hpp"
#include "gsq/weight_source.hpp"

namespace {

namespace gsq = gufo::ext::gsq;
namespace generated =
    gufo::ext::generated::extension_qwen38_flash_next_flavors;

/// Flavor storage: routed expert slabs in the flavor codebooks, or a PLE
/// table in Q5_1 (a quantization the host PLE reader does not decode).
bool ProbeReader(const gufo::core::GgufReader& reader) {
  for (const auto& tensor : reader.GetTensors()) {
    const std::string_view name = tensor.name;
    const bool expert =
        name.ends_with("ffn_gate_exps.weight") ||
        name.ends_with("ffn_up_exps.weight") ||
        name.ends_with("ffn_down_exps.weight");
    const bool flavor =
        tensor.type == gufo::core::GgmlType::kIQ3_XXS ||
        tensor.type == gufo::core::GgmlType::kIQ3_S ||
        tensor.type == gufo::core::GgmlType::kIQ2_S ||
        tensor.type == gufo::core::GgmlType::kIQ4_XS;
    if (expert && flavor) return true;
    if (name == "per_layer_token_embd.weight" &&
        tensor.type == gufo::core::GgmlType::kQ5_1) {
      return true;
    }
  }
  return false;
}

/// A Flash-Next flavor directory (or shard): any `*.gguf` it contains that
/// opens with the flavor types admitted carries flavor storage.
bool Probe(std::string_view model_root) {
  try {
    const std::filesystem::path root(model_root);
    if (std::filesystem::is_regular_file(root)) {
      std::string error;
      const auto reader = gsq::OpenFlavorGguf(root.string(), &error);
      return reader != nullptr && ProbeReader(*reader);
    }
    std::vector<std::filesystem::path> shards;
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
      if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
        shards.push_back(entry.path());
      }
    }
    std::sort(shards.begin(), shards.end());
    for (const auto& shard : shards) {
      std::string error;
      const auto reader = gsq::OpenFlavorGguf(shard.string(), &error);
      if (reader != nullptr && ProbeReader(*reader)) return true;
    }
    return false;
  } catch (const std::exception&) {
    return false;
  }
}

std::shared_ptr<gufo::ext::TextModelRunner> Create(
    const gufo::ext::TextRunnerOptions& options, std::string* error) {
  std::string stages_error;
  auto stages = gsq::CreateGsqStages(&stages_error);
  if (stages == nullptr) {
    if (error != nullptr) *error = stages_error;
    return nullptr;
  }
  return gufo::models::qwen38_flash_next::CreateQwenFlashNextTextRunner(
      options, std::move(stages), gsq::CreateGsqWeightSource(), error);
}

}  // namespace

void qwen38_flash_next_flavors_provider_register(
    gufo::ext::Registry& registry) {
  gufo::ext::TextProvider text;
  text.id = generated::kProvider_flash_next_flavors.id;
  text.probe = &Probe;
  text.probe_gguf = &ProbeReader;
  text.create = &Create;
  // GSQ coder containers fuse 3-D expert slabs (and storage types outside
  // the core table) that core cannot size; this provider reads them itself.
  text.accepts_unsized_tensors = true;
  gufo::ext::TextSpeculativeDefaults mtp;
  mtp.backend = "mtp";
  mtp.max_draft_tokens = 7;
  mtp.min_draft_tokens = 1;
  text.speculative_defaults.push_back(std::move(mtp));
  registry.AddProvider(generated::kProvider_flash_next_flavors);
  registry.AddTextProvider(std::move(text));
}
