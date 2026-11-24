#include "gsq/weight_source.hpp"

#include <cstring>
#include <vector>

#include <gufo/ext/experimental/ggml_dequant.hpp>

#include <gufo/ext/families/qwen/models/flash_next/config.hpp>
#include <gufo/ext/families/qwen/models/flash_next/ngram.hpp>
#include <gufo/ext/families/qwen/models/flash_next/weights.hpp>
#include <gufo/ext/families/qwen/text/tokenizer.hpp>

#include "gsq/format.hpp"
#include "gsq/identity.hpp"

namespace gufo::ext::gsq {
namespace {

using core::GgmlType;
using models::qwen38_flash_next::Config;

/// Read past the end of every device tensor by the vectorized loads.
constexpr std::size_t kTailMargin = 4096;

/// The row bytes of a bound tensor: the flavor geometry, else the runtime's
/// own sizing (TensorRef sizes the formats it knows, IQ4_NL included).
[[nodiscard]] std::size_t BoundRowBytes(const TensorRef& t) {
  const std::size_t flavor = FlavorRowBytes(t.type, t.cols);
  return flavor != 0 ? flavor : static_cast<std::size_t>(t.RowBytes());
}

}  // namespace

std::unique_ptr<core::GgufReader> OpenFlavorGguf(const std::string& path,
                                                 std::string* error) {
  core::GgufOpenOptions options;
  options.allow_unsized_tensors = true;
  return core::GgufReader::OpenFile(path, error, options);
}

bool IsFlavorArtifact(const std::string& path, std::string* error) {
  auto reader = OpenFlavorGguf(path, error);
  if (reader == nullptr) {
    return false;
  }
  // Flash-Next metadata, and at least one tensor the runtime cannot bind:
  // a flavor expert encoding or a Q5_1 PLE table.
  auto config = Config::FromGguf(*reader, false, nullptr);
  if (!config.has_value()) {
    if (error != nullptr) {
      error->clear();
    }
    return false;
  }
  for (const auto& t : reader->GetTensors()) {
    if (IsFlavorEncoding(t.type)) {
      return true;
    }
    if (t.type == GgmlType::kQ5_1 && t.name.find("per_layer_token_embd") !=
                                        std::string::npos) {
      return true;
    }
  }
  if (error != nullptr) {
    error->clear();
  }
  return false;
}

bool GsqWeightSource::Probe(const std::string& model_path) const {
  std::string error;
  return IsFlavorArtifact(model_path, &error);
}

bool GsqWeightSource::Open(const std::string& model_path,
                           const std::string& mtp_path, std::string* error) {
  reader_ = OpenFlavorGguf(model_path, error);
  if (reader_ == nullptr) {
    return false;
  }
  if (!mtp_path.empty()) {
    mtp_reader_ = OpenFlavorGguf(mtp_path, error);
    if (mtp_reader_ == nullptr) {
      return false;
    }
  }
  return true;
}

bool GsqWeightSource::BindTrunk(ModelWeights* weights, std::string* error) {
  return BindGsqTrunk(*reader_, weights, &states_, error);
}

bool GsqWeightSource::BindMtp(const Config& trunk, MtpWeights* mtp,
                              std::string* error) {
  if (mtp_reader_ == nullptr) {
    if (error != nullptr) {
      *error = "this flavor checkpoint has no MTP sidecar";
    }
    return false;
  }
  return BindGsqMtp(*mtp_reader_, trunk, mtp, &mtp_states_, error);
}

std::unique_ptr<::gufo::tokenization::QwenTokenizer>
GsqWeightSource::Tokenizer(const Config& config, std::string* error) {
  (void)config;
  return ::gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader_, error);
}

std::unique_ptr<NgramTable> GsqWeightSource::Ngram(const ModelWeights& weights,
                                                   std::string* error) {
  const auto& t = weights.ple_table;
  if (t.type == GgmlType::kQ5_1) {
    // The runtime decodes its own PLE formats from the mapping; a flavor
    // checkpoint's Q5_1 table streams through this row decoder instead.
    const std::size_t row_bytes = static_cast<std::size_t>(t.RowBytes());
    const std::size_t head_dim = weights.config.ple_head_dim;
    auto decode = [head_dim](const std::uint8_t* src, std::uint32_t /*row*/,
                             float* dst) {
      ::gufo::quant::DequantizeQ5_1(src, dst, head_dim);
    };
    return NgramTable::OpenEncoded(
        reader_->GetMappedRegions()[t.shard].file_descriptor, t.file_offset,
        t.rows, head_dim, row_bytes, std::move(decode), error);
  }
  return NgramTable::Open(
      reader_->GetMappedRegions()[t.shard].file_descriptor, t.file_offset,
      t.rows, weights.config.ple_head_dim, t.type, error);
}

std::span<const core::GgufMappedRegion> GsqWeightSource::Regions(
    bool mtp) const {
  if (mtp) {
    return mtp_reader_ ? mtp_reader_->GetMappedRegions()
                       : std::span<const core::GgufMappedRegion>{};
  }
  return reader_->GetMappedRegions();
}

fn::Tensor GsqWeightSource::Upload(const TensorRef& t, const UploadHint& hint,
                                   UploadContext& upload) {
  (void)hint;
  const auto* state = static_cast<const FlavorState*>(t.format_state);
  if (state == nullptr) {
    upload.Fail("tensor " + std::string(t.name) + " carries no flavor state");
    return {};
  }
  const std::size_t bytes =
      BoundRowBytes(t) * static_cast<std::size_t>(t.rows) * t.experts;
  void* blocks = nullptr;
  if (!upload.Allocate(&blocks, bytes + kTailMargin)) {
    upload.Fail("device allocation failed for " + std::string(t.name));
    return {};
  }
  if (upload.Measuring()) {
    return {};
  }
  // The encoded blocks land verbatim; the routed WMMA kernels read them
  // through the FlavorState the binding attached.
  if (!upload.CopyFromRegion(t.shard, t.file_offset, bytes, blocks) ||
      !upload.Finish()) {
    upload.Fail("upload failed for " + std::string(t.name));
    return {};
  }
  return fn::Tensor(blocks, GgmlType::kI32, static_cast<std::uint32_t>(t.cols),
                    static_cast<std::uint32_t>(t.rows),
                    static_cast<std::uint32_t>(t.experts), state);
}

std::string GsqWeightSource::Identity() const { return GsqIdentityHex(*reader_); }

std::string GsqWeightSource::DraftIdentity() const {
  return mtp_reader_ ? GsqIdentityHex(*mtp_reader_) : std::string{};
}

std::uint32_t GsqWeightSource::NativeContextLength(
    const std::string& model_path) const {
  auto reader = OpenFlavorGguf(model_path, nullptr);
  if (reader == nullptr) {
    return 0;
  }
  auto config = Config::FromGguf(*reader, false, nullptr);
  return config.has_value() ? config->context_length : 0;
}

std::string GsqWeightSource::ModelName() const {
  return std::string(
      reader_->GetMetadataString("general.name").value_or("Qwen3.8-Flash-Next"));
}

std::unique_ptr<fn::WeightSource> CreateGsqWeightSource() {
  return std::make_unique<GsqWeightSource>();
}

}  // namespace gufo::ext::gsq
