#ifndef GUFO_EXT_GSQ_WEIGHT_SOURCE_HPP_
#define GUFO_EXT_GSQ_WEIGHT_SOURCE_HPP_

// The flavor weight source: a pruned ("GSQ") GGUF opened through the
// runtime's container services with `allow_unsized_tensors`, so the reader
// admits the flavor tensor types instead of sizing them; bound by
// gsq/binder; and uploaded with the runtime's region copy. Two hooks carry
// what the runtime's own source cannot serve: a Q5_1 PLE table reaches
// NgramTable through a row decoder (OpenEncoded), and the expert trio of a
// flavor checkpoint lands verbatim as blocks with its FlavorState attached.

#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include <gufo/ext/families/qwen/models/flash_next/weight_source.hpp>
#include <gufo/ext/gguf_reader.hpp>

#include "gsq/binder.hpp"

namespace gufo::ext::gsq {

namespace fn = ::gufo::ext::qwen::flash_next;
using fn::ModelWeights;
using fn::MtpWeights;
using fn::NgramTable;
using fn::TensorRef;
using fn::UploadContext;
using fn::UploadHint;

/// Opens a flavor container: the runtime's reader with the flavor tensor
/// types admitted (split shards resolved as usual). nullptr with `error`
/// when the file is not a readable GGUF.
[[nodiscard]] std::unique_ptr<core::GgufReader> OpenFlavorGguf(
    const std::string& path, std::string* error);

/// True when `path` is a Flash-Next GGUF whose routed experts use the
/// flavor codebook quantizations (or a PLE table the runtime does not
/// decode), i.e. a checkpoint only this extension can serve.
[[nodiscard]] bool IsFlavorArtifact(const std::string& path,
                                    std::string* error);

class GsqWeightSource final : public fn::WeightSource {
public:
  [[nodiscard]] bool Probe(const std::string& model_path) const override;

  bool Open(const std::string& model_path, const std::string& mtp_path,
            std::string* error) override;

  bool BindTrunk(ModelWeights* weights, std::string* error) override;

  bool BindMtp(const fn::Config& trunk, MtpWeights* mtp,
               std::string* error) override;

  std::unique_ptr<::gufo::tokenization::QwenTokenizer> Tokenizer(
      const fn::Config& config, std::string* error) override;

  std::unique_ptr<NgramTable> Ngram(const ModelWeights& weights,
                                    std::string* error) override;

  [[nodiscard]] std::span<const core::GgufMappedRegion> Regions(
      bool mtp) const override;

  fn::Tensor Upload(const TensorRef& tensor, const UploadHint& hint,
                    UploadContext& upload) override;

  [[nodiscard]] std::string Identity() const override;

  [[nodiscard]] std::string DraftIdentity() const override;

  [[nodiscard]] std::uint32_t NativeContextLength(
      const std::string& model_path) const override;

  [[nodiscard]] std::string ModelName() const override;

private:
  std::unique_ptr<core::GgufReader> reader_;
  std::unique_ptr<core::GgufReader> mtp_reader_;
  FlavorStates states_;
  FlavorStates mtp_states_;
};

[[nodiscard]] std::unique_ptr<fn::WeightSource> CreateGsqWeightSource();

}  // namespace gufo::ext::gsq

#endif  // GUFO_EXT_GSQ_WEIGHT_SOURCE_HPP_
