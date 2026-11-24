#include "gsq/binder.hpp"

#include <gufo/ext/experimental/ggml_dequant.hpp>

#include <deque>
#include <initializer_list>
#include <utility>

namespace gufo::ext::gsq {
namespace {

using core::GgmlType;
using models::qwen38_flash_next::HcMixer;
using models::qwen38_flash_next::LayerWeights;
using models::qwen38_flash_next::TensorRef;


/// The format table of the runtime's own binder, plus this module's flavor
/// geometry for the types it does not size.
struct Size {
  std::uint64_t row_bytes;
  bool flavor;
};

[[nodiscard]] Size SizeOf(GgmlType type, std::uint64_t cols) {
  if (IsFlavorEncoding(type)) {
    return {FlavorRowBytes(type, cols), true};
  }
  TensorRef t;
  t.type = type;
  t.cols = cols;
  return {t.RowBytes(), false};
}

struct Binder {
  const core::GgufReader& reader;
  std::string* error;
  FlavorStates* states;
  bool ok{true};
  /// Set when any routed expert projection needed a FlavorState.
  bool flavor_experts{false};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  /// Binds `name` with the exact shape and one of the accepted formats, in
  /// the runtime's binder semantics; the payload of a flavor-format tensor
  /// is located the same way, but sized from the flavor geometry.
  TensorRef Get(const std::string& name, std::uint64_t cols, std::uint64_t rows,
                std::uint64_t experts, std::initializer_list<GgmlType> types,
                bool required = true) {
    TensorRef t;
    const auto* info = reader.FindTensor(name);
    if (info == nullptr) {
      if (required) {
        Fail("missing tensor " + name);
      }
      return t;
    }
    const auto& d = info->dimensions;
    const std::uint64_t got_cols = d.size() > 0 ? d[0] : 1;
    const std::uint64_t got_rows = d.size() > 1 ? d[1] : 1;
    const std::uint64_t got_experts = d.size() > 2 ? d[2] : 1;
    if (got_cols != cols || got_rows != rows || got_experts != experts ||
        d.size() > 3) {
      Fail("tensor " + name + " has shape [" + std::to_string(got_cols) + ", " +
           std::to_string(got_rows) + ", " + std::to_string(got_experts) +
           "], expected [" + std::to_string(cols) + ", " +
           std::to_string(rows) + ", " + std::to_string(experts) + "]");
      return t;
    }
    bool type_ok = false;
    for (auto type : types) {
      type_ok = type_ok || info->type == type;
    }
    const Size size = SizeOf(info->type, cols);
    if (!type_ok || size.row_bytes == 0) {
      Fail("tensor " + name + " has unsupported format " +
           std::string(core::ToString(info->type)));
      return t;
    }
    t.data = info->data;
    t.type = info->type;
    t.cols = cols;
    t.rows = rows;
    t.experts = experts;
    t.name = info->name;
    // Locate the shard so disk readers can address the payload directly, and
    // so a payload running past its shard is caught here rather than as a
    // fault later.
    const auto regions = reader.GetMappedRegions();
    const auto address = reinterpret_cast<std::uintptr_t>(info->data);
    bool inside = false;
    for (std::uint32_t i = 0; i < regions.size(); ++i) {
      const auto base = reinterpret_cast<std::uintptr_t>(regions[i].data);
      if (address >= base && address < base + regions[i].size) {
        t.shard = i;
        t.file_offset = address - base;
        inside = size.row_bytes * rows * experts <=
                 regions[i].size - t.file_offset;
        break;
      }
    }
    if (!inside) {
      Fail("tensor " + name + " is truncated");
      return TensorRef{};
    }
    return t;
  }

  HcMixer Mixer(const std::string& prefix, const Config& c, bool with_inject) {
    HcMixer m;
    const std::uint64_t hc_dim = c.HcDim();
    m.norm = Get(prefix + "_norm.weight", hc_dim, 1, 1, {GgmlType::kF32});
    m.down =
        Get(prefix + "_down.weight", hc_dim, c.hc_low_rank, 1,
            {GgmlType::kQ8_0, GgmlType::kBF16, GgmlType::kF16, GgmlType::kF32});
    m.up =
        Get(prefix + "_up.weight", c.hc_low_rank, hc_dim, 1,
            {GgmlType::kQ8_0, GgmlType::kBF16, GgmlType::kF16, GgmlType::kF32});
    if (with_inject) {
      m.inject = Get(prefix + "_inject.weight", hc_dim, c.hc_count, 1,
                     {GgmlType::kF32, GgmlType::kQ8_0, GgmlType::kBF16});
    }
    return m;
  }

  /// Routed expert projections: the runtime's formats, or the flavor
  /// codebooks (and IQ4_NL, which the runtime only uses for the PLE table).
  /// When any of the three needs the flavor route, all three carry a
  /// FlavorState so the MoE stage runs one route for the expert FFN.
  void Experts(const Config& c, std::uint32_t il, LayerWeights& l) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::uint64_t hidden = c.hidden_size;
    l.ffn_gate_exps = Get(p + "ffn_gate_exps.weight", hidden, c.expert_ff,
                          c.num_experts,
                          {GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K,
                           GgmlType::kQ5_1, GgmlType::kQ8_0, GgmlType::kIQ4_NL,
                           GgmlType::kIQ3_S, GgmlType::kIQ2_S,
                           GgmlType::kIQ4_XS});
    l.ffn_up_exps = Get(p + "ffn_up_exps.weight", hidden, c.expert_ff,
                        c.num_experts,
                        {GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K,
                         GgmlType::kQ5_1, GgmlType::kQ8_0, GgmlType::kIQ4_NL,
                         GgmlType::kIQ3_S, GgmlType::kIQ2_S,
                         GgmlType::kIQ4_XS});
    l.ffn_down_exps = Get(p + "ffn_down_exps.weight", c.expert_ff, hidden,
                          c.num_experts,
                          {GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K,
                           GgmlType::kQ5_1, GgmlType::kQ8_0, GgmlType::kIQ4_NL,
                           GgmlType::kIQ3_S, GgmlType::kIQ2_S,
                           GgmlType::kIQ4_XS});
    if (ok) {
      RouteFlavorExperts(l);
    }
  }

  /// True while serving; false when a bound expert tensor is present but no
  /// uploadable combination exists (an empty trio is the no-experts case).
  bool RouteFlavorExperts(LayerWeights& l) {
    const TensorRef* trio[3] = {&l.ffn_gate_exps, &l.ffn_up_exps,
                                &l.ffn_down_exps};
    bool flavor = false;
    for (const TensorRef* t : trio) {
      if (t->empty()) {
        continue;
      }
      if (IsFlavorEncoding(t->type) || t->type == GgmlType::kIQ4_NL) {
        flavor = true;
      }
    }
    if (!flavor) {
      return true;
    }
    for (TensorRef* t : {&l.ffn_gate_exps, &l.ffn_up_exps, &l.ffn_down_exps}) {
      if (t->empty()) {
        Fail("flavor expert FFN is incomplete");
        return false;
      }
      t->format_state = states->Add(t->type);
    }
    flavor_experts = true;
    return true;
  }

  LayerWeights Layer(const Config& c, std::uint32_t il, bool linear, bool ple,
                     bool nextn) {
    LayerWeights l;
    l.linear = linear;
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::uint64_t hidden = c.hidden_size;
    const std::uint64_t hc_dim = c.HcDim();
    const auto dense = {GgmlType::kQ8_0, GgmlType::kBF16, GgmlType::kF16,
                        GgmlType::kF32};

    l.hc_attn = Mixer(p + "hc_attn", c, true);
    l.hc_ffn = Mixer(p + "hc_ffn", c, true);

    if (linear) {
      l.ssm_qkv =
          Get(p + "attn_qkv.weight", hidden, c.SsmConvChannels(), 1, dense);
      l.ssm_gate =
          Get(p + "attn_gate.weight", hidden, c.SsmValueDim(), 1, dense);
      l.ssm_conv1d = Get(p + "ssm_conv1d.weight", c.ssm_conv_kernel,
                         c.SsmConvChannels(), 1, {GgmlType::kF32});
      l.ssm_alpha = Get(p + "ssm_alpha.weight", hidden, c.ssm_num_v_heads, 1,
                        {GgmlType::kF32, GgmlType::kBF16, GgmlType::kQ8_0});
      l.ssm_beta = Get(p + "ssm_beta.weight", hidden, c.ssm_num_v_heads, 1,
                       {GgmlType::kF32, GgmlType::kBF16, GgmlType::kQ8_0});
      l.ssm_dt =
          Get(p + "ssm_dt.bias", c.ssm_num_v_heads, 1, 1, {GgmlType::kF32});
      l.ssm_a = Get(p + "ssm_a", c.ssm_num_v_heads, 1, 1, {GgmlType::kF32});
      l.ssm_norm =
          Get(p + "ssm_norm.weight", c.ssm_head_dim, 1, 1, {GgmlType::kF32});
      l.ssm_out = Get(p + "ssm_out.weight", c.SsmValueDim(), hidden, 1, dense);
    } else {
      l.attn_q =
          Get(p + "attn_q.weight", hidden, 2 * c.AttentionQDim(), 1, dense);
      l.attn_k = Get(p + "attn_k.weight", hidden, c.AttentionKvDim(), 1, dense);
      l.attn_v = Get(p + "attn_v.weight", hidden, c.AttentionKvDim(), 1, dense);
      l.attn_out =
          Get(p + "attn_output.weight", c.AttentionQDim(), hidden, 1, dense);
      l.attn_q_norm =
          Get(p + "attn_q_norm.weight", c.head_dim, 1, 1, {GgmlType::kF32});
      l.attn_k_norm =
          Get(p + "attn_k_norm.weight", c.head_dim, 1, 1, {GgmlType::kF32});
      l.indexer_q = Get(p + "indexer.q_proj.weight", hidden,
                        c.indexer_heads * c.indexer_head_dim, 1, dense);
      l.indexer_k = Get(p + "indexer.k_proj.weight", hidden, c.indexer_head_dim,
                        1, dense);
      l.indexer_q_norm = Get(p + "indexer.q_norm.weight", c.indexer_head_dim, 1,
                             1, {GgmlType::kF32});
      l.indexer_k_norm = Get(p + "indexer.k_norm.weight", c.indexer_head_dim, 1,
                             1, {GgmlType::kF32});
    }

    if (ple) {
      const std::uint64_t ple_dim = c.PleEmbeddingDim();
      l.ple_key = Get(p + "ple_key.weight", ple_dim, hc_dim, 1, dense);
      l.ple_value = Get(p + "ple_value.weight", ple_dim, hidden, 1, dense);
      l.ple_norm_key =
          Get(p + "ple_norm_key.weight", hc_dim, 1, 1, {GgmlType::kF32});
      l.ple_norm_query =
          Get(p + "ple_norm_query.weight", hc_dim, 1, 1, {GgmlType::kF32});
      l.ple_norm_conv =
          Get(p + "ple_norm_conv.weight", hc_dim, 1, 1, {GgmlType::kF32});
      // RCO releases store this small depthwise conv in F16; the runtime
      // consumes it as F32 (executor PleConv), so widen it once at bind
      // time and hand the host-memory copy to the upload (kHostShard).
      l.ple_conv1d = Get(p + "ple_conv1d.weight", c.ple_conv_kernel, hc_dim, 1,
                         {GgmlType::kF32, GgmlType::kF16});
      if (ok && l.ple_conv1d.type == GgmlType::kF16) {
        std::vector<float> widened(static_cast<std::size_t>(l.ple_conv1d.cols) *
                                   l.ple_conv1d.rows);
        const auto* bits =
            static_cast<const std::uint16_t*>(l.ple_conv1d.data);
        for (std::size_t i = 0; i < widened.size(); ++i) {
          widened[i] = ::gufo::quant::Fp16ToFloat(bits[i]);
        }
        l.ple_conv1d.data = states->KeepHost(std::move(widened));
        l.ple_conv1d.type = GgmlType::kF32;
        l.ple_conv1d.shard = models::qwen38_flash_next::kHostShard;
        l.ple_conv1d.file_offset = 0;
      }
    }

    l.router = Get(p + "ffn_gate_inp.weight", hidden, c.num_experts, 1,
                   {GgmlType::kF32});
    Experts(c, il, l);
    l.shexp_gate_inp =
        Get(p + "ffn_gate_inp_shexp.weight", hidden, 1, 1, {GgmlType::kF32});
    l.shexp_gate =
        Get(p + "ffn_gate_shexp.weight", hidden, c.shared_expert_ff, 1, dense);
    l.shexp_up =
        Get(p + "ffn_up_shexp.weight", hidden, c.shared_expert_ff, 1, dense);
    l.shexp_down =
        Get(p + "ffn_down_shexp.weight", c.shared_expert_ff, hidden, 1, dense);

    if (nextn) {
      l.nextn_enorm =
          Get(p + "nextn.enorm.weight", hidden, 1, 1, {GgmlType::kF32});
      l.nextn_hnorm =
          Get(p + "nextn.hnorm.weight", hc_dim, 1, 1, {GgmlType::kF32});
      l.nextn_eh_proj =
          Get(p + "nextn.eh_proj.weight", 2 * hidden, hidden, 1, dense);
      l.nextn_head = Mixer(p + "nextn.hc_head", c, false);
    }
    return l;
  }
};

}  // namespace

struct FlavorStates::Impl {
  std::deque<FlavorState> states;
  std::deque<std::vector<float>> host_buffers;
};

FlavorStates::FlavorStates() = default;
FlavorStates::~FlavorStates() = default;
FlavorStates::FlavorStates(FlavorStates&&) noexcept = default;

const FlavorState* FlavorStates::Add(core::GgmlType type) {
  if (impl_ == nullptr) {
    impl_ = std::make_unique<Impl>();
  }
  impl_->states.push_back(FlavorState{type});
  return &impl_->states.back();
}

std::uint8_t* FlavorStates::KeepHost(std::vector<float> values) {
  impl_->host_buffers.push_back(std::move(values));
  return reinterpret_cast<std::uint8_t*>(impl_->host_buffers.back().data());
}

void FlavorStates::Clear() { impl_.reset(); }

bool BindGsqTrunk(const core::GgufReader& reader, ModelWeights* weights,
                  FlavorStates* states, std::string* error) {
  auto config = Config::FromGguf(reader, true, error);
  if (!config.has_value()) {
    return false;
  }
  Binder b{reader, error, states};
  ModelWeights w;
  w.config = *config;
  const Config& c = w.config;
  const auto dense = {GgmlType::kQ8_0, GgmlType::kQ6_K, GgmlType::kBF16,
                      GgmlType::kF16, GgmlType::kF32};

  {
    // The vocabulary is the embedding row count; no metadata key carries it.
    const auto* info = reader.FindTensor("token_embd.weight");
    if (info == nullptr || info->dimensions.size() != 2) {
      b.Fail("missing token_embd.weight");
      return false;
    }
    w.config.vocab_size = static_cast<std::uint32_t>(info->dimensions[1]);
  }
  w.token_embd =
      b.Get("token_embd.weight", c.hidden_size, c.vocab_size, 1, dense);
  w.output =
      b.Get("output.weight", c.hidden_size, c.vocab_size, 1, dense, false);
  if (w.output.empty()) {
    w.output = w.token_embd;
  }
  w.hc_head = b.Mixer("output_hc", c, false);
  if (c.ple_layer >= 0) {
    // The table may be Q5_1 (flavor checkpoints) as well as the runtime's
    // own formats; its rows reach the device only through NgramTable.
    const auto* info = reader.FindTensor("per_layer_token_embd.weight");
    const std::uint64_t rows = info != nullptr && info->dimensions.size() == 2
                                   ? info->dimensions[1]
                                   : 0;
    if (rows < c.ple_rows) {
      b.Fail("per_layer_token_embd.weight is missing or too short");
    } else {
      w.ple_table =
          b.Get("per_layer_token_embd.weight", c.ple_head_dim, rows, 1,
                {GgmlType::kIQ4_NL, GgmlType::kQ8_0, GgmlType::kBF16,
                 GgmlType::kQ5_1});
    }
  }
  w.layers.reserve(c.num_layers);
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    w.layers.push_back(
        b.Layer(c, il, c.IsLinearLayer(il), c.IsPleLayer(il), false));
  }
  if (!b.ok) {
    return false;
  }
  *weights = std::move(w);
  return true;
}

bool BindGsqMtp(const core::GgufReader& reader, const Config& trunk,
                MtpWeights* mtp, FlavorStates* states, std::string* error) {
  auto config = Config::FromGguf(reader, false, error);
  if (!config.has_value()) {
    return false;
  }
  Binder b{reader, error, states};
  if (!config->MtpMatches(trunk)) {
    b.Fail("MTP sidecar architecture or inference constants differ from the "
           "trunk");
    return false;
  }
  MtpWeights m;
  m.config = *config;
  m.config.vocab_size = trunk.vocab_size;
  // The draft block is a full-attention layer at index num_layers.
  m.block = b.Layer(m.config, m.config.num_layers, false, false, true);
  if (!b.ok) {
    return false;
  }
  *mtp = std::move(m);
  return true;
}

}  // namespace gufo::ext::gsq
