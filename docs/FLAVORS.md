# Qwen3.8-Flash-Next GGUF flavors

This repo carries the Gufo extension that runs **Qwen3.8-Flash-Next** artifacts
that the in-tree `qwen38_flash_next` provider cannot open. The artifacts come
from several publishers and they differ in exactly the places this engine is
strict, so each flavor needs its own pre-flight before anyone spends a download
or a load window.

Every number below was read off the container with
`tools/tests/gguf_meta.py` (bounded header/tensor-info reader, never touches
tensor payload) or measured through a llama-swap slot. Nothing here is copied
from a model card.

## What the engine actually requires

`src/model/config.cpp` validates kernel geometry by equality, so a flavor must
declare all of it:

| key | required |
|---|---|
| `num_layers` | 48 |
| `hidden_size` | 2560 |
| `hc_count` / `hc_low_rank` | 4 / 320 |
| `full_attention_interval` | 4 |
| `num_heads` / `num_kv_heads` / `head_dim` / `rotary_dim` | 24 / 2 / 256 / 64 |
| `rope_sections` | {11, 11, 10, 0} |
| `indexer_heads` / `indexer_head_dim` / `indexer_top_k` | 4 / 128 / 2048 |
| `compress_ratio` | 4 |
| `ssm_conv_kernel` / `ssm_head_dim` / `ssm_num_k_heads` / `ssm_num_v_heads` | 4 / 128 / 16 / 48 |
| `expert_count` | **512 or 256** |
| `expert_used_count` | 10 |
| `expert_ff` / `shared_expert_ff` | 640 / 640 |
| `ple.ngram_size` / `heads_per_ngram` / `head_dim` / `conv_kernel` | 3 / 8 / 160 / 4 |

Two storage constraints bite regardless of config:

- **The n-gram table.** `per_layer_token_embd.weight` is `[160, 320001536]`
  (51,200,245,760 elements, 41% of the artifact) and lives in its own trailing
  shard with its own GGUF header. `NgramTable::Open` accepts
  **IQ4_NL (18 B/32 el → 90 B/row), Q8_0 (34 → 170 B/row), Q5_1 (24 → 120
  B/row)** and BF16. Row bytes × rows must equal the shard, so a flavor that
  picks another type for this tensor is rejected before anything runs.
- **`moe_intermediate_size = 640`.** k-quants and i-quants with a 256-element
  super-block need the row divisible by 256, so `ffn_down_exps` rows of 640
  cannot be Q4_K/Q5_K/IQ4_XS-sized. Publishers fall back to IQ4_NL/Q8_0 there;
  expect the *named* quant of a flavor to describe its gate/up experts, not its
  down experts.

## Pre-flight (free, ~1 s per shard)

```bash
python3 tools/tests/gguf_meta.py <shard-1> --top 1 \
    --show split,ngram,ple,moe,expert --tensor per_layer
python3 tools/tests/gguf_meta.py <last-shard> --top 5 --tensor per_layer
```

Read three things: the `qwen4exp.*` diff against a flavor that already loads,
any `!!` REJECT on tensor types (a type the binder will not take), and whether
`split.count` is present (then Gufo joins the shards by itself —
`OpenSplitFileSet` — and the slot points at shard 1 only).

## Flavors

| flavor | experts | n-gram table | first shard | status here |
|---|---|---|---|---|
| ISTA-DASLab `GSQ-RCO-Coder-GGUF/IQ1_M` | 256 | IQ4_NL, 28.80 GB | 29.61 GB | **loads and serves** (slot `heretic`, via Abiray's abliterated re-host of the same recipe) |
| ISTA-DASLab `GSQ-RCO-GGUF/{Q2_0,IQ2_XS,IQ3_XXS,IQ3_S}` | 256 | IQ4_NL, 28.80 GB | 37.6 / 39.2 / 47.0 / 54.8 GB | not pre-flighted yet; table identical across all four, so only the dense/expert mix varies |
| AtomicChat `AD-4.27bpw-Q4_K_M-M64` | **512** | **Q5_1**, 38.40 GB | 44.87 + 11.26 GB | **loads and serves** after the Q5_1 table support (`0e5ef6d`); 3 shards, joined by core |
| ISTA-DASLab `P48NVFP4-MoESQ` | ? | ? | — | HF tree returned no `.gguf`; unresolved, needs a look before any claim |

The ISTA flavors and AtomicChat agree on **36 of 37 `qwen4exp` keys**; the only
difference is `expert_count` (256 vs 512), which the gate allows. Per-token
routed traffic is nearly the same for both (~786 MB and ~783 MB), because
AtomicChat trades fewer/larger experts for a wider top-k.

### AtomicChat specifics

- Table geometry proved on the file: `160/32 × 24 = 120 B/row × 320,001,536 =
  38,400,184,320 B` = the third shard exactly.
- Routed mix measured: `ffn_down_exps` IQ4_NL ×39 + Q2_0 ×9, gate/up IQ2_S ×20
  / IQ3_XXS ×17 / IQ3_S ×10 / IQ4_XS ×1 — every type inside the accepted set.
- It is a **reasoning** build: `reasoning_content` consumes the token budget,
  so a request with `max_tokens: 32` returns `finish_reason: length` with empty
  `content`. Budget thinking first (≥ ~512) or disable thinking.
- Measured through llama-swap (slot `atomic`, `--context 256000`, shared-Q8 MTP
  draft, batch 1): load 10.8 s, decode **53.5 tok/s** (18.7 ms/token), MTP
  draft acceptance 25/28. Process RSS stays in the low single-digit GB — the
  weights are file-backed pages and the table is read from disk, so the 56 GB
  figure is addressable bytes, not pinned RAM.
- Draft pairing: `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` is `expert_count
  512` at `blk.48`, i.e. built for this shape. The 256-expert GSQ drafter is
  not interchangeable.

## Naming note

The extension id (`qwen38_flash_next_gsq`), provider id
(`flash-next-gsq-coder`) and manifest key (`gguf:qwen4exp-gsq`) predate this
multi-flavor scope. They are load-bearing for the installed module and for the
llama-swap slots, so renaming them is a deliberate, separate change — the
provider claims on `general.architecture`, not on publisher.
