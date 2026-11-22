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

| flavor | experts | n-gram table | weights | status here |
|---|---|---|---|---|
| ISTA-DASLab `GSQ-RCO-Coder-GGUF/IQ1_M` | 256 | IQ4_NL, 28.80 GB | 29.61 GB | **loads and serves** (slot `heretic`, via Abiray's abliterated re-host of the same recipe) |
| ISTA-DASLab `GSQ-RCO-GGUF/IQ3_S` | 512 | IQ4_NL, 28.80 GB | 54.81 GB | **no binder gaps** — every stored type has geometry. Routed mix is identical to `heretic`'s (`down_exps` IQ4_NL×39 + Q2_0×9; gate/up IQ2_S×20 / IQ3_XXS×17 / IQ3_S×10 / IQ4_XS×1) at double the expert count |
| ISTA-DASLab `GSQ-RCO-GGUF/IQ3_XXS` | 512 | IQ4_NL, 28.80 GB | 47.03 GB | **will not load**: `IQ2_XS` ×20 (4.85 GB) and `IQ2_XXS` ×18 (3.89 GB) have no `FormatOf` geometry |
| ISTA-DASLab `GSQ-RCO-GGUF/IQ2_XS` | 512 | IQ4_NL, 28.80 GB | 39.22 GB | **will not load**: `IQ2_XXS` ×22 (4.76 GB) and `IQ1_M` ×6 (1.10 GB) |
| ISTA-DASLab `GSQ-RCO-GGUF/Q2_0` | 512 | IQ4_NL, 28.80 GB | 37.57 GB | **will not load**: legacy k/quants on the *dense* side — `Q3_K` ×91 (0.73 GB, including `token_embd.weight [2560, 248320]`), `Q4_0` ×16, `Q5_0` ×7 |
| AtomicChat `AD-4.27bpw-Q4_K_M-M64` | **512** | **Q5_1**, 38.40 GB | 44.87 + 11.26 GB | **loads and serves** after the Q5_1 table support (`0e5ef6d`); 3 shards, joined by core |
| ukisai `Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_S` | 512 | IQ4_NL, 28.80 GB | 44.89 + 38.84 GB | **loads and serves** (contributor, [SWIFT-1.5.md](SWIFT-1.5.md)); needed the two reader encodings fixed in PR #1 — `bool` metadata arrays, and a `u16` `split.count` where the Coder files use `u64` |
| ISTA-DASLab `P48NVFP4-MoESQ` | ? | ? | — | HF tree returned no `.gguf`; unresolved, needs a look before any claim |

So the table format is *not* what separates the flavors — every ISTA tier and
Swift 1.5 ship the same 28.80 GB IQ4_NL block, and only AtomicChat needed Q5_1.
Shard layout varies as well, and it is worth reading instead of assuming: ISTA
and AtomicChat put the table in its own trailing shard with its own header,
while Swift 1.5 declares it among shard 1's 373 tensor infos and fills shard 2
with the remaining 851 tensors. What separates
them is the storage type of ordinary weights. Missing `FormatOf` geometries,
union of the three blocked flavors:

```
IQ2_XXS   IQ2_XS   IQ1_M   Q3_K   Q4_0   Q5_0
```

All six exist in the ds41 extension's `mmq` dispatch, which is where a port
would start. `IQ3_S` is the one flavor that needs nothing: it is `heretic`'s
routing recipe at 512 experts.

AtomicChat and `heretic` agree on **36 of 37 `qwen4exp` keys**; the only
difference is `expert_count` (512 vs 256), which the gate allows — and the four
ISTA base flavors declare 512 like AtomicChat. Per-token routed traffic is
nearly the same for both (~786 MB and ~783 MB), because AtomicChat trades
fewer/larger experts for a wider top-k.

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

The extension id (`qwen38_flash_next_flavors`), provider id
(`flash-next-flavors`) and manifest key (`gguf:qwen4exp-gsq`) predate this
multi-flavor scope. They are load-bearing for the installed module and for the
llama-swap slots, so renaming them is a deliberate, separate change — the
provider claims on `general.architecture`, not on publisher.
