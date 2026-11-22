# Qwen3.8 Flash-Next for Gufo

Run **Qwen3.8-Flash-Next** `qwen4exp` checkpoints on one AMD Strix Halo: the
pruned 256-expert [GSQ-RCO](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)
coder builds, 512-expert re-quantizations of the full model, and the ~30-40 GiB
per-layer n-gram table they look up on every token — read from disk, never
resident.

This is a [Gufo](https://github.com/gufo-org/gufo) model extension. gufo 0.3
builds it as a **module**: `gufo` itself is not recompiled, so the one binary
your router already runs serves every model slot, extension included.

| Artifact | Experts | Weights | Weight license | Status here |
| --- | --- | ---: | --- | --- |
| [ISTA GSQ-RCO Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) `IQ1_M` | 256 | 29.6 GiB + 28.8 GiB table | apache-2.0 | **serves** |
| [ISTA GSQ-RCO](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) `IQ3_S` | 512 | 54.8 GiB + 28.8 GiB table | apache-2.0 | header pre-flight clean, not served here yet |
| ISTA GSQ-RCO `IQ3_XXS` / `IQ2_XS` / `Q2_0` | 512 | 47.0 / 39.2 / 37.6 GiB | apache-2.0 | **does not load** — missing quant geometries, listed in [docs/FLAVORS.md](docs/FLAVORS.md) |
| [Swift 1.5 GSQ-RCO](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) `IQ3_S` | 512 | 83.7 GiB | swift-open-license-1.0 | **serves**, see [SWIFT-1.5.md](SWIFT-1.5.md) |
| [AtomicChat AD-4.27bpw-Q4_K_M-M64](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF) | 512 | 56.1 GiB + 38.4 GiB table | qwen-community-1.0 | **serves** |
| ISTA `P48NVFP4-MoESQ` | ? | — | — | unresolved; no `.gguf` published in its tree |

## Measured

One Strix Halo (`gfx1151`), ROCm 7.2.4, 128 GiB unified, one session, MTP
speculative decoding on, greedy. `Coder IQ1_M` numbers are from
`tools/bench/slot_baseline.py` on a 3.4K-token prompt; AtomicChat's from
`/v1/chat/completions`.

| Artifact | Load | Prefill | Decode | MTP accepted |
| --- | ---: | ---: | ---: | --- |
| `Coder IQ1_M`, 256 experts | 6.5 s | 655-683 tok/s | 27.6 tok/s | 74/96 (77%) |
| AtomicChat `AD-4.27`, 512 experts | 10.8 s | — | 53.5 tok/s short runs, 36.6 tok/s over an 11.5K-token answer | 25/28 |
| Swift 1.5 `IQ3_S`, 512 experts | 13.2 s | ~1160 tok/s | 29.1-29.5 tok/s | 84% |

Decode moves with how warm the n-gram table's pages are, so read these as a
range and not a rating: the same AtomicChat slot measured 53.5 tok/s on short
answers and 36.6 tok/s on an 11.5K-token one. Swift's row was measured on the
contributor's box at `--sessions 4 --context 262144` and is reproduced in
[SWIFT-1.5.md](SWIFT-1.5.md) with retrieval, tool-call and reasoning probes.

## Quick start

```sh
gufo extension install /path/to/qwen38-flash-next-flavors-gufo-extension --yes
gufo extension list
```

`extension install` compiles against the SDK belonging to the `gufo` binary you
invoke; set `GUFO_SDK_DIR` when it is not that checkout's `build/release/sdk`.
The module is stamped with the host build identity, so it loads only into the
gufo build it was made for — rerun `gufo extension update` after every gufo
upgrade. `./scripts/build.sh` wraps the same call.

Requirements: Linux, ROCm toolchain (`gfx1151` is the qualification target),
Gufo extension API >= 0.3.37.

Then serve shard 1 — the loader follows the shard set:

```sh
hf download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF --local-dir models/gsq-coder
gufo serve llm --model models/gsq-coder/IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf \
  --context 200000 --speculative mtp --mtp-model <256-expert MTP draft>
```

## The n-gram table

Flash-Next keeps a `per_layer_token_embd.weight` table of `[160, 320001536]`
elements — 41% of the artifact. `NgramTable::Open` maps it and reads rows with
`pread`, so it costs no model memory: RSS after load stays in the low single
digits of GiB, and the ~30-40 GiB figure is addressable bytes on disk.

Its type is what a flavor has to get right. Row bytes must satisfy
`rows × row_bytes == table payload`, and only these have a decoder:

| Type | Bytes/row | Payload | Shipped by |
| --- | ---: | ---: | --- |
| `IQ4_NL` | 90 | 28.80 GB | every ISTA GSQ-RCO tier |
| `Q5_1` | 120 | 38.40 GB | AtomicChat AD builds |
| `Q8_0` | 170 | 54.40 GB | — |

## Speculative drafts pair by expert count

A draft whose `expert_count` differs from the base is rejected by design
(`MtpMatches`), which is the single most common reason a serve fails to start
here:

| Base | Draft |
| --- | --- |
| 256-expert GSQ-RCO Coder | [`Davidmg0815/Qwen3.8-Flash-Next-MTP-256exp-GGUF`](https://huggingface.co/Davidmg0815/Qwen3.8-Flash-Next-MTP-256exp-GGUF) |
| 512-expert builds (AtomicChat, Swift, ISTA base tiers) | [`unsloth` `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) |

## Reasoning models and the output budget

AtomicChat's AD builds are reasoning checkpoints and their thinking counts
inside `max_tokens`: a 32- or 220-token budget returns
`finish_reason: length` with `content` empty, which looks like a failure and is
not one. Give it room (a trivial question costs ~70 tokens, a technical
explanation ~11K), or ask for no thinking:

```json
{"model": "...", "messages": [...], "max_tokens": 100000,
 "chat_template_kwargs": {"enable_thinking": false}}
```

## Known limitations

- **Three ISTA tiers will not load.** `IQ3_XXS` stores `IQ2_XS`/`IQ2_XXS`,
  `IQ2_XS` stores `IQ2_XXS`/`IQ1_M`, and the `Q2_0` tier puts `Q3_K`/`Q4_0`/`Q5_0`
  on the dense side including `token_embd.weight`. None has `FormatOf` geometry
  in this extension yet. The full pre-flight is [docs/FLAVORS.md](docs/FLAVORS.md).
- `moe_intermediate_size = 640` blocks every quant whose super-block is 256
  elements from the routed down-projections, so a flavor's *name* describes its
  gate/up experts and never its down experts.
- Concurrency serialises: sessions buy headroom, not aggregate throughput.
- `IQ3_S` and the `P48NVFP4-MoESQ` line have not been served here.

## Two module-ABI rules that cost us a day

Both apply to any extension that forks a built-in model, and neither is
documented in the SDK:

- **Do not keep a built-in namespace.** This tree is a fork of gufo's
  `src/models/qwen38_flash_next` and originally kept
  `gufo::models::qwen38_flash_next`. The executable exports `_ZN4gufo*;` and
  dlopens modules with `RTLD_NOW|RTLD_LOCAL`, so the module's own `TextProvider`
  pointers bound to the *executable's* copies of `ProbeFlashNextGguf` and
  `create`. The host probe rejects coder artifacts, `ext::ProbeTextProvider`
  returned null, and serve reported `Failed to open GGUF: Invalid GGUF tensor
  shape or unsupported storage type: blk.1.ffn_down_exps.weight` — a
  reader-shaped message for a reader that had opened the file correctly.
  `dladdr` on the registered probe resolved to the gufo binary. `extension.cmake`
  now compiles with `-fvisibility=hidden`: hidden definitions are not
  interposable, host services still bind from the executable. Renaming the
  namespace is the other fix, and the one gufo can enforce.
- **Declare a scheduling profile.** Since API 0.3
  (`include/gufo/ext/changes/0.3/text-scheduling-profile-required.md`) the host
  refuses a runner whose `TextRunnerDescriptor::scheduling` has a zero field.
  This engine arms no layer-yield hook and captures one checkpoint per pass, so
  it declares the unsplit profile the migration note prescribes.

## Relationship to gufo-org/gufo#939

| Layer | Where it lives |
| --- | --- |
| Model support (config, weights, IQ decode kernels, tests, ~2.6k lines) | this repository (`src/model/`, `tests/`) |
| IQ block structs / codebooks / scalar accessors | vendored here; #939 proposes publishing them in core |
| Unsized and fused-expert-slab tensors | handled here, in `probe_gguf` and the extension's own reader. **No gufo fork is required**: the artifacts above load as a module on a released gufo 0.3.37. |

## Layout

```
extension.json          manifest: provider flash-next-gsq-coder, --mtp-model option
extension.cmake         gufo_ext_runtime-only targets, + vendored mmq kernels
src/model/              flash-next model fork (from #939), includes via <gufo/ext/...>
src/family/             qwen-family pieces the model needs
tests/                  projection + routed-WMMA op tests ported from #939
tools/bench/            slot baseline, peak probe, routed decode bench, kernel A/B
tools/tests/            differential kernel gate, negative control, bounded GGUF reader probe
tools/prof/             ISA mix
docs/FLAVORS.md         what each publisher's container requires of this engine
SWIFT-1.5.md            Swift 1.5 support: reader encodings fixed, served numbers
TODO.md                 honest verification status
scripts/build.sh        module install entry point
```

## Provenance and license

MIT ([LICENSE](LICENSE)), derived from [gufo-org/gufo](https://github.com/gufo-org/gufo)
`src/models/qwen38_flash_next` + `src/families/qwen` with the changes of PR #939;
`src/model/kernels/rocm/mmq/` derives from ggml/llama.cpp kernels. See
[NOTICE.md](NOTICE.md).

Model weights are **not** bundled and are not redistributable by us: ISTA's
GSQ-RCO tiers are apache-2.0, AtomicChat's and Unsloth's builds are
qwen-community-1.0, Swift 1.5 is under the swift-open-license-1.0. Check the
card of whatever you download.
