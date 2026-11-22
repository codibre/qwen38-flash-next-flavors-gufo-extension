# Qwen3.8 Flash-Next for Gufo

Run **Qwen3.8-Flash-Next** `qwen4exp` checkpoints on one AMD Strix Halo: a 48-layer
mixture-of-experts model that routes 10 of 256 or 512 experts per token and looks up
a 160-row-per-layer n-gram table on every step.

This is a [Gufo](https://github.com/gufo-org/gufo) model extension. Gufo 0.3 builds it
as a **module**: `gufo` itself is not recompiled, so the one binary your router already
runs serves every model slot, extension included.

- **Takes containers the engine does not claim.** These builds store routed experts as
  3-D slabs (`ffn_down_exps.weight`, `[640, 2560, experts]`), which the in-tree
  Flash-Next provider rejects. This extension claims them by that shape.
- **The n-gram table costs no memory.** 28.8–38.4 GiB per artifact, read with `pread`
  instead of being mapped resident: RSS after load is 2.3 GiB.
- **Three publishers, one code path.** The pruned 256-expert GSQ-RCO coder builds, the
  512-expert AtomicChat and Swift 1.5 re-quantizations, and their MTP drafts.
- **Speculative decoding with both engines.** MTP drafts plus Gufo's prompt lookup;
  both are counted in the numbers below.

| Artifact | Load | Prefill | Decode | Draft accepted |
| --- | ---: | ---: | ---: | ---: |
| Coder `IQ1_M`, 256 experts | 6.5 s | 655–683 tok/s (3.4K prompt) | 27.6 tok/s | 77% MTP |
| AtomicChat `AD-4.27`, 512 experts | 10.8–11.1 s | 1,013–1,036 tok/s (32K prompt) | 39.8–56.4 tok/s | 71–75% MTP |
| Swift 1.5 `IQ3_S`, 512 experts | 13.2 s | ~1,160 tok/s (8K cold) | 29.1–29.5 tok/s | 84% MTP |

One Strix Halo (`gfx1151`), 128 GiB unified, greedy, MTP on, one slot resident at a
time. Decode tracks how warm the n-gram table's pages are, so read these as a range and
not a rating. The AtomicChat row comes from one 32,676-token prompt holding a retrieval
needle: 32 s of prefill, then 39.8 tok/s on a long answer, 42.2 tok/s on a short one with
thinking on and 56.4 tok/s with it off, and the needle came back correct in all four ask
formats. Swift's row was measured on the contributor's box at
`--sessions 4 --context 262144`, with retrieval, tool-call and reasoning probes, and is
reproduced in [SWIFT-1.5.md](SWIFT-1.5.md).

Maintainer: Thiago Santos ([@codibre](https://github.com/codibre)).
[Report bugs here](https://github.com/codibre/qwen38-flash-next-flavors-gufo-extension/issues),
with the GGUF file names when an artifact does not load.

## Artifacts

| Build | Experts | Resident weights | N-gram table | MTP draft | Tested |
| --- | ---: | ---: | --- | --- | :---: |
| [ISTA GSQ-RCO Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) `IQ1_M` | 256 | 29.6 GiB | `IQ4_NL`, 28.8 GiB | 256-expert | Yes |
| [AtomicChat AD-4.27bpw-Q4_K_M-M64](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF) | 512 | 56.1 GiB (3 shards) | `Q5_1`, 38.4 GiB | shared `Q8_0` | Yes |
| [Swift 1.5 GSQ-RCO](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) `IQ3_S` | 512 | 54.9 GiB (2 shards) | `IQ4_NL`, 28.8 GiB | shared `Q8_0` | Yes |

The Coder row is served here from [Abiray's abliterated re-host](https://huggingface.co/Abiray/Qwen3.8-Flash-Next-Orca-Abliterated-GSQ-RCO-Coder-GGUF)
of the same recipe, with the ablation baked into the weights. Resident weights exclude
the n-gram table, which stays on disk; Swift's is declared inside shard 1 rather than
in a trailing shard of its own.

What the three were checked with, beyond loading and throughput:

| Capability | Coder `IQ1_M` | AtomicChat `AD-4.27` | Swift 1.5 `IQ3_S` |
| --- | :---: | :---: | :---: |
| Chat completions, greedy | Yes | Yes | Yes |
| Reasoning output (`reasoning_content`) | Yes | Yes | 5/5 prompts |
| Retrieval at 16K / 32K | not tested | 4/4 at 32K | 3/3, 3/3 |
| Tool calls | not tested | not tested | 8/8 |
| Concurrent sessions | 1 | 1 | 4, serialised |
| Image input (`--mmproj`) | not tested | not tested | not tested |

### Not supported yet

Four ISTA tiers and one quantization line do not load, and this repository says so
rather than listing them as supported. [docs/FLAVORS.md](docs/FLAVORS.md) has the
header-by-header pre-flight for each; in short:

- `IQ3_S` of the base GSQ-RCO line: **every stored type has geometry and the binder
  takes it**, but it has not been served here, so it is not claimed.
- `IQ3_XXS`, `IQ2_XS` and the `Q2_0` tier: stored types with no `FormatOf` geometry in
  this extension — `IQ2_XXS`, `IQ2_XS`, `IQ1_M`, `Q3_K`, `Q4_0`, `Q5_0`. Porting them
  is the next open work and has not been started.
- `P48NVFP4-MoESQ`: its Hugging Face tree publishes no `.gguf`, so nothing was read.

## Quick start

Install as a module against the `gufo` binary you already run:

```sh
git clone https://github.com/codibre/qwen38-flash-next-flavors-gufo-extension
cd qwen38-flash-next-flavors-gufo-extension
GUFO_SDK_DIR=/path/to/gufo/build/release/sdk \
  /path/to/gufo extension install "$PWD" --yes
gufo extension list
```

`extension install` compiles against the SDK of the binary you invoke and stamps the
module with that build's identity, so it loads only into the Gufo build it was made
for. Rerun `gufo extension update` after every Gufo upgrade. `scripts/build.sh` wraps
the same call. Requirements: Linux, a ROCm toolchain, Gufo extension API >= 0.3.37;
`gfx1151` is the qualification target.

Serve the 256-expert coder — point at shard 1 and the loader follows the set:

```sh
hf download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF --include "IQ1_M/*" \
  --local-dir models/gsq-coder
hf download Davidmg0815/Qwen3.8-Flash-Next-MTP-256exp-GGUF \
  mtp-Qwen3.8-Flash-Next-256exp-Q8_0.gguf --local-dir models/drafts   # 2.7 GiB

gufo serve llm \
  --model models/gsq-coder/IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf \
  --context 256000 --sessions 1 \
  --speculative mtp --mtp-model models/drafts/mtp-Qwen3.8-Flash-Next-256exp-Q8_0.gguf
```

The 512-expert builds take the same shape with the shared draft. AtomicChat is three
shards, the last of which is the whole n-gram table:

```sh
hf download AtomicChat/Qwen3.8-Flash-Next-GGUF \
  --include "Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-0000*" --local-dir models/atomic
hf download unsloth/Qwen3.8-Flash-Next-GGUF MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --local-dir models/drafts                                                     # 2.6 GiB

gufo serve llm \
  --model models/atomic/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00003.gguf \
  --context 256000 --sessions 1 \
  --speculative mtp --mtp-model models/drafts/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
```

| Option | Effect |
| --- | --- |
| `--model` | shard 1 of the set; the loader follows `split.count` and opens the rest |
| `--mtp-model` | draft for `--speculative mtp`. Its `expert_count` must match the base |
| `--context` | per session. Run at 256000 here; Swift was qualified at 262144 |
| `--sessions` | headroom, not throughput: extra sessions serialise on this engine |
| `--cache-ram-bytes` | Gufo's prompt-cache budget; not characterised for these slots |
| `--mmproj` | accepted by the server; image input has not been tested on this provider |

## Memory

Resident per serve, before session state:

| Component | Coder `IQ1_M` | AtomicChat `AD-4.27` |
| --- | ---: | ---: |
| Backbone weights | 29.6 GiB | 56.1 GiB |
| MTP draft | 2.7 GiB | 2.6 GiB |
| N-gram table | 0 (28.8 GiB on disk) | 0 (38.4 GiB on disk) |
| **Before KV** | **32.3 GiB** | **58.7 GiB** |

The only measured device footprints are the contributor's Swift runs: 90.7 GiB device
at `--sessions 4 --context 262144`, against 39.6 GiB for the in-tree `UD-Q4_K_XL`
checkpoint, with RSS of 2.3 GiB after load. KV at 256K context on our own slots has not
been captured, so it is absent from the table rather than estimated.

- Gufo's own accounting at load puts the AtomicChat slot at 71.7 GiB of the 120 GiB
  device total, `--context 256000 --sessions 1`.
- One Qwen3.8 slot at a time. Two of these builds do not coexist on a 128 Gib box;
  AtomicChat plus a resident neighbour is the usual way to find that out.
- The n-gram table is addressable bytes on disk, not memory. Its rows are read per
  token, so a slow or busy drive shows up as decode jitter, not as an allocation.

## The n-gram table

Flash-Next keeps a `per_layer_token_embd.weight` table of `[160, 320001536]` elements —
41% of the artifact. `NgramTable::Open` maps it and reads rows with `pread`, so it
costs no model memory.

Its type is what a flavor has to get right. Row bytes must satisfy
`rows × row_bytes == table payload`, and only three types have a decoder:

| Type | Bytes/row | Payload | Shipped by |
| --- | ---: | ---: | --- |
| `IQ4_NL` | 90 | 28.80 GB | every ISTA GSQ-RCO tier, and Swift 1.5 |
| `Q5_1` | 120 | 38.40 GB | AtomicChat AD builds |
| `Q8_0` | 170 | 54.40 GB | not published by anyone here |

`Q5_1` support is recent
([`0e5ef6d`](https://github.com/codibre/qwen38-flash-next-flavors-gufo-extension/commit/0e5ef6d));
before it, AtomicChat's table was rejected at bind time.

## Drafts pair by expert count

A draft whose `expert_count` differs from the base is rejected by design
(`MtpMatches`). This is the single most common reason a serve here fails to start:

| Base | Draft |
| --- | --- |
| 256-expert GSQ-RCO Coder | [`Davidmg0815/Qwen3.8-Flash-Next-MTP-256exp-GGUF`](https://huggingface.co/Davidmg0815/Qwen3.8-Flash-Next-MTP-256exp-GGUF) `mtp-...-256exp-Q8_0.gguf`, 2.7 GiB |
| 512-expert builds (AtomicChat, Swift, ISTA base tiers) | [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, 2.6 GiB |

## Troubleshooting

| Message | What it actually means |
| --- | --- |
| `Failed to open GGUF: Invalid GGUF tensor shape or unsupported storage type: blk.1.ffn_down_exps.weight` | The in-tree provider claimed the file, so this extension was not in play. Check `gufo extension list`, then `gufo extension update` — the module is stamped with the Gufo build it was compiled against. |
| Serve exits on the draft, or starts with no speculation | `expert_count` differs between base and draft. See [Drafts pair by expert count](#drafts-pair-by-expert-count). |

| An artifact is rejected for a stored tensor type | The flavor uses a quantization this extension has no geometry for. [docs/FLAVORS.md](docs/FLAVORS.md) names which. |
| `malformed GGUF header`, or far fewer tensors than published | A container encoding the reader did not accept. Two were fixed in PR #1 (`bool` metadata arrays, a `u16` `split.count`); [SWIFT-1.5.md](SWIFT-1.5.md) has the write-ups. |

## Known limitations

- **Three ISTA tiers will not load**, for want of `FormatOf` geometry on
  `IQ2_XXS`, `IQ2_XS`, `IQ1_M`, `Q3_K`, `Q4_0`, `Q5_0`. Nothing has been ported yet.
- `moe_intermediate_size = 640` blocks every quantization whose super-block is 256
  elements from the routed down-projections, so a flavor's *name* describes its gate/up
  experts and never its down experts.
- Concurrency serialises: sessions buy headroom, not aggregate throughput (measured
  4×6.2 tok/s for four concurrent requests against 29 tok/s for one).
- Image input and tool calls on the Coder and AtomicChat builds are untested, as listed
  above. Prompt-cache reuse is measured on AtomicChat: the same prompt twice prefills
  zero tokens the second time, and changing only the tail of a 32K prompt re-prefills
  1,964 of its 32,676 tokens.
- The Q2_0 decode kernel has an optimisation candidate that is bit-identical to
  production and slower, parked on `perf/q20-perm-dequant`. It is not in any release.

## Documentation

- [docs/FLAVORS.md](docs/FLAVORS.md): what each publisher's container requires of this
  engine, tier by tier, read from headers over HTTP without downloading weights.
- [docs/MODULE_ABI.md](docs/MODULE_ABI.md): how the compiled module binds to the Gufo
  build it was installed against, and the two rules that keep that binding intact.
- [SWIFT-1.5.md](SWIFT-1.5.md): Swift 1.5 support — the reader encodings fixed, the
  contributor's served numbers.
- [TODO.md](TODO.md): honest verification status, including what is not green.
- [CHANGELOG.md](CHANGELOG.md): changes by version.
- `tools/bench/`: slot baseline, peak probe, routed decode bench, kernel A/B.
- `tools/tests/`: differential kernel gate, negative control, bounded GGUF reader probe.

## License and provenance

MIT ([LICENSE](LICENSE)). The model support derives from `src/models/qwen38_flash_next`
and `src/families/qwen` in [gufo-org/gufo](https://github.com/gufo-org/gufo) — config,
weights, IQ decode kernels and their tests — carried here as an extension.
`src/model/kernels/rocm/mmq/` derives from ggml/llama.cpp kernels. See
[NOTICE.md](NOTICE.md). Unsized and fused-expert-slab tensors are handled in this
extension's own reader, so **no Gufo fork is required**: these artifacts load as a
module on a released Gufo 0.3.37.

Model weights are **not** bundled and are not redistributable by us. ISTA's GSQ-RCO
tiers are apache-2.0, AtomicChat's and Unsloth's builds are qwen-community-1.0, Swift
1.5 is under swift-open-license-1.0. Check the card of whatever you download.
