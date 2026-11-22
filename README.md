# Qwen3.8 Flash-Next GSQ — Gufo model extension

Out-of-tree [Gufo](https://github.com/gufo-org/gufo) model extension for
**Qwen3.8-Flash-Next GSQ-RCO** checkpoints: pruned 256-routed-expert variants
such as
[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF),
with native IQ-codebook quants (IQ2_S/IQ3_XXS/IQ4_XS routed experts) and the
MTP draft handshake for the 256-expert family.

It also loads **Swift 1.5 GSQ-RCO** (`ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF`,
512-expert `qwen4exp` finetune) after the reader fixes in this branch; see
[SWIFT-1.5.md](SWIFT-1.5.md) for the two metadata encodings that needed support
and the served/bare numbers.

It is a vendored fork of the in-tree `qwen38_flash_next` extension carrying
the changes of [gufo-org/gufo#939](https://github.com/gufo-org/gufo/pull/939),
packaged in the shape the maintainers asked for new model/quant support: a
self-contained extension directory built into Gufo via
`GUFO_EXTENSION_DIRS`, like [gufo-org/lfm3-gufo-extension](https://github.com/gufo-org/lfm3-gufo-extension).

## Relationship to gufo-org/gufo#939

PR #939 has three layers, and only two of them can live in an extension:

| Layer | Where it lives now |
| --- | --- |
| Model support (config, weights, IQ decode kernels, tests, ~2.6k lines) | **this repository** (`src/model/`, `tests/`) |
| IQ block structs / codebooks / scalar accessors in the extension-visible quant header | this repository vendors its own copies; upstream #939 proposes publishing them in core (`include/gufo/ext/changes/0.1/native-quant-formats.md`) |
| Rejected-GGUF extension failover (`src/core/gguf_reader.*`, `src/ext/`, `src/cli/model_open.hpp`) | **cannot live in an extension** — the core parser rejects the file before any provider is consulted. Supersedes #939's loader hunk: strict parse fails -> re-offered to opted-in providers under `TensorLayoutValidation::kFusedExpertSlabs`, with extension-declared storage types. Until it merges, build Gufo from the branch in `scripts/build.sh` (`feat/rejected-gguf-extension-fallback`, drafted at `Farenheith/gufo#2`). |

When the failover lands on Gufo main, re-point `GUFO_REF` in
`scripts/build.sh` to `gufo-org/main` and this extension is pure
out-of-tree (see [TODO.md](TODO.md)).

## Why explicit extension selection is required

Selecting `GUFO_EXTENSIONS=qwen38_flash_next_gsq` puts the build in explicit
mode: the in-tree `qwen38_flash_next` is excluded from the shipped set, so
its probe (which claims every `qwen4exp` file) never competes with ours, its
artifact key no longer collides with ours in discovery validation, and the
vendored qwen-vision translation units do not collide with the ones the
`qwen` extension links. Verified: `gufo diagnose` reports `excluded
qwen38_flash_next ... not selected in explicit mode` with `extensions.lock`
untouched. Do **not** edit `extensions.lock` to exclude the in-tree model:
the Nix release package generates `mkServe` options from the lock, and
removing the entry silently drops its declared serve options (breaks
`mk-serve-nixos-tests`).

Consequence: this extension **replaces** the in-tree Flash-Next in the binary;
it serves both the 256-expert GSQ variants and the 512-expert original.

## Install

gufo 0.3 builds and loads extensions as **modules**: `gufo` itself is not
rebuilt, so the one binary your router already runs serves every model slot,
extension included.

```sh
gufo extension install /path/to/qwen38-flash-next-gsq-gufo-extension --yes
gufo extension list
```

`extension install` compiles against the SDK that belongs to the `gufo` binary
you invoke — set `GUFO_SDK_DIR` when it is not the checkout's
`build/release/sdk`. The module is stamped with the host build identity, so it
loads only into the gufo build it was made for; rebuild after every gufo
upgrade (`gufo extension update`). `./scripts/build.sh` wraps the same call.

Compile-in-tree remains available for carrying an unmerged core hunk:

```sh
cmake --preset release \
  -DGUFO_EXTENSION_DIRS=/path/to/qwen38-flash-next-gsq-gufo-extension \
  -DGUFO_EXTENSIONS=qwen38_flash_next_gsq
cmake --build --preset release --target gufo --parallel
./build/release/gufo diagnose --section extensions
```

Requirements: Linux, AMD Strix Halo / ROCm toolchain (`gfx1151` is the
qualification target), Gufo extension API >= 0.3.37.

## Status on a released gufo (API 0.3.37)

The module loads and serves on an unmodified gufo. Verified through
llama-swap with the `IQ1_M` coder export: the provider claims the file and
generation runs (~157 tok/s prefill, ~36 tok/s decode, MTP drafts accepted).

Two rules of the module ABI mattered here, and both apply to any extension
that forks a built-in model:

- **Do not keep a built-in namespace.** This tree is a fork of gufo's
  `src/models/qwen38_flash_next` and originally kept
  `gufo::models::qwen38_flash_next`. The executable exports `_ZN4gufo*;` and
  the host dlopens modules with `RTLD_NOW|RTLD_LOCAL`, so the module's own
  `TextProvider` pointers bound to the *executable's* copies of
  `ProbeFlashNextGguf` and `create`. The host probe rejects coder artifacts,
  so `ext::ProbeTextProvider` returned null and serve reported
  `Failed to open GGUF: Invalid GGUF tensor shape or unsupported storage
  type: blk.1.ffn_down_exps.weight` — a reader-shaped message, although the
  reader had opened the file (with `allow_unsized_tensors = true` the fused 3-D
  expert slabs are sized from the next tensor offset). `dladdr` on the
  registered probe resolved to the gufo binary and the module's probe never
  ran. `extension.cmake` now compiles the module with `-fvisibility=hidden`:
  hidden definitions are not interposable, so references resolve inside the
  module while host services still bind from the executable. Renaming the
  namespace is the other fix, and is the one gufo can enforce.
- **Declare a scheduling profile.** Since API 0.3
  (`include/gufo/ext/changes/0.3/text-scheduling-profile-required.md`) the
  host refuses a runner whose `TextRunnerDescriptor::scheduling` has a zero
  field. This engine arms no layer-yield hook and captures one checkpoint per
  pass, so it declares the unsplit profile the migration note prescribes.

## Weights and usage

```sh
hf download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF --local-dir models/gsq-coder
build/release/gufo chat --model models/gsq-coder/IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
build/release/gufo serve llm --model <same shard 1> --context 200000
build/release/gufo bench --model <same shard 1> -p 2048 -n 128 -r 3
```

Sharded GGUFs: pass shard 1; the loader follows the shard set. MTP
speculative decoding with the matching 256-expert draft
(`Davidmg0815/Qwen3.8-Flash-Next-MTP-256exp-GGUF`) via
`--speculative mtp --mtp-model <draft>`; the base model's own 512-expert MTP
sidecar is rejected by design (expert-count mismatch, see
`MtpMatches`).

## Layout

```
extension.json          # manifest: provider flash-next-gsq-coder, --mtp-model option
extension.cmake         # gufo_ext_runtime-only targets (+ vendored mmq kernels)
src/model/              # flash-next model fork (from #939), includes rewritten to <gufo/ext/...>
src/family/             # qwen-family pieces needed by the model (headers; vision TUs compiled in)
tests/                  # projection + routed-WMMA op tests ported from #939
scripts/build.sh      # pinned-branch configure+build entry point
scripts/build.sh        # pinned build entry point
```

## Provenance and license

Derived from [gufo-org/gufo](https://github.com/gufo-org/gufo)
`src/models/qwen38_flash_next` + `src/families/qwen` at
`Farenheith/gufo@feat/qwen4exp-gsq-coder-support` (PR #939 head). Gufo is MIT
licensed; see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). Model weights are
distributed separately under their own licenses.
