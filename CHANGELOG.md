# Changelog

Dates are when a version was tagged here, not when it was written.

## 0.2.0 — 2026-11-22

First version submitted to [gufo-org/gufo-extension](https://github.com/gufo-org/gufo-extension).

- `fix(ple)`: accept and decode a **`Q5_1`** `per_layer_token_embd.weight`. AtomicChat's
  AD builds ship their n-gram table in `Q5_1`, which the table binder rejected
  ([`0e5ef6d`](https://github.com/codibre/qwen38-flash-next-flavors-gufo-extension/commit/0e5ef6d)).
- `fix(reader)`: accept `bool` metadata arrays (no `kBool` arm in `ReadArray`, which
  surfaced as `malformed GGUF header`) and read `split.count` at any width — Swift 1.5
  stores it as `u16` where the Coder files use `u64` (contributed by
  [@alytaphoenix](https://github.com/alytaphoenix) in PR #1).
- `docs`: README restructured around the artifacts and what was measured. New
  [docs/FLAVORS.md](docs/FLAVORS.md) with the per-tier header pre-flight and
  [docs/MODULE_ABI.md](docs/MODULE_ABI.md) with the module-ABI rules. The stale claim
  that loading requires a patched Gufo core is gone, along with two hardcoded local
  paths in `tools/tests/`.
- `chore`: version raised from 0.1.0 with the Q5_1 support, since it widens what the
  extension loads.

## 0.1.0 — never tagged

The initial extension, developed untagged across the work below. Folded into 0.2.0.

- Port of [gufo-org/gufo#939](https://github.com/gufo-org/gufo/pull/939) model support
  (config, weights, IQ decode kernels, projection and routed-WMMA tests) out of core
  and into an extension.
- Owns the artifact read end to end: `probe_gguf` accepts unsized GGUF tensors and the
  fused 3-D expert slabs, which the in-tree Flash-Next provider rejects.
- Loads as a **module** against the running `gufo` binary instead of a source build, and
  hides its symbols so they cannot be interposed by the executable
  ([docs/MODULE_ABI.md](docs/MODULE_ABI.md)).
- Declares the unsplit `TextSchedulingProfile` required since extension API 0.3.
- Baseline and kernel harnesses: slot baseline over llama-swap, DRAM/LLC/LDS peak
  probe, routed decode bench, idle-GPU kernel A/B with a differential bitwise gate and
  a negative control.
