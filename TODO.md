# TODO

## Verification status — 2026-11-22, released gufo 0.3.37, module install

No core patch, no gufo fork, no `GUFO_REF`. Everything below was produced by
`gufo extension install <dir> --yes` against an unmodified binary.

- [x] Include boundary: no `src/...` includes outside the repository except
      forwarded public headers (`<gufo/ext/...>`).
- [x] `extension.cmake` links only public surface: `gufo_ext_runtime`,
      `hip::host`, `roc::hipblas`, `roc::hipblaslt`, `Threads::Threads`. No
      `gufo_core`, no download step, no vendored binaries (`git ls-files`
      finds no `.o`/`.so`).
- [x] `gufo diagnose --section extensions` lists the extension with
      `provider flash-next-gsq-coder: available`.
- [x] Module symbol binding: `nm -D` on the installed module exports **zero**
      `gufo::models` symbols, and imports `gufo::quant::DequantizeQ5_1` from
      the host.
- [x] Serves **ISTA GSQ-RCO Coder `IQ1_M`** (256 experts, 2 shards, IQ4_NL
      table): load 6.5 s, prefill 655-683 tok/s, decode 27.6 tok/s, MTP 74/96.
- [x] Serves **AtomicChat `AD-4.27bpw-Q4_K_M-M64`** (512 experts, 3 shards,
      Q5_1 table): load 10.8 s, decode 53.5 tok/s short runs / 36.6 tok/s over
      an 11.5K-token answer, MTP 25/28. Needs the Q5_1 table decode.
- [x] Serves **Swift 1.5 `IQ3_S`** (512 experts): contributor's box, numbers
      and probes in [SWIFT-1.5.md](SWIFT-1.5.md).
- [x] Header pre-flight of every ISTA tier over HTTP Range, no weight
      downloads: [docs/FLAVORS.md](docs/FLAVORS.md).
- [x] Differential gate for the Q2_0 perm-table rewrite (byte-identical over
      2,703,384 bytes) with a negative control that moves 52.14% of payload
      bytes. The rewrite was **slower** in a paired idle-GPU A/B and is parked
      on `perf/q20-perm-dequant`; it is not in any shipped path.

## Open

- [ ] `ctest -L qwen38_flash_next_gsq` (projection ops, routed WMMA ops).
      Blocked upstream: extension test executables cannot reach core
      implementations (`gufo_core` is link-prohibited for external targets and
      `gufo_ext_runtime` exposes headers, not their objects), so out-of-tree
      tests touching `GgufReader`/`HostAllocations`/crypto cannot link. The
      same test bodies pass upstream in gufo#939's suite against
      byte-identical code paths.
- [ ] Serve ISTA base `IQ3_S` here (pre-flight says every stored type has
      geometry; nobody has pointed a slot at it).
- [ ] Missing `FormatOf` geometries, in the order that unlocks the most
      flavors: `IQ2_XS` + `IQ2_XXS` (unlocks the `IQ3_XXS` tier and most of
      `IQ2_XS`), then `IQ1_M`, then the dense-side `Q3_K`/`Q4_0`/`Q5_0` the
      `Q2_0` tier needs. All six exist in the ds41 extension's `mmq` dispatch.
- [ ] Resolve `ISTA-DASLab/Qwen3.8-Flash-Next-P48NVFP4-MoESQ`: its HF tree
      returned no `.gguf`, so it is neither supported nor ruled out.
- [ ] Windows packaging (upstream model supports the platform; vendored Linux
      only so far).
- [ ] Functional suite (upstream `functional/` registry) as an extension-side
      script.
- [ ] Registry entry in `gufo-org/gufo-extension`: needs a tagged release whose
      `extension.json` version matches, the tree digest from
      `tools/registry.py digest`, and the author checklist answered
      (source license, weight licenses, maintainers, evidence).
- [ ] Reconcile with #939 reviews: re-vendor on upstream PR changes until it
      merges, then rebase onto main and drop the pin.

## Corrections to what this file used to claim

Kept because the wrong version of this history is still in issues and PR text:

- **"Requires the gufo branch `feat/rejected-gguf-extension-fallback`."** False.
  The `Failed to open GGUF: ... blk.1.ffn_down_exps.weight` failure that
  motivated that core failover was symbol interposition, not the parser: the
  executable exports `_ZN4gufo*;`, modules are dlopened `RTLD_NOW|RTLD_LOCAL`,
  and the fork's probe pointers bound to the executable's copies of the same
  names. `dladdr` proved it. `-fvisibility=hidden` in `extension.cmake` is the
  fix, and every artifact above loads on a released gufo. `scripts/build.sh`
  takes a `GUFO` binary, not a source pin.
- **"API `>= 0.1.16`."** The manifest says `>= 0.3.37`.
- **`TextProvider::claim_file` / "Minimal claim-file design".** Retired; the
  provider claims through `probe_gguf`, and the in-tree `qwen38_flash_next`
  exclusion is a consequence of explicit extension selection, not a lock edit.

## Upstream quirks worth reporting

- With `BUILD_TESTING=ON`, core `tests/models/*` link in-tree extension targets
  even when those extensions are excluded by `GUFO_EXTENSIONS`, so an
  extension-only selection cannot configure with tests.
- Extension test executables cannot reach core implementations (see the `ctest`
  item above).
