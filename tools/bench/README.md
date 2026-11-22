# Benchmarks and profiling

Everything here measures the kernels or a running slot through llama-swap.
Nothing loads a GGUF directly and nothing starts a second engine — model
traffic goes to the router at `127.0.0.1:8642` only.

## Tools

| Tool | What it answers |
| --- | --- |
| `build.sh [peak\|routed]` | builds into `tools/bench/bin`; `routed` links the extension build tree's `kernels.o`, so it exercises the production `RoutedF16Gemm` without recompiling the 416 KB translation unit. |
| `qfn_peak.hip.cpp` | the box's ceilings: DRAM read, LLC re-read curve, LDS. The reference line for every kernel number below. |
| `qfn_routed_decode_bench.hip.cpp` | routed-expert GEMM at model shapes. `--rotate N` walks N disjoint expert sets so the timing loop is DRAM-cold the way real MoE decode is; `--tokens M` widens the live tile. |
| `slot_baseline.py <slot>` | end-to-end prefill/decode from gufo's own `timings` block. Chat endpoint, cache-busted prompt, MTP acceptance reported. |
| `../prof/isa_mix.py <kernels.o>` | opcode mix per kernel, carved out of the shipped code object. Finds instruction-level pathologies (see below). Template instantiations are selected by mangled enum value: `--find WeightTypeE42E` is Q2_0, `E12E` is Q4_K. |

## Measured baseline (2026-11-22, gfx1151, `advanced` resident)

Hardware ceilings (`qfn_peak`, first run on this box):

```
DRAM read   768 MiB:    241.7 GB/s
re-read      16 MiB:    822.8 GB/s     re-read  32 MiB:  871.1 GB/s
re-read      64 MiB:    238.1 GB/s     LDS read 16 KiB: 5382.2 GB/s
```

The LLC cliff sits between 32 and 64 MiB, which is why `--rotate` matters: a
single expert set is 11 MiB and gets served from cache at 872 GB/s, flattering
any kernel.

Routed experts at the checkpoint's real shapes — `blk.N.ffn_down_exps.weight`
is `[640, 2560, 256]` (expert_ff=640, hidden=2560, 256 experts), top-k 8:

| shape | batch | total | bytes/token |
| --- | ---: | ---: | ---: |
| Q2_0, rotate=1 (LLC-warm) | 1 | 72.0 GB/s | 10.5 MiB |
| **Q2_0, rotate=16 (DRAM-cold)** | **1** | **62.8 GB/s** | 10.5 MiB |
| Q2_0, rotate=8 | 8 | 68.3 GB/s | — |
| Q4_K, rotate=8 | 1 | 173.9 GB/s | 25.3 MiB |

Two conclusions the static reading could not give:

1. The Q2_0 route runs at **26% of the DRAM ceiling**, 2.8× below the same
   kernel's Q4_K route. Q2_0 is 0.28125 B/value against Q4_K's 0.5, so per
   *value* it is 1.5× slower, not 2.8× — it is issue-bound, not memory-bound.
2. Widening the tile from 1 to 8 tokens barely moves it (62.8 → 68.3 GB/s).
   Low occupancy is not the limiter either. Anything that only batches or
   re-tiles the existing kernel will not close this.

`isa_mix.py` on the production object says why. The Q2_0 instantiation of
`RoutedF16GEMMKernel` carries, per loop:

```
v_cvt_f32_i32 ×64   v_bfe_u32 ×58   v_cvt_f16_f32 ×48   v_wmma ×16
```

while the Q4_K instantiation has **zero** `v_cvt_f32_i32` — its nibbles go
through `v_perm` plus LDS tables. int→float conversion is quarter-rate on this
GPU, so the dequant path issues several times more work per value than the
matmul it feeds. This is the same failure ds41 documented for its IQ2 decode
(there via `v_ldexp_f32`, 129 per loop) and fixed with whole-word loads, LDS
u32 word-pair tables, sign-bit flips and loading `x` once per block.

End-to-end, same architecture, same 3407-token prompt, `/v1/chat/completions`,
temperature 0, thinking on, MTP on:

| slot | storage | prefill | decode | TTFT | MTP accepted |
| --- | --- | ---: | ---: | ---: | ---: |
| `advanced` | UD-Q4_K_XL (in-tree provider) | 1262–1267 tok/s | 54.5 tok/s (18.4 ms) | 2616 ms | 86/110 (78%) |
| `heretic` | GSQ Q2_0 (this extension) | 651–664 tok/s | 27.9 tok/s (35.9 ms) | 5072 ms | 74/96 (77%) |

The Q2_0 build reads roughly half the weight bytes of the Q4_K build and runs
roughly half as fast at both prefill and decode, with the same draft
acceptance. Acceptance being equal rules out the drafter; the difference is in
the kernels.

Earlier numbers quoted in the README (162.9 / 19.9 tok/s) came from a 6-token
prompt against `/v1/completions` and included load time. They are not a
baseline; `slot_baseline.py` is.

## Reproducing

```bash
tools/bench/build.sh                                   # peak + routed
tools/bench/bin/qfn_peak --dram 768 --llc 16 --lds 16
tools/bench/bin/qfn_routed_decode_bench --k 2560 --m 640 --experts 256 \
    --topk 8 --type q2_0 --rotate 16 --iters 40
tools/bench/slot_baseline.py heretic --prefill-tokens 2000 --decode 128
tools/bench/slot_baseline.py advanced --prefill-tokens 2000 --decode 128
```

`slot_baseline.py` switches slots through the router, so each run pays a model
load (visible as `first_response`); run the slot of interest last, and leave
whichever slot the user wants resident. The bench binaries allocate a few
hundred MiB of device memory — with a 95 GB slot resident, keep `--experts`
modest and do not raise `--dram` past 1 GiB.

## Numeric risk on any rewrite of the Q2_0 dequant

These kernels deliberately preserve the FP summation order. A decode rewrite
that keeps the same arithmetic (algebraic bias removal, table staging,
whole-word loads) can stay bit-exact — ds41's retune states it was "bit-exact
by construction and by measurement" — but any change to accumulation order
must be re-validated against the goldens before it ships.
