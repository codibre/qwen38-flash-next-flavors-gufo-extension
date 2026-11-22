#!/usr/bin/env bash
# A/B a candidate kernels.hip.cpp.o against the production one, on an IDLE GPU.
#
#   bash tools/bench/ab_kernels.sh                 # refuses to run if gufo is live
#   bash tools/bench/ab_kernels.sh --force         # run anyway (noisy numbers)
#   KERNELS_NEW=/path/others.o bash tools/bench/ab_kernels.sh
#   REPS=9 ITERS=300 bash tools/bench/ab_kernels.sh
#
# Why the user runs it: every measurement here is latency-bound on a single
# APUs's memory system. A resident model (or any other GPU work) doubles or
# halves the result -- observed 2.5 vs 69 GB/s for the SAME binary. The gate at
# the top checks the GPU is free and refuses to produce numbers otherwise.
#
# Nothing in this script touches the module store, the gufo binary, llama-swap
# or a GGUF file. It compiles two benchmark binaries plus the differential
# gate (tools/tests/q20_diff.hip.cpp: same inputs through production and
# candidate, outputs compared byte for byte) and runs them.
#
#   bash tools/bench/ab_kernels.sh --gate-only   # correctness only, seconds
#   bash tools/bench/ab_kernels.sh --build-only  # compile, no GPU work
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
out="$here/bin"
results="${RESULTS_DIR:-$here/results}"
mkdir -p "$out" "$results"

REPS="${REPS:-7}"
ITERS="${ITERS:-200}"
arch="${GPU_ARCH:-gfx1151}"
force=0
build_only=0
gate_only=0
for a in "$@"; do
  [ "$a" = "--force" ] && force=1
  [ "$a" = "--build-only" ] && build_only=1
  [ "$a" = "--gate-only" ] && gate_only=1
done

# --- production object (the one the installed module was built from) ---------
KOLD="${KERNELS_OLD:-$(find "$HOME/.local/share/gufo/extensions/qwen38_flash_next_gsq" \
        -name 'kernels.hip.cpp.o' 2>/dev/null | head -1)}"
KNEW="${KERNELS_NEW:-$out/candidate.o}"
KSRC="$repo/src/model/kernels/rocm/kernels.hip.cpp"
W8A8="${W8A8_O:-$(dirname "$KOLD")/w8a8_wave64.hip.o}"
[ -f "$KOLD" ] || { echo "production kernels.hip.cpp.o not found; set KERNELS_OLD" >&2; exit 1; }
[ -f "$W8A8" ] || { echo "w8a8_wave64.hip.o not found: $W8A8 (set W8A8_O)" >&2; exit 1; }

stamp="$(date +%Y%m%d_%H%M%S)"
log="$results/ab_${stamp}.log"
csv="$results/ab_${stamp}.csv"
exec > >(tee "$log") 2>&1

echo "=== qwen38-flash-next routed kernel A/B   $stamp ==="
echo "old (production): $KOLD"
echo "new (candidate) : $KNEW"
echo "reps=$REPS iters=$ITERS arch=$arch"

# --- GPU must be idle --------------------------------------------------------
busy="$(pgrep -af 'gufo|llama-server|vllm|python.*torch' | grep -v 'ab_kernels\|tee .*bench/results' | head -5 || true)"
# The gate compares bits, so a busy GPU cannot falsify it; only timings need a
# quiet machine.
if [ -n "$busy" ] && [ "$force" -eq 0 ] && [ "$build_only" -eq 0 ] && [ "$gate_only" -eq 0 ]; then
  echo
  echo "REFUSING: GPU is not idle. Unload the slot (llama-swap) or pass --force."
  echo "$busy"
  exit 3
fi
if [ -n "$busy" ]; then echo "WARNING: proceeding with GPU busy:"; echo "$busy"; fi
command -v rocm-smi >/dev/null && rocm-smi --showmeminfo vram 2>/dev/null | grep -i 'used\|total' | head -4
free -g | awk 'NR==2{print "host RAM free:", $7" GB"}'

cap() { systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 nice -n 10 "$@"; }

# --- build ------------------------------------------------------------------
build_candidate() {  # candidate kernels.o = this repo's source, production flags
  local ccj="$repo/build-gufo/build/release/compile_commands.json" cmd
  [ -f "$ccj" ] || {
    echo "cannot build the candidate: no compile DB at $ccj" >&2
    echo "point KERNELS_NEW at a prebuilt kernels.hip.cpp.o instead." >&2
    return 1; }
  cmd="$(python3 - "$ccj" "$KNEW" "$KSRC" <<'PY'
import json, shlex, sys
ccj, out_obj, src = sys.argv[1], sys.argv[2], sys.argv[3]
for e in json.load(open(ccj)):
    if e["file"].endswith("kernels/rocm/kernels.hip.cpp"):
        args = shlex.split(e["command"]) if "command" in e else list(e["arguments"])
        acc, skip = [], False
        for a in args:
            if skip:
                skip = False
                continue
            if a == "-o":  # drop the build-tree output path only
                skip = True
                continue
            acc.append(a)
        # The compile DB describes gufo's own tree. The candidate is THIS
        # repo's kernel source, so swap the entry's file out; compiling the
        # in-tree copy would silently bench upstream code.
        acc = [src if a == e["file"] else a for a in acc]
        assert src in acc, "source substitution failed"
        if "-c" not in acc:
            acc.append("-c")
        acc += ["-o", out_obj]
        print(" ".join(shlex.quote(x) for x in acc))
        break
else:
    sys.exit("kernels/rocm/kernels.hip.cpp is not in the compile DB")
PY
)" || { echo "could not extract the compile command from $ccj" >&2; return 1; }
  echo "--- building candidate kernels.o from $(git -C "$repo" rev-parse --short HEAD 2>/dev/null || echo 'working tree')"
  ( cd "$(dirname "$ccj")" && cap timeout 1800 bash -c "$cmd" ) || {
    echo "candidate compile failed" >&2
    return 1; }
}
if [ -f "$KNEW" ] && [ "$KNEW" -nt "$KSRC" ] && [ "${REBUILD_CANDIDATE:-0}" != 1 ]; then
  echo "--- reusing candidate object $KNEW (REBUILD_CANDIDATE=1 to rebuild)"
else
  build_candidate || exit 1
fi

if [ "$build_only" = 1 ]; then
  echo "build-only: candidate object is $KNEW; no GPU work done."
  exit 0
fi

build_bench() {  # $1 = kernels object, $2 = output binary
  echo "--- building bench ($(basename "$2"))"
  cap hipcc -O3 --offload-arch="$arch" -c "$here/qfn_routed_decode_bench.hip.cpp" \
      -o "$out/ab_bench.o" || return 1
  cap hipcc "$out/ab_bench.o" "$1" "$W8A8" -o "$2" || return 1
}
if [ "$gate_only" != 1 ]; then
  build_bench "$KOLD" "$out/ab_old" || { echo "bench(old) build failed" >&2; exit 1; }
  build_bench "$KNEW" "$out/ab_new" || { echo "bench(new) build failed" >&2; exit 1; }
fi

echo "--- building differential Q2_0 harness against the candidate"
KERNELS_O="$KNEW" W8A8_O="$W8A8" OUT_DIR="$results/diff_new" \
  bash "$repo/tools/tests/q20_diff.sh" >/dev/null || {
    echo "differential harness build failed" >&2; exit 1; }
echo "--- building differential Q2_0 harness against production"
KERNELS_O="$KOLD" W8A8_O="$W8A8" OUT_DIR="$results/diff_old" \
  bash "$repo/tools/tests/q20_diff.sh" >/dev/null || {
    echo "differential harness build failed" >&2; exit 1; }

# --- ISA: did the pathological opcodes actually go away? --------------------
echo
echo "=== ISA (Q2_0 template) ==="
for o in "$KOLD:old" "$KNEW:new"; do
  obj="${o%:*}"; lbl="${o#*:}"
  mix="$(python3 "$repo/tools/prof/isa_mix.py" --find WeightTypeE42E "$obj" 2>/dev/null || true)"
  cvt=$(grep -o 'v_cvt_f32_i32[a-z_0-9]*×[0-9]*' <<<"$mix" | head -1)
  pk=$(grep -o 'v_pack_b32_f16×[0-9]*' <<<"$mix" | head -1)
  pm=$(grep -o 'v_perm_b32×[0-9]*' <<<"$mix" | head -1)
  ins=$(grep -o '[0-9]* instructions' <<<"$mix" | head -1)
  printf '  %-4s %s | %s | %s | %s | %s\n' "$lbl" "${ins:-?}" "${cvt:-v_cvt_f32_i32:none}" \
    "${pk:-v_pack_b32_f16:none}" "${pm:-v_perm_b32:none}"
done

# --- numeric gate -----------------------------------------------------------
# The property that matters for a decode rewrite: candidate and production
# produce the SAME BITS on identical inputs. No oracle, no tolerance band, so
# no dependence on what the kernel's internal rounding order happens to be.
echo
echo "=== differential gate (candidate vs production, byte for byte) ==="
GATE_M="${GATE_M:-640}"              # checkpoint: expert_ff x hidden x experts
GATE_K="${GATE_K:-2560}"
GATE_TOPK="${GATE_TOPK:-8}"
GATE_EXPERTS="${GATE_EXPERTS:-64}"   # x topk slots per set
GATE_ROTATE="${GATE_ROTATE:-8}"      # routing sets; experts/topk covers them all
GATE_PATTERNS="${GATE_PATTERNS:-24}"  # activation shapes
for lbl in old new; do
  if ! "$results/diff_$lbl/q20_diff" --emit "$results/$lbl.bin" \
        --m "$GATE_M" --k "$GATE_K" --topk "$GATE_TOPK" \
        --experts "$GATE_EXPERTS" --rotate "$GATE_ROTATE" \
        --patterns "$GATE_PATTERNS" \
        >"$results/diff_$lbl.out" 2>&1; then
    printf '  %-4s harness run FAILED\n' "$lbl"
    tail -5 "$results/diff_$lbl.out"
    echo "log: $log"; exit 1
  fi
  printf '  %-4s %s\n' "$lbl" "$(tail -1 "$results/diff_$lbl.out")"
done
if cmp -s "$results/old.bin" "$results/new.bin"; then
  echo "  IDENTICAL: $(stat -c%s "$results/old.bin") bytes -- candidate == production"
else
  echo "  DIFFERS: $(cmp "$results/old.bin" "$results/new.bin" 2>&1 | head -1)"
  python3 - "$results/old.bin" "$results/new.bin" <<'PY'
import sys
o = open(sys.argv[1], 'rb').read(); n = open(sys.argv[2], 'rb').read()
print("  sizes: old %d bytes, new %d bytes" % (len(o), len(n)))
m = min(len(o), len(n))
d = [i for i in range(24, m) if o[i] != n[i]]        # 24-byte header, then dumps
print("  %d of %d payload bytes differ (%.4f%%)" % (len(d), m - 24,
                                                     100.0 * len(d) / (m - 24)))
for i in d[:10]:
    print("    @%8d old %02x new %02x" % (i, o[i], n[i]))
PY
  echo "  candidate is not bit-exact; skipping timing."
  echo "log: $log"; exit 1
fi
if [ "$gate_only" = 1 ]; then
  echo "gate-only run finished: identical. log: $log"
  exit 0
fi

# --- A/B, interleaved -------------------------------------------------------
echo
echo "=== A/B (interleaved old/new; $ROUNDS rounds, each the median of $REPS reps x $ITERS iters) ==="
dims=(--k 2560 --experts 256 --topk 8)
# Q4_K tiles need m % 256 == 0, so the control cannot share Q2_0's m=640.
dims_for() { case "$1" in q4_k) echo --m 768 ;; *) echo --m 640 ;; esac; }
echo "variant,type,rot,rep,ms_med,gbs_med,gbs_at_min" >"$csv"
run_one() {  # $1 variant $2 binary $3 type $4 rotate $5 rep
  local v="$1" bin="$2" ty="$3" r="$4" rep="$5" line
  line="$("$bin" "${dims[@]}" $(dims_for "$ty") --type "$ty" --rotate "$r" --iters "$ITERS" --reps "$REPS" 2>/dev/null \
         | grep TOTAL)"
  [ -n "$line" ] || { echo "  !! no TOTAL from $v/$ty/rotate=$r" >&2; return 1; }
  python3 - "$v" "$ty" "$r" "$rep" "$line" <<'PY' >>"$csv"
import re, sys
v, ty, r, rep, line = sys.argv[1:6]
m = re.search(r"TOTAL\s+med\s+([\d.]+) ms\s+([\d.]+) GB/s.*?best\s+([\d.]+) GB/s", line)
if not m:
    print(f"{v},{ty},{r},{rep},NA,NA,NA"); raise SystemExit
med, gmed, gbest = (float(x) for x in m.groups())
print(f"{v},{ty},{r},{rep},{med:.4f},{gmed:.2f},{gbest:.2f}")
PY
  printf '  %-3s %-6s rot=%-2s rep%s  %s\n' "$v" "$ty" "$r" "$rep" \
    "$(grep TOTAL <<<"$(echo "$line")" | sed 's/^ *//')"
}
ROUNDS="${AB_ROUNDS:-5}"
for rep in $(seq 1 "$ROUNDS"); do
  for ty in q2_0 q4_k; do
    for r in 1 16; do
      run_one old "$out/ab_old" "$ty" "$r" "$rep"
      run_one new "$out/ab_new" "$ty" "$r" "$rep"
    done
  done
done

# --- verdict ----------------------------------------------------------------
echo
echo "=== verdict (best-of runs) ==="
python3 - "$csv" <<'PY'
import csv, sys, collections
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["gbs_med"] != "NA"]
best = collections.defaultdict(lambda: [0.0, 0.0])
for r in rows:
    key = (r["variant"], r["type"], r["rot"])
    best[key][0] = max(best[key][0], float(r["gbs_med"]))
    best[key][1] = max(best[key][1], float(r["gbs_at_min"]))
for ty in sorted({k[1] for k in best}):
    for rot in sorted({k[2] for k in best if k[1] == ty}):
        o, n = best.get(("old", ty, rot)), best.get(("new", ty, rot))
        if not o or not n:
            continue
        d = (n[0] / o[0] - 1) * 100 if o[0] else 0
        print(f"  {ty:6s} rot={rot:<3s} old {o[0]:6.1f} -> new {n[0]:6.1f} GB/s  ({d:+.0f}%)   "
              f"[best-run: {o[1]:.1f} -> {n[1]:.1f}]")
print("\n  rot=1 is LLC-resident, rot=16 walks disjoint expert sets (DRAM-cold).")
print("  q4_k is the control: it must not regress.")
PY
echo
echo "differential gate: candidate output is byte-identical to production"
echo "log: $log"
echo "csv: $csv"
