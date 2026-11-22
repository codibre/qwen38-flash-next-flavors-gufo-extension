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
# or a GGUF file. It compiles two benchmark binaries plus the bitwise dequant
# gate (one-hot extraction against the float reference) and runs them.
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
for a in "$@"; do
  [ "$a" = "--force" ] && force=1
  [ "$a" = "--build-only" ] && build_only=1
done

# --- production object (the one the installed module was built from) ---------
KOLD="${KERNELS_OLD:-$(find "$HOME/.local/share/gufo/extensions/qwen38_flash_next_gsq" \
        -name 'kernels.hip.cpp.o' 2>/dev/null | head -1)}"
KNEW="${KERNELS_NEW:-$out/candidate.o}"
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
if [ -n "$busy" ] && [ "$force" -eq 0 ] && [ "$build_only" -eq 0 ]; then
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
  cmd="$(python3 - "$ccj" "$KNEW" <<'PY'
import json, shlex, sys
ccj, out_obj = sys.argv[1], sys.argv[2]
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
if [ -f "$KNEW" ] && [ "${REBUILD_CANDIDATE:-0}" != 1 ]; then
  echo "--- reusing candidate object $KNEW (REBUILD_CANDIDATE=1 to rebuild)"
else
  build_candidate || exit 1
fi

build_bench() {  # $1 = kernels object, $2 = output binary
  echo "--- building bench ($(basename "$2"))"
  cap hipcc -O3 --offload-arch="$arch" -c "$here/qfn_routed_decode_bench.hip.cpp" \
      -o "$out/ab_bench.o" || return 1
  cap hipcc "$out/ab_bench.o" "$1" "$W8A8" -o "$2" || return 1
}
build_bench "$KOLD" "$out/ab_old" || { echo "bench(old) build failed" >&2; exit 1; }
build_bench "$KNEW" "$out/ab_new" || { echo "bench(new) build failed" >&2; exit 1; }

echo "--- building bitwise dequant gate against the candidate"
KERNELS_O="$KNEW" W8A8_O="$W8A8" OUT_DIR="$results/be_new" \
  bash "$repo/tools/tests/bitexact_q20.sh" >/dev/null || {
    echo "bitwise gate build failed" >&2; exit 1; }
echo "--- building bitwise dequant gate against production"
KERNELS_O="$KOLD" W8A8_O="$W8A8" OUT_DIR="$results/be_old" \
  bash "$repo/tools/tests/bitexact_q20.sh" >/dev/null || {
    echo "bitwise gate build failed" >&2; exit 1; }

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
echo
echo "=== bitwise dequant gate (one-hot extraction, no tolerance band) ==="
gfail=0
for lbl in old new; do
  if "$results/be_$lbl/bitexact_q20" --m 640 --k 640 --experts 3 \
       >"$results/be_$lbl.out" 2>&1; then
    printf '  %-4s %s\n' "$lbl" "$(tail -1 "$results/be_$lbl.out")"
  else
    printf '  %-4s FAIL\n' "$lbl"
    grep -m5 'MISMATCH\|elements checked' "$results/be_$lbl.out"
    gfail=1
    [ "$lbl" = new ] && { echo "  candidate is not bit-exact; skipping timing."; }
  fi
done
if [ "$gfail" -ne 0 ]; then
  echo "  ('old' failing means the gate itself is wrong, not the kernel.)"
  echo "log: $log"; exit 1
fi

# --- A/B, interleaved -------------------------------------------------------
echo
echo "=== A/B (interleaved old/new, median of $REPS reps of $ITERS iters) ==="
dims=(--k 2560 --m 640 --experts 256 --topk 8)
echo "variant,type,rot,rep,ms_med,gbs_med,gbs_at_min" >"$csv"
run_one() {  # $1 variant $2 binary $3 type $4 rotate $5 rep
  local v="$1" bin="$2" ty="$3" r="$4" rep="$5" line
  line="$("$bin" "${dims[@]}" --type "$ty" --rotate "$r" --iters "$ITERS" --reps "$REPS" 2>/dev/null \
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
for rep in $(seq 1 3); do
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
    best[key][1] = max(best[key][1], float(r["gbs_best"]))
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
echo "bitwise dequant gate: PASS on both (old and new == float reference)"
echo "log: $log"
echo "csv: $csv"
