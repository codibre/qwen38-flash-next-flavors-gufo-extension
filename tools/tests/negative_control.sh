#!/usr/bin/env bash
# Negative control for the differential gate.
#
# A gate that always says "identical" is worthless, so this builds a candidate
# whose Q2_0 perm selector is wrong in exactly one field and checks that the
# gate notices. Run it once after any change to tools/tests/q20_diff.* or to
# the Q2_0 branch of RoutedF16GEMMKernel.
#
#   bash tools/tests/negative_control.sh
#
# Needs tools/bench/results/new.bin, i.e. one prior
#   bash tools/bench/ab_kernels.sh --gate-only
#
# Builds objects under tools/bench/bin (gitignored). No engine, no GGUF, no
# module store, and it never measures time.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../.." && pwd)
bin="$repo/tools/bench/bin"
results="$repo/tools/bench/results"
KSRC="$repo/src/model/kernels/rocm/kernels.hip.cpp"
ccj=$repo/build-gufo/build/release/compile_commands.json
# Where the *production* kernels.hip.cpp.o lives, for the old-vs-new
# differential. For a module install that is inside the extension store, whose
# path embeds the absolute checkout it was compiled from, so it is passed in
# rather than guessed:
#   KERNELS_OLD=$(dirname "$(find "${GUFO_EXT_STORE:-$HOME/.local/share/gufo/extensions}" \
#     -name kernels.hip.cpp.o | head -1)") bash tools/tests/negative_control.sh
if [ -z "${KERNELS_OLD:-}" ]; then
  echo "KERNELS_OLD is not set: point it at the directory holding the production" >&2
  echo "kernels.hip.cpp.o, e.g. under" >&2
  echo "  ${GUFO_EXT_STORE:-$HOME/.local/share/gufo/extensions}/<ext>/source/build-gufo/..." >&2
  exit 1
fi
D=$KERNELS_OLD
W8A8=${W8A8_O:-$D/w8a8_wave64.hip.o}
A=${GATE_ARGS:-"--m 640 --k 2560 --topk 8 --experts 64 --rotate 8 --patterns 24"}

[ -f "$results/new.bin" ] || { echo "run 'bash tools/bench/ab_kernels.sh --gate-only' first"; exit 1; }
mkdir -p "$bin"

# 1. break the candidate in one field: the odd-element selector should pick
#    table entry 2*q0+1; make it 2*q0+3, which is out of range for q0 = 3.
sed 's/((2U \* q0 + 1U) << 8)/((2U * q0 + 3U) << 8)/' "$KSRC" > "$bin/broken_src.hip.cpp"
if ! cmp -s "$KSRC" "$bin/broken_src.hip.cpp"; then
  echo "injected fault: odd-element selector 2*q0+1 -> 2*q0+3"
else
  echo "source no longer contains the selector pattern this control patches"; exit 1
fi

# 2. compile it with the extension's own compile command (gufo's checkout DB
#    compiles gufo's in-tree file, which would silently test the wrong source).
[ -f "$ccj" ] || { echo "no compile DB at $ccj"; exit 1; }
cmd=$(python3 - "$ccj" "$bin/broken.o" "$bin/broken_src.hip.cpp" <<'PY'
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
            if a == "-o":
                skip = True
                continue
            acc.append(src if a == e["file"] else a)
        assert src in acc, "source substitution failed"
        if "-c" not in acc:
            acc.append("-c")
        acc += ["-o", out_obj]
        print(" ".join(shlex.quote(x) for x in acc))
        break
else:
    sys.exit("kernels/rocm/kernels.hip.cpp not in compile DB")
PY
)
echo "compiling the broken candidate (~1-2 min) ..."
rm -f "$bin/broken.o"
( cd "$(dirname "$ccj")" && systemd-run --user --scope -p MemoryMax=8G \
    -p MemorySwapMax=0 nice -n 10 timeout 1800 bash -c "$cmd" ) >/dev/null 2>&1
[ -f "$bin/broken.o" ] || { echo "broken candidate compile failed"; exit 1; }

# 3. run it through the same harness and compare against the real dump.
OUT_DIR="$results/diff_bad" KERNELS_O="$bin/broken.o" W8A8_O="$W8A8" \
  bash "$here/q20_diff.sh" >/dev/null 2>&1
# shellcheck disable=SC2086
"$results/diff_bad/q20_diff" --emit "$results/bad.bin" $A | tail -1

if cmp -s "$results/new.bin" "$results/bad.bin"; then
  echo "CONTROL FAILED: a wrong selector produced identical output, so the gate"
  echo "is not observing the dequant. Do not trust a PASS from it."
  exit 1
fi
python3 - "$results/new.bin" "$results/bad.bin" <<'PY'
import sys
o = open(sys.argv[1], 'rb').read(); n = open(sys.argv[2], 'rb').read()
m = min(len(o), len(n))
d = sum(1 for i in range(24, m) if o[i] != n[i])
print("control detected it: %d of %d payload bytes differ (%.2f%%)"
      % (d, m - 24, 100.0 * d / max(m - 24, 1)))
PY
echo "The gate is sensitive; the real candidate moved none of these bits."
