#!/usr/bin/env bash
# Build the differential Q2_0 harness against ONE kernels object.
#
#   KERNELS_O=/path/kernels.hip.cpp.o bash tools/tests/q20_diff.sh
#   bin/q20_diff/q20_diff --emit dump.bin
#
# KERNELS_O is the only kernel code under test: production
# kernels.hip.cpp.o for the baseline, a rebuilt object for the candidate. Two
# dumps, one per object, are compared byte for byte by tools/bench/ab_kernels.sh
# (or by hand). Builds only -- it touches no model, no module store, no gufo.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"

K="${KERNELS_O:?usage: KERNELS_O=<kernels.hip.cpp.o> bash tools/tests/q20_diff.sh}"
ARCH="${GPU_ARCH:-gfx1151}"
out="${OUT_DIR:-$here/bin/q20_diff}"
W8A8="${W8A8_O:-$(dirname "$K")/w8a8_wave64.hip.o}"
EXTRA="${EXTRA_OBJS:-}"
[[ -f "$K" ]] || { echo "missing kernels object: $K" >&2; exit 1; }
[[ -f "$W8A8" ]] || { echo "missing w8a8 object: $W8A8" >&2; exit 1; }

mkdir -p "$out"
hipcc -O2 --offload-arch="$ARCH" -c "$repo/tools/tests/q20_diff.hip.cpp" \
  -o "$out/q20_diff.o"
hipcc --offload-arch="$ARCH" -o "$out/q20_diff" "$out/q20_diff.o" "$K" "$W8A8" $EXTRA
echo "built: $out/q20_diff  <-  $K"
