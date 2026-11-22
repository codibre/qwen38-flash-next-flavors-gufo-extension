#!/usr/bin/env bash
# Build the bitwise Q2_0 dequant gate against an arbitrary kernels.hip.cpp.o.
#   KERNELS_O=/path/kernels.hip.cpp.o bash tools/tests/bitexact_q20.sh [--m 640 --k 640 --experts 3]
# Compile/link only; running the binary needs the GPU (a few hundred KB of it).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
out="${OUT_DIR:-$here/bin}"
mkdir -p "$out"
KERNELS_O="${KERNELS_O:?set KERNELS_O to the kernels.hip.cpp.o under test}"
[ -f "$KERNELS_O" ] || { echo "no such object: $KERNELS_O" >&2; exit 1; }
W8A8="${W8A8_O:-$(dirname "$KERNELS_O")/w8a8_wave64.hip.o}"
extra=()
[ -f "$W8A8" ] && extra+=("$W8A8")
if [ -n "${EXTRA_OBJS:-}" ]; then
  extra=(); for o in $EXTRA_OBJS; do
    [ -f "$o" ] || { echo "EXTRA_OBJS entry missing: $o" >&2; exit 1; }
    extra+=("$o"); done
fi
arch="${GPU_ARCH:-gfx1151}"
hipcc -O2 --offload-arch="$arch" -o "$out/bitexact_q20.o" -c "$here/bitexact_q20.hip.cpp" || exit 1
hipcc "$out/bitexact_q20.o" "$KERNELS_O" "${extra[@]}" -o "$out/bitexact_q20" || exit 1
echo "built $out/bitexact_q20 (kernels.o: $KERNELS_O)"
