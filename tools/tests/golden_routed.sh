#!/usr/bin/env bash
# Golden gate for the routed WMMA ops (F64 oracle, incl. Q2_0), built against an
# ARBITRARY kernels.hip.cpp.o so a candidate kernel object can be proven
# numerically before it ever reaches a model slot.
#
#   KERNELS_O=/path/kernels.hip.cpp.o bash tools/tests/golden_routed.sh
#
# Never touches the module store, the gufo binary or llama-swap. Allocates a few
# MB on the GPU only.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(dirname "$(dirname "$here")")"
out="${OUT_DIR:-$repo/tools/tests/bin}"
mkdir -p "$out"

KERNELS_O="${KERNELS_O:?set KERNELS_O to the kernels.hip.cpp.o under test}"
[ -f "$KERNELS_O" ] || { echo "no such object: $KERNELS_O" >&2; exit 1; }
obj_dir="$(dirname "$KERNELS_O")"
w8a8="$obj_dir/w8a8_wave64.hip.o"
extra=()
[ -f "$w8a8" ] && extra+=("$w8a8")
# A candidate object compiled outside the build tree has no sibling objects.
if [ -n "${EXTRA_OBJS:-}" ]; then
  extra=()
  for o in $EXTRA_OBJS; do
    [ -f "$o" ] || { echo "EXTRA_OBJS entry missing: $o" >&2; exit 1; }
    extra+=("$o")
  done
fi

arch="${GPU_ARCH:-gfx1151}"
build="$repo/build-gufo/build/release"

# Same defines/includes as the production HIP target, but WITHOUT -ffast-math:
# the oracle's double-precision reference must stay strictly ordered.
set -u
clang=/opt/rocm-7.2.4/lib/llvm/bin/clang++
"$clang" \
  -DENGINE_ENABLE_HIP=1 -DQFN_ROCM_BUILD=1 -D__HIP_PLATFORM_AMD__=1 \
  -I"$repo" -I"$repo/src" \
  -I"$build/extensions/gufo" -I"$build/src/ext/public-include" \
  -isystem "$repo/build-gufo/include" -isystem "$repo/src/model/kernels/rocm/mmq" \
  -O2 -g -DNDEBUG -std=c++20 --offload-arch="$arch" \
  -Wno-unused-command-line-argument \
  -o "$out/golden_routed.o" -x hip -c "$repo/tests/routed_wmma_ops_test.cpp"

hipcc "$out/golden_routed.o" "$KERNELS_O" "${extra[@]}" \
  -L/opt/rocm/lib -lamdhip64 -o "$out/golden_routed"

echo "built $out/golden_routed (kernels.o: $KERNELS_O)"
