#!/usr/bin/env bash
# Standalone microbenchmarks for the gfx1151 box.
#
#   tools/bench/build.sh            # build everything into tools/bench/bin
#   tools/bench/build.sh peak       # only qfn_peak
#   tools/bench/build.sh routed     # only qfn_routed_decode_bench (links kernels.o)
#
# routed_decode_bench links against the extension build tree's kernels.o so it
# exercises the real RoutedF16Gemm without recompiling the 416 KB translation
# unit. Kernels.o lives under the installed extension's build directory;
# override with KERNELS_O=/path/to/kernels.hip.cpp.o.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="$here/bin"
mkdir -p "$out"

arch="${GUFO_GPU_ARCH:-gfx1151}"
# The extension build tree keeps its archives next to the objects; linking the
# archives (not a bare kernels.o) resolves the cross-translation-unit calls,
# e.g. kernels.hip.cpp -> w8a8_wave64.hip.
BUILD_DIR="${GUFO_EXT_BUILD_DIR:-$(dirname "$(find "$HOME/.local/share/gufo/extensions/qwen38_flash_next_gsq" -name 'libgufo_qwen38_flash_next_gsq_hip.a' 2>/dev/null | head -1)")}"
KERNELS_O="${KERNELS_O:-$(find "$HOME/.local/share/gufo/extensions/qwen38_flash_next_gsq" -name 'kernels.hip.cpp.o' 2>/dev/null | head -1)}"

if [ "$#" -gt 0 ]; then want=("$@"); else want=(peak routed); fi

for w in "${want[@]}"; do
  case "$w" in
    peak)
      echo "hipcc -> $out/qfn_peak"
      hipcc -O3 --offload-arch="$arch" -o "$out/qfn_peak" "$here/qfn_peak.hip.cpp"
      ;;
    routed|routed_decode_bench)
      [ -n "$KERNELS_O" ] || { echo "kernels.hip.cpp.o not found; set KERNELS_O" >&2; exit 1; }
      echo "hipcc -> $out/qfn_routed_decode_bench   (kernels.o: $KERNELS_O)"
      # Redeclared ABI must match the object exactly; fail loudly otherwise.
      # Capture first: `nm | grep -q` under pipefail reports SIGPIPE (141),
      # because grep exits before nm finishes.
      syms="$(nm -C "$KERNELS_O" 2>/dev/null || true)"
      for sym in RoutedCompact RoutedTiles RoutedF16Gemm RoutedCompactRows; do
        case "$syms" in
          *" T gufo::models::qwen38_flash_next::rocm::$sym("*) ;;
          *) echo "kernels.o does not define $sym (stale object?)" >&2; exit 1 ;;
        esac
      done
      # Compile, then link: handing the prebuilt .o to hipcc in one step makes
      # the driver pass `-x hip` to it as well and it tries to *compile* the
      # object file.
      hipcc -O3 --offload-arch="$arch" -c "$here/qfn_routed_decode_bench.hip.cpp" \
        -o "$out/qfn_routed_decode_bench.o"
      # Link only the objects the bench actually needs. Passing whole archives
      # drags device_model.cpp.o (and with it the GSQ reader and the SDK),
      # which the benchmark never calls.
      obj_dir="$(dirname "$KERNELS_O")"
      extra=()
      for o in w8a8_wave64.hip.o; do
        [ -f "$obj_dir/$o" ] && extra+=("$obj_dir/$o")
      done
      hipcc "$out/qfn_routed_decode_bench.o" "$KERNELS_O" "${extra[@]}" \
        -o "$out/qfn_routed_decode_bench"
      ;;
    *) echo "unknown bench '$w'" >&2; exit 2 ;;
  esac
done
