#!/usr/bin/env bash
# Build and install qwen38_flash_next_gsq as a gufo extension MODULE.
#
# gufo 0.3+ builds extensions against the SDK that ships with the gufo binary
# you already run: gufo itself is NOT rebuilt, so one binary serves every model
# slot. The module is stamped with the host build identity and refuses to load
# into a different gufo build, which is why this script takes the binary rather
# than a source pin.
#
#   GUFO=/path/to/gufo ./scripts/build.sh
#
# Environment:
#   GUFO              gufo binary to build against (default: gufo on PATH)
#   GUFO_SDK_DIR      SDK tree of that binary; set it when the SDK is not
#                     where the binary looks (build/release/sdk in a checkout)
#   CMAKE_BUILD_PARALLEL_LEVEL
#                     compile jobs (keep this at 2-4 on a box whose model slot
#                     is resident: the HIP link is memory-hungry)
set -euo pipefail

EXT_DIR=$(cd "$(dirname "$0")/.." && pwd)
GUFO=${GUFO:-gufo}
export GUFO_SDK_DIR

if ! command -v "$GUFO" >/dev/null 2>&1 && [[ ! -x "$GUFO" ]]; then
  echo "error: '$GUFO' is not runnable; set GUFO to your gufo binary" >&2
  exit 1
fi

# Building an extension loads nothing: no GPU, no weights.
"$GUFO" extension install "$EXT_DIR" --yes
echo
"$GUFO" extension list

echo
echo "Served by llama-swap / gufo as soon as the slot for the artifact starts:"
echo "  curl http://127.0.0.1:8642/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"heretic\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'"
