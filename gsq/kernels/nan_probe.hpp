#ifndef GUFO_EXT_GSQ_KERNELS_NAN_PROBE_HPP_
#define GUFO_EXT_GSQ_KERNELS_NAN_PROBE_HPP_

// Env-gated (GSQ_NAN) non-finite probes for the flavor MoE path. Kernel
// launches record offenders into pinned memory; a monitor thread prints
// them, so nothing here perturbs stream order or graph capture beyond one
// read-only kernel per scan.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::ext::gsq {

[[nodiscard]] bool NanDbgEnabled();

// One pass id per MoE stage invocation; scans and launch checks report it.
void NanPassEnter();

void NanScanF32(const float* d, std::size_t n, const char* what, int layer,
                std::uint32_t ntok, hipStream_t stream);

void NanScanF16(const __half* d, std::size_t n, const char* what, int layer,
                std::uint32_t ntok, hipStream_t stream);

// Throttled hipGetLastError report at a named site (debug only; clears
// sticky errors, so never active when the env gate is off).
void NanLaunchCheck(const char* site, int layer);

// CPU-side scan (throttled fprintf) for host decoders like the Q5_1 PLE.
void NanScanHostReport(const char* what, int layer, std::uint32_t row,
                       const float* v, std::size_t n);

}  // namespace gufo::ext::gsq

#endif  // GUFO_EXT_GSQ_KERNELS_NAN_PROBE_HPP_
