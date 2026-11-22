// Machine ceilings for the gfx1151 box, measured, not assumed.
//
// Ported in spirit from ds41-gufo-extension's tools/bench (MIT, llama.cpp
// lineage) after the 2026-11 review: their DECODE/PREFILL docs quote a DRAM
// read ceiling of 240 GB/s, a last-level-cache re-read band of 750-920 GB/s
// and an LDS band of 6280 GB/s on this same APU. Decode for GSQ Q2_0 experts
// is bandwidth-bound, so every kernel-side claim has to be priced against
// these numbers instead of against a guess.
//
//   hipcc -O3 --offload-arch=gfx1151 -o qfn_peak qfn_peak.hip.cpp
//
// Allocates at most --dram MiB (default 1024) so it stays usable while a
// model is resident.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CK(x)                                                              \
  do {                                                                     \
    hipError_t e_ = (x);                                                   \
    if (e_ != hipSuccess) {                                                \
      std::fprintf(stderr, "hip error %s at %s:%d\n", hipGetErrorString(e_), \
                   __FILE__, __LINE__);                                    \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

// Grid-stride read of uint4 blocks; the accumulator is written out per thread
// so nothing in the loop is dead.
__global__ void ReadBandwidthKernel(const uint4* __restrict__ data,
                                    std::size_t n_vec, float* __restrict__ sink) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  unsigned long long acc = 0;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n_vec; i += stride) {
    const uint4 v = data[i];
    acc += static_cast<unsigned long long>(v.x) + v.y + v.z + v.w;
  }
  if (acc == 0xdeadbeefULL) sink[0] = 1.0f;
}

// LDS band: each wave reads a 16 KiB shared tile repeatedly as float4.
__global__ void LdsBandwidthKernel(float* __restrict__ sink) {
  __shared__ float tile[4096];  // 16 KiB
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) tile[i] = i * 0.5f;
  __syncthreads();
  const float4* v = reinterpret_cast<const float4*>(tile);
  float acc = 0.0f;
#pragma unroll 8
  for (int r = 0; r < 256; ++r) {
    const float4 x = v[(threadIdx.x + r * 7) & 1023];
    acc += x.x + x.y + x.z + x.w;
  }
  if (acc == 12345.678f) sink[0] = acc;
}

int main(int argc, char** argv) {
  std::size_t dram_mib = 1024;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--dram") && i + 1 < argc) dram_mib = std::atoll(argv[++i]);
  }
  int device = 0;
  CK(hipGetDevice(&device));
  hipDeviceProp_t prop{};
  CK(hipGetDeviceProperties(&prop, device));
  std::printf("device: %s (%s)  LDS/SM=%zu KB  LLC=%d MB\n", prop.name,
              prop.gcnArchName, prop.sharedMemPerMultiprocessor / 1024,
              static_cast<int>(prop.l2CacheSize / (1024 * 1024)));
  float* sink = nullptr;
  CK(hipMalloc(&sink, sizeof(float)));

  // 1. DRAM read bandwidth.
  const std::size_t bytes = dram_mib * 1024ull * 1024ull;
  uint4* data = nullptr;
  CK(hipMalloc(&data, bytes));
  CK(hipMemset(data, 0x3c, bytes));
  const std::size_t n_vec = bytes / sizeof(uint4);
  const int blocks = prop.multiProcessorCount * 32;
  hipEvent_t a, b;
  CK(hipEventCreate(&a));
  CK(hipEventCreate(&b));
  for (int i = 0; i < 2; ++i) ReadBandwidthKernel<<<blocks, 256>>>(data, n_vec, sink);
  CK(hipDeviceSynchronize());
  CK(hipEventRecord(a));
  for (int i = 0; i < 5; ++i) ReadBandwidthKernel<<<blocks, 256>>>(data, n_vec, sink);
  CK(hipEventRecord(b));
  CK(hipEventSynchronize(b));
  float ms = 0.0f;
  CK(hipEventElapsedTime(&ms, a, b));
  std::printf("DRAM read  %4zu MiB: %8.1f GB/s\n", dram_mib,
              bytes * 5.0 / (ms * 1e-3) / 1e9);
  CK(hipFree(data));

  // 2. Re-read ramp: how far the last-level cache lifts repeated reads.
  for (std::size_t mib : {1ull, 4ull, 16ull, 32ull, 64ull, 128ull}) {
    const std::size_t rb = mib * 1024ull * 1024ull;
    uint4* small = nullptr;
    CK(hipMalloc(&small, rb));
    CK(hipMemset(small, 0x5a, rb));
    const std::size_t rv = rb / sizeof(uint4);
    const int reps = mib <= 16 ? 64 : 16;
    for (int i = 0; i < 4; ++i) ReadBandwidthKernel<<<blocks, 256>>>(small, rv, sink);
    CK(hipDeviceSynchronize());
    CK(hipEventRecord(a));
    for (int i = 0; i < reps; ++i) ReadBandwidthKernel<<<blocks, 256>>>(small, rv, sink);
    CK(hipEventRecord(b));
    CK(hipEventSynchronize(b));
    CK(hipEventElapsedTime(&ms, a, b));
    std::printf("re-read    %4zu MiB: %8.1f GB/s\n", mib,
                rb * static_cast<double>(reps) / (ms * 1e-3) / 1e9);
    CK(hipFree(small));
  }

  // 3. LDS band (aggregate over the device).
  const int lds_blocks = prop.multiProcessorCount * 8;
  for (int i = 0; i < 2; ++i) LdsBandwidthKernel<<<lds_blocks, 256>>>(sink);
  CK(hipDeviceSynchronize());
  CK(hipEventRecord(a));
  for (int i = 0; i < 20; ++i) LdsBandwidthKernel<<<lds_blocks, 256>>>(sink);
  CK(hipEventRecord(b));
  CK(hipEventSynchronize(b));
  CK(hipEventElapsedTime(&ms, a, b));
  // 256 reads of a float4 per thread, per launch.
  const double lds_bytes =
      static_cast<double>(lds_blocks) * 256.0 * 256.0 * 16.0 * 20.0;
  std::printf("LDS read        16 KiB tile: %8.1f GB/s\n", lds_bytes / (ms * 1e-3) / 1e9);

  CK(hipFree(sink));
  return 0;
}
