# Qwen3.8 Flash-Next flavors: the pruned ("GSQ") GGUF checkpoints whose
# routed experts use codebook quantizations the Flash-Next runtime does not
# bind (IQ3_S / IQ2_S / IQ4_XS / IQ3_XXS) and whose PLE table may be Q5_1.
# Everything the runtime already understands — container, config, tokenizer,
# attention, shared experts, sampler, speculative engine — is imported from
# the qwen38_flash_next extension; this module adds only the flavor binding,
# the weight source, the MoE stage override and the routed WMMA GEMMs that
# read the codebook blocks.

# Host code: the flavor binding to Flash-Next's weight descriptors and the
# artifact identity.
gufo_extension_target(
  NAME gufo_qwen38_flash_next_flavors
  SOURCES
    extension.cpp
    gsq/binder.cpp
    gsq/identity.cpp
  LINK_LIBRARIES
    gufo::qwen38_flash_next
    gufo_ext_core
    gufo_ext_runtime
)

if(TARGET gufo_qwen38_flash_next_flavors)
  target_include_directories(gufo_qwen38_flash_next_flavors PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
endif()

# The providers: the stages, the weight source and the routed WMMA GEMMs
# Flash-Next's runner takes, built with Flash-Next's HIP options.
gufo_extension_target(
  NAME gufo_qwen38_flash_next_flavors_hip
  SOURCES
    gsq/provider.cpp
    gsq/stages.cpp
    gsq/weight_source.cpp
  LINK_LIBRARIES
    gufo_qwen38_flash_next_flavors
    gufo::qwen38_flash_next_hip
    gufo::qwen_family
    gufo_ext_runtime
    hip::host
)

if(TARGET gufo_qwen38_flash_next_flavors_hip)
  gufo_extension_hip_sources(gufo_qwen38_flash_next_flavors_hip
    SOURCES
      gsq/kernels/routed_wmma.hip.cpp
      gsq/kernels/rows.hip.cpp
      gsq/kernels/nan_probe.hip.cpp
  )
  target_compile_definitions(gufo_qwen38_flash_next_flavors_hip PRIVATE
    ENGINE_ENABLE_HIP=1)
  target_compile_options(gufo_qwen38_flash_next_flavors_hip PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
    $<$<COMPILE_LANGUAGE:HIP>:-fno-finite-math-only>
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-command-line-argument>
  )
endif()

# gufo_ext_runtime only declares the public services; these link their
# implementations into executables that stand apart from `gufo`.
set(_qwen38_flash_next_flavors_test_libs gufo_ext_core gufo_ext_cache)
if(ENGINE_ENABLE_HIP)
  list(APPEND _qwen38_flash_next_flavors_test_libs gufo_ext_hip)
endif()

# TODO(PR): port gsq_bind_test / gsq_routed_wmma_test to the migrated
# API (bind via GsqWeightSource; kernels via gsq/kernels/routed_wmma.hpp)
# before merge.

install(FILES
    "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
    "${GUFO_EXTENSION_BUILD_ROOT}/NOTICE.md"
  DESTINATION share/licenses/gufo/extensions/qwen38_flash_next_flavors)
