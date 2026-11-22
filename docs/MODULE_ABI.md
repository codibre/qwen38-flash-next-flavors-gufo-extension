# Module ABI rules for Gufo extensions

Two rules cost us a day of debugging in April. Both apply to **any extension that
forks a built-in Gufo model**, and neither is stated in the SDK headers. Written
here so the next person does not rediscover them.

## 1. Do not keep a built-in namespace

This tree is a fork of gufo's `src/models/qwen38_flash_next` and originally kept
the namespace `gufo::models::qwen38_flash_next`.

**What happened.** The `gufo` executable exports `_ZN4gufo*`, and dlopens modules
with `RTLD_NOW | RTLD_LOCAL` — no `RTLD_DEEPBIND`. So the module's own
`TextProvider` vtable pointers bound to the *executable's* copies of
`ProbeFlashNextGguf` and `create`, not to the module's. The host probe does not
claim GSQ coder artifacts, so `ext::ProbeTextProvider` returned null, and serve
failed with:

```text
Failed to open GGUF: Invalid GGUF tensor shape or unsupported storage type: blk.1.ffn_down_exps.weight
```

A reader-shaped message about a reader that had opened the file correctly. The
tensor is fine; the provider was never ours.

**How it was localised.** `dladdr()` on the function pointer the module had
registered. It resolved to the `gufo` binary rather than to the module's `.so`,
which is only possible if the symbol was interposed at load time.

**The fix that works.** `extension.cmake` compiles the module with
`-fvisibility=hidden -fvisibility-inlines-hidden`. Hidden definitions are not
interposable, so the module binds its own copies, while host services still bind
out of the executable. Verified on the installed module:

```sh
nm -D <module>.so | grep -c ' T _ZN4gufo6models'   # 0
```

**What does not work.** `-Bsymbolic` from `extension.cmake`: the final module
target is generated *after* the included file runs, so link options added there
do not reach it.

**The fix Gufo can enforce.** Rename the forked namespace. An extension that
keeps the built-in name is one `dlopen` away from silently running host code.

## 2. Declare a scheduling profile

Since API 0.3
(`include/gufo/ext/changes/0.3/text-scheduling-profile-required.md`) the host
refuses a runner whose `TextRunnerDescriptor::scheduling` has a zero field:

```text
text runner declares no scheduling profile (TextRunnerDescriptor::scheduling)
```

This engine arms no layer-yield hook and captures one checkpoint per pass, so it
declares the unsplit profile the migration note prescribes:

```cpp
.scheduling = gufo::ext::TextSchedulingProfile{
    .pass_tokens = 512, .max_layer_groups = 1, .max_yield_decode_width = 1},
```

Whether a different profile would help prefill or decode has not been measured.
