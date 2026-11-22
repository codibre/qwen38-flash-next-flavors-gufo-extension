#!/usr/bin/env python3
"""Read a GGUF header (metadata + tensor-info list) with a HARD byte cap.

This never touches tensor payload: it reads at most --cap bytes from the head
of the file, parses KV metadata and the tensor-info table, prints a summary and
exits. Memory and time are bounded by the cap, not by model size, so it is safe
next to a resident slot. No GPU, no engine, no llama-swap call.
"""
import argparse
import struct
import sys
from collections import Counter

# GGUF v3 scalar types (ggml.h / gguf.spec).
T_UINT8, T_INT8, T_UINT16, T_INT16 = 0, 1, 2, 3
T_UINT32, T_INT32, T_FLOAT32, T_BOOL = 4, 5, 6, 7
T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 8, 9, 10, 11, 12

# GGML storage types, ids copied from gufo's GgmlType (src/core/gguf_reader.hpp)
# plus the extension-local compat id 42 (quant_ids.hpp: GSQ stores Q2_0 there).
# (name, block, block_bytes, accepted_by_the_binder's FormatOf)
GGML_TYPE_SIZE = {
    0: ("F32", 1, 4, True),
    1: ("F16", 1, 2, True),
    2: ("Q4_0", 32, 18, False),
    3: ("Q4_1", 32, 20, False),
    6: ("Q5_0", 32, 24, False),
    7: ("Q5_1", 32, 24, True),
    8: ("Q8_0", 32, 34, True),
    9: ("Q8_1", 32, 40, False),
    10: ("Q2_K", 256, 84, False),
    11: ("Q3_K", 256, 104, False),
    12: ("Q4_K", 256, 144, True),
    13: ("Q5_K", 256, 176, True),
    14: ("Q6_K", 256, 210, True),
    15: ("Q8_K", 256, 136, False),
    16: ("IQ2_XXS", 256, 66, False),
    17: ("IQ2_XS", 256, 74, False),
    18: ("IQ3_XXS", 256, 98, True),
    19: ("IQ1_S", 256, 52, False),
    20: ("IQ4_NL", 32, 18, True),
    21: ("IQ3_S", 256, 110, True),
    22: ("IQ2_S", 256, 82, True),
    23: ("IQ4_XS", 256, 136, True),
    24: ("IQ1_M", 256, 56, False),
    26: ("I32", 1, 4, False),
    27: ("I16", 1, 2, False),
    28: ("I8", 1, 1, False),
    29: ("IQ1_M?", 256, 56, False),
    30: ("BF16", 1, 2, True),
    34: ("F64", 1, 8, False),
    42: ("Q2_0(compat)", 64, 18, True),
}



class Cursor:
    def __init__(self, buf):
        self.b = buf
        self.i = 0

    def take(self, n):
        if self.i + n > len(self.b):
            raise EOFError("header larger than the read cap")
        v = self.b[self.i:self.i + n]
        self.i += n
        return v

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        n = self.u64()
        if n > 1 << 20:
            raise ValueError("absurd string length")
        return self.take(n).decode("utf-8", "replace")


def skip_value(c, t, depth=0):
    if depth > 4:
        raise ValueError("array nesting too deep")
    if t in (T_UINT8, T_INT8, T_BOOL):
        c.take(1)
    elif t in (T_UINT16, T_INT16):
        c.take(2)
    elif t in (T_UINT32, T_INT32, T_FLOAT32):
        c.take(4)
    elif t in (T_UINT64, T_INT64, T_FLOAT64):
        c.take(8)
    elif t == T_STRING:
        c.string()
    elif t == T_ARRAY:
        et = c.u32()
        n = c.u64()
        if n > 1 << 22:
            raise ValueError("absurd array length")
        for _ in range(n):
            skip_value(c, et, depth + 1)
    else:
        raise ValueError(f"unknown gguf value type {t}")


def kv_scalar(c, t):
    if t == T_STRING:
        return c.string()
    fmt = {T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h",
           T_UINT32: "<I", T_INT32: "<i", T_FLOAT32: "<f", T_UINT64: "<Q",
           T_INT64: "<q", T_FLOAT64: "<d", T_BOOL: "<?"}.get(t)
    if fmt is None:
        skip_value(c, t)
        return "<skipped>"
    n = struct.calcsize(fmt)
    return struct.unpack(fmt, c.take(n))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--cap", type=int, default = 64 << 20,
                    help="max bytes to read (default 64 MiB)")
    ap.add_argument("--show", default="ngram,ple,expert,moe",
                    help="comma list of substrings: print every KV whose key "
                         "or value matches")
    ap.add_argument("--tensor", action="append", default=[],
                    help="print the full info for a tensor name (repeatable)")
    ap.add_argument("--top", type=int, default=12,
                    help="list the N largest tensors by encoded size")
    args = ap.parse_args()

    with open(args.path, "rb") as f:
        head = f.read(args.cap)
    if head[:4] != b"GGUF":
        sys.exit(f"{args.path}: not a GGUF")
    c = Cursor(head)
    c.take(4)
    version = c.u32()
    n_tensors = c.u64()
    n_kv = c.u64()
    print(f"GGUF v{version}: {n_tensors} tensors, {n_kv} kv pairs "
          f"(read {len(head) / 1e6:.1f} MB of header)")

    meta = {}
    for _ in range(n_kv):
        k = c.string()
        t = c.u32()
        if t == T_ARRAY:
            et, n = c.u32(), c.u64()
            vals = [kv_scalar(c, et) for _ in range(n)]
            meta[k] = vals if n <= 12 else f"[{n} x {et}]"
        else:
            meta[k] = kv_scalar(c, t)
    print(f"header consumed to byte {c.i} ({c.i / 1e6:.1f} MB)")

    keys = [k for k in meta if any(s in k.lower() for s in
                                   [x.strip().lower() for x in args.show.split(",") if x.strip()])]
    for k in sorted(set(list(keys) + [
            "general.architecture", "general.name", "general.quantization_version",
            "qwen4exp.quantized_type_size"])):
        if k in meta:
            print(f"  {k} = {meta[k]}")

    infos = []
    for _ in range(n_tensors):
        name = c.string()
        nd = c.u32()
        dims = [c.u64() for _ in range(nd)]
        tid = c.u32()
        off = c.u64()
        infos.append((name, dims, tid, off))
    print(f"tensor-info table ends at byte {c.i} ({c.i / 1e6:.1f} MB)")

    def geom(tid):
        return GGML_TYPE_SIZE.get(tid, (f"raw#{tid}", 0, 0, False))

    def enc(dims, tid):
        _, blk, bts, _ = geom(tid)
        n = 1
        for d in dims:
            n *= d
        return (n // blk) * bts if blk else 0

    by_type = Counter()
    bytes_by_type = Counter()
    ok_by_name = {}
    for name, dims, tid, off in infos:
        nm, _, _, ok = geom(tid)
        by_type[nm] += 1
        bytes_by_type[nm] += enc(dims, tid)
        ok_by_name[nm] = ok
    print("tensors by storage type (REJECT = the binder's FormatOf has no "
          "geometry for it, so the artifact will not load):")
    total = 0
    for nm, cnt in by_type.most_common():
        total += bytes_by_type[nm]
        print(f"  {nm:14s} {cnt:6d} tensors {bytes_by_type[nm] / 1e9:9.3f} GB"
              f"{'  REJECT' if not ok_by_name[nm] else ''}")
    print(f"  {'TOTAL':14s} {len(infos):6d} tensors {total / 1e9:9.3f} GB")

    print(f"largest tensors:")
    for name, dims, tid, off in sorted(infos, key=lambda i: -enc(i[1], i[2]))[:args.top]:
        nm, blk, bts, ok = geom(tid)
        print(f"  {enc(dims, tid) / 1e9:9.3f} GB  {name:40s} dims={dims} "
              f"type={tid}({nm}){' REJECT' if not ok else ''}")

    # Group by tensor family (layer index folded away) so the routed vs dense
    # byte split is readable at a glance.
    fam = {}
    for name, dims, tid, off in infos:
        parts = name.split(".")
        parts = ["*" if p.startswith("blk") or p.isdigit() else p for p in parts]
        key = ".".join(parts)
        nm, _, _, _ = geom(tid)
        e = fam.setdefault(key, [0, 0, Counter()])
        e[0] += 1
        e[1] += enc(dims, tid)
        e[2][nm] += 1
    print("bytes by tensor family (top 14):")
    for key, (cnt, byts, types) in sorted(fam.items(), key=lambda kv: -kv[1][1])[:14]:
        tstr = ",".join(f"{k}x{v}" for k, v in types.most_common())
        print(f"  {byts / 1e9:9.3f} GB {cnt:5d}  {key:38s} [{tstr}]")

    for want in args.tensor:
        hits = [i for i in infos if want in i[0]]
        if not hits:
            print(f"  !! no tensor matching '{want}'")
        for name, dims, tid, off in hits[:8]:
            nm, blk, bts, ok = geom(tid)
            print(f"  {name}: dims={dims} type={tid}({nm}) "
                  f"block={bts}B/{blk}el offset={off} "
                  f"row_bytes={(dims[0] // blk) * bts if blk else '?'}"
                  f"{' REJECT' if not ok else ''}")
    print(f"OK: {args.path}")


if __name__ == "__main__":
    main()
