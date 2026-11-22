#!/usr/bin/env python3
"""Opcode mix of a HIP kernel, read from the shipped code object.

Port of the ISA-mix check the ds41 notes credit with finding the quarter-rate
`v_ldexp_f32` decode pathology. Their `tools/prof/isa_mix.py` is not in this
checkout, so this carves the GCN code objects straight out of a host object's
`.hip_fatbin` (roc-obj-extract is deprecated and rejects fatbin-bearing
objects) and disassembles them with llvm-objdump.

  tools/prof/isa_mix.py <kernels.hip.cpp.o> [--find Q2_0] [--top 15]

Reports per-kernel opcode counts plus the ratios that matter on gfx1151:
quarter-rate transcendentals, per-element selects, narrow (sub-32-bit) global
loads, and LDS ops.
"""
import argparse
import collections
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import os

SLOW = {
    "v_ldexp_f32": "quarter-rate transcendental (2-bit/8-bit exponent decode)",
    "v_exp_f32": "quarter-rate exp",
    "v_rcp_f32": "quarter-rate reciprocal",
    "v_div_f32": "quarter-rate divide",
    "v_cndmask_b32": "select (sign ternary pattern)",
    "v_cmp_lt_f32": "compare feeding a select",
    "global_load_ushort": "narrow 16-bit global load (fragmented weights)",
    "global_load_uchar": "narrow 8-bit global load",
    "global_load_ushort": "narrow 16-bit global load",
    "ds_read_u16": "narrow LDS read (bank-conflict prone)",
    "ds_read_b16": "narrow LDS read",
    "ds_write_b16": "narrow LDS write",
    "v_permt2_b32": "byte permute",
    "v_perm_b32": "byte permute",
    "v_readfirstlane_b32": "SGPR move (loop-carried pressure)",
}

OBJDUMP_CANDIDATES = [
    shutil.which("llvm-objdump"),
    "/opt/rocm/llvm/bin/llvm-objdump",
]


def objdump():
    for c in OBJDUMP_CANDIDATES:
        if c and os.path.exists(c):
            return c
    sys.exit("llvm-objdump not found (rocm-llvm package)")


def carve_code_objects(path):
    """Return GCN ELF blobs embedded in a host object's fatbin container."""
    data = open(path, "rb").read()
    out = []
    for m in re.finditer(b"\x7fELF", data):
        off = m.start()
        if off + 64 > len(data):
            continue
        e_machine = struct.unpack_from("<H", data, off + 18)[0]
        if e_machine != 224:  # EM_AMDGPU
            continue
        # Extend to the section-table end; good enough for disassembly.
        e_shoff = struct.unpack_from("<Q", data, off + 40)[0]
        e_shentsize = struct.unpack_from("<H", data, off + 58)[0]
        e_shnum = struct.unpack_from("<H", data, off + 60)[0]
        end = min(len(data), off + e_shoff + e_shentsize * max(e_shnum, 1))
        out.append(data[off:end])
    return out


def disassemble(blob, od):
    with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=False) as f:
        f.write(blob)
        name = f.name
    try:
        r = subprocess.run([od, "-d", name], capture_output=True, text=True,
                           timeout=900)
        if r.returncode != 0:
            sys.exit(f"llvm-objdump failed: {r.stderr[:400]}")
        return r.stdout
    finally:
        os.unlink(name)


def split_functions(text):
    """{symbol: [opcode, ...]} from objdump output."""
    funcs, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.+)>:$", line.strip())
        if m:
            cur = m.group(1)
            funcs[cur] = []
            continue
        if cur is None:
            continue
        # llvm-objdump on GCN prints "<TAB>opcode operands  // ADDR: raw"
        # with no address prefix, unlike the x86 output.
        m = re.match(r"^\s+([a-z][a-z0-9_.]*)", line)
        if m:
            funcs[cur].append(m.group(1))
    return {k: v for k, v in funcs.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obj")
    ap.add_argument("--find", default="", help="only symbols containing this")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    blobs = carve_code_objects(args.obj)
    if not blobs:
        sys.exit("no EM_AMDGPU code object found in " + args.obj)
    print(f"{len(blobs)} code object(s) carved")
    od = objdump()
    for i, blob in enumerate(blobs):
        funcs = split_functions(disassemble(blob, od))
        picked = {k: v for k, v in funcs.items() if args.find in k} or funcs
        print(f"\n=== code object {i}: {len(funcs)} kernels "
              f"({len(picked)} matching '{args.find or '*'}') ===")
        for name, ops in sorted(picked.items(),
                                key=lambda kv: -len(kv[1]))[:8]:
            c = collections.Counter(ops)
            total = len(ops)
            short = name if len(name) < 96 else name[:93] + "..."
            print(f"\n{short}\n  {total} instructions")
            print("  top: " + ", ".join(f"{o}×{n}" for o, n in c.most_common(args.top)))
            hits = []
            for op, why in SLOW.items():
                n = sum(v for k, v in c.items() if op in k)
                if n:
                    hits.append(f"    {op:22s} {n:5d}  ({100.0*n/total:4.1f}%) — {why}")
            if hits:
                print("  slow/narrow:")
                print("\n".join(hits))
            else:
                print("  slow/narrow: none")


if __name__ == "__main__":
    main()
