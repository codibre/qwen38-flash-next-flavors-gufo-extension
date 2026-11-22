#!/usr/bin/env python3
"""Syntax-check the two patched translation units with the module build's own
flags, so this proves what the module build would compile. Caps itself."""
import json, subprocess, sys, shlex

DB = ("/home/tosol/projects/personal/qwen38-flash-next-gsq-gufo-extension/"
      "build-gufo/build/release/compile_commands.json")
WANT = ["src/model/ngram.cpp", "src/model/weights.cpp"]

db = json.load(open(DB))
by_src = {}
for e in db:
    by_src.setdefault(e["file"].replace("\\", "/"), e)

rc = 0
for want in WANT:
    hit = [e for f, e in by_src.items() if f.endswith(want)]
    if not hit:
        print(f"!! no compile_commands entry for {want}")
        rc = 1
        continue
    e = hit[0]
    cmd = e.get("command")
    if not cmd:
        cmd = " ".join(shlex.quote(a) for a in e["arguments"])
    argv = shlex.split(cmd)
    # swap the output-producing step for a syntax-only pass
    out = []
    skip = False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in ("-o", "-c"):
            if a == "-o":
                skip = True
            continue
        out.append(a)
    out += ["-fsyntax-only", "-Wno-error"]
    print(f"-- {want}")
    p = subprocess.run(out, cwd=e["directory"], capture_output=True, text=True)
    tail = (p.stderr or "").strip().splitlines()
    if p.returncode != 0:
        rc = 1
        print(f"   FAIL rc={p.returncode}")
        print("\n".join(tail[:40]))
    else:
        warn = [l for l in tail if "warning" in l]
        print(f"   OK rc=0 ({len(warn)} warning lines)")
        for l in warn[:8]:
            print("     " + l[:180])
sys.exit(rc)
