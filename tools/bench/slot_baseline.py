#!/usr/bin/env python3
"""Real prefill/decode baseline for one llama-swap slot.

Reads gufo's own `timings` block from /v1/completions — no stopwatch guessing.
All model traffic goes through llama-swap (127.0.0.1:8642); this never touches
the engine directly.

  tools/bench/slot_baseline.py advanced
  tools/bench/slot_baseline.py heretic --prefill-tokens 4000 --decode 128
"""
import argparse
import json
import subprocess
import sys

BASE = "http://127.0.0.1:8642"


def post(path, payload, timeout=1200):
    r = subprocess.run(
        ["curl", "-s", f"{BASE}{path}", "-H", "content-type: application/json",
         "-d", json.dumps(payload)],
        capture_output=True, text=True, timeout=timeout)
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        raise SystemExit(f"bad response from {path}: {r.stdout[:400]}")


def complete(slot, prompt, max_tokens, extra=None):
    # Chat, not raw completion: these are instruct models and answer a bare
    # completion with an immediate EOS (0 generated tokens, finish=stop).
    body = {"model": slot, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": False}
    body.update(extra or {})
    j = post("/v1/chat/completions", body)
    if "timings" not in j:
        raise SystemExit(f"no timings (error?): {json.dumps(j)[:400]}")
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slot")
    ap.add_argument("--prefill-tokens", type=int, default=2000)
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    # Loading is llama-swap's job: it queues the first request behind the
    # model load. This build has no /load or /started endpoint (both 404), so
    # readiness is proven by a real completion answering, retries included.
    import time
    t_load = None
    for attempt in range(6):
        t0 = time.perf_counter()
        try:
            complete(args.slot, "Reply with the single word: ready", 4)
            t_load = time.perf_counter() - t0
            break
        except SystemExit as e:
            print(f"  warmup attempt {attempt + 1} failed: {e}", flush=True)
            time.sleep(10)
    else:
        raise SystemExit(f"slot {args.slot} never answered")

    unit = ("The registration module validates each submission before writing "
            "it to the ledger. Auditors compare the ledger against the archive "
            "and flag any drift between the two records. ")
    approx = 18  # tokens per unit sentence, measured below
    prompt = unit * max(1, args.prefill_tokens // approx)

    # gufo keeps a prefix cache, so repeats of one prompt measure nothing
    # after the first hit. Each repeat gets its own unique head.
    pre, pt = [], 0
    for i in range(args.repeats):
        head = (f"Bench pass {i} reference id {1000 + i * 7919}. "
                "Ignore this id; answer the question at the end.\n")
        short = complete(args.slot, head + prompt, 1)
        ps = short["timings"]["prompt_per_second"]
        cached = short["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
        if cached:
            print(f"  pass {i}: {cached} cached tokens, skipped")
            continue
        pre.append(ps)
        pt = short["usage"]["prompt_tokens"]

    longj = complete(args.slot, "Bench decode run. " + prompt, args.decode)
    t = longj["timings"]
    u = longj["usage"]
    if not u.get("completion_tokens"):
        print(f"  decode returned no tokens; raw: {json.dumps(longj)[:500]}")
    print(f"slot={args.slot}  first_response={t_load:.1f}s (includes any load)")
    print(f"  prompt_tokens={pt}  completion_tokens={u['completion_tokens']}")
    if pre:
        print(f"  prefill : {min(pre):7.1f} .. {max(pre):7.1f} tok/s "
              f"(last long-run {t['prompt_per_second']:.1f}, "
              f"{len(pre)} uncached passes)")
    else:
        print(f"  prefill : all passes were cache hits")
    pps = t.get("predicted_per_second") or 0.0
    print(f"  decode  : {pps:7.1f} tok/s "
          f"({(1000.0 / pps) if pps else 0:.2f} ms/token)")
    g = j_usage = u
    if u.get("draft_tokens"):
        acc = u.get("draft_tokens_accepted", 0) / u["draft_tokens"]
        print(f"  MTP     : {u['draft_tokens_accepted']}/{u['draft_tokens']} "
              f"accepted ({acc:.0%}) — decode is draft-assisted")
    gu = u.get("gufo", {})
    if gu:
        print(f"  ttft    : {gu.get('ttft_ms', 0):.0f} ms   "
              f"mean_inter_token {gu.get('mean_inter_token_ms', 0):.2f} ms   "
              f"prefill_chunks {gu.get('prefill_chunks')}")
    print(f"  note    : ms/token covers routed experts + attention + shared "
          f"MLP + sampler, not one kernel")


if __name__ == "__main__":
    main()
