#!/usr/bin/env python3
"""ninfer_btop - a btop-style live dashboard for a ninfer inference engine.

Polls the engine's Prometheus metrics endpoint and renders:
  * scrolling bar graph: batch size  (ninfer_requests_running)
  * scrolling bar graph: prefill t/s (rate of ninfer_prefill_tokens_total)
  * scrolling bar graph: decode t/s  (rate of ninfer_generation_tokens_total)
  * totals: input (prompt) / output (generation) tokens
  * Resources and Cache (checkpoint reuse) panels, side by side:
    left = KV pool / state slots / host context gauges; right = "Tok hit"
    (token-level input hit) and "Req hit" (request-level hit) ratios,
    "State pool" pressure, a colored pool verdict, and two context-transfer
    bars on a shared scale: "Spill (d2h)" (device→host) vs "Recall (h2d)"
    (host→device), read per-direction from ninfer_context_transfer_bytes_total
  * request lanes: running / waiting / prefilling / decode-ready / paused
  * utilization at the bottom: (prefill time + decode time) / uptime,
    estimated from a startup benchmark of the engine

On startup the script benchmarks the engine: one large uncached prompt
(max prefill t/s) and an 8-way parallel burst (max decode t/s and the
effective max concurrency). Use --no-calib to skip the benchmark.

Standard library only (curses + urllib). No packages to install.

The cloud-API pricing baseline ($ per million tokens) is read from pricing.yaml
next to this script; select which model's row to use with --pricing-model
(default qwen3.8-27b). If pricing.yaml is missing, the built-in qwen3.8-27b
row is used instead.

Usage:
  python3 ninfer_btop.py --url URL [-i INTERVAL_MS] [--text] [--no-calib]
      [--burst N] [--pricing-model KEY] [--pricing-file PATH]

Keys in TUI mode:
  q / Esc  quit        s  pause/resume polling        r  reset graph auto-scale
"""

import argparse
import curses
import json
import os
import random
import signal
import sys
import threading
import time
import urllib.request
from collections import deque


# ---------------------------------------------------------------- metrics

def _split_labels(s):
    # Split a Prometheus label-group string on commas that are OUTSIDE
    # quoted values, so a comma inside a value cannot split a label.
    parts, cur, in_q = [], [], False
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if in_q:
            if ch == "\\" and i + 1 < n:          # keep escaped char pair
                cur.append(ch); cur.append(s[i + 1]); i += 2; continue
            if ch == '"':
                in_q = False
            cur.append(ch)
        else:
            if ch == '"':
                in_q = True
                cur.append(ch)
            elif ch == ",":
                parts.append("".join(cur)); cur = []
            else:
                cur.append(ch)
        i += 1
    if cur:
        parts.append("".join(cur))
    return parts


def _unescape_label(v):
    # Undo Prometheus label-value escaping in one left-to-right pass so
    # escapes are not double-processed:  backslash+backslash -> backslash,
    # backslash+quote -> quote, backslash+n -> newline.
    out, i, n = [], 0, len(v)
    while i < n:
        if v[i] == "\\" and i + 1 < n:
            nxt = v[i + 1]
            if nxt == "n":
                out.append("\n"); i += 2; continue
            if nxt == '"':
                out.append('"'); i += 2; continue
            if nxt == "\\":
                out.append("\\"); i += 2; continue
        out.append(v[i]); i += 1
    return "".join(out)


def fetch_metrics(url: str, timeout: float = 5.0) -> dict:
    """Fetch and parse a Prometheus text endpoint.

    Returns {name: value} for scalar series (unlabeled, or the first
    labeled series per name), plus {name + "__labels": {k: v}} for label
    context and {name + "__series": [(labels, value), ...]} holding every
    series of that name (needed to read per-direction counters such as the
    context-transfer bytes split by direction). Label values are unescaped
    and label groups are split on commas outside quoted values, so a comma,
    quote, backslash or newline inside a value survives intact.
    """
    with urllib.request.urlopen(url, timeout=timeout) as r:
        text = r.read().decode("utf-8", "replace")
    out: dict = {}
    series: dict = {}   # name -> [(labels_dict, value), ...] for EVERY series
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head, _, value = line.rpartition(" ")
        if not _:
            continue
        name, _, labels = head.partition("{")
        try:
            val = float(value)
        except ValueError:
            continue
        kv = {}
        if labels:
            for part in _split_labels(labels.rstrip("}")):
                k, _, v = part.partition("=")
                kv[k.strip()] = _unescape_label(v.strip().strip('"'))
        series.setdefault(name, []).append((kv, val))
        if name not in out:
            if labels:
                out[name + "__labels"] = kv
            out[name] = val
    for nm, lst in series.items():
        out[nm + "__series"] = lst
    return out


def g(m: dict, name: str, default=None):
    v = m.get(name)
    return v if isinstance(v, float) or isinstance(v, int) else default


def human(n, unit=""):
    for mult in (1e12, 1e9, 1e6, 1e3):
        if abs(n) >= mult:
            return f"{n / mult:.2f}{'T' if mult == 1e12 else 'G' if mult == 1e9 else 'M' if mult == 1e6 else 'K'}{unit}"
    return f"{n:.0f}{unit}"


# ---------------------------------------------------------------- startup calibration

# set by main() before the UI: {"ok": True, "concurrency", "prefill_tps",
# "decode_tps", "duration"} or {"ok": False, "error": ...}; None = not run
CALIB = None

# Cloud-API cost baseline, $ per million tokens. The per-model rates live in
# pricing.yaml next to this script (one row per model). The qwen3.8-27b row
# below is the built-in fallback used when pricing.yaml is absent, and is the
# source of truth for the CLOUD_* aliases so existing references keep
# resolving. Select a model's row at runtime with --pricing-model.
DEFAULT_PRICING = {
    "qwen3.8-27b": {"input": 0.5, "cached_input": 0.05, "output": 3.0},
}
_DEFAULT_ROW = DEFAULT_PRICING["qwen3.8-27b"]
CLOUD_IN_RATE = _DEFAULT_ROW["input"]           # new input tokens
CLOUD_CACHE_RATE = _DEFAULT_ROW["cached_input"]  # cached input tokens (cache hit = 10% of input)
CLOUD_OUT_RATE = _DEFAULT_ROW["output"]          # output tokens
CLOUD_LABEL = "cloud API"


def pricing_file_path():
    """Absolute path of pricing.yaml, located next to this script."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "pricing.yaml")


def load_pricing(path=None):
    """Load the cloud-API pricing baseline ($/M tokens) from a tiny 2-level YAML.

    Returns {model_key: {"input": float, "cached_input": float, "output": float}}.

    Hand-rolled parser (no external deps): skip blank lines and full-line
    ``#`` comments; a non-indented ``key:`` line starts a new model row; an
    indented ``field: value`` line (value a float) adds to the current row.
    If the file is missing/unreadable or yields no rows, fall back to the
    built-in DEFAULT_PRICING so the script and tests still work without the
    YAML. Importing this module never reads the file.
    """
    if path is None:
        path = pricing_file_path()
    rows, cur = {}, None
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return {k: dict(v) for k, v in DEFAULT_PRICING.items()}
    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            continue
        if line[:1].isspace():
            # indented "field: value" line adds a float to the current row
            if cur is None:
                continue
            field, sep, value = line.strip().partition(":")
            if not sep or not field.strip() or not value.strip():
                continue
            try:
                cur[field.strip()] = float(value.strip())
            except ValueError:
                pass
        else:
            # non-indented "key:" line starts a new model row
            key = line.split(":", 1)[0].strip()
            if not key:
                continue
            cur = {}
            rows[key] = cur
    if not rows:
        return {k: dict(v) for k, v in DEFAULT_PRICING.items()}
    return rows

_ALPHA = "abcdefghijklmnopqrstuvwxyz0123456789"


def money(v):
    """compact dollar formatter: $0.42 / $1,193 / $1.2M"""
    if v < 10:
        return f"${v:.2f}"
    if v < 1_000_000:
        return f"${v:,.0f}"
    return f"${v / 1e6:,.1f}M"


def _rand_lines(n, width=48):
    """n uncached random filler lines (~15 tokens each)."""
    return "\n".join(
        "ln%03d %s" % (i, "".join(random.choice(_ALPHA) for _ in range(width)))
        for i in range(n)
    )


def _chat_first_chunk(api, model, prompt, max_tokens, timeout=120):
    """Stream a chat completion to the end; return (monotonic time of the
    first content/reasoning chunk, total elapsed)."""
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0,
                       "stream": True}).encode()
    req = urllib.request.Request(api, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            if line[6:].strip() == "[DONE]":
                break
            try:
                o = json.loads(line[6:].strip())
            except ValueError:
                continue
            d = (o.get("choices") or [{}])[0].get("delta", {})
            if d.get("content") or d.get("reasoning_content"):
                if first is None:
                    first = time.monotonic()
    return first, time.monotonic() - t0


def calibrate(url, log=print, burst=8):
    """Benchmark the engine: max prefill t/s (one large uncached prompt) and,
    with `burst` parallel requests, the effective max concurrency and max
    decode t/s. Values are best-effort on a shared engine: if other users'
    traffic overlaps the test, the numbers drift. Returns a dict."""
    base = url.split("/metrics", 1)[0].rstrip("/")
    api = base + "/v1/chat/completions"
    m0 = fetch_metrics(url)
    model = (m0.get("ninfer_model_info__labels") or {}).get("model_name", "")
    if not model:
        with urllib.request.urlopen(base + "/v1/models", timeout=5) as r:
            model = json.load(r)["data"][0]["id"]

    t_start = time.monotonic()

    # phase A: max prefill t/s — one big uncached prompt, a couple decode steps
    log("  prefill: one large uncached prompt…")
    pre0 = g(fetch_metrics(url), "ninfer_prefill_tokens_total", 0.0)
    prompt_a = ("[calib %s] %s\nUsing the lines above as a glossary, write a short "
                "technical note on tree attention for speculative decoding."
                % (time.time_ns(), _rand_lines(250)))
    t_sent = time.monotonic()
    first_at, _ = _chat_first_chunk(api, model, prompt_a, 16)
    if first_at is None:
        raise RuntimeError("engine returned no tokens during the prefill test")
    pre1 = g(fetch_metrics(url), "ninfer_prefill_tokens_total", 0.0)
    prefill_tps = max(1.0, (pre1 - pre0) / (first_at - t_sent))

    # phase B: max decode t/s + max concurrency — `burst` parallel requests
    log(f"  decode: {burst} parallel requests…")
    gen0 = g(fetch_metrics(url), "ninfer_generation_tokens_total", 0.0)
    stop = threading.Event()
    samples, lock = [], threading.Lock()

    def sampler():
        while not stop.is_set():
            try:
                mm = fetch_metrics(url, timeout=1.5)
                with lock:
                    samples.append((time.monotonic(),
                                    g(mm, "ninfer_requests_running", 0.0),
                                    g(mm, "ninfer_generation_tokens_total", 0.0)))
            except Exception:
                pass
            time.sleep(0.05)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t_b = time.monotonic()

    def worker(n):
        try:
            p = ("[calib %s w%d] %s\nSummarize the glossary above in one sentence."
                 % (time.time_ns(), n, _rand_lines(25)))
            _chat_first_chunk(api, model, p, 240, timeout=90)
        except Exception:
            pass

    wts = [threading.Thread(target=worker, args=(n,), daemon=True) for n in range(burst)]
    for w in wts:
        w.start()
    for w in wts:
        w.join(timeout=45)
    stop.set()
    th.join(timeout=2)
    duration = time.monotonic() - t_start

    with lock:
        samples.sort(key=lambda x: x[0])
    concurrency = int(max((r for _, r, _ in samples), default=0.0))
    decode_tps = 0.0
    for (ta, _ra, ga), (tb, _rb, gb) in zip(samples, samples[1:]):
        dt = tb - ta
        if dt > 0.01 and gb >= ga:
            decode_tps = max(decode_tps, (gb - ga) / dt)
    if concurrency <= 0:
        concurrency = int(g(m0, "ninfer_max_concurrency", 0.0))
    if decode_tps <= 0:
        gen1 = g(fetch_metrics(url), "ninfer_generation_tokens_total", 0.0)
        decode_tps = max(1.0, (gen1 - gen0) / max(time.monotonic() - t_b, 0.1))

    return {"ok": True, "model": model, "concurrency": concurrency,
            "prefill_tps": prefill_tps, "decode_tps": decode_tps,
            "duration": duration}


def run_calibration(args):
    """Run the startup benchmark once; stores the result in CALIB."""
    global CALIB
    print("ninfer_btop: calibrating engine…", flush=True)
    try:
        CALIB = calibrate(args.url, log=lambda m: print(m, flush=True),
                          burst=args.burst)
    except Exception as e:
        CALIB = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        print(f"  calibration failed: {CALIB['error']} — utilization shows n/a", flush=True)
        return
    c = CALIB
    print(f"  done in {c['duration']:.1f}s: max concurrency {c['concurrency']}  ·  "
          f"prefill {c['prefill_tps']:,.0f} t/s  ·  decode {c['decode_tps']:,.0f} t/s",
          flush=True)


# ---------------------------------------------------------------- state

def cache_verdict(slots_used, slots_cap, xfer_rate, miss_ratio):
    """Classify on-device state-cache health from /metrics-only signals.

    Returns "n/a" (no pool / cache off), "ok", "tight", or "small" (the cache
    is too small). Three independent signals each add one point; two or more
    means the device state pool is almost certainly undersized:

      - the state pool pinned near capacity (finished-sequence checkpoints
        are being evicted rather than kept resident)
      - active host<->device state transfer (state constantly shuffled)
      - a high miss ratio (most input recompute instead of cache reuse)
    """
    if not slots_cap:
        return "n/a"
    signals = 0
    if slots_used / slots_cap >= 0.95:                  # pool pinned -> evictions
        signals += 1
    if xfer_rate is not None and xfer_rate > 0.05:      # host<->device state churn
        signals += 1
    if miss_ratio is not None and miss_ratio >= 0.6:    # most input recomputed
        signals += 1
    if signals >= 2:
        return "small"
    if signals == 1:
        return "tight"
    return "ok"


class Engine:
    def __init__(self, url, interval=0.5, rates=None, label=None):
        self.url = url
        self.interval = interval
        # display name for the "saved vs …" figure: the selected pricing
        # model key plus "Cloud API" (e.g. "qwen3.8-27b Cloud API"), or the
        # generic "cloud API" when no pricing model is supplied
        self.cloud_label = f"{label} Cloud API" if label else CLOUD_LABEL
        self.ok = True
        self.last_err = ""
        self.last_good = None
        self.model = ""
        self.start_time = None
        self.m = {}
        # cloud-API pricing row ($/M: input, cached_input, output); fall back
        # to the built-in qwen3.8-27b row when none is supplied. Accepts either
        # a {input, cached_input, output} dict or three floats in that order.
        if rates is None:
            rates = DEFAULT_PRICING["qwen3.8-27b"]
        elif not isinstance(rates, dict) and len(rates) == 3:
            rates = {"input": rates[0], "cached_input": rates[1],
                     "output": rates[2]}
        self.rates = {
            "input": float(rates.get("input", 0.0)),
            "cached_input": float(rates.get("cached_input", 0.0)),
            "output": float(rates.get("output", 0.0)),
        }
        # previous sample for inter-poll rates
        self._last_t = None
        self._last_gen = None
        self._last_prefill = None
        self._last_xfer = None
        self._last_spill = None
        self._last_recall = None

    def poll(self):
        try:
            m = fetch_metrics(self.url)
            self.ok = True
            self.last_err = ""
            self.last_good = time.time()
            lbl = m.get("ninfer_model_info__labels", {})
            self.model = lbl.get("model_name", lbl.get("model", ""))
            backend = lbl.get("speculative_backend", "")
            if backend and backend != "none":
                self.model = f"{self.model} (spec: {backend})"
            st = g(m, "ninfer_server_start_time_seconds")
            self.start_time = st
            now = time.monotonic()
            gen = g(m, "ninfer_generation_tokens_total", 0.0)
            pre = g(m, "ninfer_prefill_tokens_total", 0.0)
            # inter-poll rates: the engine commits decode tokens per decode step
            # and prefill tokens per prefill step, so both track the live phase
            # of each request (decode t/s stays 0 during the prefill phase)
            toks = None
            ptok = None
            xfer_rate = None
            spill_rate = None
            recall_rate = None
            xfer = g(m, "ninfer_context_transfer_seconds_total", 0.0)
            # per-direction context-transfer bytes: d2h = device->host spill,
            # h2d = host->device recall (summed across state / main_kv /
            # backend_kv resources)
            ser = m.get("ninfer_context_transfer_bytes_total__series") or []
            spill_tot = sum(v for (kv, v) in ser if kv.get("direction") == "d2h")
            recall_tot = sum(v for (kv, v) in ser if kv.get("direction") == "h2d")
            if self._last_gen is not None:
                dt = now - self._last_t
                if dt > self.interval * 0.5:
                    if gen >= self._last_gen:
                        toks = (gen - self._last_gen) / dt
                    if pre >= self._last_prefill:
                        ptok = (pre - self._last_prefill) / dt
                    if self._last_xfer is not None and xfer >= self._last_xfer:
                        xfer_rate = (xfer - self._last_xfer) / dt
                    if self._last_spill is not None and spill_tot >= self._last_spill:
                        spill_rate = (spill_tot - self._last_spill) / dt
                    if self._last_recall is not None and recall_tot >= self._last_recall:
                        recall_rate = (recall_tot - self._last_recall) / dt
            self._last_t, self._last_gen, self._last_prefill = now, gen, pre
            self._last_xfer = xfer
            self._last_spill, self._last_recall = spill_tot, recall_tot
            # lifetime utilization: estimated busy time (tokens / calibrated
            # max rates) over engine uptime. An underestimate by design:
            # cached prompt tokens and time below max rate are not counted.
            util = None
            if CALIB and CALIB.get("ok") and st:
                up = time.time() - st
                if up > 1.0:
                    busy = (pre / max(CALIB["prefill_tps"], 1.0)
                            + gen / max(CALIB["decode_tps"], 1.0))
                    util = min(1.0, max(0.0, busy / up))
            # lifetime money saved vs the cheapest cloud API (self-host ~ $0)
            in_tot = g(m, "ninfer_prompt_tokens_total", 0.0)
            in_cache = g(m, "ninfer_prompt_tokens_cached_total", 0.0)
            out_tot = g(m, "ninfer_generation_tokens_total", 0.0)
            cloud_cost = ((in_tot - in_cache) / 1e6 * self.rates["input"]
                          + in_cache / 1e6 * self.rates["cached_input"]
                          + out_tot / 1e6 * self.rates["output"])
            # cache reuse + is the on-device state cache too small?
            root = g(m, "ninfer_root_selections_total", 0.0)
            chkp = g(m, "ninfer_checkpoint_selections_total", 0.0)
            slots_used = g(m, "ninfer_device_state_used_slots", 0.0)
            slots_cap = g(m, "ninfer_device_state_capacity_slots", 0.0)
            tok_hit = (in_cache / in_tot) if in_tot > 0 else None
            tok_miss = (pre / in_tot) if in_tot > 0 else None
            req_hit = (chkp / (chkp + root)) if (chkp + root) > 0 else None
            pool_sat = (slots_used / slots_cap) if slots_cap > 0 else None
            cache_state = cache_verdict(slots_used, slots_cap, xfer_rate, tok_miss)
            self.m = m
            return {
                "batch": g(m, "ninfer_requests_running", 0.0),
                "toks_per_s": toks,
                "prefill_tps": ptok,
                "input_total": in_tot,
                "output_total": out_tot,
                "input_cached": in_cache,
                "prefill": g(m, "ninfer_prefill_tokens_total", 0.0),
                "cloud_cost": cloud_cost,
                "kv_used": g(m, "ninfer_device_kv_used_pages", 0.0),
                "kv_cap": g(m, "ninfer_device_kv_capacity_pages", 0.0),
                "slots_used": slots_used,
                "slots_cap": slots_cap,
                "host_used": g(m, "ninfer_host_context_used_bytes", 0.0),
                "host_cap": g(m, "ninfer_host_context_capacity_bytes", 0.0),
                "host_peak": g(m, "ninfer_host_context_peak_bytes", 0.0),
                "running": g(m, "ninfer_requests_running", 0.0),
                "waiting": g(m, "ninfer_requests_waiting", 0.0),
                "paused": g(m, "ninfer_requests_paused", 0.0),
                "prefilling": g(m, "ninfer_requests_prefilling", 0.0),
                "decode_ready": g(m, "ninfer_requests_decode_ready", 0.0),
                "ready": g(m, "ninfer_engine_ready", 0.0),
                "max_concurrency": g(m, "ninfer_max_concurrency", 0.0),
                "util": util,
                "tok_hit": tok_hit,
                "req_hit": req_hit,
                "pool_sat": pool_sat,
                "xfer_rate": xfer_rate,
                "spill_rate": spill_rate,
                "recall_rate": recall_rate,
                "cache_state": cache_state,
            }
        except Exception as e:
            self.ok = False
            self.last_err = f"{type(e).__name__}: {e}"
            return None


class Graph:
    """btop-style scrolling bar history. Newest sample on the left."""

    def __init__(self, width, floor=0.0):
        self.hist = deque(maxlen=width)
        self.scale = None  # None => autoscale to window max
        self.floor = floor  # autoscale never goes below this (e.g. max_concurrency)

    def add(self, v):
        self.hist.append(None if v is None else max(0.0, float(v)))

    def add_blank(self):
        self.hist.append(None)

    @property
    def values(self):
        return [v for v in self.hist if v is not None]

    def max_for_scale(self):
        vals = self.values
        m = max(vals) if vals else 0.0
        if self.scale is not None:
            return max(m, self.scale)
        return max(m, self.floor)

    def set_autoscale(self):
        self.scale = None

    def set_fixed_scale(self, v):
        self.scale = max(v, 1e-9)

    def bar_heights(self, rows):
        """Return per-column integer bar heights for `rows` tall plot area."""
        mx = self.max_for_scale()
        out = []
        for v in self.hist:
            if v is None or mx <= 0:
                out.append(0)
            else:
                out.append(int(round(rows * min(1.0, v / mx))))
        return out


# ---------------------------------------------------------------- text mode

def run_text(url, rates=None, label=None):
    eng = Engine(url, rates=rates, label=label)
    s = eng.poll()
    if s is None:
        print(f"error: cannot reach {url}: {eng.last_err}")
        return 1
    print(f"engine : {url}  ({'ready' if s['ready'] else 'not ready'})")
    if eng.model:
        print(f"model  : {eng.model}")
    print(f"batch  : running={s['running']:.0f} waiting={s['waiting']:.0f} "
          f"prefilling={s['prefilling']:.0f} decode_ready={s['decode_ready']:.0f}")
    print(f"input  : {human(s['input_total'])} tokens total "
          f"(cached {human(s['input_cached'])}, prefill {human(s['prefill'])})")
    print(f"output : {human(s['output_total'])} tokens total")
    u = s.get("util")
    if u is not None:
        print(f"util   : {u * 100:5.1f}%  (est. (prefill time + decode time) / uptime, from startup calibration)")
    else:
        print("util   : n/a  (benchmark skipped with --no-calib)")
    cc = s.get("cloud_cost")
    if cc is not None:
        print(f"saved  : ≈{money(cc)} vs {eng.cloud_label}  "
              f"({eng.rates['input']} $/M in, {eng.rates['cached_input']} $/M cached in, "
              f"{eng.rates['output']} $/M out)")
    kv = s['kv_used'] / s['kv_cap'] * 100 if s['kv_cap'] else 0
    print(f"kv     : {s['kv_used']:.0f}/{s['kv_cap']:.0f} pages ({kv:.1f}%)   "
          f"slots {s['slots_used']:.0f}/{s['slots_cap']:.0f}   "
          f"host ctx {human(s['host_used'])}/{human(s['host_cap'])}")
    _th, _rh, _ps = s.get("tok_hit"), s.get("req_hit"), s.get("pool_sat")
    _bits = []
    if _th is not None:
        _bits.append(f"input hit {_th * 100:.1f}%")
    if _rh is not None:
        _bits.append(f"request hit {_rh * 100:.1f}%")
    if _ps is not None:
        _bits.append(f"state pool {_ps * 100:.0f}%")
    _tag = {"ok": "ok", "tight": "TIGHT (pool near full)",
            "small": "TOO SMALL (raise --device-state-slots)"}.get(
                s.get("cache_state", "n/a"), "n/a (no on-device state cache)")
    print(f"cache  : {', '.join(_bits) if _bits else 'n/a'}   [{_tag}]")
    return 0


# ---------------------------------------------------------------- curses UI

def make_colors(scr):
    if not curses.has_colors():
        return {}
    curses.start_color()
    curses.use_default_colors()
    c = {}
    pairs = {
        "title": (curses.COLOR_CYAN, -1),
        "ok": (curses.COLOR_GREEN, -1),
        "bad": (curses.COLOR_RED, -1),
        "batch": (curses.COLOR_GREEN, -1),
        "toks": (curses.COLOR_YELLOW, -1),
        "label": (curses.COLOR_CYAN, -1),
        "dim": (curses.COLOR_WHITE, -1),
        "bar_dim": (curses.COLOR_BLACK, -1),
        "val": (curses.COLOR_WHITE, -1),
        "kv": (curses.COLOR_MAGENTA, -1),
        "peak": (curses.COLOR_GREEN, -1),
    }
    for i, (k, (fg, bg)) in enumerate(pairs.items(), start=1):
        try:
            curses.init_pair(i, fg, bg)
            c[k] = i
        except curses.error:
            pass
    return c


def cstr(scr, c, key, s):
    attr = curses.color_pair(c[key]) if key in c else 0
    return s, attr


def metric_stat_segments(graph, cur_str):
    """Banner suffix for a graph panel title: (cur=..,max=..,avg=..,min=..).

    Returns a list of (text, is_peak) segments, empty when there are no
    samples yet. The max= token is flagged is_peak so draw_panel paints it in
    the highlighted peak color. min/avg/max cover the whole plotted window;
    cur is the latest sample, passed in as cur_str.
    """
    vals = graph.values
    if not vals:
        return []
    mn, mx = min(vals), max(vals)
    av = sum(vals) / len(vals)
    return [
        (f"(cur={cur_str}", False),
        (f",max={mx:,.0f}", True),
        (f",avg={av:,.0f}", False),
        (f",min={mn:,.0f}", False),
        (")", False),
    ]


def _stat_attr(c, is_peak):
    """Attribute for a title-banner stat segment: peak value highlighted, rest dim."""
    if is_peak:
        return (curses.color_pair(c["peak"]) | curses.A_BOLD) if "peak" in c else curses.A_REVERSE
    return curses.color_pair(c.get("dim", 0))


def draw_panel(scr, y, title, key, c, width, stats=None):
    """Draw a panel title row.

    When ``stats`` (segments from ``metric_stat_segments``) is given, the
    suffix "─  (cur=..,max=..,avg=..,min=..)" is appended right after the
    title with the peak value highlighted; otherwise the title bar is dashed
    out to the full width as before.
    """
    w = width - 1  # leave the last column empty (see note in frame())
    if stats:
        suffix_len = 3 + sum(len(t) for t, _ in stats)  # "─  " + the stat text
        avail = max(0, w - 1 - suffix_len)  # title room after the leading "─"
        lead = "─" + title[:avail] + "─  "
        try:
            scr.addnstr(y, 0, lead[:w], w)
        except curses.error:
            pass
        x = len(lead[:w])
        for text, is_peak in stats:
            if x >= w:
                break
            n = min(len(text), w - x)
            if n <= 0:
                break
            try:
                scr.addnstr(y, x, text[:n], n, _stat_attr(c, is_peak))
            except curses.error:
                break
            x += n
    else:
        avail = w - 1  # columns after the leading "─"
        body = title[:avail]  # truncate the title only if it overflows
        scr.addnstr(y, 0, "─" + body + "─" * (avail - len(body)), w)
    return y + 1


def draw_split_panel(scr, y, left_title, right_title, c, width):
    """One title row holding two labeled columns, split near the middle.

    The left column occupies columns [0, split) and the right column
    [split, width-1). Lets the Resources and Cache panels run side by side in
    half the vertical space of two stacked panels. Returns (y + 1, split) so
    callers write their content rows into the two columns one row below the
    titles.
    """
    w = width - 1
    split = (w + 1) // 2
    for x0, x1, title in ((0, split, left_title), (split, w, right_title)):
        span = x1 - x0
        body = "─" + title
        txt = (body + "─" * max(0, span - len(body)))[:span]
        try:
            scr.addnstr(y, x0, txt, span)
        except curses.error:
            pass
    return y + 1, split


def _xfer_bar(rate, peak, width=12):
    """Small horizontal bar for a transfer rate, scaled to ``peak`` (both
    bytes/s). Filled cells over empty cells; all-empty for no data or a zero
    peak, so the two spill/recall bars share one scale and the longer bar is
    the dominant direction."""
    if rate is None or peak <= 0:
        return "░" * width
    filled = int(round(width * (rate / peak)))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def _rate_val(rate):
    """Human rate label for a transfer rate; an em dash when there is no data."""
    if rate is None:
        return "   —"
    return human(rate, "B/s")


def draw_bar_graph(scr, y, graph: Graph, rows, width, key, c):
    heights = graph.bar_heights(rows)
    hist_vals = list(graph.hist)
    peak_v = max((v for v in hist_vals if v is not None), default=0.0)
    plot_w = min(width - 14, len(heights))
    base_attr = curses.color_pair(c.get(key, 0))
    # window-max samples are highlighted (bold peak color, or reverse video
    # when the terminal has no colors)
    peak_attr = (curses.color_pair(c["peak"]) | curses.A_BOLD) if "peak" in c else curses.A_REVERSE
    for col in range(plot_w):
        h = heights[col]
        attr = peak_attr if (peak_v > 0 and hist_vals[col] == peak_v) else base_attr
        for r in range(rows):
            y_ = y + (rows - 1 - r)
            if r < h:
                ch = "█"
            elif r == h:
                ch = "▄" if h > 0 else " "
            else:
                continue
            try:
                scr.addstr(y_, 1 + col, ch, attr)
            except curses.error:
                pass
    # baseline
    try:
        scr.addnstr(y + rows, 1, "─" * plot_w, plot_w)
    except curses.error:
        pass
    # the cur/min/max/avg figures live in the panel title banner now
    return y + rows + 1


def frame(scr, eng, graphs, paused, c):
    scr.erase()
    try:
        max_y, max_x = scr.getmaxyx()
    except Exception:
        return
    width = max_x - 1

    s = eng.m and eng.last_frame
    model = eng.model or "?"
    if eng.start_time:
        up = time.time() - eng.start_time
        h_, rem = divmod(int(up), 3600)
        m_, sec = divmod(rem, 60)
        uptime = f"{h_:02d}:{m_:02d}:{sec:02d}"
    else:
        uptime = "?"
    status = "READY" if (eng.ok and s and s["ready"]) else ("DOWN" if not eng.ok else "BUSY")
    status_key = "ok" if status == "READY" else "bad"
    head = f"ninfer btop ─ {model} ─ engine: {status} ─ up {uptime} ─ {eng.url}"
    if paused:
        head += "  [paused]"
    try:
        # never fill the last column: a cursor parked at EOL puts ncurses in
        # pending-wrap state and its relative row moves land 1-2 rows low
        scr.addnstr(0, 0, head[:width - 1], width - 1, curses.color_pair(c.get("title", 0)))
    except curses.error:
        pass

    if CALIB and CALIB.get("ok"):
        cd = CALIB
        peak_attr = (curses.color_pair(c["peak"]) | curses.A_BOLD) if "peak" in c else curses.A_REVERSE
        lbl = curses.color_pair(c.get("label", 0))
        x = 0
        for txt, a in ((f"calibrated in {cd['duration']:.1f}s:  concurrency ", lbl),
                       (f"{cd['concurrency']}", peak_attr),
                       ("  ·  prefill ", lbl),
                       (f"{cd['prefill_tps']:,.0f} t/s", peak_attr),
                       ("  ·  decode ", lbl),
                       (f"{cd['decode_tps']:,.0f} t/s", peak_attr)):
            n = min(len(txt), width - x)
            if n <= 0:
                break
            try:
                scr.addnstr(1, x, txt[:n], n, a)
            except curses.error:
                break
            x += n
    elif CALIB and CALIB.get("error"):
        try:
            scr.addnstr(1, 0, f"calibration failed: {CALIB['error'][:width - 25]}  (util n/a)",
                        width - 1, curses.color_pair(c.get("dim", 0)))
        except curses.error:
            pass

    # first free row below the header block (header is row 0); the calibration
    # line (or its failed message) is row 1 and util now lives only in the footer
    if CALIB is not None:
        y = 2          # header + calibration / "calibration failed" line
    else:
        y = 1          # header only (--no-calib)
    if not eng.ok:
        # the banner sits below the header/calibration block and pushes the
        # panels down instead of sharing their rows
        try:
            scr.addnstr(y, 0, f"cannot reach engine: {eng.last_err}",
                        min(len(eng.last_err) + 20, width - 1), curses.color_pair(c.get("bad", 0)))
            scr.addnstr(y + 1, 0, f"last good sample: {eng.last_good}", width - 1)
        except curses.error:
            pass
        y += 2
    g_batch = graphs["batch"]
    g_toks = graphs["toks"]
    # fixed rows: header (2 if calibrated else 1) + 3 titles + 3 baselines
    # + 1 lane + 1 totals title + 2 totals + 1 shared Resources/Cache
    # title + 5 shared rows (incl. spill/recall bars) + 1 footer; the
    # cur/min/max/avg figures sit in the panel titles, plots take the rest
    fixed = 19 if CALIB is not None else 18
    plot_rows = max(1, min(6, (max_y - fixed) // 3))

    # --- batch panel
    b = (s or {}).get("running")
    cur_b = f"{b:.0f}" if b is not None else "—"
    y = draw_panel(scr, y, "Batch size  (ninfer_requests_running)", "batch", c, width,
                   metric_stat_segments(g_batch, cur_b))
    y = draw_bar_graph(scr, y, g_batch, plot_rows, width, "batch", c)
    if s:
        lane = (f"running {s['running']:.0f}  ·  waiting {s['waiting']:.0f}  ·  "
                f"prefilling {s['prefilling']:.0f}  ·  decode-ready {s['decode_ready']:.0f}  ·  "
                f"paused {s['paused']:.0f}")
        try:
            scr.addnstr(y, 1, lane[: width - 2], width - 2)
        except curses.error:
            pass
        y += 1

    # --- prefill tokens/s panel (prompt processing; decode t/s is flat
    #     during the prefill phase, this shows what the engine is doing then)
    g_pre = graphs.get("prefill")
    if g_pre is not None:
        p = (s or {}).get("prefill_tps")
        cur_p = f"{p:,.0f}" if p is not None else "—"
        y = draw_panel(scr, y, "Prefill t/s  (prompt processing)", "toks", c, width,
                       metric_stat_segments(g_pre, cur_p))
        y = draw_bar_graph(scr, y, g_pre, plot_rows, width, "toks", c)

    # --- decode tokens/s panel
    t = (s or {}).get("toks_per_s")
    cur_t = f"{t:,.0f}" if t is not None else "—"
    y = draw_panel(scr, y, "Decode t/s  (generation)", "toks", c, width,
                   metric_stat_segments(g_toks, cur_t))
    y = draw_bar_graph(scr, y, g_toks, plot_rows, width, "toks", c)

    # --- totals panel
    y = draw_panel(scr, y, "Token totals", "title", c, width)
    if s:
        lines = [
            f"  Input  tokens : {s['input_total']:>14,.0f}   (prefill {human(s['prefill'])}, cached from checkpoints {human(s['input_cached'])})",
            f"  Output tokens : {s['output_total']:>14,.0f}",
        ]
        for i, ln in enumerate(lines):
            try:
                scr.addnstr(y + i, 0, ln[:width - 1], width - 1, curses.color_pair(c.get("val", 0)))
            except curses.error:
                pass
        y += len(lines)
    else:
        try:
            scr.addnstr(y, 1, "waiting for first sample…", width - 2)
        except curses.error:
            pass
        y += 1

    # --- resources + cache, side by side: one shared title row + 5 content
    #     rows (Resources on the left, Cache on the right incl. the spill/recall
    #     transfer bars). Halves the vertical space the two panels used to need
    #     when stacked.
    if y + 6 <= max_y - 1:
        if s:
            kv_pct = f"{s['kv_used'] / s['kv_cap'] * 100:4.1f}%" if s["kv_cap"] else "  n/a"
            sl_pct = f"{s['slots_used'] / s['slots_cap'] * 100:4.1f}%" if s["slots_cap"] else "  n/a"
            host_pct = f"{s['host_used'] / s['host_cap'] * 100:4.1f}%" if s["host_cap"] else "  n/a"
            th, rh = s.get("tok_hit"), s.get("req_hit")
            in_line = (f"  Tok hit : {th * 100:5.1f}%" if th is not None
                       else "  Tok hit :   n/a  (no input yet)")
            req_line = (f"  Req hit : {rh * 100:5.1f}%" if rh is not None
                        else "  Req hit :   n/a  (no bindings yet)")
            pool = f"  State pool : {s['slots_used']:>4.0f}/{s['slots_cap']:.0f} {sl_pct}"
            sr, rr = s.get("spill_rate"), s.get("recall_rate")
            peak = max(sr or 0.0, rr or 0.0)   # shared scale: longer bar = dominant
            left = [f"  KV     : {s['kv_used']:>6,.0f}/{s['kv_cap']:,.0f}  {kv_pct}",
                    f"  Slots  : {s['slots_used']:>4.0f}/{s['slots_cap']:.0f}  {sl_pct}",
                    f"  Host   : {human(s['host_used'])}/{human(s['host_cap'])}  {host_pct}",
                    "", ""]
            right = [in_line, req_line, pool,
                     "  Spill  (d2h)  " + _xfer_bar(sr, peak) + "  " + _rate_val(sr),
                     "  Recall (h2d)  " + _xfer_bar(rr, peak) + "  " + _rate_val(rr)]
        else:
            left = ["  KV     : —", "  Slots  : —", "  Host   : —", "", ""]
            right = ["  Tok hit : —", "  Req hit : —", "  State pool : —",
                     "  Spill  (d2h)  " + _xfer_bar(None, 0) + "  —",
                     "  Recall (h2d)  " + _xfer_bar(None, 0) + "  —"]
            pool = ""
        y, split = draw_split_panel(scr, y, "Resources", "Cache (checkpoint reuse)", c, width)
        leftw = split - 1
        rightw = width - 1 - split
        for i in range(5):
            try:
                scr.addnstr(y + i, 0, left[i][:leftw], leftw)
                scr.addnstr(y + i, split, right[i][:rightw], rightw)
            except curses.error:
                pass
        if s:
            verdicts = {"ok": ("ok", "ok"),
                        "tight": ("tight — pool near full", "toks"),
                        "small": ("TOO SMALL — raise --device-state-slots", "bad"),
                        "n/a": ("n/a (no on-device state cache)", "dim")}
            vtxt, vkey = verdicts.get(s.get("cache_state", "n/a"),
                                      (s.get("cache_state", ""), "dim"))
            x = split + len(pool) + 1
            if x < width - 1:
                try:
                    scr.addnstr(y + 2, x, vtxt[:width - 1 - x], width - 1 - x,
                                curses.color_pair(c.get(vkey, 0)))
                except curses.error:
                    pass
        y += 5

    # --- footer (live utilization at the bottom)
    y = max_y - 1
    util = (s or {}).get("util")
    if util is not None:
        ukey = "ok" if util < 0.7 else ("toks" if util < 0.9 else "bad")
        utext = f" UTIL {util * 100:5.1f}%"
    else:
        ukey, utext = "dim", "UTIL n/a "
    _cc = (s or {}).get("cloud_cost")
    if _cc is not None:
        stext, skey = f" ≈{money(_cc)} saved vs {eng.cloud_label}", "ok"
    else:
        stext, skey = "", "dim"
    x = 0
    for txt, key in ((" q quit·s pause·r reset scale", "dim"),
                     (utext, ukey),
                     (stext, skey)):
        if x >= width:
            break
        n = min(len(txt), width - x - 1)
        if n <= 0:
            break
        try:
            scr.addnstr(y, x, txt[:n], n, curses.color_pair(c.get(key, 0)))
        except curses.error:
            break
        x += n
    scr.refresh()



def main_loop(stdscr, args):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)
    c = make_colors(stdscr)

    max_y, max_x = stdscr.getmaxyx()
    gwidth = max(20, max_x - 16)
    graphs = {"batch": Graph(gwidth), "toks": Graph(gwidth), "prefill": Graph(gwidth)}
    interval = args.interval / 1000.0
    eng = Engine(args.url, interval, args.pricing_rates, label=args.pricing_model)
    paused = False
    last_poll = 0.0
    running = True

    while running:
        now = time.time()
        if not paused and now - last_poll >= interval:
            last_poll = now
            s = eng.poll()
            eng.last_frame = s
            if s is not None:
                if s["max_concurrency"] > 0:
                    graphs["batch"].floor = s["max_concurrency"]
                graphs["batch"].add(s["running"])
                graphs["toks"].add(s["toks_per_s"])
                graphs["prefill"].add(s["prefill_tps"])
            else:
                graphs["batch"].add_blank()
                graphs["toks"].add_blank()
                graphs["prefill"].add_blank()

        # resync graph width on terminal resize
        ny, nx = stdscr.getmaxyx()
        if abs(nx - 16 - gwidth) > 2:
            gwidth = max(20, nx - 16)
            for k in graphs:
                old = list(graphs[k].hist)
                graphs[k] = Graph(gwidth)
                graphs[k].hist.extend(old[-gwidth:])

        frame(stdscr, eng, graphs, paused, c)

        ch = stdscr.getch()
        if ch == -1:
            # small sleep so we don't spin when no key
            time.sleep(0.05)
            continue
        if ch in (ord("q"), ord("Q"), 27):
            running = False
        elif ch == ord("s"):
            paused = not paused
        elif ch == ord("r"):
            graphs["batch"].set_autoscale()
            graphs["toks"].set_autoscale()

    return 0


def main():
    ap = argparse.ArgumentParser(description="btop-style dashboard for a ninfer engine")
    ap.add_argument("--url", required=True, help="metrics endpoint (required, e.g. http://host:port/metrics)")
    ap.add_argument("-i", "--interval", type=int, default=500, help="poll interval in ms (default 500)")
    ap.add_argument("--text", action="store_true", help="one-shot plain-text stats, no TUI")
    ap.add_argument("--no-calib", action="store_true",
                    help="skip the startup benchmark (utilization shows n/a)")
    ap.add_argument("--burst", type=int, default=8,
                    help="parallel requests for the calibration burst (default 8)")
    ap.add_argument("--pricing-model", default="qwen3.8-27b",
                    help="model key in pricing.yaml selecting the cloud-API pricing "
                         "row (default qwen3.8-27b)")
    ap.add_argument("--pricing-file", default=None,
                    help="path to a pricing YAML file "
                         "(default: pricing.yaml next to the script)")
    args = ap.parse_args()

    pricing = load_pricing(args.pricing_file)
    if args.pricing_model not in pricing:
        print(f"error: unknown --pricing-model {args.pricing_model!r}; "
              f"available keys: {', '.join(sorted(pricing))}", file=sys.stderr)
        sys.exit(2)
    args.pricing_rates = pricing[args.pricing_model]

    if not args.no_calib:
        run_calibration(args)
    if args.text:
        sys.exit(run_text(args.url, args.pricing_rates, args.pricing_model))
    signal.signal(signal.SIGTTOU, signal.SIG_IGN)  # like top: safe even as a background job
    try:
        curses.wrapper(main_loop, args)
    except curses.error as e:
        print(f"curses error (run inside a real terminal?): {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
