#!/usr/bin/env python3
"""ninfer_btop - a btop-style live dashboard for a ninfer inference engine.

Polls the engine's Prometheus metrics endpoint and renders:
  * scrolling bar graph: batch size  (ninfer_requests_running)
    (height follows the engine's max concurrency: one braille character
    resolves 4 dot levels = 4 concurrent requests, so the panel is
    ceil(max_concurrency/4) chars high. The scale is the panel's total
    dot count (4 or 8, from the height) - one dot level is always one
    request, and overload clips at the top; for ninfer's cap of 8 the
    panel is 1 or 2 chars high)
  * scrolling bar graph: prefill t/s (rate of ninfer_prefill_tokens_total)
  * scrolling bar graph: decode t/s  (rate of ninfer_generation_tokens_total)
    (btop's braille glyphs: two samples per character - each sample is a
    single vertical dot line, the newer sample in the left dot column, so
    a character may be a rising/falling glyph - and four vertical levels
    per text line; the graphs scroll left to right: the newest sample
    enters at the left edge, and each poll shifts every dot line by
    exactly half a character, so the motion can never jump)
  * totals: input (prompt) / output (generation) tokens
  * Device KV / Host KV / Cache (checkpoint reuse): three side-by-side
    panels sharing one title row (btop-style). The KV columns show the
    engine's unified context pool per domain (engine >= 2026-10-23 state/KV
    unification) — capacity, in use (device only), cached and lifetime peak
    bytes, read from ninfer_context_pool_{bytes,live_bytes,peak_bytes}; the
    cache panel holds "Tok hit" (token-level input hit) and "Req hit"
    (request-level hit) ratios and two context-transfer bars: "Spill (d2h)"
    (device→host) vs "Recall (h2d)" (host→device), read per-direction from
    ninfer_context_transfer_bytes_total
  * request lanes: running / waiting / prefilling / decode-ready / paused
    (the paused figure is painted red whenever non-zero)
  * utilization at the bottom: (prefill time + decode time) / uptime,
    estimated from a startup benchmark of the engine

On startup the script benchmarks the engine: one large uncached prompt
(max prefill t/s) and an 8-way parallel burst (max decode t/s and the
effective max concurrency). The result is cached in ~/.ninfer_btop.json and
reused on the next start while the engine's start time is unchanged (a
restart re-benchmarks automatically); --re-calib drops the cache and forces
a fresh benchmark. The device/host tier split of the token hit needs
--bytes-per-tok (exact context bytes per token, see bytes_per_token.md).

Standard library only (curses + urllib). No packages to install.

The cloud-API pricing baseline ($ per million tokens) is read from pricing.yaml
next to this script; select which model's row to use with --pricing-model
(default qwen3.8-27b). If pricing.yaml is missing, the built-in qwen3.8-27b
row is used instead.

Usage:
  python3 ninfer_btop.py --url URL [-i INTERVAL_MS] [--text] [--re-calib]
      [--burst N] [--bytes-per-tok N] [--pricing-model KEY] [--pricing-file PATH]

Keys in TUI mode:
  q / Esc  quit
"""

import argparse
import builtins
import curses
import json
import math
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

def _pool(m: dict, name: str, domain: str, kind=None):
    """Sum a labeled context-pool gauge over all ranks for one domain
    (device|host) and, optionally, one breakdown kind (capacity / in_use /
    cached / ...). 0.0 when the engine build does not report the metric
    (pre-2026-10-23 builds without the unified context pool)."""
    tot = 0.0
    for kv, v in m.get(name + "__series") or []:
        if kv.get("domain") == domain and (kind is None or kv.get("kind") == kind):
            tot += v
    return tot


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
    mA = fetch_metrics(url)
    pre0 = g(mA, "ninfer_prefill_tokens_total", 0.0)
    prompt_a = ("[calib %s] %s\nUsing the lines above as a glossary, write a short "
                "technical note on tree attention for speculative decoding."
                % (time.time_ns(), _rand_lines(250)))
    t_sent = time.monotonic()
    first_at, _ = _chat_first_chunk(api, model, prompt_a, 16)
    if first_at is None:
        raise RuntimeError("engine returned no tokens during the prefill test")
    mB = fetch_metrics(url)
    pre1 = g(mB, "ninfer_prefill_tokens_total", 0.0)
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
            "duration": duration,
            "server_start_time": g(m0, "ninfer_server_start_time_seconds")}


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


CALIB_FILE = os.path.expanduser("~/.ninfer_btop.json")


def server_start_time(url):
    """The engine's start time (epoch seconds) as reported by /metrics;
    None when the endpoint or the metric is unavailable."""
    try:
        m = fetch_metrics(url)
    except (OSError, ValueError):
        return None
    return g(m, "ninfer_server_start_time_seconds")


def load_calib_cache(url, probe=server_start_time):
    """The cached calibration from ~/.ninfer_btop.json. Valid only while the
    same engine runs: the cache and the live endpoint must agree on the URL
    and the server start time (a restart is a new engine). Returns the
    cached CALIB dict, or None to force a fresh benchmark."""
    try:
        with open(CALIB_FILE) as f:
            j = json.load(f)
    except FileNotFoundError:
        return None
    except OSError as e:
        # a cache that exists but cannot be read will not be overwritable
        # either: fail now, before spending minutes on a fresh benchmark
        print(f"error: cannot read calibration cache {CALIB_FILE}: {e}",
              file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        # corrupt cache: re-benchmark and let the fresh run rewrite it
        print(f"note: calibration cache {CALIB_FILE} is not valid JSON "
              f"({e}); re-benchmarking", file=sys.stderr)
        return None
    if j.get("url") != url:
        return None
    st = probe(url)
    cached = j.get("server_start_time")
    if st is None or not cached or st != cached:
        return None
    c = {k: j[k] for k in ("model", "concurrency", "prefill_tps", "decode_tps",
                           "duration") if k in j}
    c["ok"] = True  # only successful calibrations are ever cached
    return c


def save_calib_cache(url, calib):
    """Persist a successful calibration for the next startup; a failed
    benchmark is never cached."""
    st = calib.get("server_start_time")
    if not st:
        return
    try:
        with open(CALIB_FILE, "w") as f:
            json.dump({"url": url, "server_start_time": st,
                       **{k: calib[k] for k in ("model", "concurrency",
                                                "prefill_tps", "decode_tps",
                                                "duration") if k in calib}},
                      f, indent=2)
    except OSError as e:
        # a silent failure would make the next start re-benchmark even
        # though this run calibrated: surface it
        print(f"error: cannot write calibration cache {CALIB_FILE}: {e}",
              file=sys.stderr)
        sys.exit(1)


def resolve_calib(url, re_calib=False, run=run_calibration,
                  probe=server_start_time):
    """Decide where the startup calibration comes from: the
    ~/.ninfer_btop.json cache when the engine's start time is unchanged, a
    fresh benchmark otherwise. `re_calib` (the --re-calib flag) drops the
    cache and always benchmarks. Sets the global CALIB and refreshes the
    cache file after a successful run. A missing or corrupt cache simply
    triggers a fresh benchmark; a cache file that exists but cannot be
    read, written, or removed aborts with an error (silently falling back
    would mislead the user about which calibration is in effect)."""
    global CALIB
    if re_calib:
        try:
            os.remove(CALIB_FILE)
        except FileNotFoundError:
            pass
        except OSError as e:
            # swallowing this would leave the stale cache in place and make
            # the re-calibration look like it was persisted
            print(f"error: cannot remove calibration cache {CALIB_FILE}: {e}",
                  file=sys.stderr)
            sys.exit(1)
    CALIB = None if re_calib else load_calib_cache(url, probe=probe)
    if CALIB is None:
        run()
    if CALIB and CALIB.get("ok"):
        save_calib_cache(url, CALIB)


# ---------------------------------------------------------------- state

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
            spill_rate = None
            recall_rate = None
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
                    if self._last_spill is not None and spill_tot >= self._last_spill:
                        spill_rate = (spill_tot - self._last_spill) / dt
                    if self._last_recall is not None and recall_tot >= self._last_recall:
                        recall_rate = (recall_tot - self._last_recall) / dt
            self._last_t, self._last_gen, self._last_prefill = now, gen, pre
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
            # cache reuse
            root = g(m, "ninfer_root_selections_total", 0.0)
            chkp = g(m, "ninfer_checkpoint_selections_total", 0.0)
            tok_hit = (in_cache / in_tot) if in_tot > 0 else None
            req_hit = (chkp / (chkp + root)) if (chkp + root) > 0 else None
            # tier split of the token hits: lifetime host->device recall
            # bytes / bytes-per-token (--bytes-per-tok) ~ how many cached
            # tokens were served from host RAM (clamped to the cached
            # total — it also absorbs non-hit restores); the remainder was
            # served in place from VRAM. None without the flag.
            tok_dev = tok_host = None
            bpt = CALIB.get("bytes_per_tok") if CALIB else None
            if bpt and in_tot > 0:
                host_tok = min(in_cache, recall_tot / bpt)
                tok_host = host_tok / in_tot
                tok_dev = (in_cache - host_tok) / in_tot
            # unified context pool, per domain (device = KV, host = state),
            # summed across ranks; 0.0 on pre-unification engine builds
            dev_cap = _pool(m, "ninfer_context_pool_bytes", "device", "capacity")
            dev_inuse = _pool(m, "ninfer_context_pool_live_bytes", "device", "in_use")
            dev_cached = _pool(m, "ninfer_context_pool_live_bytes", "device", "cached")
            dev_peak = _pool(m, "ninfer_context_pool_peak_bytes", "device")
            host_cap = _pool(m, "ninfer_context_pool_bytes", "host", "capacity")
            host_inuse = _pool(m, "ninfer_context_pool_live_bytes", "host", "in_use")
            host_cached = _pool(m, "ninfer_context_pool_live_bytes", "host", "cached")
            host_peak = _pool(m, "ninfer_context_pool_peak_bytes", "host")
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
                "dev_cap": dev_cap,
                "dev_inuse": dev_inuse,
                "dev_cached": dev_cached,
                "dev_peak": dev_peak,
                "host_cap": host_cap,
                "host_cached": host_cached,
                "host_peak": host_peak,
                "running": g(m, "ninfer_requests_running", 0.0),
                "waiting": g(m, "ninfer_requests_waiting", 0.0),
                "paused": g(m, "ninfer_requests_paused", 0.0),
                "prefilling": g(m, "ninfer_requests_prefilling", 0.0),
                "decode_ready": g(m, "ninfer_requests_decode_ready", 0.0),
                "ready": g(m, "ninfer_engine_ready", 0.0),
                "max_concurrency": g(m, "ninfer_max_concurrency", 0.0),
                "ctx_window": g(m, "ninfer_max_context_tokens", 0.0),
                "util": util,
                "tok_hit": tok_hit,
                "req_hit": req_hit,
                "tok_dev": tok_dev,
                "tok_host": tok_host,
                "spill_rate": spill_rate,
                "recall_rate": recall_rate,
            }
        except Exception as e:
            self.ok = False
            self.last_err = f"{type(e).__name__}: {e}"
            return None


class Graph:
    """Scrolling bar history. The newest sample sits at the left edge: bars
    enter on the left and scroll out on the right. Two samples per drawn
    character (the newer in the left dot line), so a redraw after each poll
    shifts every dot line by exactly half a character - it can never jump,
    and the buffer holds twice as many samples as the drawn column count.

    `pinned` graphs (the batch graph) map their samples against the plot's
    total dot count (4*rows, from the panel height): one dot level is
    always exactly one request, and overload clips at the top of the plot.
    `floor` is the engine's max_concurrency, which sizes the panel height.
    Other graphs autoscale to the window max."""

    def __init__(self, width, floor=0.0, pinned=False):
        self.hist = deque(maxlen=width)
        self.floor = floor  # engine max_concurrency (sizes the batch panel)
        self.pinned = pinned  # True: scale is the plot's total dot count

    def add(self, v):
        self.hist.append(None if v is None else max(0.0, float(v)))

    def add_blank(self):
        self.hist.append(None)

    @property
    def values(self):
        return [v for v in self.hist if v is not None]

    def dots(self, rows=1):
        """Dot heights (0..4*rows) for the current window of a `rows`-char
        plot: pinned graphs scale against the plot's total dot count, the
        rest autoscale to the window max."""
        total = 4 * rows
        return compute_dots(list(self.hist),
                            max=total if self.pinned else None,
                            total_dots=total)


# ---------------------------------------------------------------- text mode

def _pool_body(cap, inuse, cached, peak):
    """One line of context-pool figures for --text output; a note when the
    engine build does not report the pool at all."""
    if not cap:
        return "n/a (pool not reported by this engine build)"
    body = f"cap {human(cap)}"
    if inuse is not None:
        body += f"  in-use {human(inuse)} ({inuse / cap * 100:.1f}%)"
    body += f"  cached {human(cached)} ({cached / cap * 100:.1f}%)"
    body += f"  peak {human(peak)}"
    return body


def run_text(url, rates=None, label=None):
    eng = Engine(url, rates=rates, label=label)
    s = eng.poll()
    if s is None:
        print(f"error: cannot reach {url}: {eng.last_err}")
        return 1
    print(f"engine : {url}  ({'ready' if s['ready'] else 'not ready'})")
    if eng.model:
        print(f"model  : {eng.model}")
    if s.get("ctx_window"):
        cw = s["ctx_window"]
        print(f"context: {cw:,.0f} tok  ({cw / 1024:.1f} KiToks)")
    print(f"batch  : running={s['running']:.0f} waiting={s['waiting']:.0f} "
          f"prefilling={s['prefilling']:.0f} decode_ready={s['decode_ready']:.0f}")
    print(f"input  : {human(s['input_total'])} tokens total "
          f"(cached {human(s['input_cached'])}, prefill {human(s['prefill'])})")
    print(f"output : {human(s['output_total'])} tokens total")
    u = s.get("util")
    if u is not None:
        print(f"util   : {u * 100:5.1f}%  (est. (prefill time + decode time) / uptime, from startup calibration)")
    else:
        print("util   : n/a  (no calibration data)")
    cc = s.get("cloud_cost")
    if cc is not None:
        print(f"saved  : ≈{money(cc)} vs {eng.cloud_label}  "
              f"({eng.rates['input']} $/M in, {eng.rates['cached_input']} $/M cached in, "
              f"{eng.rates['output']} $/M out)")
    print(f"kv     : device  {_pool_body(s['dev_cap'], s['dev_inuse'], s['dev_cached'], s['dev_peak'])}")
    print(f"         host    {_pool_body(s['host_cap'], None, s['host_cached'], s['host_peak'])}")
    _th, _rh = s.get("tok_hit"), s.get("req_hit")
    _bits = []
    if _th is not None:
        if s.get("tok_dev") is not None and s.get("tok_host") is not None:
            _bits.append(f"input hit {_th * 100:.1f}% "
                         f"({s['tok_dev'] * 100:.1f}% device, "
                         f"{s['tok_host'] * 100:.1f}% host)")
        else:
            _bits.append(f"input hit {_th * 100:.1f}%")
    if _rh is not None:
        _bits.append(f"request hit {_rh * 100:.1f}%")
    print(f"cache  : {', '.join(_bits) if _bits else 'n/a'}")
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


def draw_triple_panel(scr, y, left_title, mid_title, right_title, c, width):
    """One title row holding three labeled columns, each a third of the width.

    The columns occupy [0, s1), [s1, s2) and [s2, width-1). Lets three
    panels (Device KV / Host KV / Cache) run side by side in a quarter of
    the vertical space of three stacked panels. Returns (y + 1, s1, s2) so
    callers write their content rows into the three columns one row below
    the titles.
    """
    w = width - 1
    s1 = w // 3
    s2 = w - 1 - w // 3
    for x0, x1, title in ((0, s1, left_title), (s1, s2, mid_title),
                          (s2, w, right_title)):
        span = x1 - x0
        body = "─" + title
        txt = (body + "─" * max(0, span - len(body)))[:span]
        try:
            scr.addnstr(y, x0, txt, span)
        except curses.error:
            pass
    return y + 1, s1, s2


def _xfer_row(label, graph, width=13):
    """One cache-panel transfer row: a one-row braille sparkline over the
    graph's window (the same dots and double-wide characters as the main
    bar graphs - each character holds two adjacent samples, the newer in
    the left dot line), autoscaled to its own window peak, with that peak
    as the side label for scale. Unlike a one-poll bar, a sub-frame transfer burst (a few-MS
    recall DMA) stays visible for the whole graph window. The row is label
    + width + 1 + the peak label, which at 120 cols fits the ⅓-width cache
    column with the longest label (…MB/s, 10 chars). Like the main graphs
    the newest sample is on the left, and a partial window grows from that
    edge."""
    if graph is None:
        return label + " " * (width + 1) + "—"
    hist = list(graph.hist)
    n = min(2 * width, len(hist))
    hist = hist[-n:]  # the newest n samples (the row reads newest first)
    dots = compute_dots(hist, total_dots=4)  # autoscale to its own window peak
    vals = [v for v in hist if v is not None]
    mx = max(vals) if vals else 0.0
    if n % 2:
        dots.append(0)
    spark = "".join(dots_to_chars(dots[::-1], 1)[0]).ljust(width)  # partial window grows from the left
    return label + spark + " " + _rate_val(mx)


def _rate_val(rate):
    """Human rate label for a transfer rate; an em dash when there is no data."""
    if rate is None:
        return "   —"
    return human(rate, "B/s")


def _kv_row(label, val, cap):
    """Labeled context-pool row for the Device KV / Host KV panels: value plus
    the share of pool capacity; 'n/a' when the engine build does not report a
    pool capacity for that domain."""
    if not cap:
        return f"  {label:<9} : n/a"
    pct = f"  ({val / cap * 100:4.1f}%)" if val else ""
    return f"  {label:<9} : {human(val)}{pct}"


# Braille glyph table for the bar graphs, taken from btop (aristocratos/btop,
# Symbols::graph_symbols["braille_up"] in src/btop_draw.cpp) - btop's default
# graph symbol. btop packs two samples into one character (older = left dot
# column, newer = right dot column); we use the table's single-sample
# diagonal - BRAILLE[level][level] - so each character is a flat column of
# dots (never a rising/falling glyph) for exactly one sample. Each text row
# packs four dot rows, so a `rows`-row plot resolves 4*rows vertical levels.
# Index: [left level][right level], levels 0..4 (0 = empty cell).
BRAILLE = (
    (" ", "⢀", "⢠", "⢰", "⢸"),
    ("⡀", "⣀", "⣠", "⣰", "⣸"),
    ("⡄", "⣄", "⣤", "⣴", "⣼"),
    ("⡆", "⣆", "⣦", "⣶", "⣾"),
    ("⡇", "⣇", "⣧", "⣷", "⣿"),
)


def compute_dots(samples, max=None, total_dots=4):
    """Dot heights (0..total_dots) for the samples of a bar graph.

    `max` is the value that fills the plot: the plot's total dot count for
    pinned graphs (the batch graph - one dot = one request), or None to
    autoscale to the window max (prefill/decode, spill/recall).
    no_dots = ceil(v / (max / total_dots)): every non-zero sample draws at
    least one dot, and overshoots clip at total_dots."""
    if max is None:  # the parameter shadows the builtin max
        vals = [v for v in samples if v is not None]
        max = builtins.max(vals) if vals else 0.0
    if max <= 0:
        return [0] * len(samples)
    step = max / total_dots
    return [0 if v is None else min(total_dots, math.ceil(v / step))
            for v in samples]


def dots_to_chars(dots, rows):
    """The braille characters for a `rows`-row plot from a dots array.

    One sample per single dot line: a braille character is double-wide and
    holds the dot lines of two adjacent samples - the newer sample in the
    left dot column, the older in the right (so a character may be a
    rising/falling glyph). An odd-length array is padded with a trailing
    zero. Returns a rows x len//2 grid of 5x5 glyphs (indexed
    [left level][right level]); grid row 0 is the top of the plot, so a
    sample fills up from the last row. A sample with d dots occupies dots
    [(rows-1-r)*4, (rows-r)*4) of row r's band."""
    if len(dots) % 2:
        dots = [*dots, 0]
    grid = []
    for r in range(rows):
        band = (rows - 1 - r) * 4  # first dot of this row's band
        line = []
        for c in range(len(dots) // 2):
            l = min(4, max(0, dots[2 * c] - band))
            h = min(4, max(0, dots[2 * c + 1] - band))
            line.append(BRAILLE[l][h])
        grid.append(line)
    return grid


def draw_bar_graph(scr, y, graph: Graph, rows, width, key, c):
    hist_vals = list(graph.hist)
    dots = graph.dots(rows)  # one dot height per sample
    # two samples per braille character: the plot is half the buffer wide;
    # the buffer holds oldest -> newest, but the newest sample is on the
    # left, so the window is read newest first
    n = min(len(hist_vals), 2 * (width - 14))
    vals = hist_vals[-n:][::-1]
    grid = dots_to_chars(dots[-n:][::-1], rows)
    plot_w = min(width - 14, (n + 1) // 2)
    peak_v = max((v for v in hist_vals if v is not None), default=0.0)
    base_attr = curses.color_pair(c.get(key, 0))
    # window-max samples are highlighted (bold peak color, or reverse video
    # when the terminal has no colors); a glyph holds two dot lines, so a
    # character is painted peak when either of its samples is the window max
    peak_attr = (curses.color_pair(c["peak"]) | curses.A_BOLD) if "peak" in c else curses.A_REVERSE
    # the newest sample is on the left: the leftmost character holds it and
    # the second-newest sample, and the graph scrolls left to right (bars
    # enter left, exit right); the newer sample always sits in the left dot
    # line of its character, so each poll shifts every dot line by exactly
    # half a character - the pairing never flips, it cannot jump; a partial
    # (not-yet-full) window grows from the left edge
    for col in range(plot_w):
        pair = vals[2 * col:2 * col + 2]  # pair's samples, newer first
        attr = (peak_attr if (peak_v > 0
                              and any(v == peak_v for v in pair if v is not None))
                else base_attr)
        for r in range(rows):
            ch = grid[r][col]
            if ch == " ":
                continue
            try:
                scr.addstr(y + r, 1 + col, ch, attr)
            except curses.error:
                pass
    # baseline
    try:
        scr.addnstr(y + rows, 1, "─" * plot_w, plot_w)
    except curses.error:
        pass
    # the cur/min/max/avg figures live in the panel title banner now
    return y + rows + 1


def frame(scr, eng, graphs, c):
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
    try:
        # never fill the last column: a cursor parked at EOL puts ncurses in
        # pending-wrap state and its relative row moves land 1-2 rows low
        scr.addnstr(0, 0, head[:width - 1], width - 1, curses.color_pair(c.get("title", 0)))
    except curses.error:
        pass

    # status line (row 1): calibration figures, plus the engine's context
    # window in KiToks (1 KiToks = 1024 tokens) when the engine reports it
    ctx = (s or {}).get("ctx_window")
    ctx_txt = f"{ctx / 1024:.1f} KiToks" if ctx else ""
    if CALIB and CALIB.get("ok"):
        cd = CALIB
        peak_attr = (curses.color_pair(c["peak"]) | curses.A_BOLD) if "peak" in c else curses.A_REVERSE
        lbl = curses.color_pair(c.get("label", 0))
        segs = [(f"calibrated in {cd['duration']:.1f}s:  concurrency ", lbl),
                (f"{cd['concurrency']}", peak_attr),
                ("  ·  prefill ", lbl),
                (f"{cd['prefill_tps']:,.0f} t/s", peak_attr),
                ("  ·  decode ", lbl),
                (f"{cd['decode_tps']:,.0f} t/s", peak_attr)]
        if ctx_txt:
            segs += [("  ·  ctx ", lbl), (ctx_txt, peak_attr)]
        x = 0
        for txt, a in segs:
            n = min(len(txt), width - x)
            if n <= 0:
                break
            try:
                scr.addnstr(1, x, txt[:n], n, a)
            except curses.error:
                break
            x += n
    elif CALIB and CALIB.get("error"):
        msg = f"calibration failed: {CALIB['error'][:width - 25]}  (util n/a)"
        if ctx_txt:
            msg += f"  ·  ctx {ctx_txt}"
        try:
            scr.addnstr(1, 0, msg[:width - 1], width - 1, curses.color_pair(c.get("dim", 0)))
        except curses.error:
            pass
    elif ctx_txt:
        # no calibration data: the status line still carries the context window
        try:
            scr.addnstr(1, 0, ctx_txt, width - 1, curses.color_pair(c.get("dim", 0)))
        except curses.error:
            pass

    # row 0 header, row 1 status line; the panels start on row 2
    y = 2
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
    # fixed rows: header + status line + 3 titles + 3 baselines
    # + 1 lane + 1 totals title + 2 totals + 1 shared Device KV/Host KV/Cache
    # title + 4 shared rows (incl. the spill/recall bars) + 1 footer; the
    # cur/min/max/avg figures sit in the panel titles, plots take the rest
    fixed = 18
    # the batch panel's height follows the engine's max concurrency (kept in
    # Graph.floor, updated on every poll): one braille character resolves 4
    # dot levels, so N concurrent requests need ceil(N/4) characters
    # (<=4 -> 1 char, 5-8 -> 2, 9-12 -> 3, ...). The panel's scale is its
    # own total dot count (4*rows), which makes each dot level exactly one
    # request. ninfer caps concurrency at 8, so its batch graph is 1 or 2
    # chars high. Before the first successful poll (/metrics is unreachable
    # until after the engine's warmup, and then max_concurrency is in every
    # scrape) the panel height falls back to the even three-way split; the
    # scale stays pinned to the height's dot count.
    if g_batch.floor > 0:
        rows_batch = max(1, min(6, -(-int(g_batch.floor) // 4)))
    else:
        rows_batch = max(1, min(6, (max_y - fixed) // 3))
    # prefill/decode share the rows the batch panel does not use
    plot_rows = max(1, min(6, (max_y - fixed - rows_batch) // 2))

    # --- batch panel
    b = (s or {}).get("running")
    cur_b = f"{b:.0f}" if b is not None else "—"
    y = draw_panel(scr, y, "Batch size  (ninfer_requests_running)", "batch", c, width,
                   metric_stat_segments(g_batch, cur_b))
    y = draw_bar_graph(scr, y, g_batch, rows_batch, width, "batch", c)
    if s:
        # paused is the one lane number that signals a problem: a live
        # request stalled on a capacity event, so paint it red when non-zero
        lane_base = (f"running {s['running']:.0f}  ·  waiting {s['waiting']:.0f}  ·  "
                     f"prefilling {s['prefilling']:.0f}  ·  "
                     f"decode-ready {s['decode_ready']:.0f}  ·  ")
        lane_end = f"paused {s['paused']:.0f}"
        end_attr = curses.color_pair(c.get("bad", 0)) if s["paused"] else 0
        x = 1
        for txt, attr in ((lane_base, 0), (lane_end, end_attr)):
            n = min(len(txt), width - 1 - x)
            if n <= 0:
                break
            try:
                scr.addnstr(y, x, txt[:n], n, attr)
            except curses.error:
                break
            x += n
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

    # --- device KV + host KV + cache, three side-by-side panels sharing one
    #     title row (btop-style): a quarter of the space three stacked
    #     panels would need
    if y + 5 <= max_y - 1:
        if s:
            left = [f"  Capacity  : " + (human(s['dev_cap']) if s['dev_cap'] else "n/a"),
                    _kv_row("In use", s["dev_inuse"], s["dev_cap"]),
                    _kv_row("Cached", s["dev_cached"], s["dev_cap"]),
                    _kv_row("Peak", s["dev_peak"], s["dev_cap"])]
            mid = [f"  Capacity  : " + (human(s['host_cap']) if s['host_cap'] else "n/a"),
                   _kv_row("Cached", s["host_cached"], s["host_cap"]),
                   _kv_row("Peak", s["host_peak"], s["host_cap"]),
                   ""]
            th, rh = s.get("tok_hit"), s.get("req_hit")
            td, ths = s.get("tok_dev"), s.get("tok_host")
            if td is not None and ths is not None:
                # tier split (approximate, see poll()); no space between
                # the groups: at 120 cols this column is 38 chars wide and
                # the line must fit, 100.0% included
                in_line = (f"  Tok hit : {td * 100:4.1f}% (device)"
                           f"{ths * 100:4.1f}% (host)")
            elif th is not None:
                in_line = f"  Tok hit : {th * 100:5.1f}%"
            else:
                in_line = "  Tok hit :   n/a  (no input yet)"
            req_line = (f"  Req hit : {rh * 100:5.1f}%" if rh is not None
                        else "  Req hit :   n/a  (no bindings yet)")
            right = [in_line, req_line,
                     _xfer_row("  Spill  (d2h)  ", graphs.get("xfer_d2h")),
                     _xfer_row("  Recall (h2d)  ", graphs.get("xfer_h2d"))]
        else:
            left = ["  Capacity  : —", "  In use    : —", "  Cached    : —",
                    "  Peak      : —"]
            mid = ["  Capacity  : —", "  Cached    : —", "  Peak      : —", ""]
            right = ["  Tok hit : —", "  Req hit : —",
                     _xfer_row("  Spill  (d2h)  ", graphs.get("xfer_d2h")),
                     _xfer_row("  Recall (h2d)  ", graphs.get("xfer_h2d"))]
        y, s1, s2 = draw_triple_panel(scr, y, "Device KV", "Host KV",
                                      "Cache (checkpoint reuse)", c, width)
        w0 = s1 - 1
        w1 = s2 - 1 - s1
        w2 = width - 1 - s2
        for i in range(4):
            try:
                scr.addnstr(y + i, 0, left[i][:w0], w0)
                scr.addnstr(y + i, s1, mid[i][:w1], w1)
                scr.addnstr(y + i, s2, right[i][:w2], w2)
            except curses.error:
                pass
        y += 4

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
    for txt, key in ((" q quit", "dim"),
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
    # two samples per drawn character: histories are twice the column count
    graphs = {"batch": Graph(2 * gwidth, pinned=True),  # pinned: scale = plot's dot count (1 dot = 1 request)
              "toks": Graph(2 * gwidth), "prefill": Graph(2 * gwidth),
              # transfer sparklines: fixed 13-char width (26 samples ~ 13 s
              # at the default 500 ms poll), independent of terminal width
              "xfer_d2h": Graph(26), "xfer_h2d": Graph(26)}
    interval = args.interval / 1000.0
    eng = Engine(args.url, interval, args.pricing_rates, label=args.pricing_model)
    last_poll = 0.0
    running = True

    while running:
        now = time.time()
        # each poll appends exactly one sample (half a character); the newer
        # sample always occupies the left dot line of its character, so a
        # redraw after every poll shifts every dot line by exactly half a
        # character - the pairing never flips, the motion cannot jump
        if now - last_poll >= interval:
            last_poll = now
            s = eng.poll()
            eng.last_frame = s
            if s is not None:
                if s["max_concurrency"] > 0:
                    graphs["batch"].floor = s["max_concurrency"]
                graphs["batch"].add(s["running"])
                graphs["toks"].add(s["toks_per_s"])
                graphs["prefill"].add(s["prefill_tps"])
                graphs["xfer_d2h"].add(s.get("spill_rate"))
                graphs["xfer_h2d"].add(s.get("recall_rate"))
            else:
                for k in graphs:
                    graphs[k].add_blank()
            frame(stdscr, eng, graphs, c)

        # resync graph width on terminal resize
        ny, nx = stdscr.getmaxyx()
        if abs(nx - 16 - gwidth) > 2:
            gwidth = max(20, nx - 16)
            for k in ("batch", "toks", "prefill"):  # xfer sparklines stay 13 chars
                old = list(graphs[k].hist)
                g2 = Graph(gwidth, floor=graphs[k].floor, pinned=graphs[k].pinned)
                g2.hist.extend(old[-gwidth:])
                graphs[k] = g2
            frame(stdscr, eng, graphs, c)

        ch = stdscr.getch()
        if ch == -1:
            # small sleep so we don't spin when no key
            time.sleep(0.05)
            continue
        if ch in (ord("q"), ord("Q"), 27):
            running = False

    return 0


def main():
    ap = argparse.ArgumentParser(description="btop-style dashboard for a ninfer engine")
    ap.add_argument("--url", required=True, help="metrics endpoint (required, e.g. http://host:port/metrics)")
    ap.add_argument("-i", "--interval", type=int, default=500,
                help="poll interval in ms (default 500); each poll shifts the "
                     "bar graphs by half a character (one dot line)")
    ap.add_argument("--text", action="store_true", help="one-shot plain-text stats, no TUI")
    ap.add_argument("--re-calib", action="store_true",
                    help="drop the ~/.ninfer_btop.json calibration cache and "
                         "re-run the startup benchmark (default: reuse the "
                         "cache while the engine's start time is unchanged)")
    ap.add_argument("--bytes-per-tok", type=float, default=None, metavar="N",
                    help="context bytes per token for the device/host hit split "
                         "(exact value from the model architecture: full-attention "
                         "layers x 2 x kv_heads x (head_dim x dtype_bytes + scale "
                         "overhead; e.g. INT8 group-64 = 256+8 B per 256-dim head, "
                         "plus the MTP draft layer if pooled). Required for the "
                         "device/host tier split; without it the Tok hit line "
                         "shows the combined figure")
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

    resolve_calib(args.url, re_calib=args.re_calib,
                  run=lambda: run_calibration(args))
    if args.bytes_per_tok:
        global CALIB
        CALIB = dict(CALIB or {})
        CALIB["bytes_per_tok"] = args.bytes_per_tok
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
