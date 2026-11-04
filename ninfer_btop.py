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
    - the rightmost 6 braille columns of each graph are period summaries,
      labeled MhHDWL on the baseline row: M last 5 minutes, h last
      half-hour, H last hour, D last day, W last week, L this monitor
      process's life-time. Each
      column's left dot line is the period average, its right dot line the
      period max, in the graph's own dot scale (a period max above the
      current display scale clips at the top)
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
--bytes-per-tok (exact context bytes per token, see docs/bytes_per_token.md).

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
import curses
import json
import os
import random
import signal
import sys
import threading
import time
import urllib.request

from bar_graphs import compute_dots, dots_to_chars


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
            sse = "data: "
            if not line.startswith(sse):
                continue
            if line[len(sse):].strip() == "[DONE]":
                break
            try:
                o = json.loads(line[len(sse):].strip())
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
    # phase A probe: one large uncached prompt, a few decode steps are enough
    # to reach the first chunk; phase B workers use short prompts but generate
    # enough tokens to fill the decode window
    probe_lines, probe_max_tokens = 250, 16
    burst_lines, burst_max_tokens = 25, 240

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
                % (time.time_ns(), _rand_lines(probe_lines)))
    t_sent = time.monotonic()
    first_at, _ = _chat_first_chunk(api, model, prompt_a, probe_max_tokens)
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
                 % (time.time_ns(), n, _rand_lines(burst_lines)))
            _chat_first_chunk(api, model, p, burst_max_tokens, timeout=90)
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


# width of the one-row spill/recall graphs in characters; fixed and
# independent of the terminal width (buffer = 2 samples per character)
XFER_COLS = 13


def _xfer_row(label, graph, width=XFER_COLS):
    """One cache-panel transfer row: a one-row braille graph over the
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
    chars = "".join(dots_to_chars(dots[::-1], 1)[0]).ljust(width)  # partial window grows from the left
    return label + chars + " " + _rate_val(mx)


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
        import frame_chrome
        sys.exit(frame_chrome.run_text(args.url, args.pricing_rates, args.pricing_model))
    signal.signal(signal.SIGTTOU, signal.SIG_IGN)  # like top: safe even as a background job
    import frame_chrome  # lazy: keeps `import ninfer_btop` free of screen code
    try:
        curses.wrapper(frame_chrome.main_loop, args, CALIB)
    except curses.error as e:
        print(f"curses error (run inside a real terminal?): {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # frame_chrome imports ninfer_btop by name; running this file makes the
    # module __main__, so alias it: frame_chrome then reuses this module
    # (live CALIB, one Engine/Graph class) instead of importing a second copy
    sys.modules["ninfer_btop"] = sys.modules[__name__]
    main()
