#!/usr/bin/env python3
"""ninfer_btop metrics & calibration: the /metrics HTTP client, the
startup benchmark, the ~/.ninfer_btop.json calibration cache, and the
cloud-API pricing data.

Extracted from ninfer_btop.py so metrics/calibration edits stop fighting
the dashboard code for the same file. Stdlib-only leaf: imports nothing
from ninfer_btop / tui / bar_graphs; both of them import THIS
module at top level.

The live calibration state is private (_CALIB), set at startup by
resolve_calib() (cache hit or fresh benchmark) and accessed only
through the get_calib()/set_calib() pair - consumers must not bind the
value, because the writers re-bind it. ninfer_engine_state's
NinferEngineState reads it per poll (lifetime utilization, the
bytes-per-token tier split); main() passes get_calib()'s current value to
the chrome as a value.
"""
import json
import os
import random
import sys
import threading
import time
import urllib.request

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


def get_metric(m: dict, name: str, default=None):
    v = m.get(name)
    return v if isinstance(v, float) or isinstance(v, int) else default

def pool(m: dict, name: str, domain: str, kind=None):
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

# Startup calibration state: {"ok": True, "concurrency", "prefill_tps",
# "decode_tps", "duration"} or {"ok": False, "error": ...}; None = not run.
# Private - freely re-bound by the writers (run_calibration/resolve_calib).
# Consumers read it through get_calib()/set_calib(), never by importing
# the value itself: a from-imported name would be a snapshot and would not
# see later re-bindings.
_CALIB = None

def get_calib():
    """Current calibration state (dict or None)."""
    return _CALIB

def set_calib(calib):
    """Store calibration state (dict or None) - the write half of
    get_calib(); used by main() for --bytes-per-tok and by tests for
    fixtures."""
    global _CALIB
    _CALIB = calib

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
    pre0 = get_metric(mA, "ninfer_prefill_tokens_total", 0.0)
    prompt_a = ("[calib %s] %s\nUsing the lines above as a glossary, write a short "
                "technical note on tree attention for speculative decoding."
                % (time.time_ns(), _rand_lines(probe_lines)))
    t_sent = time.monotonic()
    first_at, _ = _chat_first_chunk(api, model, prompt_a, probe_max_tokens)
    if first_at is None:
        raise RuntimeError("engine returned no tokens during the prefill test")
    mB = fetch_metrics(url)
    pre1 = get_metric(mB, "ninfer_prefill_tokens_total", 0.0)
    prefill_tps = max(1.0, (pre1 - pre0) / (first_at - t_sent))

    # phase B: max decode t/s + max concurrency — `burst` parallel requests
    log(f"  decode: {burst} parallel requests…")
    gen0 = get_metric(fetch_metrics(url), "ninfer_generation_tokens_total", 0.0)
    stop = threading.Event()
    samples, lock = [], threading.Lock()

    def sampler():
        while not stop.is_set():
            try:
                mm = fetch_metrics(url, timeout=1.5)
                with lock:
                    samples.append((time.monotonic(),
                                    get_metric(mm, "ninfer_requests_running", 0.0),
                                    get_metric(mm, "ninfer_generation_tokens_total", 0.0)))
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
        concurrency = int(get_metric(m0, "ninfer_max_concurrency", 0.0))
    if decode_tps <= 0:
        gen1 = get_metric(fetch_metrics(url), "ninfer_generation_tokens_total", 0.0)
        decode_tps = max(1.0, (gen1 - gen0) / max(time.monotonic() - t_b, 0.1))

    return {"ok": True, "model": model, "concurrency": concurrency,
            "prefill_tps": prefill_tps, "decode_tps": decode_tps,
            "duration": duration,
            "server_start_time": get_metric(m0, "ninfer_server_start_time_seconds")}


def run_calibration(args):
    """Run the startup benchmark once; stores the result (via set_calib)."""
    global _CALIB
    print("ninfer_btop: calibrating engine…", flush=True)
    try:
        result = calibrate(args.url, log=lambda m: print(m, flush=True),
                           burst=args.burst)
    except Exception as e:
        result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    _CALIB = result
    if not result.get("ok"):
        print(f"  calibration failed: {result['error']} — utilization shows n/a", flush=True)
        return
    print(f"  done in {result['duration']:.1f}s: max concurrency {result['concurrency']}  ·  "
          f"prefill {result['prefill_tps']:,.0f} t/s  ·  decode {result['decode_tps']:,.0f} t/s",
          flush=True)


CALIB_FILE = os.path.expanduser("~/.ninfer_btop.json")


def server_start_time(url):
    """The engine's start time (epoch seconds) as reported by /metrics;
    None when the endpoint or the metric is unavailable."""
    try:
        m = fetch_metrics(url)
    except (OSError, ValueError):
        return None
    return get_metric(m, "ninfer_server_start_time_seconds")


def load_calib_cache(url, probe=server_start_time):
    """The cached calibration from ~/.ninfer_btop.json. Valid only while the
    same engine runs: the cache and the live endpoint must agree on the URL
    and the server start time (a restart is a new engine). Returns the
    cached calibration dict, or None to force a fresh benchmark."""
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
    cache and always benchmarks. Stores the calibration state (read it
    back via get_calib()) and refreshes the cache file after a successful
    run. A missing or corrupt cache simply
    triggers a fresh benchmark; a cache file that exists but cannot be
    read, written, or removed aborts with an error (silently falling back
    would mislead the user about which calibration is in effect)."""
    global _CALIB
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
    _CALIB = None if re_calib else load_calib_cache(url, probe=probe)
    if _CALIB is None:
        run()
    if _CALIB and _CALIB.get("ok"):
        save_calib_cache(url, _CALIB)
