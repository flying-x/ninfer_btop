"""ninfer_btop engine state: the engine's live state, refreshed per poll.

NinferEngineState: the poll loop over the engine's /metrics endpoint -
tokens/s rates, gauge labels, the unified context pool per domain, cache
hits, the device/host tier split, and the spill/recall transfer rates.

Imports only the stdlib-only leaf metrics_calib (the /metrics client and
calibration cache); it imports neither ninfer_btop nor tui, so
`import ninfer_engine_state` is cycle-free.
"""
import time

from metrics_calib import (
    CLOUD_LABEL,
    DEFAULT_PRICING,
    fetch_metrics,
    get_calib,
    get_metric,
    pool,
)


# ---------------------------------------------------------------- state

class NinferEngineState:
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
            st = get_metric(m, "ninfer_server_start_time_seconds")
            self.start_time = st
            now = time.monotonic()
            gen = get_metric(m, "ninfer_generation_tokens_total", 0.0)
            pre = get_metric(m, "ninfer_prefill_tokens_total", 0.0)
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
            calib = get_calib()
            if calib and calib.get("ok") and st:
                up = time.time() - st
                if up > 1.0:
                    busy = (pre / max(calib["prefill_tps"], 1.0)
                            + gen / max(calib["decode_tps"], 1.0))
                    util = min(1.0, max(0.0, busy / up))
            # lifetime money saved vs the cheapest cloud API (self-host ~ $0)
            in_tot = get_metric(m, "ninfer_prompt_tokens_total", 0.0)
            in_cache = get_metric(m, "ninfer_prompt_tokens_cached_total", 0.0)
            out_tot = get_metric(m, "ninfer_generation_tokens_total", 0.0)
            cloud_cost = ((in_tot - in_cache) / 1e6 * self.rates["input"]
                          + in_cache / 1e6 * self.rates["cached_input"]
                          + out_tot / 1e6 * self.rates["output"])
            # cache reuse
            root = get_metric(m, "ninfer_root_selections_total", 0.0)
            chkp = get_metric(m, "ninfer_checkpoint_selections_total", 0.0)
            tok_hit = (in_cache / in_tot) if in_tot > 0 else None
            req_hit = (chkp / (chkp + root)) if (chkp + root) > 0 else None
            # tier split of the token hits: lifetime host->device recall
            # bytes / bytes-per-token (--bytes-per-tok) ~ how many cached
            # tokens were served from host RAM (clamped to the cached
            # total — it also absorbs non-hit restores); the remainder was
            # served in place from VRAM. None without the flag.
            tok_dev = tok_host = None
            bpt = calib.get("bytes_per_tok") if calib else None
            if bpt and in_tot > 0:
                host_tok = min(in_cache, recall_tot / bpt)
                tok_host = host_tok / in_tot
                tok_dev = (in_cache - host_tok) / in_tot
            # unified context pool, per domain (device = KV, host = state),
            # summed across ranks; 0.0 on pre-unification engine builds
            dev_cap = pool(m, "ninfer_context_pool_bytes", "device", "capacity")
            dev_inuse = pool(m, "ninfer_context_pool_live_bytes", "device", "in_use")
            dev_cached = pool(m, "ninfer_context_pool_live_bytes", "device", "cached")
            dev_peak = pool(m, "ninfer_context_pool_peak_bytes", "device")
            host_cap = pool(m, "ninfer_context_pool_bytes", "host", "capacity")
            host_cached = pool(m, "ninfer_context_pool_live_bytes", "host", "cached")
            host_peak = pool(m, "ninfer_context_pool_peak_bytes", "host")
            self.m = m
            return {
                "batch": get_metric(m, "ninfer_requests_running", 0.0),
                "toks_per_s": toks,
                "prefill_tps": ptok,
                "input_total": in_tot,
                "output_total": out_tot,
                "input_cached": in_cache,
                "prefill": get_metric(m, "ninfer_prefill_tokens_total", 0.0),
                "cloud_cost": cloud_cost,
                "dev_cap": dev_cap,
                "dev_inuse": dev_inuse,
                "dev_cached": dev_cached,
                "dev_peak": dev_peak,
                "host_cap": host_cap,
                "host_cached": host_cached,
                "host_peak": host_peak,
                "running": get_metric(m, "ninfer_requests_running", 0.0),
                "waiting": get_metric(m, "ninfer_requests_waiting", 0.0),
                "paused": get_metric(m, "ninfer_requests_paused", 0.0),
                "prefilling": get_metric(m, "ninfer_requests_prefilling", 0.0),
                "decode_ready": get_metric(m, "ninfer_requests_decode_ready", 0.0),
                "ready": get_metric(m, "ninfer_engine_ready", 0.0),
                "max_concurrency": get_metric(m, "ninfer_max_concurrency", 0.0),
                "ctx_window": get_metric(m, "ninfer_max_context_tokens", 0.0),
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
