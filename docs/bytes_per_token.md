# bytes-per-token for the device/host hit split

From the [ninfer_btop](https://github.com/flying-x/ninfer_btop) project — a btop-style TUI dashboard for the NInfer inference engine.

`ninfer_btop` splits the token-level input hit into device- and
host-served tiers by converting lifetime host→device recall bytes into
tokens: `host_tokens = h2d_bytes / bpt` (clamped to the cached-token
total). The split is only as good as `bpt` — the engine moves exactly
the pooled context bytes per token, and those are fixed by the model
architecture and the KV dtype. Pass the exact value with
`--bytes-per-tok N` (without the flag the split is off and the Tok hit
line shows the combined figure). An earlier version measured bpt with a
startup probe; that was a noisy lower bound on a busy, pinned-pool
engine (31.1 KB/tok vs the exact 33 KB for Qwen3.8-27B), and a low
reading lets lifetime h2d volume saturate the clamp, drifting the split
to `0.0% (device)` over uptime — so the value is now supplied, not
measured.

## Formula

    bpt = layers_fa x 2 (K+V) x kv_heads x (head_dim x dtype_bytes + scale overhead)

plus one full-attention layer's worth if the MTP draft layer's KV is
pooled. Only **full-attention** layers contribute: hybrid models store
per-token KV in a subset of layers, the rest carry fixed per-request
state.

ninfer KV storage per 256-dim head, per token, per plane
(`src/core/paged_kv_storage.h`, `src/ops/kv_cache/`):

| dtype | layout | bytes/plane |
|---|---|---|
| INT8 group-64 | 256 x 1B + 4 x fp16 scales (per 64-elem group) | 264 |
| FP8 e4m3 group-128 | 256 x 1B + 2 x 1B power-of-2 scales (per 128-elem group) | 258 |

## Qwen3.8-27B (Qwen3_5 architecture, base Qwen/Qwen3.8-27B)

- 64 hybrid layers: 16 full-attention (every 4th; 24 q-heads,
  **4 kv-heads, head_dim 256**), 48 GDN linear-attention layers
- GDN state: fixed per-request ring, ~1.5 MB total, moved in one lump
  on spill/recall — negligible per token
- MTP: 1 full-attention draft layer (same 4 kv-heads x 256 dim)

| KV dtype | bpt (no MTP) | bpt (MTP pooled) |
|---|---|---|
| FP8 e4m3 group-128 | **33,024** | 35,088 |
| INT8 group-64 | 33,792 | 35,904 |

In use: `--kv-dtype fp8 --spec mtp` → `--bytes-per-tok 33024`, which
yields a sane live split (e.g. `59.9% (device)30.0% (host)` against a
59.9%+30.0% of an ~89% input hit).


## Best practices (for agent readers)

- When a tool converts byte counters to token counts (or divides lifetime
  counters to derive a rate), do not probe-measure the conversion factor at
  runtime on a busy engine: the reading is a noisy lower bound, and one low
  value skews the whole derivation over uptime (here the clamp saturates
  and the split drifts to `0.0% (device)`).
- Compute the exact factor from the model architecture and KV storage
  layout (formula above) and expose it as an explicit parameter so the user
  pins it (`--bytes-per-tok N`).
