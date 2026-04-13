# Architecture

This document describes the internal architecture of `metalgrow` — what the
pieces are, how they fit together, and the seams we intend to extend.

## Goals

- **Apple Metal first.** Default execution path is PyTorch's MPS backend on
  Apple Silicon. CUDA and CPU are supported as first-class fallbacks, never
  as an afterthought.
- **Pluggable upscaling core.** The choice of super-resolution model is an
  implementation detail. The public API and CLI must not change when the
  backbone is swapped.
- **Small surface.** One `Upscaler` class, one CLI entrypoint, one device
  selector. No hidden globals, no framework-y indirection.

## High-level flow

```
┌────────────┐      ┌────────────┐      ┌──────────────┐      ┌────────────┐
│    CLI     │ ───▶ │  Upscaler  │ ───▶ │   Backbone   │ ───▶ │   Device   │
│ (typer)    │      │  (public)  │      │ (SR model)   │      │ (MPS/CUDA) │
└────────────┘      └────────────┘      └──────────────┘      └────────────┘
        │                  │                   │
        ▼                  ▼                   ▼
     parse         image I/O + tensor      torch.nn.Module
     flags         conversion              (bicubic today,
                                            learned later)
```

1. **CLI** (`metalgrow.cli`) parses arguments and resolves paths.
2. **Upscaler** (`metalgrow.upscaler`) is the stable public interface. It
   owns image I/O, tensor conversion, and orchestrates the backbone call.
3. **Backbone** (`metalgrow.backbones`) is the actual super-resolution
   operation. `bicubic` is always available (no weights); learned
   backbones like `realesrgan-x2` / `realesrgan-x4` load weights via
   `metalgrow.weights` and run through spandrel.
4. **Device** (`metalgrow.device`) picks the torch device once per process
   using a strict preference order.

## Modules

### `metalgrow.device`

Single responsibility: return a `torch.device` given a preference string
(`auto | mps | cuda | cpu`). Resolution order under `auto`:

1. `mps` if `torch.backends.mps.is_available()`
2. `cuda` if `torch.cuda.is_available()`
3. `cpu`

This module **never** imports model code — it stays cheap so tests and CLI
`info` stay fast.

### `metalgrow.upscaler`

The `Upscaler` class is the public API. Responsibilities:

- Hold the resolved `torch.device`.
- Convert PIL → float tensor in `[0, 1]`, run the backbone, convert back.
- Clamp outputs and return a PIL image.
- Provide a file-level convenience (`upscale_file`) used by the CLI.

The backbone call is intentionally a single line so swapping backbones
doesn't reshape the surrounding code.

### `metalgrow.backbones`

Package with a `Backbone` ABC, a name→factory registry, and one module
per backbone. Each backbone accepts a `(device, dtype)` pair and exposes
`upscale(tensor, scale)`. `supported_scales` is `None` for analytical
resamplers (any scale) or a fixed tuple (e.g. `(2.0, 4.0)` for
Real-ESRGAN).

### `metalgrow.weights`

Small registry + cache layer for weight files. Downloads on first use,
verifies SHA-256 on every call, removes corrupt files so a retry can
re-download. Cache location overridable via `METALGROW_CACHE_DIR`.

### `metalgrow.cli`

A thin `typer` app. Two commands:

- `upscale SRC DST --scale --device --backbone --dtype` — the real work.
- `info` — prints torch version and backend availability. Useful for
  verifying MPS is picked up on a fresh Apple Silicon machine.

The CLI must never contain model logic. If a flag needs model state, it
belongs on `Upscaler`.

## Extension points

### Plugging in a learned backbone

Backbones live in `metalgrow/backbones/` and subclass `Backbone`
(`base.py`). Each backbone owns its own preprocess/postprocess and tiling
strategy; `Upscaler` only knows the `upscale(tensor, scale)` contract.
Register a new backbone by calling `register(name, factory)` at import
time — see `bicubic.py` and `realesrgan.py` for the pattern. To expose it
on the CLI, nothing extra is needed: `--backbone` reads from the registry.

### Tiled inference

`RealESRGANBackbone` tiles 256×256 with a 16-px overlap to bound memory on
large inputs. Tiling lives on the backbone rather than `Upscaler` because
the optimal tile size is backbone-specific (receptive field, VRAM
profile). The bicubic backbone needs no tiling.

### Model registry

`metalgrow.weights.REGISTRY` maps a backbone name to a `WeightSpec`
(filename, URL, SHA-256). `ensure_weight(name)` downloads into
`~/.cache/metalgrow/` on first use and verifies the checksum every call.
Override the cache location with `METALGROW_CACHE_DIR`. A CLI subcommand
for listing/removing cached weights is future work.

## Device notes

### MPS quirks

The MPS backend is our primary target but still has rough edges. How
metalgrow handles them:

- **Automatic fallback.** When `get_device` resolves to MPS, it sets
  `PYTORCH_ENABLE_MPS_FALLBACK=1` so ops missing on the MPS backend
  silently execute on CPU instead of aborting the forward pass. Set
  `METALGROW_DISABLE_MPS_FALLBACK=1` to opt out — useful when you want
  hard failures to locate an unsupported op.
- **Known op gaps.** Real-ESRGAN's RRDBNet runs end-to-end on MPS, but
  expect occasional fallbacks on exotic kernels. The fallback is
  transparent but adds a host↔device round-trip per affected op.
- **`float32` default.** `fp16` on MPS is roughly 1.5–2× faster but
  noticeably noisier for SR (halos, color banding on smooth gradients).
  Opt in with `--dtype fp16` when speed matters more than fidelity;
  stick with `fp32` for production renders.
- **`torch.compile`.** MPS support is improving but still produces
  correctness regressions on some arches. Not enabled by default.

### CUDA

Supported but not the primary target. Keep the code path identical — no
CUDA-specific branches in `Upscaler`.

### CPU

Must stay correct and testable. All unit tests run on CPU in CI so
contributors without Apple Silicon can work on the project.

## Testing strategy

- **Unit tests** run on CPU only and assert shape/value invariants of the
  public API. They must not download weights.
- **Integration tests** (future) will exercise the learned backbone on a
  tiny fixture image and assert output determinism given a fixed seed.
- **Benchmarks** (future) live under `benchmarks/` and are run manually;
  they are not part of CI.

## Non-goals

- A GUI. This is a library + CLI.
- Training. `metalgrow` does inference only; training happens upstream in
  the backbone's own repo.
- Video. Frame-by-frame video SR is a separate concern and would warrant
  its own package.
