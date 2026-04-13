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
3. **Backbone** is the actual super-resolution operation. Today this is
   `torch.nn.functional.interpolate` (bicubic); tomorrow it will be a
   learned `nn.Module` (Real-ESRGAN, SwinIR, etc.).
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

The backbone call is intentionally a single line so it can be replaced with
a learned model without reshaping the surrounding code.

### `metalgrow.cli`

A thin `typer` app. Two commands:

- `upscale SRC DST --scale --device` — the real work.
- `info` — prints torch version and backend availability. Useful for
  verifying MPS is picked up on a fresh Apple Silicon machine.

The CLI must never contain model logic. If a flag needs model state, it
belongs on `Upscaler`.

## Extension points

### Plugging in a learned backbone

The intended replacement for the bicubic baseline is a learned `nn.Module`
loaded once per `Upscaler` instance. The expected shape of that change:

1. Add `metalgrow/backbones/realesrgan.py` exposing a `load_model(device)`
   function that returns an `nn.Module` in `eval()` mode with weights
   loaded from a cache directory.
2. In `Upscaler.__init__`, load the backbone lazily (first call) so tests
   and `metalgrow info` stay fast.
3. Replace the `F.interpolate` call in `Upscaler.upscale` with a model
   forward pass. Pre/post-processing (tensor layout, value range) belongs
   on the backbone module, not in `Upscaler`.

### Tiled inference

Large images won't fit in MPS memory at 4× scale. The plan is a `tile` and
`tile_pad` parameter on `Upscaler.upscale` that slices the input, runs the
backbone per tile, and stitches with overlap blending. This lives inside
`Upscaler` because it's backbone-agnostic.

### Model registry

Weights should not be committed. A small registry will map model names to
download URLs + SHA256 + target path under `~/.cache/metalgrow/`. The CLI
will gain `metalgrow models {list,download,rm}`.

## Device notes

### MPS quirks

- Some ops still fall back to CPU silently. Keep an eye on the
  `PYTORCH_ENABLE_MPS_FALLBACK=1` env var when adding a learned backbone.
- `float16` on MPS is faster but numerically noisier for SR — default to
  `float32` until we benchmark per-model.
- `torch.compile` support on MPS is improving but not yet a default.

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
