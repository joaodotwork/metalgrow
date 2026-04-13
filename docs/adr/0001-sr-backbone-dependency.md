# ADR 0001: Super-resolution backbone dependency

- **Status:** Accepted
- **Date:** 2026-04-13
- **Context milestone:** M2 — Learned SR backbone (issue #6)

## Context

Issue #6 requires integrating Real-ESRGAN as the first learned SR backbone.
The backbone contract from issue #5 lets us swap implementations freely, so
the question is purely: how do we obtain the RRDBNet architecture and future
SR arches (SwinIR, HAT, …) on the roadmap?

## Options considered

### A. Depend on `basicsr` / `realesrgan` (upstream pip packages)

- Zero model code to maintain.
- Pulls in a heavy transitive graph (opencv, tb-nightly, facexlib, gfpgan,
  scipy, …) that dwarfs the rest of metalgrow's dependencies.
- Maintenance cadence on `basicsr` has been uneven.

### B. Depend on `spandrel` ([chaiNNer-org/spandrel](https://github.com/chaiNNer-org/spandrel), MIT)

- Actively maintained; supports a wide catalog of SR architectures
  (Real-ESRGAN, SwinIR, HAT, DAT, …) behind a single uniform loader.
- Auto-detects architecture from weight files — no per-arch glue code.
- Slim transitive graph: adds only `einops` and `safetensors` on top of
  torch/torchvision.
- Smoke-tested locally: loads `RealESRGAN_x4plus.pth` as `ESRGAN`/RRDBNet
  and produces the expected `16x16 → 64x64` forward pass.

### C. Vendor RRDBNet architecture (~150 LOC, BSD-3 / Apache-2.0)

- No new runtime dependencies. Full control over the model code.
- Cost compounds: every new arch on the roadmap (SwinIR, HAT, …) means
  more vendoring, more license notices, more code to keep current.

## Decision

**Adopt spandrel.**

M2 is explicitly about plugging in multiple learned backbones; spandrel's
design matches that directly. The `Backbone` interface from #5 keeps the
dependency swappable — if spandrel ever becomes a liability we can drop in
vendored arches behind the same interface without touching the rest of the
pipeline.

## Consequences

- Add `spandrel>=0.4` to `pyproject.toml` and `requirements.txt`.
- Weight files stay metalgrow's responsibility: we manage download +
  checksum + cache layout in `src/metalgrow/weights.py`, then hand the
  local path to `spandrel.ModelLoader`.
- Future backbones (SwinIR, HAT, …) only need a new entry in the weights
  registry plus a thin `Backbone` subclass.
