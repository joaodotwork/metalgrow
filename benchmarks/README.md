# Benchmarks

Reproducible benchmark harness for metalgrow backbones.

## Running locally

```bash
# All backbones × scales on the auto-selected device, skip uncached weights
python -m benchmarks.run --skip-uncached

# Pin to CPU (works everywhere, useful for contributor machines)
python -m benchmarks.run --device cpu --backbones bicubic --scales 2,4

# Write a markdown fragment into the docs
python -m benchmarks.run --device auto --output docs/benchmarks.md
```

### Inputs

The runner generates synthetic RGB noise with a fixed seed (`SEED = 0`). No
fixtures are downloaded and the exact same pixel data is produced on every
machine, so results are only shaped by device + backbone + scale + size —
not by which image was handy.

### Metrics

- **Median (s)** — wall time of the median iteration, after `--warmup` iters.
- **MP/s** — output megapixels per second = `out_w * out_h / median_s`.
- **Peak (MiB)** — device-side peak allocation during the measured run:
  - `cuda`: `torch.cuda.max_memory_allocated()` after `reset_peak_memory_stats()`
  - `mps`: delta of `torch.mps.driver_allocated_memory()` around the run
  - `cpu`: not reported (no reliable device-side counter; RSS deltas are noisy)

## CUDA numbers

GitHub's hosted runners don't have GPUs, so CUDA rows are produced on a
CUDA-enabled runner (cloud or self-hosted) via `workflow_dispatch`. The
rendered markdown is committed under `docs/benchmarks.md`; JSON output can
be combined across devices with any script that concatenates `results[]`.

## Not part of CI

This harness intentionally lives outside `tests/` — it is **not** executed
on push/PR. A long-running benchmark in CI produces flaky numbers (noisy
neighbours, runner variability) and blocks contributor workflows. Run it
manually when weights, backbones, or kernels change.
