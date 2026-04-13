# Benchmarks

Reproducible numbers for `(device, backbone, scale)` across the supported
backbones. See [`benchmarks/README.md`](../benchmarks/README.md) for how to
reproduce these locally.

- **Inputs:** seeded RGB noise (`SEED=0`), 256×256 by default.
- **MP/s:** output megapixels per second at the median of `--iters` runs.
- **Peak (MiB):** device-side peak allocation (CUDA: `max_memory_allocated`;
  MPS: delta of `driver_allocated_memory`; CPU: not reported).

## Results

_Run `python -m benchmarks.run --output docs/benchmarks.md` to regenerate
this section. The file is overwritten on each run — commit the update
alongside any change that could shift performance (new backbone, kernel
path, tile defaults)._
