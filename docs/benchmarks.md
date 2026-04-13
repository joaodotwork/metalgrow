# Benchmarks

Reproducible numbers for `(device, backbone, scale)` across the supported
backbones. See [`benchmarks/README.md`](../benchmarks/README.md) for how to
reproduce these locally.

- **Inputs:** seeded RGB noise (`SEED=0`), 256×256 by default.
- **MP/s:** output megapixels per second at the median of `--iters` runs.
- **Peak (MiB):** device-side peak allocation (CUDA: `max_memory_allocated`;
  MPS: high-water of `driver_allocated_memory` from before model construction;
  CPU: not reported).

## Results

Host: Darwin arm64 · Python 3.11.15 · torch 2.11.0 · single-chip Apple Silicon.
Numbers taken from **isolated per-backbone runs** — the MPS allocator pool
does not shrink between cells in one process, so the combined `--backbones a,b,c`
invocation only reports truthful peak memory for the first cell.

| Device | Backbone        | Scale | Median (s) | MP/s   | Peak (MiB) |
| ------ | --------------- | ----- | ---------- | ------ | ---------- |
| mps    | `bicubic`       | 2     | 0.003      | 94.1   | 40         |
| mps    | `bicubic`       | 4     | 0.007      | 156.5  | —          |
| mps    | `realesrgan-x2` | 2     | 0.175      | 1.50   | 1130       |
| mps    | `realesrgan-x4` | 4     | 0.708      | 1.48   | 1098       |
| mps    | `swinir-x2`     | 2     | 2.287      | 0.11   | 1144       |
| mps    | `swinir-x4`     | 4     | 2.214      | 0.47   | 1136       |
| cpu    | `bicubic`       | 2     | 0.002      | 123.1  | —          |
| cpu    | `bicubic`       | 4     | 0.007      | 143.4  | —          |
| cpu    | `realesrgan-x2` | 2     | 1.902      | 0.14   | —          |
| cpu    | `realesrgan-x4` | 4     | 7.655      | 0.14   | —          |
| cpu    | `swinir-x2`     | 2     | 5.282      | 0.05   | —          |
| cpu    | `swinir-x4`     | 4     | 5.568      | 0.19   | —          |

### CUDA

Not yet populated — GitHub's hosted runners don't expose GPUs, so CUDA
numbers require a CUDA-enabled runner (self-hosted or cloud). Run:

```bash
python -m benchmarks.run --device cuda --format markdown >> docs/benchmarks.md
```

on such a runner and commit the appended section.

## Reading the numbers

- **Directional, not absolute.** Another M-series chip, a thermal cycle, or
  a different torch build will move the numbers. Re-run locally when a
  decision depends on an exact value.
- **MP/s** is the useful cross-backbone metric — it normalises for scale
  (a 4× model producing 4× more output pixels isn't a free win).
- **Peak on MPS** is the *working set size*, not a strict activation peak.
  It counts the MPS allocator pool's reservation, which torch grows but
  rarely returns. Practical interpretation: "about this much GPU memory
  will be held by this workload".
- **Peak on CPU** is omitted. RSS deltas are too noisy to be useful — the
  Python process, PyTorch's intern caches, and the OS page cache all move.

See [models.md](./models.md) for the human-readable backbone comparison
and guidance on which to pick.
