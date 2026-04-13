# Models

`metalgrow` ships three SR backbone families — one analytical baseline and
two learned models. Pick one on the CLI (`--backbone <name>`) or from Python
(`Upscaler(backbone=...)`).

| Name             | Type        | Scales | Quality (subjective) | Speed (CPU) | Speed (MPS) | Memory     | Best for                          |
| ---------------- | ----------- | ------ | -------------------- | ----------- | ----------- | ---------- | --------------------------------- |
| `bicubic`        | analytical  | any    | low                  | very fast   | very fast   | negligible | smoke tests, baselines, CI        |
| `realesrgan-x2`  | CNN (RRDB)  | 2      | high                 | slow        | fast        | ~1.1 GiB   | photo restoration, real-world SR  |
| `realesrgan-x4`  | CNN (RRDB)  | 4      | high                 | slow        | fast        | ~1.1 GiB   | photo restoration, real-world SR  |
| `swinir-x2`      | transformer | 2      | very high            | slow        | medium      | ~1.1 GiB   | clean photographic content        |
| `swinir-x4`      | transformer | 4      | very high            | slow        | medium      | ~1.1 GiB   | clean photographic content        |

## How to choose

- **Bicubic** is the default for a reason: zero deps, instantaneous, and
  always available. Use it when you only need an obvious size change with no
  perceptual restoration (e.g. CI, layout previews).
- **Real-ESRGAN** is the workhorse for noisy or compressed inputs (web
  imagery, JPEG artefacts, mixed sources). It hallucinates plausible texture
  aggressively, which is great on real-world content and a liability on
  synthetic / line-art content.
- **SwinIR** trades runtime for sharper, more faithful detail on clean
  photographic input. The bundled weights are the *classical SR* DIV2K
  variant — trained on clean images, so it doesn't denoise as aggressively as
  Real-ESRGAN. Slower per pixel due to the attention layers, and a handful
  of ops fall back to CPU on MPS.

## Benchmark snapshot

Numbers from an Apple Silicon dev machine, `torch 2.11`, 256×256 synthetic
input, median of 3 iterations after 1 warm-up iter. Peak memory is the
high-water mark of `torch.mps.driver_allocated_memory()` across weights +
activations. Reproduce with `python -m benchmarks.run` — see
[benchmarks.md](./benchmarks.md).

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

Read these as directional, not absolute — another M-series chip, a thermal
cycle, or a different torch build will move the numbers. CUDA rows are
produced separately on a GPU runner and published to
[benchmarks.md](./benchmarks.md).

**Observations from this run:**

- Real-ESRGAN on MPS is **10–50×** faster than CPU at the same scale.
- SwinIR on MPS is only **~2×** faster than CPU — its attention layers hit
  MPS op-gaps and silently fall back to CPU (see
  [architecture.md § MPS quirks](./architecture.md#mps-quirks)).
- Bicubic is effectively free. MPS dispatch overhead sometimes makes it
  slower than CPU at tiny sizes; that flips at larger inputs.
- Peak memory is dominated by the weights (~67 MB on disk) plus the MPS
  allocator pool's reservation — roughly a gigabyte in practice.

## Memory & tiling

Both learned backbones run through `Upscaler`'s tiler with backbone-specific
defaults:

| Backbone        | `default_tile` | `default_tile_pad` |
| --------------- | -------------- | ------------------ |
| `realesrgan-*`  | 256            | 16                 |
| `swinir-*`      | 128            | 16                 |

SwinIR's smaller tile reflects the higher activation memory of the attention
blocks. Override either with `--tile` / `--tile-pad` (or `tile=` /
`tile_pad=` in Python). `--tile 0` disables tiling entirely.

## Adding a new backbone

See [`architecture.md`](./architecture.md) for the plugin interface. In
short: subclass `Backbone`, register it via `metalgrow.backbones.register`,
and (if it needs weights) add a `WeightSpec` entry in
`metalgrow.weights.REGISTRY` so the model registry CLI
(`metalgrow models …`) can manage the cache.
