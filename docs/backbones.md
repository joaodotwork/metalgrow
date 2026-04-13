# Backbones

`metalgrow` ships three SR backbones today. They're all selected via
`--backbone <name>` on the CLI or `Upscaler(backbone=...)` from Python.

| Name             | Type        | Scales | Quality (subjective) | Speed (CPU) | Speed (MPS) | Memory     | Best for                          |
| ---------------- | ----------- | ------ | -------------------- | ----------- | ----------- | ---------- | --------------------------------- |
| `bicubic`        | analytical  | any    | low                  | very fast   | very fast   | negligible | smoke tests, baselines, CI        |
| `realesrgan-x2`  | CNN (RRDB)  | 2      | high                 | medium      | fast        | medium     | photo restoration, real-world SR  |
| `realesrgan-x4`  | CNN (RRDB)  | 4      | high                 | medium      | fast        | medium     | photo restoration, real-world SR  |
| `swinir-x2`      | transformer | 2      | very high            | slow        | medium      | high       | clean photographic content        |
| `swinir-x4`      | transformer | 4      | very high            | slow        | medium      | high       | clean photographic content        |

## How to choose

- **Bicubic** is the default for a reason: zero deps, instantaneous, and
  always available. Use it when you only need an obvious size change with no
  perceptual restoration (e.g. CI, layout previews).
- **Real-ESRGAN** is the workhorse for noisy or compressed inputs (web
  imagery, JPEG artefacts, mixed sources). It hallucinates plausible texture
  aggressively, which is great on real-world content and a liability on
  synthetic / line-art content.
- **SwinIR** trades runtime and memory for sharper, more faithful detail on
  clean photographic input. The bundled weights are the *classical SR* DIV2K
  variant — trained on clean images, so it doesn't denoise as aggressively as
  Real-ESRGAN. Slower per pixel due to the attention layers.

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
