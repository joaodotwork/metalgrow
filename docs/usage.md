# Usage

Two entry points: a `typer` CLI (`metalgrow …`) and a small Python API
(`from metalgrow import Upscaler`). Both route through the same
`Upscaler` class, so feature parity is automatic.

## CLI

### Single file

```bash
metalgrow upscale input.jpg out.png --scale 2
metalgrow upscale input.jpg out.png --scale 4 --backbone realesrgan-x4
metalgrow upscale input.png out.png --scale 2 --device cpu --dtype fp32
```

The destination directory is created if it doesn't exist. Alpha channels
are preserved: RGB is upscaled through the backbone, the alpha plane is
upscaled bicubically in parallel, and the two are recombined.

### Directories and globs

Point `SRC` at a directory or a glob — `DST` must be a directory. Matched
inputs are upscaled into `DST` with their original filenames.

```bash
metalgrow upscale ./photos ./photos-upscaled --scale 2 --backbone realesrgan-x2
metalgrow upscale "./photos/*.png" ./out --scale 2 --skip-existing
metalgrow upscale ./photos ./out --workers 8
```

- `--skip-existing` leaves already-processed outputs in place (useful to
  resume an interrupted run).
- `--workers` controls parallel I/O (open/save). Inference stays serial on
  the chosen device — the GPU is a shared resource.
- Glob patterns go in quotes so the shell doesn't expand them first.

### Managing weights

Learned backbones download their weights on first use into
`~/.cache/metalgrow/` (or `$METALGROW_CACHE_DIR`). Every download is
sha256-verified — a corrupt file is removed on the next run so a retry can
re-download cleanly.

```bash
metalgrow models list                       # see what's cached
metalgrow models download realesrgan-x4     # pre-fetch before going offline
metalgrow models rm swinir-x4               # free up disk
```

### Info & diagnostics

```bash
metalgrow info
# → torch version, MPS/CUDA availability, device auto-selection
```

If something goes wrong on MPS, try `--device cpu` to confirm it's a
backend issue, and see [architecture.md § MPS quirks](./architecture.md#mps-quirks)
for the escape hatches (fallback toggle, dtype).

### Flag reference

| Flag                  | Default       | Description                                                     |
| --------------------- | ------------- | --------------------------------------------------------------- |
| `--scale`, `-s`       | `2.0`         | Upscale factor (1.01–8.0). Learned backbones lock this to their trained scale. |
| `--device`, `-d`      | `auto`        | `auto` \| `mps` \| `cuda` \| `cpu`                              |
| `--backbone`, `-b`    | `bicubic`     | `bicubic` \| `realesrgan-x{2,4}` \| `swinir-x{2,4}`             |
| `--dtype`             | `fp32`        | `fp32` \| `fp16`. fp16 is MPS-only, ~2× faster, noticeably noisier on smooth gradients. |
| `--tile`              | backbone hint | Tile size in input px for tiled inference. `0` disables.        |
| `--tile-pad`          | backbone hint | Context padding per tile edge. Sized to the backbone's receptive field. |
| `--skip-existing`     | off           | Batch mode only. Skip outputs that already exist.               |
| `--workers`, `-j`     | `4`           | Batch mode only. Parallel I/O workers (inference stays serial). |

## Library

```python
from PIL import Image
from metalgrow import Upscaler

upscaler = Upscaler(backbone="realesrgan-x2", device="auto")
result = upscaler.upscale(
    Image.open("input.jpg").convert("RGB"),
    scale=2.0,
    tile=256,        # optional; backbone ships with sensible defaults
    tile_pad=16,
)
result.save("out.png")
```

### `Upscaler(backbone, device, dtype)`

- `backbone` — name from `metalgrow.backbones.list_backbones()`. Default `"bicubic"`.
- `device` — `"auto"` | `"mps"` | `"cuda"` | `"cpu"`. `auto` prefers
  MPS → CUDA → CPU. Resolved once at construction.
- `dtype` — `torch.float32` (default) or `torch.float16`.

### `Upscaler.upscale(image, scale, tile=None, tile_pad=None)`

Takes a `PIL.Image`, returns a `PIL.Image`. Alpha handling is automatic:
RGBA inputs run the backbone on the RGB plane and bicubic the alpha.
`tile` / `tile_pad` override the backbone's defaults; leave them `None`
to auto-tile whenever either image dimension exceeds the default tile size.

### `Upscaler.upscale_file(src, dst, scale, tile=None, tile_pad=None)`

File-in, file-out convenience. Creates `dst.parent` if needed. Used by
the CLI's single-file path.

### Batch mode from Python

```python
from pathlib import Path
from metalgrow import Upscaler
from metalgrow.batch import upscale_batch

upscaler = Upscaler(backbone="realesrgan-x2")
upscale_batch(
    upscaler,
    sources=list(Path("./photos").glob("*.jpg")),
    dst_dir=Path("./out"),
    scale=2.0,
    skip_existing=True,
    workers=4,
)
```

Inference is serial (one image at a time on the device); `workers` only
parallelises image I/O around the inference step.

## Environment variables

| Name                              | Effect                                                    |
| --------------------------------- | --------------------------------------------------------- |
| `METALGROW_CACHE_DIR`             | Override weight cache location (default `~/.cache/metalgrow`). |
| `PYTORCH_ENABLE_MPS_FALLBACK`     | Set to `1` when MPS is selected; metalgrow sets this automatically unless you opt out. |
| `METALGROW_DISABLE_MPS_FALLBACK`  | Set to `1` to prevent metalgrow from enabling the MPS fallback — useful for locating an unsupported op. |

## Common pitfalls

- **"ValueError: backbone … supports scales [2.0, 4.0]"** — learned backbones
  bake the scale into the weights. Match `--scale` to the backbone suffix:
  `realesrgan-x2` ↔ `--scale 2`, `realesrgan-x4` ↔ `--scale 4`.
- **Slow first run** — learned backbones download weights (67–130 MB) on
  first use. Pre-fetch with `metalgrow models download <name>` to keep
  timing runs honest.
- **Out-of-memory on large images** — `Upscaler` auto-tiles when either
  dimension exceeds the backbone's `default_tile`. If you're still hitting
  OOM, drop the tile size (`--tile 128`) or switch from SwinIR to
  Real-ESRGAN.
- **Bicubic looks bad** — it's a baseline, not an upscaler. Switch to a
  learned backbone for anything user-facing.
- **SwinIR is slow on MPS** — several attention-adjacent ops still fall
  back to CPU on the MPS backend. Expect roughly 2× the wall time of
  Real-ESRGAN at the same scale. See [models.md](./models.md#benchmark-snapshot).
