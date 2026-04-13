# metalgrow

> AI-powered image upscaler accelerated on Apple Metal (MPS).

[![CI](https://github.com/joaodotwork/metalgrow/actions/workflows/ci.yml/badge.svg)](https://github.com/joaodotwork/metalgrow/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-MPS-000000.svg?logo=apple&logoColor=white)](https://developer.apple.com/metal/pytorch/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](#roadmap)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing)

`metalgrow` runs super-resolution on Apple Silicon GPUs through PyTorch's MPS
backend, with graceful fallbacks to CUDA and CPU. It ships with a bicubic
baseline so the pipeline is runnable end-to-end, and exposes a clean seam for
plugging in a learned backbone (Real-ESRGAN, SwinIR, etc.).

---

## Table of contents

- [Features](#features)
- [Requirements](#requirements)
- [Install](#install)
- [Usage](#usage)
- [Project layout](#project-layout)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Features

- 🍎 **Apple Metal first** — PyTorch MPS backend, zero config on Apple Silicon
- 🔁 **Portable** — automatic fallback to CUDA / CPU when MPS isn't available
- 🧩 **Pluggable backbones** — bicubic, Real-ESRGAN (x2/x4), SwinIR (x2/x4)
- 🧱 **Tiled inference** — process arbitrarily large images with feathered overlap blending
- 📂 **Batch mode** — upscale whole directories or globs with a progress bar
- 📦 **Model registry** — `metalgrow models` manages cached weights with sha256 verification
- 🧪 **Tested** — pytest suite covering the CPU baseline, tiling, batch mode, and registry

## Requirements

- Python **3.11+**
- macOS with Apple Silicon for MPS acceleration (optional — CPU/CUDA also work)

## Install

```bash
git clone https://github.com/joaodotwork/metalgrow.git
cd metalgrow
uv venv && source .venv/bin/activate
uv pip install -e .
```

Or with plain pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Usage

### CLI

Single file:

```bash
metalgrow info
metalgrow upscale input.jpg out.png --scale 2
metalgrow upscale input.jpg out.png --scale 4 --device mps --backbone realesrgan-x4
```

Whole directory or glob (writes to a target directory, mirroring filenames):

```bash
metalgrow upscale ./photos ./photos-upscaled --scale 2 --backbone realesrgan-x2
metalgrow upscale "./photos/*.png" ./out --scale 2 --skip-existing
```

| Flag                  | Default       | Description                                                     |
| --------------------- | ------------- | --------------------------------------------------------------- |
| `--scale`, `-s`       | `2.0`         | Upscale factor (1.01–8.0)                                       |
| `--device`, `-d`      | `auto`        | `auto` \| `mps` \| `cuda` \| `cpu`                              |
| `--backbone`, `-b`    | `bicubic`     | `bicubic` \| `realesrgan-x{2,4}` \| `swinir-x{2,4}`             |
| `--dtype`             | `fp32`        | `fp32` \| `fp16` (fp16 is MPS-only and noisier)                 |
| `--tile`              | backbone hint | Tile size in input px for tiled inference (`0` disables)        |
| `--tile-pad`          | backbone hint | Context padding per tile edge (covers backbone receptive field) |
| `--skip-existing`     | off           | Batch mode: skip outputs that already exist                     |
| `--workers`, `-j`     | `4`           | Batch mode: parallel I/O workers (inference stays serial)       |

#### Manage cached model weights

```bash
metalgrow models list
metalgrow models download realesrgan-x4
metalgrow models rm realesrgan-x4
```

Weights cache to `~/.cache/metalgrow/` by default; override with the
`METALGROW_CACHE_DIR` env var. Every download is sha256-verified.

See [`docs/models.md`](./docs/models.md) for a quality / speed / memory
comparison (with benchmark numbers) and guidance on which backbone to pick,
or [`docs/usage.md`](./docs/usage.md) for the full CLI / library reference.

### Library

```python
from PIL import Image
from metalgrow import Upscaler

upscaler = Upscaler(backbone="realesrgan-x2", device="auto")  # auto | mps | cuda | cpu
result = upscaler.upscale(
    Image.open("input.jpg").convert("RGB"),
    scale=2.0,
    tile=256,        # optional — backbone has sensible defaults
    tile_pad=16,
)
result.save("out.png")
```

## Project layout

```
src/metalgrow/
  device.py            # device auto-selection (MPS → CUDA → CPU)
  upscaler.py          # Upscaler class + tiled inference with overlap blending
  batch.py             # directory / glob batch mode
  weights.py           # weight registry, sha256 verification, cache management
  cli.py               # typer CLI (upscale, info, models)
  backbones/           # bicubic, realesrgan, swinir; plugin registry
tests/
docs/
  usage.md             # CLI recipes, library API, env vars, pitfalls
  models.md            # backbone comparison, benchmark snapshot, tiling
  benchmarks.md        # full (device × backbone × scale) tables
  architecture.md      # module layout, data flow, extension points
```

## Roadmap

- [x] Integrate a learned SR backbone (Real-ESRGAN default)
- [x] Tiled inference for large images
- [x] Batch / directory mode
- [x] Model registry + weight download
- [x] Second backbone family (SwinIR)
- [x] GitHub Actions CI (lint + test on macOS + Linux)
- [x] Benchmarks: MPS vs CPU (CUDA pending GPU runner)
- [ ] v0.1.0 release

## Contributing

Issues and PRs are welcome. Please run `ruff` and `pytest` before opening a PR.

## License

MIT — see [LICENSE](LICENSE).

---

<sub>Made with 🔩 on Apple Silicon.</sub>
