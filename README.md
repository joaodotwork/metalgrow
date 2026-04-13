# metalgrow

> AI-powered image upscaler accelerated on Apple Metal (MPS).

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
- 🧩 **Pluggable backbone** — swap the upscaling core without touching the CLI
- 🪶 **Tiny surface** — one `Upscaler` class, one CLI entrypoint
- 🧪 **Tested** — pytest suite with a CPU-only baseline test

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

```bash
metalgrow info
metalgrow upscale input.jpg out.png --scale 2
metalgrow upscale input.jpg out.png --scale 4 --device mps
```

| Flag        | Default | Description                           |
| ----------- | ------- | ------------------------------------- |
| `--scale`   | `2.0`   | Upscale factor (1.01–8.0)             |
| `--device`  | `auto`  | `auto` \| `mps` \| `cuda` \| `cpu`    |

### Library

```python
from PIL import Image
from metalgrow import Upscaler

upscaler = Upscaler(device="auto")  # auto | mps | cuda | cpu
result = upscaler.upscale(Image.open("input.jpg").convert("RGB"), scale=2.0)
result.save("out.png")
```

## Project layout

```
src/metalgrow/
  device.py      # device auto-selection (MPS → CUDA → CPU)
  upscaler.py    # Upscaler class — swap in a learned backbone here
  cli.py         # typer CLI
tests/
```

## Roadmap

- [ ] Integrate a learned SR backbone (Real-ESRGAN default)
- [ ] Tiled inference for large images
- [ ] Batch / directory mode
- [ ] Model registry + weight download
- [ ] Benchmarks: MPS vs CPU vs CUDA
- [ ] GitHub Actions CI (lint + test on macOS + Linux)

## Contributing

Issues and PRs are welcome. Please run `ruff` and `pytest` before opening a PR.

## License

MIT — see [LICENSE](LICENSE).

---

<sub>Made with 🔩 on Apple Silicon.</sub>
