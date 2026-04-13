# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] — 2026-04-13

### Fixed

- **PyPI project description.** `pyproject.toml` now declares
  `readme = "README.md"` so hatchling includes the long description in the
  built distributions. v0.1.0 shipped without one and rendered as *"The
  author of this package has not provided a project description"* on PyPI
  (#26). Package contents are otherwise identical to v0.1.0.

## [0.1.0] — 2026-04-13

First public release. Everything below is an introduction rather than a diff
against a prior version.

### Added

- **Core pipeline.** `Upscaler` class (public API), a single `torch.device`
  selector (`auto → mps → cuda → cpu`), and a `typer` CLI with `upscale`,
  `info`, and `models {list,download,rm}` commands.
- **Backbones.** Pluggable registry behind a `Backbone` ABC. Ships with
  `bicubic` (analytical, weightless), Real-ESRGAN (`realesrgan-x2`,
  `realesrgan-x4`), and SwinIR (`swinir-x2`, `swinir-x4`). Learned backbones
  route through [spandrel](https://github.com/chaiNNer-org/spandrel).
- **Tiled inference.** `Upscaler._tiled_forward` with feathered overlap
  blending, auto-triggered when either image dimension exceeds the
  backbone's `default_tile`. Backbones declare tile geometry; the control
  flow is shared.
- **Batch mode.** `metalgrow upscale SRC DST` accepts a directory or glob
  and writes mirrored filenames into `DST` with `--skip-existing` and
  parallel I/O workers. Inference stays serial on the device.
- **Model registry.** `metalgrow.weights` caches weights under
  `~/.cache/metalgrow/` (overridable via `METALGROW_CACHE_DIR`) with
  sha256 verification on every call; corrupt files are removed so a retry
  can re-download. `metalgrow models` inspects and manages the cache.
- **Alpha handling.** `Upscaler` runs RGB-only backbones on the RGB plane
  and bicubic-upscales the alpha channel in parallel, then recombines —
  preserving transparency without corrupting it.
- **MPS ergonomics.** Automatic `PYTORCH_ENABLE_MPS_FALLBACK=1` so missing
  ops fall back to CPU silently instead of aborting; opt out with
  `METALGROW_DISABLE_MPS_FALLBACK=1`.
- **Benchmarks.** Reproducible harness at `benchmarks/run.py` driven by
  seeded synthetic fixtures, with device-aware peak-memory reporting and
  markdown / JSON output. Not part of CI.
- **Release automation.** Tag-triggered workflow builds sdist + wheel,
  publishes to PyPI via trusted publishing (OIDC), and attaches artifacts
  to a GitHub Release. See [`docs/releasing.md`](docs/releasing.md).
- **Docs.** Usage guide (CLI + library), model comparison with benchmark
  numbers, full benchmark tables, architecture overview, and release
  process. See [`docs/README.md`](docs/README.md).
- **CI.** Lint (ruff) + test (pytest) matrix on macOS and Linux,
  Python 3.11 + 3.12.

### Known limitations

- CUDA benchmark numbers are not yet populated — GitHub-hosted runners
  don't expose GPUs. Run `benchmarks/run.py` on any CUDA-enabled machine
  and append to [`docs/benchmarks.md`](docs/benchmarks.md).
- SwinIR on MPS is only ~2× faster than CPU at these sizes: several
  attention-adjacent ops fall back to CPU via the MPS fallback. Tracked
  as a quality-of-life issue against upstream torch.
- Peak memory on MPS is measured as the high-water mark of
  `driver_allocated_memory`, which includes allocator pool reservations —
  interpret it as "working set size", not a strict activation peak.

[Unreleased]: https://github.com/joaodotwork/metalgrow/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/joaodotwork/metalgrow/releases/tag/v0.1.1
[0.1.0]: https://github.com/joaodotwork/metalgrow/releases/tag/v0.1.0
