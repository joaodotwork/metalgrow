# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Install (editable, with uv):
```bash
uv venv && source .venv/bin/activate
uv pip install -e . -r requirements-dev.txt   # dev deps: ruff, pytest, pre-commit
```

Run the CLI:
```bash
metalgrow info                                    # print torch + backend availability
metalgrow upscale input.jpg out.png --scale 2     # bicubic baseline
metalgrow upscale input.jpg out.png -s 4 -d mps   # force device
```

Lint / test (contributors are expected to run both before a PR):
```bash
ruff check .
ruff format .
pytest                                            # full suite (CPU-only)
pytest tests/test_upscaler.py::test_name -v       # single test
```

When adding a learned backbone, watch for silent CPU fallback:
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 metalgrow upscale ...
```

## Architecture

See `docs/architecture.md` for full rationale and `docs/adr/` for design decisions.

- `metalgrow.device` — resolves a `torch.device` from a preference string (`auto | mps | cuda | cpu`). Order under `auto`: MPS → CUDA → CPU. Must stay import-cheap (no model code) so `metalgrow info` and tests stay fast.
- `metalgrow.backbones` — pluggable SR backbones. `base.Backbone` is the ABC (accepts `[N,C,H,W]` float tensor in `[0,1]` on `self.device`, returns upscaled tensor of same dtype/device; declares `name`, `supported_scales`, `input_channels`, `default_tile`, `default_tile_pad` as class vars). `registry` is the `register` / `get_backbone` / `list_backbones` API. Built-ins: `bicubic` (weightless), `realesrgan`, `swinir`.
- `metalgrow.weights` — weight cache + download (`~/.cache/metalgrow/`) with SHA256 verification and TTY-aware progress. Backbones must call into here, not download inline.
- `metalgrow.upscaler` — the `Upscaler` class is the stable public API. Owns PIL↔tensor conversion, alpha handling (`alpha_mode`: `separate` = RGB→backbone, alpha→bicubic when `backbone.input_channels == 3`; `ink` = composite over white, backbone, luminance→alpha via `metalgrow.ink`; `auto` picks ink for single-colour transparent art), output clamping, file I/O, and **tiled inference** (`_tiled_forward` with feathered overlap blending, auto-triggered when image > `backbone.default_tile`). Holds one `Backbone` instance; never contains model logic itself.
- `metalgrow.ink` — single-ink detection (`is_single_ink`) and the `InkMatte` coverage mapping used by `alpha_mode="ink"`. See ADR 0003.
- `metalgrow.bitmap` — 1-bit line-art masters (`to_bitmap`): coverage resampled to target ppi in strips bit-identical to a single pass, then thresholded.
- `metalgrow.batch` — directory / glob batch mode, progress reporting, skip-existing logic. Wraps `Upscaler` for multi-file runs.
- `metalgrow.cli` — thin typer app. Commands: `upscale`, `bitmap`, `descreen`, `info`, `models {list,download,rm}`, batch when `SRC` is a directory or glob. Must never contain model logic; flags that need model state belong on `Upscaler` or `batch`.

### Invariants to preserve when extending

- **Apple Metal first, but no device-specific branches in `Upscaler` or backbones.** CUDA and CPU paths must be identical to MPS at the call site.
- **New backbones go under `metalgrow/backbones/<name>.py`**, subclass `Backbone`, and self-register via `registry.register(name, factory)` in `backbones/__init__.py`. Lazy weight load on first `upscale()` call so `info` and unit tests stay weightless. Pre/post-processing (tensor layout, value range, fp16 quirks) lives on the backbone, not on `Upscaler`.
- **Tiled inference lives on `Upscaler`** (`_tiled_forward`) and is backbone-agnostic. Backbones declare `default_tile` / `default_tile_pad`; the Upscaler auto-tiles when the input exceeds the tile size. New backbones should set sensible defaults rather than forcing callers to pass them.
- **Unit tests run CPU-only** and must not download weights — gate weighted tests behind a marker/skip. Integration tests are future work; benchmarks (`benchmarks/`) run manually and are not part of CI.
- **Default dtype is float32.** fp16 is opt-in via `--dtype fp16`; numerically noisier for SR until benchmarked per-model.
- **Significant design decisions get an ADR** under `docs/adr/NNNN-slug.md` (see `0001-sr-backbone-dependency.md` for the format).

Non-goals: GUI, training, video SR.

## Session workflow

Contributors: use an ordinary feature branch and open a PR against `main`. Releases follow [`docs/releasing.md`](docs/releasing.md).

The maintainer runs sessions with Plantas, which is local tooling, not part of the repo (`.claude/` is gitignored). When `plantas` and `.plantas.config.json` are present:

- Always `plantas list` before starting work.
- Use `plantas start <id> "<desc>" [branch] [areas]` / `plantas end <id>` — never manipulate worktrees or `sessions.json` directly.
- In session worktrees use `plantas merge` (not `git merge`) because `.claude/` is symlinked.
- In the main worktree use `plantas sync` (not `git pull`) so coordination files don't block the pull.
- `context.md` is auto-regenerated by the post-commit hook — never edit it manually.
