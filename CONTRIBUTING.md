# Contributing to metalgrow

Thanks for your interest in contributing! This guide covers the local setup
and the conventions we follow.

## Dev setup

We use [`uv`](https://docs.astral.sh/uv/) but plain `pip` also works.

```bash
git clone https://github.com/joaodotwork/metalgrow.git
cd metalgrow

# uv
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# or plain pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Install the pre-commit hooks once per clone:

```bash
pre-commit install
```

This runs `ruff` and `ruff-format` on every commit. The same checks run in
CI, so installing the hook saves a round trip.

## Running tests and lint

```bash
pytest               # full test suite (CPU only, no weights required)
ruff check .         # lint
ruff format .        # auto-format
ruff format --check . # CI-equivalent format check
```

The test suite is intentionally CPU-only so it passes on any machine and on
Linux CI runners without Apple Silicon.

## Branch and PR conventions

- Branch from `master`. Name branches by intent: `feat/...`, `fix/...`,
  `chore/...`, `docs/...`, `test/...`.
- Keep commits focused. Prefer one commit per issue when a branch closes
  multiple issues; otherwise group related changes.
- Commit subject line uses the same conventional prefix (`feat:`, `fix:`,
  `chore:`, `docs:`, `test:`, `ci:`) followed by a short imperative summary.
- Reference the issue in the commit body with `Closes #N`.
- Before opening a PR: `ruff check .`, `ruff format --check .`, and
  `pytest` must all pass. CI will re-run them on macOS and Linux.

## Adding a new backbone

The public API (`Upscaler`, CLI flags) is stable. New super-resolution
models plug in behind it. See
[`docs/architecture.md`](docs/architecture.md#plugging-in-a-learned-backbone)
for the expected shape of that change — briefly:

1. Add `src/metalgrow/backbones/<name>.py` exposing a `load_model(device)`
   that returns an `nn.Module` in `eval()` mode.
2. Load the backbone lazily in `Upscaler.__init__` so `metalgrow info` and
   unit tests stay fast.
3. Replace the `F.interpolate` call in `Upscaler.upscale` with the model
   forward pass. Pre/post-processing belongs on the backbone, not on
   `Upscaler`.

Weights must not be committed; a model registry is planned (see the M3
milestone).

## Reporting issues

Bugs, questions, and feature requests are all welcome on the
[issue tracker](https://github.com/joaodotwork/metalgrow/issues). Include
your OS, Python version, and the output of `metalgrow info` when relevant.
