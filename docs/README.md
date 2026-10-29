# metalgrow docs

- [Usage](./usage.md) — CLI recipes, the Python API, env vars, common pitfalls.
- [Models](./models.md) — bicubic / Real-ESRGAN (incl. anime) / SwinIR comparison, benchmark snapshot, tiling defaults.
- [Benchmarks](./benchmarks.md) — full `(device × backbone × scale)` tables; how to reproduce.
- [Architecture](./architecture.md) — module layout, data flow, and extension points.
- [Releasing](./releasing.md) — tag → PyPI workflow, one-time setup, rollback.
- Decisions: [0001 SR backbone dependency](./adr/0001-sr-backbone-dependency.md) ·
  [0002 descreen](./adr/0002-descreen.md) · [0003 ink mode](./adr/0003-alpha-ink-mode.md).
