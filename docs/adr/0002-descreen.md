# ADR 0002: Descreening halftone print before upscaling

- **Status:** Accepted
- **Date:** 2026-10-01
- **Context:** issue #29

## Context

Scanned newspaper and magazine pages carry the printer's halftone screen: a
regular dot lattice. SR backbones treat the dots as detail and sharpen them
into moiré, so the screen has to be removed *before* the backbone runs.

The motivating job is a set of 1940s Norwegian newspaper page scans (200 dpi,
RGB) going into an academic publication. The requirement was to remove the
screen **without otherwise altering** the pages. Measuring them showed:

- 45° / 135° dot screens at a 2.83–3.09 px period (≈65–71 lpi), with
  harmonics on the axes and at the spectrum corners;
- **several different screens on one page** (e.g. 2.88 px and 3.00 px photos
  side by side), each smeared over a few spectral bins by paper warp;
- dense body type whose line pitch makes compact peaks on the vertical
  frequency axis, and whose stroke rhythm can make peaks on the horizontal
  axis.

## Options considered

### A. Blur / low-pass

Removes the screen, but also softens type and linework everywhere. Rejected,
because it alters exactly what must be preserved.

### B. Global FFT notch at the detected lattice points

This is the textbook approach. Prototyped first, it failed on real pages:
it found only the dominant screen (other photos were left screened), and
widening it to catch all of them notched the text-line ridge, which put
visible horizontal ripples into every line of type.

### C. Learned descreening model

There's no well-established, permissively licensed model, and it would
bring the hallucination risk we're trying to avoid for archival material.

### D. Local (windowed) notch filter with lattice validation

A short-time Fourier transform: 256 px windows at 50 % overlap with
sqrt-Hann analysis and synthesis, which reconstructs exactly when nothing
is changed. Each window is judged on its own:

1. **Prominence** is log-magnitude over a horizontal *and* a vertical 1-D
   spectral background, whichever is smaller. Compact peaks score high;
   axis-aligned ridges (text-line rhythm, rules) score ≈0.
2. **Lattice check:** the strongest seed peak needs a partner at least 30°
   away in direction and within 5 % in period. That's what a square dot
   screen looks like. Text either has all its peaks on one axis, or on two
   axes at unrelated periods, so it fails the check.
3. **Notch:** seeds grow by hysteresis into connected, moderately prominent
   bins, which picks up smeared fundamentals and harmonics. The region is
   made conjugate-symmetric and feathered by about 1 bin. Frequencies below
   0.08 cycles/px are never touched.
4. Windows with no valid lattice are passed through unchanged.

## Decision

**Option D**, as `metalgrow.descreen`, opt-in through `--descreen` on
`upscale` and a standalone `metalgrow descreen` command. `Upscaler` owns the
step and runs it before the backbone, so batch mode inherits it.

It runs on CPU in float32 whatever the inference device is. A full
newspaper page takes about 3 s, which is small next to SR inference, and
results are identical across MPS, CUDA and CPU. This keeps the "no
device-specific branches" invariant.

## Consequences

- On the four motivating pages, only the photos change. Text, rules and
  blank paper come back unchanged, and the photos lose 7–30× of their
  screen energy.
- Line screens (a 1-D halftone) are deliberately not handled, because the
  lattice check rejects them. They are rare in newspaper print.
- Very small or very light photos whose screen sits near the threshold may
  be left untouched rather than half-processed. A missed photo stays as it
  was, which is the safe failure mode for archival work. `threshold` is a
  keyword argument on the library API for tuning.
- The window size (256 px) suits screens of roughly 2–10 px period. That
  covers common newspaper and magazine rulings scanned at 150–600 dpi.
