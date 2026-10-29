# ADR 0003: Ink mode for single-colour artwork

- **Status:** Accepted
- **Date:** 2026-10-28

## Context

The RGBA path in `Upscaler` runs the backbone on the RGB plane and
bicubic-upscales the alpha channel. That is right for a photo with a cut-out
mask, where the detail is in the colour and the alpha is a smooth outline.

It fails for **ink-on-transparent artwork**: a scanned drawing with the paper
knocked out. All the detail is in the alpha channel and the RGB plane is one
flat colour. The backbone receives a uniform image, and the drawing itself
gets bicubic. The output is no better than `--backbone bicubic`.

The motivating job was 26 vintage ink and linocut illustrations for a
restaurant's print identity, needed at A3 / 300 dpi. They range from 350 to
2,700 px, and their RGB is (0,0,0) everywhere, with 7–28 % of pixels holding
partial alpha (edges and printing grit). One of them was an opaque
black-on-paper scan whose paper also had to be removed.

## Options considered

### A. Run the backbone on alpha directly

Feed alpha as a grey image. This works, but a coverage map is white-on-black,
while SR models are trained on natural images where ink is dark on light.
The backbone also never sees the ink colour, and opaque scans would need a
separate keying step.

### B. Composite over white paper, upscale, read luminance back as coverage

Composite the artwork over white, which gives a normal dark-on-light
drawing, squarely in-distribution for the backbone. Upscale it, then map
luminance back to coverage, `(paper − L) / (paper − ink)`, and emit that as
alpha over a flat plane of the ink colour. An opaque scan on paper is the
same image with alpha ≡ 1, so it takes the same path and its paper comes
out transparent.

### C. A learned matting model

This is heavy, adds a dependency, and solves a harder problem (hair, glass,
soft shadows) than single-colour ink.

## Decision

**Option B**, as `metalgrow.ink`, opt-in through `--alpha-mode ink`
(`Upscaler(alpha_mode="ink")`). `separate` stays the default, so existing
behaviour does not change. `auto` picks ink mode when the image has
transparency and at least 99.5 % of its visible pixels sit within 24/255 of
one colour. Otherwise it falls back to `separate`. An opaque scan is not
auto-detected; it needs `ink` explicitly.

- **Ink colour** is the median of the darkest pixels over white (within 0.02
  luma of the minimum). An earlier version took the median colour of the
  opaque pixels, which broke on an opaque scan carrying a few stray alpha
  values: its opaque pixels were mostly paper, so the "ink" came out white.
  `test_opaque_scan_with_stray_alpha_still_finds_dark_ink` pins this.
- **Paper level** is the 99th-percentile luma. A scan's paper is often a few
  levels below 255, and this knocks it out to exactly alpha 0.
- **Neutral ink returns `LA`**, so it prints as a single ink. On that path
  `reapply` drops an RGB ICC profile, which is invalid on a grey image.
- The resample to the exact target scale runs on a float plane, so the
  coverage mapping sees no 8-bit banding.

## Consequences

- The backbone now sees the drawing. On the motivating set, a 4x Real-ESRGAN
  pass gives clean ink edges where `separate` gave bicubic.
- Picking a backbone matters more on this material:
  - `swinir-x4` (classical-SR weights) faithfully reproduces the blocky
    pixels of sources that were already enlarged.
  - `realesrgan-anime-x4`, added alongside this change, gives the crispest
    edges, but invents hatch strokes in grit and drops faint specks.
  - `realesrgan-x4` kept the most printing grit with no invented strokes.
- Multi-colour artwork is out of scope; `auto` leaves it on `separate`.
- `--dpi` is added alongside, because print jobs are specified as "N dpi"
  rather than "keep the physical size". It overrides the rescaled DPI tag.
