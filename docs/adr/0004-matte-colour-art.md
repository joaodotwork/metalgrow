# ADR 0004: Matte for colour art on paper

- **Status:** Accepted
- **Date:** 2026-11-09

## Context

The Vanguard's Velvet book places 19 full-colour drink illustrations on
coloured page backgrounds. They are renders on near-white paper, upscaled 4x
with `metalgrow upscale`, and the paper has to come out:

- the **glass** (the drink in its vessel, with highlights) must stay **solid**:
  a page colour showing through the drink reads as a printing fault;
- the **cast shadow** must stay **translucent**, so it falls on whatever
  colour the page is;
- the paper itself must be **fully transparent**, including its faint unevenness.

This was done by a standalone script, `matte_drinks.py`, kept in a Drive
working folder with no version control or tests. Its `--seal` option (added
2026-11-06) is in the pipeline of every illustration the book prints. This
ADR records its move into metalgrow as `metalgrow matte` (#37).

### Why ink mode doesn't cover it

Ink mode (ADR 0003) assumes **one ink**. It reads luminance back as coverage
over a single ink colour, which is right for line art and wrong here twice:

- it would flatten a multi-colour drawing to one colour;
- it would make every light pixel translucent, the glass highlights included,
  so the page shows through the drink.

`auto` already refuses multi-colour art for that reason. Matting colour art
needs to know *where the glass is*, not just how dark each pixel is.

## Options considered

### A. Plain colour-to-alpha everywhere

Unmix every pixel against the paper colour: alpha is the smallest value for
which `pixel = alpha·colour + (1 − alpha)·paper` has a colour in gamut. The
shadow comes out right, translucent sepia that recomposites exactly over the
paper. The glass does not: its highlights are as light as the paper, so they
become holes, and the whole drink goes partly see-through.

### B. A learned matting model

Heavy, a new model dependency, and trained for portrait or object mattes. It
would not reproduce the book's "solid glass, translucent shadow" convention
without retraining.

### C. Colour-to-alpha outside a glass region found by flood fill

The approach `matte_drinks.py` takes:

1. **Paper** is the median colour of a 16 px band along all four edges.
2. **Unmix** every pixel against it (Option A), which gives `a` and an
   un-premultiplied colour.
3. **Glass** is every pixel the paper cannot reach from the border through
   light pixels (`a < T`, `T = 0.5`). The glass outline is dark, so the flood
   stops at it, and the highlights inside it stay glass. Enclosed regions under
   `MIN_AREA` (0.4 % of the image) are shadow hatching, not glass, and stay
   outside. Holes inside the glass are filled.
4. **Inside the glass the render is kept as drawn, at alpha 1**, except a ring
   `EDGE` (3) px wide at the boundary. That ring keeps the unmixed matte, so the
   antialiased outline does not carry a paper-white fringe onto a dark page.
5. **Outside the glass**, the unmixed alpha and colour are used: the shadow is
   translucent.
6. **Haze.** No paper is perfectly flat, and unmixing leaves alpha of
   1–8/255 across the background. Alpha below `HAZE` (4/255) is cleared and the
   rest rescaled to stay continuous. That shifts a recomposite over the paper
   by at most `HAZE`.

**Seal** (opt-in) covers glass whose outline is not closed: renders with white
highlights reaching the edge of the glass and no dark outline there. Without
it, the paper floods in through the highlight and leaves a hole. Seal:

- closes gaps up to `SEAL_R` (0.6 % of the width) between *neutral* strokes,
  those with chroma ≤ `SEAL_CHROMA`;
- keeps as glass every area of at least `SEAL_HOLE` (0.05 %) that the closed
  strokes, together with any other strong pixels, then enclose;
- counts a sealed area only within each row's span of the glass found in step 3.

Cast shadows are hatched in warm sepia, above `SEAL_CHROMA`, so their gaps
never close. The row span stops a sealed scrap below or beside the drink from
becoming glass. Seal is opt-in because it changes which pixels are glass. A
plain run is unchanged by its addition.

## Decision

**Option C**, ported verbatim from `matte_drinks.py` into `metalgrow.matte`.
It is exposed as `metalgrow matte SRC DST [--seal] [--dpi] [--names]`.

- **Parity is the acceptance test.** Same constants, same float64 arithmetic,
  same 8-bit quantisation, and the same PNG settings (`--dpi` default 300,
  zlib level 6). On the 19 Velvet sources, metalgrow's decoded RGBA and pHYs
  match the script's exactly, with and without `--seal`.
- **scipy is an optional extra, `metalgrow[matte]`.** The algorithm needs
  `label`, `sum`, `distance_transform_edt`, `binary_fill_holes` and
  `binary_erosion`. Reimplementing those in numpy would be slow on 4000 px
  images, and parity would have to prove the rewrite. A required dependency
  would make every upscale-only install larger. `metalgrow.matte` imports
  scipy lazily and raises `MatteDependencyError` with the install hint. The
  CLI prints that hint as one line and exits 1. scipy is in the dev extra,
  so tests and CI run the matte.
- **Pure functions, no torch.** `metalgrow.matte` holds no model code and does
  not import torch. The package `__init__` now loads `Upscaler` and
  `get_device` lazily (PEP 562), so `import metalgrow.matte` stays torch-free.
  The CLI module still imports torch at load time; that is outside this change.
- **The Vanguard's `--drinks` naming becomes a generic `--names MAP.csv`.**
  Each row is `source filename,output filename`. Unlisted sources are written
  as `<stem>.png`. These are errors, raised before anything is written:
  - a source listed twice;
  - two sources mapped to the same output;
  - a mapped source that isn't in the input;
  - an output that isn't a plain `.png` file name.

  metalgrow never parses the Vanguard's markdown. An explicit map is also
  easier to check than a template.
- **Batch** takes a file, directory or glob, like the other commands, with no
  `_NNNN` filename filter.

## Consequences

- The Velvet pipeline becomes `metalgrow upscale … && metalgrow matte --seal …`,
  versioned and tested, and the Drive script can be retired.
- Tests are CPU-only, on small synthetic renders. Each rule (opaque glass,
  EDGE ring, flood threshold, MIN_AREA, border median, haze clear, every part
  of seal, and the install hint) has a test that fails when the rule is mutated.
- The thresholds are tuned for renders on near-white paper with dark or
  neutral glass outlines. Art on toned paper with low-contrast outlines may
  need its own constants. They stay module constants until a second use
  case shows which ones need to be options.
- The matte is per-pixel, so it is only as sharp as its input. Upscale first.
