"""Batch / directory upscaling.

Reads and writes happen on a thread pool so I/O can overlap with model
inference, but the inference call itself is serialized on the main thread:
running multiple GPU forwards concurrently from Python threads doesn't speed
anything up and risks contention on MPS, so we prefetch reads and queue
writes around a single in-flight forward.
"""

from __future__ import annotations

import glob as glob_module
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from metalgrow.upscaler import Upscaler

IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})


@dataclass(frozen=True)
class BatchResult:
    processed: int
    skipped: int
    failed: int

    @property
    def total(self) -> int:
        return self.processed + self.skipped + self.failed


def discover_inputs(src: str) -> list[Path]:
    """Resolve ``src`` to a sorted list of input image paths.

    Accepts a single file path, a directory (non-recursive walk), or a glob
    pattern containing wildcards. Directories filter to known image
    extensions; globs trust the user.
    """
    if any(ch in src for ch in "*?["):
        matches = sorted(Path(p) for p in glob_module.glob(src))
        return [p for p in matches if p.is_file()]
    path = Path(src)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    raise FileNotFoundError(src)


def plan_outputs(inputs: Iterable[Path], dst_dir: Path) -> list[tuple[Path, Path]]:
    """Map each input to ``dst_dir / input.name``."""
    return [(src, dst_dir / src.name) for src in inputs]


def _load(src: Path) -> tuple[Image.Image, str]:
    image = Image.open(src)
    has_alpha = image.mode in ("RGBA", "LA") or "transparency" in image.info
    mode = "RGBA" if has_alpha else "RGB"
    return image.convert(mode), mode


def _save(image: Image.Image, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    image.save(dst)


def _prefetch(
    executor: ThreadPoolExecutor,
    items: Iterable[tuple[Path, Path]],
    depth: int,
) -> Iterator[tuple[Path, Path, Future]]:
    """Stream items keeping at most ``depth`` reads in flight at once."""
    pending: deque[tuple[Path, Path, Future]] = deque()
    iterator = iter(items)

    def submit_next() -> bool:
        try:
            src, dst = next(iterator)
        except StopIteration:
            return False
        pending.append((src, dst, executor.submit(_load, src)))
        return True

    for _ in range(max(1, depth)):
        if not submit_next():
            break

    while pending:
        item = pending.popleft()
        yield item
        submit_next()


def run_batch(
    upscaler: Upscaler,
    items: list[tuple[Path, Path]],
    *,
    scale: float,
    tile: int | None,
    tile_pad: int | None,
    workers: int = 4,
    skip_existing: bool = False,
    progress: bool = True,
) -> BatchResult:
    """Run ``upscaler`` over each (src, dst) pair, returning counts."""
    pending = [(s, d) for s, d in items if not (skip_existing and d.exists())]
    skipped = len(items) - len(pending)
    processed = 0
    failed = 0
    write_futs: list[Future] = []

    columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        with Progress(*columns, disable=not progress) as bar:
            task = bar.add_task("upscaling", total=len(pending))
            for src, dst, read_fut in _prefetch(pool, pending, depth=workers):
                try:
                    image, _mode = read_fut.result()
                except Exception as exc:  # noqa: BLE001 — surface but keep batch going
                    failed += 1
                    bar.console.log(f"[red]read failed[/red] {src}: {exc}")
                    bar.advance(task)
                    continue
                try:
                    out = upscaler.upscale(image, scale=scale, tile=tile, tile_pad=tile_pad)
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    bar.console.log(f"[red]upscale failed[/red] {src}: {exc}")
                    bar.advance(task)
                    continue
                write_futs.append(pool.submit(_save, out, dst))
                processed += 1
                bar.advance(task)

        for fut in write_futs:
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                failed += 1
                processed -= 1
                # Best-effort surfacing; a dedicated logger would be nicer here.
                print(f"write failed: {exc}")

    return BatchResult(processed=processed, skipped=skipped, failed=failed)
