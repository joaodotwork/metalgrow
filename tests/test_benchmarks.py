"""Smoke tests for the benchmark harness.

The real runs are manual (see ``benchmarks/README.md``); these tests only
verify that the module imports, one tiny bicubic iteration on CPU completes,
and the renderers produce the expected shape.
"""

from __future__ import annotations

from benchmarks import run as br


def test_synth_image_deterministic():
    a = br._synth_image(32)
    b = br._synth_image(32)
    assert a.size == (32, 32)
    assert a.tobytes() == b.tobytes()


def test_allowed_scales_bicubic_accepts_all():
    assert br._allowed_scales("bicubic", (2.0, 4.0)) == [2.0, 4.0]


def test_allowed_scales_realesrgan_filters_to_native():
    assert br._allowed_scales("realesrgan-x2", (2.0, 4.0)) == [2.0]
    assert br._allowed_scales("realesrgan-x4", (2.0, 4.0)) == [4.0]
    assert br._allowed_scales("realesrgan-x4", (2.0,)) == []


def test_allowed_scales_swinir_filters_to_native():
    assert br._allowed_scales("swinir-x2", (2.0, 4.0)) == [2.0]
    assert br._allowed_scales("swinir-x4", (2.0, 4.0)) == [4.0]


def test_bench_one_bicubic_cpu_runs():
    result = br.bench_one(
        device_pref="cpu",
        backbone="bicubic",
        scale=2.0,
        size=16,
        warmup=0,
        iters=1,
        skip_uncached=True,
    )
    assert result.device == "cpu"
    assert result.backbone == "bicubic"
    assert result.iters == 1
    assert result.median_s >= 0.0
    assert result.mp_per_s > 0.0
    assert result.peak_mib is None


def test_run_suite_skip_uncached_records_skip_note():
    run = br.run_suite(
        device_pref="cpu",
        backbones=("bicubic",),
        scales=(2.0,),
        size=16,
        warmup=0,
        iters=1,
        skip_uncached=True,
    )
    assert len(run.results) == 1
    md = br.to_markdown(run)
    assert "bicubic" in md
    assert "| Device |" in md


def test_to_json_is_valid():
    import json

    run = br.run_suite(
        device_pref="cpu",
        backbones=("bicubic",),
        scales=(2.0,),
        size=16,
        warmup=0,
        iters=1,
        skip_uncached=True,
    )
    payload = json.loads(br.to_json(run))
    assert "host" in payload and "torch" in payload and "results" in payload
    assert payload["results"][0]["backbone"] == "bicubic"
