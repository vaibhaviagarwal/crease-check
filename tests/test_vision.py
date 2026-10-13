"""Photo reader: the pieces, and whole photos with known answers."""

import cv2
import numpy as np
import pytest

from crease.checks import check
from crease.examples import EXAMPLES
from crease.synth import render_photo
from crease.vision import PhotoError, order_corners, read_photo, thin


def test_thinning_leaves_a_one_pixel_line():
    img = np.zeros((60, 200), np.uint8)
    cv2.line(img, (10, 30), (190, 30), 255, 9)
    sk = thin(img)
    cols = (sk > 0).sum(axis=0)[20:180]
    assert cols.max() == 1 and cols.min() == 1


def test_corner_order():
    pts = np.array([[90, 10], [10, 12], [12, 95], [95, 90]])
    tl, tr, br, bl = order_corners(pts)
    assert tuple(tl) == (10, 12) and tuple(br) == (95, 90)
    assert tuple(tr) == (90, 10) and tuple(bl) == (12, 95)


def test_not_an_image():
    with pytest.raises(PhotoError, match="not an image"):
        read_photo(b"definitely not a jpeg")


def test_blank_paper_has_no_creases():
    img = np.full((600, 800, 3), (70, 80, 90), np.uint8)
    cv2.rectangle(img, (150, 80), (650, 520), (228, 236, 242), -1)
    ok, buf = cv2.imencode(".jpg", img)
    with pytest.raises(PhotoError, match="No red or blue lines"):
        read_photo(buf.tobytes())


@pytest.mark.parametrize("name", ["waterbomb", "square-twist", "broken-twist"])
@pytest.mark.parametrize("seed", [1001, 1002])
def test_photo_gives_the_same_verdict_as_the_file(name, seed):
    pattern = EXAMPLES[name][1]()
    truth = check(pattern)
    res = read_photo(render_photo(pattern, seed, "easy"))
    got = check(res.pattern, angle_tol=1.0)
    assert got.flat_foldable == truth.flat_foldable
    assert got.creases == truth.creases
    assert len(got.failing_vertices) == len(truth.failing_vertices)


def test_mountain_and_valley_colours_are_kept_apart():
    pattern = EXAMPLES["waterbomb"][1]()
    res = read_photo(render_photo(pattern, 1003, "easy"))
    kinds = [a for a in res.pattern.assignments if a in "MV"]
    assert kinds.count("M") == 2 and kinds.count("V") == 4


def test_benchmark_quick_run():
    """A small slice of bench/vision_benchmark.py, so CI notices regressions."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
    from vision_benchmark import run, summarise
    s = summarise(run(1, 2, 1000, ("easy",)))["easy"]
    assert s["recall"] > 0.97 and s["precision"] > 0.97
    assert s["verdict"] >= 0.8


# --- screenshots / exports from design software ------------------------------

def _grid_miura():
    from crease.examples import miura_ori
    # 45-degree zigzags on a 16 x 16 grid: every vertex is a grid point.
    return miura_ori(8, 16, lean=1 / 16)


@pytest.mark.parametrize("seed", [0, 1])
def test_digital_grid_pattern_is_read_exactly(seed):
    from crease.synth import render_digital
    pattern = _grid_miura()
    truth = check(pattern)
    assert truth.flat_foldable
    res = read_photo(render_digital(pattern, seed, grid=16))
    got = check(res.pattern, angle_tol=1.0)
    assert res.stats["kind"] == "digital crease pattern"
    assert res.stats["grid"] == 16 and res.stats["method"] == "grid"
    assert got.flat_foldable
    assert got.creases == truth.creases and got.vertices_checked == truth.vertices_checked


def test_digital_off_grid_pattern_uses_line_finding():
    from crease.synth import render_digital
    pattern = EXAMPLES["broken-twist"][1]()
    res = read_photo(render_digital(pattern, 3, grid=None))
    got = check(res.pattern, angle_tol=1.0)
    assert res.stats["kind"] == "digital crease pattern"
    assert res.stats["method"] == "lines"
    assert len(got.failing_vertices) == 2


def test_marker_photos_still_use_the_marker_profile():
    res = read_photo(render_photo(EXAMPLES["waterbomb"][1](), 1001, "easy"))
    assert res.stats["kind"] == "photo of a drawing"
