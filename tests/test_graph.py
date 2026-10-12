"""Turning loose segments into a clean planar graph."""

import random

from hypothesis import given, settings
from hypothesis import strategies as st

from crease.graph import cleanup
from crease.model import Pattern


def P(*segs):
    return Pattern.from_segments(list(segs))


def test_crossing_lines_get_a_vertex():
    out = cleanup(P(((0, 0), (1, 1), "M"), ((0, 1), (1, 0), "V")), add_border=False)
    assert len(out.vertices) == 5
    assert len(out.edges) == 4
    assert (0.5, 0.5) in [tuple(round(c, 9) for c in v) for v in out.vertices]


def test_t_junction_splits_the_top_bar():
    out = cleanup(P(((0, 0), (2, 0), "M"), ((1, 0), (1, 1), "V")), add_border=False)
    assert len(out.edges) == 3


def test_endpoints_that_nearly_meet_are_merged():
    out = cleanup(P(((0, 0), (1, 0), "M"), ((1 + 1e-9, 0), (1, 1), "V")), add_border=False)
    assert len(out.vertices) == 3


def test_duplicate_crease_kept_once():
    out = cleanup(P(((0, 0), (1, 0), "M"), ((1, 0), (0, 0), "M")), add_border=False)
    assert len(out.edges) == 1


def test_mountain_and_valley_on_the_same_line_becomes_unassigned():
    out = cleanup(P(((0, 0), (1, 0), "M"), ((0, 0), (1, 0), "V")), add_border=False)
    assert out.assignments == ["U"]
    assert any("both mountain and valley" in n for n in out.notes)


def test_overlapping_collinear_segments():
    out = cleanup(P(((0, 0), (2, 0), "M"), ((1, 0), (3, 0), "M")), add_border=False)
    assert len(out.edges) == 3          # 0-1, 1-2, 2-3
    assert len(out.vertices) == 4


def test_border_beats_crease_on_the_paper_edge():
    out = cleanup(P(((0, 0), (1, 0), "B"), ((0, 0), (1, 0), "M")), add_border=False)
    assert out.assignments == ["B"]


def test_missing_border_is_added():
    out = cleanup(P(((0, 0), (1, 1), "M")))
    assert out.count("B") == 4
    assert out.notes


# --- the grid finds exactly what brute force finds -------------------------

def _random_segments(seed: int, n: int):
    rng = random.Random(seed)
    segs = []
    for _ in range(n):
        p = (rng.random(), rng.random())
        q = (rng.random(), rng.random())
        segs.append((p, q, rng.choice("MV")))
    return segs


@settings(max_examples=25, deadline=None)
@given(st.integers(0, 10**9), st.integers(65, 220))
def test_grid_matches_brute_force(seed, n):
    pat = P(*_random_segments(seed, n))
    fast = cleanup(pat, add_border=False, use_grid=True)
    slow = cleanup(pat, add_border=False, use_grid=False)
    assert len(fast.vertices) == len(slow.vertices)
    assert sorted(fast.edges) == sorted(slow.edges)


def test_large_pattern_is_fast():
    import time
    from crease.examples import miura_ori
    t = time.perf_counter()
    out = cleanup(miura_ori(30, 30))
    assert time.perf_counter() - t < 5.0
    assert len(out.edges) > 1700
