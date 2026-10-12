"""Whole-pattern checks, fixes and file formats."""

import json
import math
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from crease.checks import apply_fix, check, describe_fix
from crease.examples import EXAMPLES, broken_twist, random_vertex, vertex_pattern
from crease.io import PatternError, parse, parse_cp, parse_fold, to_cp, to_fold
from crease.model import Pattern


@pytest.mark.parametrize("name", ["square-twist", "waterbomb", "miura-ori"])
def test_real_bases_fold(name):
    r = check(EXAMPLES[name][1]())
    assert r.flat_foldable, [v.message for v in r.violations]
    assert r.vertices_checked > 0


def test_miura_vertex_count():
    r = check(EXAMPLES["miura-ori"][1]())
    assert r.vertices_checked == 25      # 5 x 5 interior grid points


def test_broken_twist_has_exactly_its_two_planted_mistakes():
    r = check(broken_twist())
    pts = {tuple(round(c, 3) for c in v.point) for v in r.violations}
    assert pts == {(0.35, 0.35), (0.65, 0.65)}
    rules = {v.rule for v in r.violations}
    assert {"kawasaki", "maekawa"} <= rules
    kaw = next(v for v in r.violations if v.rule == "kawasaki")
    assert kaw.details["alternating_sum_deg"] == pytest.approx(174.0, abs=1e-6)
    assert abs(kaw.fix["by_deg"]) == pytest.approx(6.0)


def test_applying_every_suggested_fix_makes_it_fold():
    pat = check(broken_twist()).pattern
    for _ in range(4):
        r = check(pat)
        if r.flat_foldable:
            break
        v = next(v for v in r.violations if v.fix)
        pat = apply_fix(r.pattern, v)
    assert check(pat).flat_foldable


def test_fix_descriptions_read_well():
    assert describe_fix({"flip": ["C3"], "to": ["V"]}) == "Flip C3 to valley"
    assert describe_fix({"rotate": "C5", "by_deg": -6.0}) == "Rotate crease C5 by 6.0° clockwise"


def test_odd_degree_is_reported():
    segs = [((0.5, 0.5), (1, 0.5), "M"), ((0.5, 0.5), (0, 0), "M"), ((0.5, 0.5), (0, 1), "V")]
    r = check(Pattern.from_segments(segs))
    assert [v.rule for v in r.violations] == ["even_degree"]


def test_unassigned_creases_are_flagged_not_guessed():
    p = EXAMPLES["waterbomb"][1]()
    p.assignments = ["U" if a == "M" else a for a in p.assignments]
    r = check(p)
    assert not r.flat_foldable
    assert any(v.rule == "unassigned" for v in r.violations)


def _transform(p: Pattern, angle: float, scale: float, mirror: bool, dx: float, dy: float) -> Pattern:
    c, s = math.cos(angle), math.sin(angle)
    pts = []
    for x, y in p.vertices:
        if mirror:
            x = -x
        pts.append((scale * (c * x - s * y) + dx, scale * (s * x + c * y) + dy))
    return Pattern(pts, list(p.edges), list(p.assignments))


@settings(max_examples=40, deadline=None)
@given(st.sampled_from(sorted(EXAMPLES)), st.floats(0, 2 * math.pi),
       st.floats(0.01, 500), st.booleans(), st.floats(-1e3, 1e3), st.floats(-1e3, 1e3))
def test_verdict_survives_rotation_scaling_and_mirroring(name, angle, scale, mirror, dx, dy):
    base = check(EXAMPLES[name][1]())
    moved = check(_transform(EXAMPLES[name][1](), angle, scale, mirror, dx, dy))
    assert moved.flat_foldable == base.flat_foldable
    assert len(moved.failing_vertices) == len(base.failing_vertices)


@settings(max_examples=60, deadline=None)
@given(st.integers(0, 10**9), st.integers(1, 4))
def test_random_foldable_vertex_patterns_pass(seed, crimps):
    rng = random.Random(seed)
    angles, assigns = random_vertex(rng, crimps)
    centre = (rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7))
    r = check(vertex_pattern(angles, assigns, centre))
    assert r.flat_foldable, [v.message for v in r.violations]


# --- files ------------------------------------------------------------------

def test_fold_round_trip():
    p = check(EXAMPLES["square-twist"][1]()).pattern
    back = parse_fold(json.dumps(to_fold(p)))
    assert check(back).flat_foldable
    assert len(back.edges) == len(p.edges)


def test_cp_round_trip():
    p = check(EXAMPLES["miura-ori"][1]()).pattern
    back = parse_cp(to_cp(p))
    assert check(back).flat_foldable


def test_parse_picks_format():
    p = EXAMPLES["waterbomb"][1]()
    assert parse(json.dumps(to_fold(p)), "x.fold").edges
    assert parse(to_cp(p), "x.cp").edges
    assert parse(to_cp(p)).edges              # sniffed


@pytest.mark.parametrize("bad, msg", [
    ("{not json", "Not valid JSON"),
    ('{"vertices_coords": [[0,0]]}', "needs vertices_coords and edges_vertices"),
    ('{"vertices_coords": [[0,0],[1,1]], "edges_vertices": [[0,5]]}', "does not exist"),
    ('{"vertices_coords": [[0,0],[1,1]], "edges_vertices": [[0,1]], "edges_assignment": ["Q"]}', "Unknown edge assignment"),
])
def test_bad_fold_files_get_clear_errors(bad, msg):
    with pytest.raises(PatternError, match=msg):
        parse_fold(bad)


def test_bad_cp_file():
    with pytest.raises(PatternError, match="Line 2"):
        parse_cp("2 0 0 1 1\n2 0 0 one 1\n")
