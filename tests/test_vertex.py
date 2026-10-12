"""Single-vertex rules, checked against textbook cases and a brute force."""

import itertools
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from crease import vertex as vx
from crease.examples import random_vertex
from oracle import oracle_foldable


# --- textbook cases --------------------------------------------------------

def test_straight_fold():
    assert vx.crimp_foldable([180, 180], ["M", "M"])
    assert not vx.crimp_foldable([180, 180], ["M", "V"])   # half up, half down
    assert not vx.crimp_foldable([170, 190], ["M", "M"])   # bent line


def test_four_right_angles():
    # Two crossing folds: three of one kind, one of the other.
    assert vx.crimp_foldable([90] * 4, ["M", "M", "M", "V"])
    assert not vx.crimp_foldable([90] * 4, ["M", "M", "V", "V"])  # Maekawa 0
    assert not vx.crimp_foldable([90] * 4, ["M"] * 4)             # Maekawa 4


def test_kawasaki():
    assert vx.kawasaki_error([45, 90, 45, 45, 90, 45]) == 0
    assert vx.kawasaki_error([90, 135, 84, 51]) == pytest.approx(-6)


def test_big_little_big_catches_what_maekawa_misses():
    # Sectors 90, 135, 90, 45: the 45 is a strict minimum, so creases 3 and
    # 0 must disagree. This assignment passes Maekawa (3 M, 1 V) but has
    # them agreeing.
    s, a = [90, 135, 90, 45], ["M", "V", "M", "M"]
    assert vx.maekawa_ok(a)
    assert vx.big_little_big(s, a, 1e-6) == [3]
    assert not vx.crimp_foldable(s, a)


def test_odd_degree_never_folds():
    assert not vx.crimp_foldable([120, 120, 120], ["M", "M", "V"])


# --- against the brute force ----------------------------------------------

def _sectors(k: int, rng: random.Random) -> list[float]:
    """k even and k odd sectors, each set summing to 180, on a 15° grid so
    that ties between sectors (the hard case) are common."""
    def part():
        cuts = sorted(rng.sample(range(1, 12), k - 1)) if k > 1 else []
        return [15 * (b - a) for a, b in zip([0] + cuts, cuts + [12])]
    ev, od = part(), part()
    return [x for pair in zip(ev, od) for x in pair]


@pytest.mark.parametrize("seed", range(160))
def test_crimp_matches_brute_force_on_every_assignment(seed):
    rng = random.Random(seed)
    for k in (1, 2, 3):
        s = _sectors(k, rng)
        for a in itertools.product("MV", repeat=2 * k):
            assert vx.crimp_foldable(s, list(a)) == oracle_foldable(s, list(a)), (s, a)


@settings(max_examples=60, deadline=None)
@given(st.integers(0, 10**9), st.integers(1, 3))
def test_generated_vertices_are_accepted_by_both(seed, crimps):
    angles, assigns = random_vertex(random.Random(seed), crimps)
    s = vx.sectors_from_angles(angles)
    assert vx.crimp_foldable(s, assigns, 1e-6)
    if len(s) <= 8:
        assert oracle_foldable(s, assigns, 1e-6)


# --- properties -------------------------------------------------------------

@settings(max_examples=200, deadline=None)
@given(st.integers(0, 10**9), st.integers(0, 5))
def test_never_rejects_a_foldable_vertex(seed, crimps):
    """The promise in the README: no false alarms on real folds."""
    angles, assigns = random_vertex(random.Random(seed), crimps)
    s = vx.sectors_from_angles(angles)
    assert vx.crimp_foldable(s, assigns, 1e-6)
    assert vx.maekawa_ok(assigns)
    assert abs(vx.kawasaki_error(s)) < 1e-6
    assert vx.big_little_big(s, assigns, 1e-6) == []


@settings(max_examples=200, deadline=None)
@given(st.integers(0, 10**9), st.integers(0, 5), st.data())
def test_one_flip_always_breaks_it(seed, crimps, data):
    angles, assigns = random_vertex(random.Random(seed), crimps)
    s = vx.sectors_from_angles(angles)
    i = data.draw(st.integers(0, len(assigns) - 1))
    flipped = list(assigns)
    flipped[i] = "V" if flipped[i] == "M" else "M"
    assert not vx.crimp_foldable(s, flipped, 1e-6)   # Maekawa moves by 2


@settings(max_examples=200, deadline=None)
@given(st.integers(0, 10**9), st.integers(0, 5), st.integers(0, 20))
def test_symmetries(seed, crimps, shift):
    angles, assigns = random_vertex(random.Random(seed), crimps)
    s = vx.sectors_from_angles(angles)
    swapped = ["V" if a == "M" else "M" for a in assigns]
    assert vx.crimp_foldable(s, swapped, 1e-6)        # turn the paper over
    k = shift % len(s)
    assert vx.crimp_foldable(s[k:] + s[:k], assigns[k:] + assigns[:k], 1e-6)  # relabel
    # Mirror image: reverse the order of sectors and creases together.
    ms = s[::-1]
    ma = [assigns[0]] + assigns[1:][::-1]
    assert vx.crimp_foldable(ms, ma, 1e-6)


# --- fixes -------------------------------------------------------------------

def test_min_flips_finds_the_single_flip():
    s, a = [90, 135, 90, 45], ["V", "V", "V", "V"]
    flips = vx.min_flips(s, a, 1e-6)
    assert flips is not None and len(flips) == 1
    trial = list(a)
    trial[flips[0]] = "M"
    assert vx.crimp_foldable(s, trial)


def test_min_flips_gives_up_when_kawasaki_fails():
    assert vx.min_flips([90, 135, 84, 51], ["V", "V", "V", "M"], 1e-6) is None


@settings(max_examples=100, deadline=None)
@given(st.floats(-20, 20).filter(lambda e: abs(e) > 0.1))
def test_rotation_fix_cancels_the_error(err):
    s = [90.0 + err, 135.0 - err, 90.0, 45.0]   # even sectors sum to 180 + err
    for k, d in vx.rotation_fix(s, vx.kawasaki_error(s)):
        fixed = list(s)
        fixed[k - 1] += d
        fixed[k] -= d
        assert vx.kawasaki_error(fixed) == pytest.approx(0, abs=1e-9)
