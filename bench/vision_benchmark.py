"""Score the photo reader on synthetic photos with known answers.

    python bench/vision_benchmark.py             # full run, writes reports/
    python bench/vision_benchmark.py --quick     # smaller, for CI

For each test pattern and difficulty level it renders photos
(crease/synth.py), reads them back (crease/vision.py) and compares:

* crease recall:    share of the true crease length that was found
* crease precision: share of the detected crease length that is real
* M/V accuracy:     of the found length, share with the right colour
* verdict:          does the photo get the same pass/fail as the file?
* exact:            same verdict AND the same failing points

The vision parameters were tuned on seeds below 1000. Reported numbers
use seeds from 1000 up, which the tuning never saw.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crease.checks import check                      # noqa: E402
from crease.examples import EXAMPLES, random_vertex, vertex_pattern   # noqa: E402
from crease.graph import cleanup                     # noqa: E402
from crease.synth import render_photo                # noqa: E402
from crease.vision import read_photo                 # noqa: E402

NEAR = 0.015       # a true crease counts as found within 1.5% of the side
ANGLE = 5.0        # ... and with a direction within 5 degrees


def _creases(p):
    out = []
    for (i, j), a in zip(p.edges, p.assignments):
        if a in ("M", "V"):
            out.append((p.vertices[i], p.vertices[j], a))
    return out


def _dist_to_seg(pt, a, b):
    (px, py), (ax, ay), (bx, by) = pt, a, b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def _dir(a, b):
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0


def coverage(src, dst):
    """Length of src creases lying on some dst crease; and of that, how much
    has the same kind. Sampled every 0.5% of the side."""
    total = found = same = 0.0
    for a, b, kind in src:
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(2, int(L / 0.005))
        th = _dir(a, b)
        for k in range(n):
            t = (k + 0.5) / n
            pt = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            total += L / n
            best = None
            for c, d, k2 in dst:
                dth = abs(_dir(c, d) - th)
                if min(dth, 180 - dth) > ANGLE:
                    continue
                dd = _dist_to_seg(pt, c, d)
                if dd <= NEAR and (best is None or dd < best[0]):
                    best = (dd, k2)
            if best:
                found += L / n
                same += (L / n) * (best[1] == kind)
    return total, found, same


def same_failures(r1, r2) -> bool:
    a = [v.point for v in r1.violations]
    b = [v.point for v in r2.violations]
    fa = {tuple(round(c, 2) for c in p) for p in a}
    match = lambda p, pool: any(math.hypot(p[0] - q[0], p[1] - q[1]) < 0.03 for q in pool)
    return all(match(p, b) for p in a) and all(match(p, a) for p in b) and (len(fa) > 0) == (len(b) > 0)


def test_patterns(n_random: int, seed: int):
    pats = [(name, f()) for name, (_, f) in EXAMPLES.items()]
    rng = random.Random(seed)
    for k in range(n_random):
        angles, assigns = random_vertex(rng, rng.randint(1, 3), grid=22.5)
        centre = (rng.uniform(0.35, 0.65), rng.uniform(0.35, 0.65))
        pats.append((f"random-{k}", vertex_pattern(angles, assigns, centre)))
    return pats


def run(photos: int, n_random: int, first_seed: int, levels=("easy", "normal", "hard")):
    rows = []
    for name, pat in test_patterns(n_random, first_seed):
        truth = check(pat)
        true_c = _creases(cleanup(pat))
        for level in levels:
            for s in range(photos):
                seed = first_seed + zlib.crc32(f"{name}/{level}/{s}".encode()) % 100_000
                img = render_photo(pat, seed, level)
                t = time.perf_counter()
                res = read_photo(img)
                rep = check(res.pattern, angle_tol=1.0)
                ms = (time.perf_counter() - t) * 1000
                det_c = _creases(rep.pattern)
                tt, tf, ts = coverage(true_c, det_c)
                dt, df, _ = coverage(det_c, true_c)
                rows.append({
                    "pattern": name, "level": level, "seed": seed,
                    "recall": tf / tt if tt else 1.0,
                    "precision": df / dt if dt else 0.0,
                    "mv": ts / tf if tf else 0.0,
                    "verdict": rep.flat_foldable == truth.flat_foldable,
                    "exact": rep.flat_foldable == truth.flat_foldable and same_failures(rep, truth),
                    "ms": ms,
                })
    return rows


def summarise(rows):
    out = {}
    for level in sorted({r["level"] for r in rows}, key=["easy", "normal", "hard"].index):
        rs = [r for r in rows if r["level"] == level]
        out[level] = {
            "photos": len(rs),
            "recall": statistics.mean(r["recall"] for r in rs),
            "precision": statistics.mean(r["precision"] for r in rs),
            "mv_accuracy": statistics.mean(r["mv"] for r in rs),
            "verdict": statistics.mean(r["verdict"] for r in rs),
            "exact": statistics.mean(r["exact"] for r in rs),
            "median_ms": statistics.median(r["ms"] for r in rs),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tune", action="store_true", help="use the tuning seeds instead")
    ap.add_argument("--photos", type=int, default=3)
    ap.add_argument("--random", type=int, default=16)
    args = ap.parse_args()
    first = 0 if args.tune else 1000
    if args.quick:
        rows = run(1, 4, first, ("easy", "normal"))
    else:
        rows = run(args.photos, args.random, first)
    summary = summarise(rows)
    print(f"{'level':8s} {'photos':>6s} {'recall':>7s} {'prec':>7s} {'M/V':>7s} {'verdict':>8s} {'exact':>7s} {'ms':>6s}")
    for level, s in summary.items():
        print(f"{level:8s} {s['photos']:6d} {s['recall']:7.1%} {s['precision']:7.1%} {s['mv_accuracy']:7.1%} "
              f"{s['verdict']:8.1%} {s['exact']:7.1%} {s['median_ms']:6.0f}")
    worst = sorted(rows, key=lambda r: (r["exact"], r["recall"]))[:8]
    print("\nworst cases:")
    for r in worst:
        print(f"  {r['pattern']:14s} {r['level']:6s} seed {r['seed']:6d}  recall {r['recall']:.2f} "
              f"precision {r['precision']:.2f} exact {r['exact']}")
    if not args.quick and not args.tune:
        reports = ROOT / "reports"
        reports.mkdir(exist_ok=True)
        (reports / "vision_benchmark.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    return summary


if __name__ == "__main__":
    main()
