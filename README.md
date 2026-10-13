# Crease Check

[![tests](https://github.com/vaibhaviagarwal/crease-check/actions/workflows/tests.yml/badge.svg)](https://github.com/vaibhaviagarwal/crease-check/actions/workflows/tests.yml)

**Live: [crease-check-smoky.vercel.app](https://crease-check-smoky.vercel.app)** · [API docs](https://crease-check-smoky.vercel.app/docs)

Will an origami crease pattern fold flat? Upload a design file, a phone
photo of a pattern drawn in marker, or a screenshot from design software,
and Crease Check finds every point that can't fold, says which rule it
breaks, and suggests the smallest fix.

## What it does

- **Checks every vertex exactly.** The three classic rules (Maekawa,
  Kawasaki, big-little-big) are necessary but not sufficient, so each vertex
  is also run through the crimp reduction, which is exact for a single
  vertex. A brute-force checker that tries every layer order agrees with it
  on all 13,440 vertex-and-assignment cases in the tests.
- **Suggests fixes and can apply them.** The fewest crease flips that make a
  vertex fold (avoiding flips that would break the vertex at the crease's
  other end), or the one crease rotation that fixes an angle. The tests
  apply every suggested fix to a broken pattern and confirm it then folds.
- **Reads photos and screenshots.** Red for mountain folds, blue for valley.
  Classical computer vision with OpenCV, no neural network. On 180
  synthetic marker photos it had never seen, it found over 99.9% of crease
  length and gave the same verdict as the original file for 95% of
  normal-difficulty photos. Screenshots of grid-based designs are read
  exactly by detecting the design grid.
- **Scales.** A 60 × 60 Miura-ori grid (7,080 creases, 3,481 vertices) checks
  in about 0.35 s, because a spatial grid replaces the all-pairs
  line-crossing search.

## The scope, stated plainly

Crease Check proves that every point folds flat *on its own*. Whether a
whole sheet folds flat is NP-hard to decide in general (Bern and Hayes,
1996), because flaps far apart can collide. So a pass means "no point is
broken", not "guaranteed to fold". The API labels this `"scope": "local"`.

## How a check works

```
file / photo ─► segments ─► planar graph ─► per-vertex rules ─► report + fixes
```

1. **Parse** FOLD (`.fold`, the research community's JSON format) or ORIPA
   (`.cp`) files.
2. **Clean up the geometry** (`crease/graph.py`): split lines where they
   cross, merge points that nearly touch, drop duplicates, and resolve
   clashing assignments. Candidate crossings come from a uniform grid
   traversal rather than comparing every pair. A property test checks the
   grid finds exactly what brute force finds.
3. **Check each interior vertex** (`crease/vertex.py`): even degree,
   Kawasaki, Maekawa, big-little-big, then the exact crimp test.
4. **Suggest fixes** (`crease/checks.py`): a minimal flip search over subsets
   of up to 3 creases, or a single-crease rotation that cancels the Kawasaki
   error. A rotation prefers a crease whose far end is on the paper edge, so
   the fix doesn't disturb another vertex.

## Reading a photo

`crease/vision.py`, about 900 lines:

| Step | Method |
|---|---|
| Find the paper | Brightness-minus-saturation mask, Otsu threshold, largest quadrilateral; a lighting-corrected second reading takes over when a shadow clips the first |
| Straighten | Perspective warp to an 800 × 800 square |
| Separate inks | HSV thresholds for red and blue |
| Thin strokes | Zhang-Suen skeleton (vectorised NumPy) |
| Find lines | Probabilistic Hough transform, merged by angle and offset, refit by weighted PCA, gaps bridged when the gap itself is inked |
| Recover misses | Leftover ink that is long and thin becomes a line too |
| Snap (optional) | Angles within 5° of a multiple of 22.5° snap to it |
| Make creases meet | Ends move to junctions within reach, and dangling ends extend forward to the next line |
| Solve junctions | Union-find clusters crossings, then rounds of least-squares vertex placement and sideways line shifts make the drawing consistent |

Before any of that is shown, a sanity check asks whether the result looks
like a crease pattern at all (lines reaching the paper's edge, few loose
ends). If it doesn't, the page says so and explains how to take a photo
that works, rather than reporting dozens of meaningless failures.

### Screenshots and exports from design software

Thin, exact lines on a dense grid need different handling from marker
strokes, so the reader first measures stroke width. On an 800-px sheet,
software lines are about 3-5 px and marker strokes 9-12 px. Thin lines
switch it to a digital profile:

- straighten at 1600 px, skip speck removal (it erases thin lines), accept
  paler anti-aliased colour, and tighten every distance tolerance;
- use the outer extent of the creases as the paper's edge, since software
  draws a frame just inside the image;
- **detect the design grid.** Most designed patterns, box pleating
  especially, put every vertex on an N × N grid. The smallest N that
  nearly every line end sits on is the grid. Then each half-step between
  neighbouring grid points, horizontal, vertical and diagonal, is tested
  directly for red, blue or no ink. That reads the geometry exactly rather
  than estimating it. It's used only when those steps account for at
  least 97% of the ink, so patterns with creases off the grid keep the
  line-finding result.

On a 32 × 32 box-pleated pattern, an 856-crease screenshot, this reads
386 vertices and flags exactly one, which turns out to be a real error in
the image: four valley folds meet at one point.

### Measured, not eyeballed

`bench/vision_benchmark.py` draws patterns the way a person would (wobbly
strokes that overshoot or stop short), photographs them in software on a
desk at an angle with uneven light, blur, noise and JPEG compression, and
scores the reader against the known answer. Parameters were tuned on seeds
below 1000; these numbers use seeds from 1000 up, which tuning never saw.

| Difficulty | Photos | Crease recall | Precision | M/V correct | Same verdict | Same failing points |
|---|---|---|---|---|---|---|
| Easy | 60 | >99.9% | 99.9% | >99.9% | 96.7% | 96.7% |
| Normal | 60 | >99.9% | >99.9% | >99.9% | 95.0% | 95.0% |
| Hard | 60 | >99.9% | 99.8% | >99.9% | 88.3% | 86.7% |

About 0.3 s per photo. Recall is nearly perfect, so the remaining errors
are almost all in how lines meet: one junction split in two flips the
verdict. Dense patterns (Miura-ori, 60 creases) cause most of them.

### Limits

- **Hand-drawn angles are only as good as the hand.** With snapping on, a
  deliberate error under 5° from the grid is snapped away; with snapping
  off, wobble makes angles uncertain by about a degree. Patterns off the
  22.5° grid can only be judged to about ±1°.
- **Synthetic photos are not real photos.** Real marker, paper and lighting
  will differ; a set of real photos is the next test.
- **Very short creases** (under about 4.5% of the paper's width) are treated
  as noise in marker photos.

## API

FastAPI, with interactive docs at `/docs`.

| Method | Path | |
|---|---|---|
| GET | `/api/examples` | Built-in patterns |
| GET | `/api/examples/{id}` | One of them, checked |
| POST | `/api/check` | JSON: a FOLD object, or `{"cp": "..."}` |
| POST | `/api/check/file` | Upload `.fold` or `.cp` |
| POST | `/api/photo` | Upload a photo or screenshot |
| POST | `/api/fix` | Apply one suggested fix, re-check |
| GET | `/api/checks/recent` | Latest saved checks |
| GET | `/api/stats` | Which rules fail most, across all checks |

```json
{
  "flat_foldable": false,
  "scope": "local",
  "checked": { "vertices": 4, "creases": 12 },
  "failing_vertices": 2,
  "violations": [
    { "rule": "kawasaki", "point": [0.35, 0.35],
      "message": "Every other angle should add up to 180°. These add up to 174.0°.",
      "fix": { "rotate": "C5", "by_deg": 6.0 },
      "fix_text": "Rotate crease C5 by 6.0° counter-clockwise" }
  ]
}
```

With `DATABASE_URL` set, every check is saved to PostgreSQL
(`sql/schema.sql`); `/api/stats` aggregates them with `unnest` and
`percentile_cont`. Without it, storage is skipped and nothing else changes.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m uvicorn crease_api.main:app --reload       # http://localhost:8000
```

`samples/` has three synthetic photos to try the photo upload with.

With a database, using Docker:

```bash
docker compose up --build
```

## Tests

```bash
python -m pytest                       # 233 tests, ~45 s
python bench/vision_benchmark.py       # full photo benchmark, ~90 s
```

- **Vertex rules** against textbook cases and a brute-force layer-order
  search, plus property tests (Hypothesis): no false alarm on randomly
  generated foldable vertices, any single flip is caught, and verdicts
  survive rotation, relabelling, mirroring and turning the paper over.
- **Geometry cleanup**: crossings, T-junctions, near-misses, overlaps, and
  the spatial grid against brute force on random line sets.
- **Patterns and fixes**: the built-in bases fold; the broken demo has
  exactly its two planted mistakes; applying the suggested fixes repairs it.
- **Photos and screenshots**: thinning, corner order, bad input, whole
  synthetic photos, screenshot-style images read exactly through grid
  detection, and the not-a-pattern check.
- **API**: every endpoint, error responses, the fix round trip, and saving
  to a real PostgreSQL database (in CI, a Postgres 16 service container).

The random vertices come from running the crimp test backwards: start
from a straight fold and "un-crimp" repeatedly. That makes them foldable
by construction, which is what lets a property test say "never rejects a
foldable vertex".

## Layout

```
crease/            the engine (no web code)
  model.py         the Pattern data structure
  io.py            FOLD and .cp reading and writing
  graph.py         planar cleanup with a spatial grid
  vertex.py        single-vertex rules, crimp test, fix search
  checks.py        whole-pattern report, fixes, apply_fix
  examples.py      built-in bases and the random vertex generator
  vision.py        photo or screenshot → pattern
  synth.py         synthetic photos and screenshots for the tests and benchmark
crease_api/        FastAPI app and PostgreSQL storage
api/index.py       Vercel entry point (restores the original path after Vercel's rewrite)
public/            the web page (HTML, CSS, plain JS)
bench/             photo benchmark
tests/             pytest suite, including the brute-force oracle
```

## References

- J. Maekawa; T. Kawasaki (1989): the vertex conditions.
- T. Hull, "The combinatorics of flat folds: a survey" (2002): crimps and
  the big-little-big lemma.
- M. Bern and B. Hayes, "The complexity of flat origami" (1996).
- E. Demaine et al., the FOLD file format.
