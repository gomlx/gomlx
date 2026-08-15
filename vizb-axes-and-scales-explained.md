# Axes and scales in vizb, from first principles

Learning doc, not part of any PR — kept untracked, same as the other personal notes
(`vizb-full-journey-explained.md`, `vizb-integration-explained.md`). Those two cover *what
happened* and *what the code does*; this one is neither — it's the conceptual layer underneath
both: what an axis actually *is*, why linear and logarithmic scales behave differently, and how
that maps onto the specific bugs found in vizb. Built chunk by chunk, each one only depending on
the chunk before it.

## Chunk 1: What Jan actually wants from this plot

Strip away every bug report and feature request, and the underlying need is simple: he's training
ML models, and at every so many steps the training loop logs a metric (loss, accuracy, ...) to a
file. He wants a chart of **metric value vs. training step**, so he can look at the shape of the
curve and answer questions like:

- Is the loss still going down, or has it plateaued?
- Did model A converge faster than model B?
- What happened early in training (the first few hundred steps), when the curve usually moves
  fastest and is hardest to see once the X-axis also has to cover the last 100,000 steps?

That last question is *why* log scale matters to him at all — it's not an aesthetic preference,
it's the only way to see "step 1 to step 100" in the same chart as "step 1,000 to step 100,000"
without the early part being crushed into a few invisible pixels at the left edge. Everything
else in this doc — categorical vs. continuous axes, linear vs. log, tick generation — only matters
because of this one real goal.

## Chunk 2: What an axis actually *is*

An axis is a rule for turning a value into a pixel position on screen. That's it. Every other
concept in this doc (categorical, linear, log, ticks) is just a different flavor of that one rule.

Two fundamentally different flavors:

- **Categorical**: like a bus-stop sign that lists stops in order — Stop 1, Stop 2, Stop 3 — evenly
  spaced on the sign, no matter whether Stop 2 is 1 mile or 50 miles down the road from Stop 1. The
  sign only encodes *order*, never *distance*. There's no number underneath a stop name; you can't
  do arithmetic on "Stop 2."
- **Continuous**: like a ruler, where the physical distance between two marks represents the real
  numeric distance between the values. If Stop 2 is 50 miles away and Stop 3 is only 1 mile past
  that, a ruler-style map shows a long gap then a short one, because it places things by their
  actual value.

vizb's line chart puts the X axis through the categorical path unconditionally — every step number
becomes a bus-stop label, not a ruler position. That's chunk 5's first bug.

## Chunk 3: Linear vs. logarithmic — two different continuous rulers

Once an axis is continuous (a ruler, not a bus-stop sign), there's still a choice for *how* the
ruler's marks relate to the numbers: linear or logarithmic.

- **Linear**: equal *differences* in value get equal spacing on screen. The distance from 10 to 20
  looks the same as the distance from 1,010 to 1,020 — both are "+10."
- **Logarithmic**: equal *ratios* in value get equal spacing on screen. The distance from 1 to 10
  (×10) looks the same as the distance from 10,000 to 100,000 (also ×10) — regardless of the fact
  that the second jump is 9,000 times bigger in absolute terms.

Concrete example, using real step numbers: 1, 10, 100, 1,000, 10,000, 100,000.

- On a **linear** X-axis, steps 1 through 10,000 would all be crushed into the first 10% of the
  chart's width, since 100,000 dominates the range. You'd see nothing of early training.
- On a **log** X-axis, each ×10 jump (1→10, 10→100, 100→1,000, ...) gets the *same* width — so
  early training (steps 1–100) gets just as much visual room as late training (steps 10,000–
  100,000). This is exactly the property Jan needs (chunk 1).

The same logic applies to a Y-axis of loss/metric *values* — a metric that drops from 2.0 to 0.05
over training spans almost two orders of magnitude, so a log Y-axis shows the late-training
fine detail (0.05 to 0.06, say) with the same visual weight as the early, coarser changes (2.0 to
1.5).

## Chunk 4: Ticks are a *separate* decision from the axis rule

A "tick" is one of the small labeled marks along an axis — the printed numbers, like the marks on
a ruler. This is easy to conflate with the axis rule itself (chunk 3), but they're two independent
decisions:

1. **The axis rule**: how do I convert a value into a pixel position? (linear or log — chunk 3)
2. **Tick selection**: given that rule, which specific values do I bother printing labels for?

A log axis does *not* imply the tick labels themselves must be spaced by a constant ratio. Good
tick-generation logic picks whatever values are most *readable* for the actual data range, which
depends on how many orders of magnitude that range spans:

- **Wide range** (spans several decades, e.g. steps 1 to 100,000): decade ticks make sense —
  `1, 10, 100, 1k, 10k, 100k`, each a clean power of ten, with minor ticks at `2, 5` between them.
  This is what Plotly generates for the X-axis in our reference chart (verified directly, not
  assumed — see chunk 6).
- **Narrow range** (spans less than one decade, e.g. a loss value moving from 0.12 to 0.30):
  decade ticks would be useless — the only clean powers of ten nearby are `0.1` and `1.0`, both
  outside or barely touching the actual data. A good tick generator instead falls back to plain,
  evenly-**stepped** decimals: `0.12, 0.14, 0.16, 0.18, 0.20, ..., 0.30`. This confirmed directly
  (see chunk 6): those ticks increase by a constant *difference* (+0.02 each), not a constant
  *ratio* — i.e. they're linearly stepped, even though they're being drawn on a log-scale axis.
  That's not a contradiction: the axis (chunk 3) is still using the log transform to place points;
  it's only the *choice of which numbers to label* that falls back to linear-looking steps when a
  log-native tick set (decades) would be too sparse to be useful.

vizb's current bug (chunk 5) is that it doesn't do *either* of these — for a narrow range, it
prints only the two extreme values and stops, generating no intermediate ticks of either kind.

## Chunk 5: How this actually breaks down in vizb — four separate issues

Easy to blur together into "the axes are broken," but hands-on investigation (chunk 6) found four
genuinely distinct problems, each with its own root cause, found and fixed (or reported)
independently:

1. **Value rounding** (a data-precision bug, nothing to do with axes at all): `vizb line` used to
   round every value to a fixed 2 decimal places during CSV→JSON conversion, regardless of
   magnitude — so a converging loss curve with values like `0.914273581, 0.913891247, ...` all
   collapsed to `0.91`, showing up as a flat, misleading line segment. **Fixed upstream in v0.18.0**
   (traced, reported, and confirmed by us — see `vizb-full-journey-explained.md` §11).
2. **Log-axis domain (min/max) computation** (chunk 3's rule, applied wrong): the Y-axis log scale
   used to clamp its floor to a power-of-10 boundary at ≥1 — so for data entirely between 0 and 1
   (like a loss of 0.12–0.30), the computed range included a huge empty band up to 1, making the
   toggle look like it did nothing at all. **Fixed upstream in v0.18.2**
   ([goptics/vizb#345](https://github.com/goptics/vizb/issues/345) /
   [#348](https://github.com/goptics/vizb/pull/348)) — confirmed both via an isolated repro and
   against the real `img_loss` chart.
3. **Log-axis tick generation** (chunk 4, currently absent): even with the domain fixed, vizb's Y
   axis under log scale currently prints *only* the two extreme values as ticks, none in between —
   unlike the reference behavior in chunk 4. **Open**
   ([goptics/vizb#351](https://github.com/goptics/vizb/issues/351)).
4. **X-axis is always categorical** (chunk 2, the axis rule itself): vizb's line chart routes every
   X value through the categorical path regardless of whether the values are genuinely numeric —
   confirmed to be a vizb implementation choice, not an ECharts limitation (chunk 6), likely
   because vizb's core use case (comparing named benchmarks) has no numbers on the X-axis at all.
   **Open**, feature request drafted with fahimfaisaal (not yet filed as a formal issue) — he's
   asked for the concrete use case and a proposed CLI surface, which we've supplied (extending
   `-S/--scale` to optionally target an axis, e.g. `--scale x:log`).

Note how #1 and #2 could easily be mistaken for the same bug (both were reported as "the values
look wrong" / "the log toggle isn't working") — but they're different code paths (CSV→JSON value
conversion vs. chart-rendering axis-domain computation), which is exactly why we double-checked
before assuming the `#337`/`#338` fix (unrelated: an on-chart `--show-labels` text-formatting fix)
resolved #1's rounding bug. It didn't — #1 needed its own separate report and fix.

## Chunk 6: How every claim above was actually verified, not assumed

Every fact in chunks 3–5 was confirmed hands-on, specifically because "this library obviously
supports X" or "this must be the same bug as that other one" turned out to be wrong more than
once during this investigation:

- **Whether ECharts supports a continuous/log X-axis at all** (chunk 5, item 4): built a real,
  minimal, standalone ECharts page (no vizb involved), rendered the same data as
  `xAxis.type: "category"` and `xAxis.type: "log"`, and screenshotted both — confirmed the log
  version correctly space-by-ratio positions `1, 10, 100, 1,000, 10,000, 100,000`. This is what
  turned "is this even possible?" into "this is a vizb choice, not a library limitation."
- **Whether the value-rounding bug and the log-axis bug were the same thing** (chunk 5, items 1 vs.
  2): re-tested the exact rounding repro against the vizb version that had already merged the
  `#337`/`#338` fix, and confirmed the rounding was still present — proving they were unrelated
  code paths, not assuming a fix for one covered the other.
- **What "properly ticked" actually looks like, in real numbers** (chunk 4): rather than guess
  what a fixed vizb *should* show, generated the same 0.12–0.30 dataset through the old `go-plotly`
  library (still present in the repo, unused by `cmd/gomlx_checkpoints` since PR #447, but usable
  standalone) with an explicit log Y-axis, screenshotted it, and read off the actual tick values —
  `0.12, 0.14, ..., 0.30` — then checked whether they were log-ratio-spaced or linear-stepped by
  computing the actual ratios between consecutive ticks (they shrink: ×1.167, ×1.143, ×1.125, ...,
  ×1.071 — not constant, so linear-stepped, not log-stepped).
- **Whether each upstream fix actually landed**, rather than trusting a linked PR: for both the
  rounding fix (v0.18.0) and the log-domain fix (v0.18.2), reinstalled the specific vizb version,
  re-ran the *exact original repro*, and inspected the real output (JSON values, or a rendered
  screenshot) before considering either resolved.

The general pattern across all of this: when a claim is checkable (does the library support X? is
this the same bug as that? what does correct output actually look like?), build the smallest
possible example that checks it, rather than reasoning from what "should" be true.
