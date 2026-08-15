# The full vizb journey: before, why, and everything since

Learning doc, not part of any PR — kept untracked, same as the other personal notes
(`vizb-integration-explained.md`, `plot-config-refactor-explained.md`). This one is the
narrative/chronological version: what existed before, why it changed, what the maintainer
actually asked for at each step, and how we got to where things stand now. For deep technical
detail on the rendering pipeline itself, see `vizb-integration-explained.md`.

## 1. What existed before, and why it had to change

`gomlx_checkpoints` is a standalone CLI tool (`cmd/gomlx_checkpoints`) for inspecting a trained
model's checkpoint directory — model size, hyperparameters, variable shapes, and training metrics.
The metrics part (`--plot`) used to render an interactive HTML chart using `go-plotly`, a Go
library, wired in via `ui/gonb/plotly`.

The problem: `go-plotly` (and `gonb`, which it depended on for notebook display) were **hard**
dependencies of the entire `gomlx` module. Every single person who imports `gomlx` for anything —
training a model, running inference, none of which touches `--plot` at all — pulled those
libraries into their own `go.mod` regardless. That's the core thing the maintainer
(janpfeifer) wanted fixed: a rarely-used CLI convenience feature shouldn't tax everyone's
dependency tree.

The chosen fix: replace `go-plotly` with [vizb](https://vizb.goptics.org/), an *external CLI
tool* instead of another Go library. Shelling out via `os/exec` to an optionally-installed binary
is a **soft** dependency — nothing changes in `go.mod`, and the tool works fine without it, except
for the one flag that needs it. This is the single idea the rest of the work builds on.

## 2. Phase 1: the actual swap

Implemented the rendering pipeline: `Point[]` data (already existed, unrelated to this change —
see §8) gets pivoted into a CSV, shaped into a vizb "DataSet JSON" via `vizb line`, combined across
metric types via `vizb merge`, and rendered to one self-contained HTML file via `vizb ui`. A
`checkVizbAvailable()` check up front gives a clear, actionable error (with the exact install
command) if `vizb` isn't on `PATH`, rather than a cryptic failure three subprocess calls deep.

While testing this by hand (not something the maintainer had flagged yet — found through actually
running `--plot -loop N` and watching what happened), a UX bug turned up: every `-loop` tick opened
a *new* browser tab, because the output path wasn't stable across iterations. Fixed with: a
session-stable output path, a guard so the browser only opens once, an mtime check to skip
re-rendering when nothing changed, and a `<meta http-equiv="refresh">` tag so an already-open tab
keeps itself current on its own.

**First iteration of that fix used a deterministic (hashed) output path** — the maintainer caught
a real problem with that: two unrelated, concurrent invocations of the tool against the same
checkpoint directory could collide on the same hashed filename. Switched to a random path picked
once per process instead — simpler *and* safer, per his explicit suggestion. This is a good
example of the collaboration pattern throughout: propose something, get concrete pushback with a
reason, land on something better than the first attempt.

Also folded `cmd/gomlx_checkpoints`'s own `go.mod` back into the root module — it had been split
out originally only because of `go-plotly`/`gonb`, which were now gone, so the split no longer
served a purpose.

## 3. PR #447: review feedback and merge

Opened as PR #447. One specific, generalizable piece of review feedback:

> Instead of using global variables, what about creating a small type `PlotBuilder` struct,
> ... rename `BuildPlot` to `NewPlotBuilder`... This makes it easier to test: no fear of
> forgetting if some global was properly set, and allows tests to run in parallel.

`BuildPlots` and four package-level variables (output path cache, "has rendered once" flag, etc.)
became a `PlotBuilder` type constructed once in `main()` and threaded down through the call chain.
The general lesson, not vizb-specific: package-level mutable state hides dependencies (a function
signature that doesn't reveal everything it reads/mutates), makes tests need an explicit
reset-and-restore ritual to avoid leaking into each other, and makes `t.Parallel()` unsafe. None of
that is true once the same state lives on a constructed value instead.

PR #447 was merged into `main` as a squash-merge (`b8a981cb`), including both the original swap and
the `PlotBuilder` review fix.

## 4. Real-world testing surfaces two bugs and three feature requests

The maintainer then actually tried to use the merged tool for something real: generating a demo
image for the README from two actual FlowMatching training checkpoints ("model-a"/"model-b"). He
sent the real fixture data along with the feedback:

> Plots are not displayed together when their "metric_type" is the same. ... E.g.: Loss on
> validation and Loss on training: one wants to see them together in the same plot to have an
> idea of when they diverge.
>
> Multiple checkpoints is not being handled correctly: that's one of the things I use it a lot
> for, compare how 2 (or more) model training sessions are doing.

Plus three feature requests: a `-plot_title` flag, a box listing which models are being compared,
and a box mapping each metric's short label to its full description.

**Investigation, using his real data, not guessing**: dumping the intermediate JSON `vizb line`
produces showed duplicate `"type"` values within a single dataset — vizb's chart renderer groups
series by that field, so two different metrics sharing a label silently collide into what looks
like one broken series. Confirmed with a minimal isolated repro against `vizb line` directly
(two `--select` columns with an identical label → duplicate `"type"` preserved as-is in the
output, not deduped). Traced the actual cause upstream: `Point.Short` (built in `ui/plot/plot.go`,
a different, unrelated part of the codebase) appends a per-dataset suffix that came out empty for
this fixture's eval datasets, so e.g. "Mean Loss on train" and "Mean Loss on validation" both
reduced to the identical `"#loss()"`.

**The fix** (`disambiguateShortLabels`, scoped entirely to `cmd/gomlx_checkpoints`, not touching
`ui/plot` — that's separate territory, see §8) detects collisions within one metric type's line
set and disambiguates only where needed. This single fix explained *both* reported bugs — the
multi-checkpoint case had the same collisions, just compounded by more lines competing for the
same short labels.

**The three features** were implemented as a sidebar (not stacked boxes above the charts, which
would push them below the fold): a fixed, independently-scrolling column on the left, styled
purely via `#app`'s stable `id` attribute (confirmed only `id="app"` exists statically in vizb's
output — everything else is built by its own JS after load, so styling by that one stable anchor
never depends on vizb's internal, unstable DOM). Each sidebar entry got a color-coded dot matching
its model — deliberately *not* trying to match vizb's own chart-line colors, since those can
change live via vizb's own in-browser theme dropdown, independent of anything set at generation
time.

## 5. Discovering PR #447 was already merged, and restructuring branches

Partway through implementing the bug fixes, checking `origin/main` revealed PR #447 had already
been squash-merged — meaning the branch this new work was built on (`dev/vizb-integration`) was
now stale, and continuing to add commits to it risked opening a PR against an already-closed one.
Restructured: fast-forwarded local `main`, created a fresh `dev/vizb-plot-fixes` branch from it for
the bug-fix/feature work, and separately moved the *other*, unrelated in-progress work (the Phase 0
`ui/plot.Config` refactor, a separate PR requested earlier by the maintainer — decoupling
`plotly.PlotConfig`'s generic scheduling logic from its rendering logic) back onto its own
`dev/plot-config-refactor` branch, also rebased onto the fresh `main`. Filed
[gomlx/gomlx#450](https://github.com/gomlx/gomlx/issues/450) to track the bug-fix work before
opening its PR.

## 6. The clickable-legends question — and the real fix that came from asking

Next maintainer feedback: the old `go-plotly` plots let you click a legend entry to hide/show that
line, and he wondered whether vizb (built on Apache ECharts) supports the same, pointing at an
[ECharts stacked-line demo](https://echarts.apache.org/examples/en/editor.html?c=line-stack) where
multiple lines share one chart with a clickable legend.

**Checked hands-on rather than assuming**: rendered the real multi-model comparison (19 series in
the "loss" panel) and confirmed — visually, in a real browser — that vizb's `-c line` chart type
at the time rendered **one chart per series**, not one chart with overlaid lines and a shared
legend. Also tested vizb's `--stack` flag directly against our data shape first, in case it already
covered this: `vizb line` reported `flag "stack" skipped: requires axis "y" (present: [x])` — it
only applied to a different ("grouped") axis mode than the one ("solo") we were using.

**Filed as a feature request, deliberately not a bug report** — the reasoning: we only knew two
things for certain (it wasn't a flag we were missing, and it wasn't what we needed), neither of
which proves the one-chart-per-series behavior was *unintended* on vizb's side. Framed the ask
neutrally ("would it be possible... at least as an option?") rather than asserting something was
broken.

**Kept the actual GitHub issue short**, per the maintainer's own advice from experience getting
responses from open-source maintainers: state the ask plainly with a reference link, hold the
detailed justification (the `--stack` test, the JSON dump) in reserve for if they ask. This worked:
[goptics/vizb#336](https://github.com/goptics/vizb/issues/336) got a detailed, technical reply from
a vizb maintainer (fahimfaisaal) within about 40 minutes.

**The reply reframed the whole problem**: "solo multi `--select`" mode (what this tool used)
is intentionally one-chart-per-series — not a gap, a documented design. The actual path to
multiple series on one chart with a shared legend is vizb's *group* mode, using `--col-axis y` to
put column names on the Y axis as series:

```bash
vizb line test.csv -n loss -g step -p x --col-axis y --select "col0{Alpha},col1{Beta}" -o loss.html
```

Group mode's "aggregation" (summing rows sharing a key) had been the specific reason solo mode was
chosen back in Phase 1 — but the reply clarified aggregation only fires when rows *actually* share
a key; with one value per (step, series), which is always true here, there's nothing to sum, so
nothing is lost. `--stack` was also clarified as a different thing entirely: a stacked *area* chart
(part-to-whole), not a plain multi-line overlay.

**Verified before writing any code**, same discipline as always: a minimal 2-series test first (one
chart, both lines, clickable legend — confirmed), then the full real production shape (19 series,
misaligned steps producing blank CSV cells, `vizb merge` across multiple metric types) — confirmed
blank cells are correctly omitted per-series (not zero-filled or erroring) and the merge/render
pipeline still produces one panel per metric type. Only then was `runVizbLine` actually changed
(`cmd/gomlx_checkpoints/vizb.go`) from solo mode to `-g step -p x --col-axis y`, with the stale
Phase-1-era test comment about "the exact failure mode of vizb's --group mode" corrected to reflect
the real understanding.

**A bonus, unplanned discovery** while verifying in a real browser: hovering a legend entry (not
clicking) triggers ECharts' own built-in "emphasis" effect — dimming every other series
temporarily. Distinct from click-to-toggle (which also works), and something neither side asked
for — it came free with switching to group mode, since that's just standard ECharts legend
behavior once a chart actually has multiple real series sharing it.

## 7. The full-name-on-hover question — and a genuine dead end, found by testing

Follow-up from the maintainer: could a tooltip with each metric's *full* name (not just its short
legend label) be added on hover — and specifically, could it be built by post-processing the
generated HTML ourselves, rather than needing vizb's help?

**Built it and it looked right in principle**: a small injected `<script>` that maps every known
short label to its full description, then uses a `MutationObserver` (since vizb's legend doesn't
exist in the DOM until its own JS finishes rendering) to find matching text and set the native
`title` attribute on it — no dependency on vizb's internal class names, only on the label text
itself appearing somewhere in the page, which is guaranteed since it's literally the string we gave
vizb. Exact text matching (not a substring check) was a deliberate choice, made *because* an early
probe script (using a looser substring check, only for manual investigation) surfaced a real
false-positive: `"#loss() (1)"` is a substring of `"#2 #loss() (1)"`, a different line's label
entirely.

**Testing it live surfaced a real, fundamental blocker** — not a bug in the script:

1. A devtools probe confirmed our tagging logic genuinely worked: the right `<span>` elements had
   the right `title` attributes set, with the correct full-description text.
2. The tooltip still never visually appeared on hover.
3. Right-clicking → "Inspect" directly on the visible legend text (not the console, the actual
   browser inspector) revealed why: vizb's legend renders onto a `<canvas>`, not real interactive
   DOM. A native tooltip can only trigger when the browser hit-tests an actual hovered element —
   canvas pixels are never that, regardless of how correctly a `title` attribute is set somewhere
   else in a parallel (likely accessibility-tree) DOM structure that happens to contain matching
   text.

**Checked for an escape hatch before giving up**: no `--renderer`/`svg`/`canvas` flag exists
anywhere in `vizb`'s CLI (checked `vizb ui --help`, `vizb line --help`, `vizb --help` directly) —
confirmed there's no way to force a DOM/SVG-based legend instead of canvas from the outside.

**Reverted the code rather than keep it**: shipping a script that silently tags invisible
parallel-DOM elements no user will ever actually hover is worse than not having it — it adds real
complexity (a page-wide `MutationObserver` in every generated file) for zero user-visible benefit.
The investigation and its conclusion are preserved directly in `injectPageExtras`'s doc comment
instead, so nobody re-discovers this the hard way. The sidebar's "Metric labels" list — already
built, already working — remains the reliable answer to "see the full name," exactly the fallback
the maintainer had offered as acceptable from the start ("we will still need a glossary box... or
a mouse over").

**Closed the loop with vizb**, keeping the message short per the same lesson from §6: thanked them,
confirmed the group-mode fix works end-to-end against real data, and asked one small, low-pressure
follow-up — whether `--select` could support a short legend label *and* a separate long name for
vizb's own tooltip system specifically (not ours), since that system already handles canvas
hit-testing correctly (visible in the "step / Y-axis / Total" boxes that already appear when
hovering a *data point*, as opposed to a legend entry) in a way an external DOM hack fundamentally
cannot.

## 8. Two related but genuinely separate pieces of work, and why they stayed separate

Worth being explicit about, since it comes up repeatedly: `ui/plot` (the `Point` type, the
`training_plot_points.json` file format, `plot.Config`) is a different, more foundational layer
than `cmd/gomlx_checkpoints` (the CLI tool that reads that data and renders it). The vizb work
lives entirely in the latter. A separate, maintainer-requested refactor of the former — making
`ui/gonb/plotly.PlotConfig` and `ui/gonb/margaid.Plots` thin wrappers around a new shared
`ui/plot.Config` type — was deliberately kept on its own branch and treated as its own PR
throughout, even when both pieces of work were in flight in the same working directory at the same
time. The `Point.Short` collision root-caused in §4 is a good illustration of why that boundary
matters in practice: the actual *cause* of the empty dataset-suffix lives in `ui/plot`, but the
*fix* belongs entirely in `cmd/gomlx_checkpoints`, because that's the only place uniqueness of the
label actually matters — reaching into `ui/plot` to "properly" fix the root cause would have meant
touching the other PR's territory for no real benefit.

## 9. Status at that point

- **Merged**: the original vizb swap + `PlotBuilder` refactor (PR #447, now part of `main`).
- **In progress on `dev/vizb-plot-fixes`**, tracked by
  [gomlx/gomlx#450](https://github.com/gomlx/gomlx/issues/450): label disambiguation, the sidebar
  (`-plot_title`, models-compared box, metric-labels box), the group-mode switch, and the reverted
  hover-tooltip attempt.
- **Resolved upstream**: [goptics/vizb#336](https://github.com/goptics/vizb/issues/336) — the
  group-mode recipe now in use came from that thread.
- **Separate, still in progress**: the `ui/plot.Config` refactor on `dev/plot-config-refactor`,
  its own future PR.

What follows (§10 onward) is everything that happened *after* this point — round-2 review feedback
from the maintainer, several more upstream vizb findings, and a substantial sidebar redesign.

## 10. Round-2 feedback: five points, tackled in maintainer-approved order

Jan tried the fixed tool against the real `model-a`/`model-b` fixture (now saved locally at
`models-examples/`) and sent five more points. Asked him which to prioritize before touching code;
he said 1, 3, 4 first, then 2, then investigate 5.

1. **Drop "for model X" repetition** in descriptions — the "Models compared" box already
   establishes which model is which, so `createPlotLines` stopped appending it.
2. **Dedupe the glossary across models** — a metric measured on both models (e.g. `T/~loss`) was
   showing up twice, once per model. Fixed with `buildDescriptionEntries`, deduplicating by
   `(coreShort, desc)`. **Found a real bug along the way**: `createPlotLines` builds its line list
   by ranging over a Go map, whose iteration order is randomized per process — so two models'
   independent train/validation disambiguation suffixes could get mismatched purely by chance,
   silently breaking the dedup. Fixed by sorting lines by `desc` before disambiguating, so the same
   underlying metric always gets the same suffix regardless of which model produced it. Locked in
   with `TestDisambiguateShortLabels_ConsistentAcrossModels`, using deliberately shuffled input.
3. **Unicode model markers** (`❶`/`❷` instead of `#1`/`#2`) — "#" already means something else in
   these metric names ("average", vs "~" for "moving average"), so `#1 #loss()` read ambiguously.
   Falls back to `[N]` brackets past 10 models (circled-digit block only covers 1–10).
4. **Smooth curves** — `--smooth` on `vizb line`, confirmed via `--help` and the live JSON output
   (`"smooth":true`).
5. **Investigate, don't assume**: X-axis categorical-vs-continuous, and Y-axis log-scale toggle
   reliability. Covered in §12.

## 11. The value-rounding bug: found, worked around, fixed upstream, workaround reverted

Jan reported (via a `compare.html` reference file he'd built independently, *not* generated by our
tool — worth noting since it briefly caused confusion about whether the sidebar labels were
vizb-native): "it seems to round the values: in the original plot there were no horizontal straight
lines."

**Traced precisely, not guessed**: confirmed the source data has full float64 precision, confirmed
our own CSV output preserves it (`strconv.FormatFloat(value, 'f', -1, 64)`), then confirmed via
direct JSON inspection that `vizb line` itself was rounding every value to a *fixed* 2 decimal
places during CSV→JSON conversion, regardless of magnitude (`2.123` → `2.12`, but also
`914.273581` → `914.27` — always exactly 2 decimals). For metrics that vary by less than 0.01
between steps — extremely common on converging training curves — this turns real variation into a
flat, misleading line segment. Built a clean, isolated 5-point repro (`precision.csv`) showing 5
genuinely different declining values all collapsing to `0.91`.

**Built a workaround before the upstream fix existed**: scale every value up by `1e6` before `vizb
line` sees it (so its fixed 2-decimal rounding preserves ~6 more decimal places of the original),
then scale the resulting JSON's values back down before rendering — confirmed `vizb ui` doesn't
re-round when rendering, so this genuinely survived to the final chart. Verified end-to-end against
the real `model-a`/`model-b` fixture: zero rounding collisions across 21 series.

**Filed upstream, short and simple**: `fahimfaisaal` (vizb maintainer) confirmed it as a real bug
within about an hour and shipped a fix in v0.18.0.

**Verified the fix concretely, then removed the workaround entirely** — rather than assume a
version bump fixes things: reinstalled `vizb@latest`, re-ran the exact `precision.csv` repro
against the confirmed new version, saw all 5 values pass through untouched. Only then deleted
`vizbValuePrecisionScale`, `scaleLineValues`, and `rescaleDataSetValues` from `vizb.go` — no reason
to keep dead complexity once the real fix landed. `runVizbLine`'s doc comment now states the
`vizb >= v0.18.0` requirement plainly, linking the upstream confirmation.

**A real environment bug found along the way**: `~/.local/bin/vizb` (an old v0.16.1) was shadowing
the newly-installed `~/go/bin/vizb` (v0.18.0) on `PATH` — meaning even after upgrading, every
invocation of plain `vizb` was silently still hitting the buggy old binary. Diagnosed with
`which -a`, confirmed via `go version -m` on each binary, fixed by removing the stale one. This
wasn't a one-off: a regression test (`TestRunVizbLine_PreservesNarrowRangePrecision`) caught the
exact same shadowing issue a second time later in the session, purely by failing when it should
have passed — a good example of a test doing its job.

## 12. Two more upstream findings, both investigated hands-on before being reported

**Y-axis log-scale toggle silently fails on narrow-value-range data.** Isolated by direct
comparison: a wide-range repro (values spanning ~59×) reshapes correctly when toggled to
logarithmic; a narrow-range repro (values spanning ~2.5×) shows the toggle as selected but the
chart never changes. This isn't mathematically expected — log transform is well-defined and
non-trivial even for a 2.5× range, just less dramatic than a 59× range, so a complete *lack* of any
visible change (not just a smaller change) points at an internal threshold/rounding edge case in
vizb's own axis-bounds computation. Filed as
[goptics/vizb#345](https://github.com/goptics/vizb/issues/345), kept short (one minimal repro, no
prescribed diagnosis) per the same lesson from §6.

**X-axis is always categorical, even for purely numeric data.** Jan asked directly whether ECharts
(which vizb is built on) could support a continuous/log X-axis. Rather than answer from memory,
built a real, minimal, standalone ECharts page (no vizb involved) with the *same* data rendered two
ways — `xAxis.type: "category"` vs `xAxis.type: "log"` — and screenshotted the result: the log
version correctly spaces `1, 10, 100, 1,000, 10,000, 100,000` by equal ratio, proving the
*library* supports it. So this is a vizb implementation choice (probably because vizb's core use
case — comparing named benchmarks — genuinely has no numbers on the X-axis), not an ECharts
limitation. A short feature-request draft for this is ready, not yet filed.

**Hit the same sandboxing class of bug twice while building that demo**: the snap-confined
Chromium used for verification screenshots can't write to `/tmp` at all (`--screenshot` and
`--dump-dom` both silently produced nothing there, no error) and can't read arbitrary `file://`
paths outside `$HOME` either — both needed routing through `$HOME` instead. Separately, the user's
own browser opens local HTML files through an `xdg-desktop-portal` path
(`/run/user/1000/doc/<hash>/...`) that exposes *only* the single opened file, not sibling files in
the same directory — meaning a demo page with `<script src="echarts.min.js">` (a sibling file
reference) rendered as blank white boxes until the library was inlined directly into the HTML
instead of referenced externally. Neither issue was about vizb or our own code — both were local
environment quirks worth documenting so they're not mistaken for something else next time.

## 13. Sidebar redesign: from ad hoc styling to a validated design system

Asked to make the sidebar "clean, standard, industry" — how a design-conscious engineer would
build it, not just functional. Applied the repo's `dataviz` skill (a design-system-agnostic method:
pick the form, assign color by the job it does, validate the palette with a script rather than
eyeballing it, then apply mark/spacing conventions) rather than picking colors and fonts by feel.

- **Structure**: `legendEntry` split from one pre-concatenated string into `term`/`detail` —
  `<dl>`/`<dt>`/`<dd>`, the semantically correct element for label→description pairs, replacing
  plain `<li>` bullets. Metric short-codes render in `<code>` (they're literal identifiers, not
  prose); checkpoint paths too.
- **Typography/tokens**: `system-ui` font stack, uppercase muted section headers with a hairline
  rule, CSS custom properties for surface/text/border colors that flip automatically under
  `prefers-color-scheme: dark`.
- **Color**: swapped an ad hoc 6-color palette for the `dataviz` skill's validated 8-hue
  categorical order — actually ran `validate_palette.js` against both a light and dark sidebar
  surface (not assumed to pass just because it's the reference palette) before using it.
- **Per-metric-type grouping**: "Metric labels" split into one section per chart (`Metric labels —
  loss`, `Metric labels — img_loss`, ...) in the same order the charts themselves render, so a
  reader looking at one chart isn't scrolling past every other chart's labels. `plotLineInfo`
  gained a `metricType` field to make this possible; `buildDescriptionEntries` groups by it.
- **Per-row color, not per-model color**: initially every sidebar row cycled through the palette by
  position (mirroring "every chart series gets its own color"). Confirmed via a `Plot()`-level
  integration test (`TestPlot_SidebarReflectsCurrentMetricsAcrossRepeatedCalls`) that the sidebar
  genuinely regenerates fresh on each `-loop` iteration rather than going stale.

## 14. The color-matching breakthrough — and walking back an earlier "not feasible" claim

Asked whether the sidebar's swatch colors could match vizb's own chart-line colors. Initial answer,
based on §7's canvas-rendering finding, was that this wasn't feasible — vizb assigns chart colors
internally, with no way to know or influence them from outside.

**That answer was wrong, and re-checking (rather than repeating it) found why**: `vizb line --help`
has a `--theme` flag — "Embed a color theme on the dataset... bare `#hex,#hex,...` palette" — never
previously noticed. Tested directly: passing a hex list embeds it into the DataSet JSON's `themes`
field, and it survives verbatim into the rendered chart, in the same order as the series, including
correctly repeating a hex at two positions when two series should share a color.

**A real design tension surfaced once this became possible**: the chart plots one line *per model
per metric* (two differently-colored lines for a shared metric), but the sidebar already collapses
that to *one* glossary entry — so a single swatch can't exactly match two differently-colored chart
lines at once. Given the explicit choice (force the same color per metric, accepting that two
models' lines for a shared metric become the same chart color, relying on the `❶`/`❷` marker text
to tell them apart instead of color): `buildMetricTypeSection` was factored out of
`buildDescriptionEntries` so `Plot()` can compute the exact same, deterministically-ordered
per-metric-type entry list *before* calling `runVizbLine`, assign each distinct metric a
`swatchColorHex` slot, and pass matching colors to `--theme`. Verified against the real fixture
end-to-end (not just unit-tested): parsed the actual embedded `themes` JSON and the actual rendered
sidebar CSS, cross-checked all 19 series — zero mismatches.

**Then a second real usability issue surfaced from the first**: reusing the same palette for
"Models compared" swatches meant a model's dot could coincidentally match an unrelated metric's
color elsewhere on the page, implying a connection that doesn't exist (colors now mean "which
metric," consistently, everywhere else). Fixed by dropping the swatch from "Models compared"
entirely — the `❶`/`❷` marker already uniquely identifies each model, so the color was never
load-bearing there, only decorative, and had become actively misleading.

## 15. Where things stand now

Jan's read after all of the above: "We could merge your PR as is for now, and do another PR for
when vizb fixes [the Y-axis log-scale bug] (and maybe add support for [continuous X-axis])." In
other words: what's on `dev/vizb-plot-fixes` is considered mergeable *now*, with the two open
upstream items (§12) tracked as follow-up work once vizb ships them, not blockers.

- **Merged**: PR #447 (original vizb swap + `PlotBuilder`), part of `main`.
- **Ready to merge, tracked by [gomlx/gomlx#450](https://github.com/gomlx/gomlx/issues/450)**:
  label disambiguation with the desc-derived `(train)`/`(validation)` suffix, group-mode rendering,
  the redesigned sidebar (validated palette, per-metric-type sections, chart-matching colors), all
  five round-2 feedback points, `--smooth` curves.
- **Resolved upstream**: [goptics/vizb#336](https://github.com/goptics/vizb/issues/336) (group
  mode) and the value-rounding bug (fixed in v0.18.0, our workaround since removed).
- **Open upstream, not blocking**: [goptics/vizb#345](https://github.com/goptics/vizb/issues/345)
  (Y-axis log-scale, narrow range) filed; the continuous-X-axis feature request drafted, not yet
  filed.
- **Separate, still in progress**: the `ui/plot.Config` refactor on `dev/plot-config-refactor`.
