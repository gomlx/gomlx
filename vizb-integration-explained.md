# The vizb integration, code by code, first principles

Learning doc, not part of any PR — kept untracked, same as the other personal notes
(`vizb-full-journey-explained.md`, `plot-config-refactor-explained.md`). That other doc is the
narrative/chronological version (what happened, when, why); this one is the code-level companion —
every file, every function, why it exists, and (for the one piece that got built and then removed)
why it doesn't exist anymore. Every snippet below is the real, current code, read fresh from disk,
not from memory.

## 1. First principles: what this tool does, and the boundary that shapes everything

`gomlx_checkpoints` is a CLI tool that inspects a trained model's checkpoint directory.
`--plot` is one flag among several (`--summary`, `--params`, `--vars`, `--metrics`) — the one that
renders training metrics as an interactive HTML chart.

The data it renders already existed before any of this work started, in `ui/plot/plot.go`: a
`Point` (metric name, a compact "short" name, a "type" used to group related metrics onto the same
chart, a step, a value) and a plain JSON-lines file (`training_plot_points.json`) written during
training and read back later by this tool. That split — training writes data, a *separate* process
reads and renders it — is a real architectural boundary, not incidental, and it's why the vizb work
never touches `ui/plot` at all: everything below lives entirely in `cmd/gomlx_checkpoints`.

`--plot` used to render that data with `go-plotly`, a Go library. The problem: `go-plotly` (and
`gonb`, which it needed for notebook display) were **hard** dependencies of the whole `gomlx`
module — every importer got them in their `go.mod`, whether they ever touched `--plot` or not. The
fix: replace it with an *external CLI tool*, [vizb](https://vizb.goptics.org/), shelled out to via
`os/exec`. That's a **soft** dependency — nothing in `go.mod` changes, and everything except
`--plot` itself works with or without `vizb` installed:

```go
// cmd/gomlx_checkpoints/vizb.go
func checkVizbAvailable() error {
	if _, err := exec.LookPath("vizb"); err != nil {
		return errors.New("--plot requires the `vizb` CLI tool, which was not found on your PATH.\n" +
			"Install it with:\n\n\tgo install github.com/goptics/vizb@latest\n\n" +
			"or see https://vizb.goptics.org/ for other install options.")
	}
	return nil
}
```

Called once, at the very top of `PlotBuilder.Plot` (§4) — fail fast with an actionable error,
rather than a cryptic failure three subprocess calls deep.

## 2. The rendering pipeline (`cmd/gomlx_checkpoints/vizb.go`)

Three `vizb` subcommands, chained together, once per metric type, then merged:

```
[]plotLineInfo (one metric type)
  → writeLinesAsCSV   (pivot to a wide CSV: one row per step, one column per line)
  → vizb line          (CSV → one DataSet JSON, one series per column)
  → [repeat per metric type]
  → vizb merge          (combine all per-type JSONs into one, only if there's more than one type)
  → vizb ui              (JSON → one self-contained HTML file)
```

### `writeLinesAsCSV`: why a "wide" CSV with blank cells

The data isn't guaranteed aligned — e.g. training loss logged every step, eval loss only every 100.
So the CSV can't just be "one row per (step, line)"; it has to be one row per *distinct step across
every line*, with a blank cell wherever a given line has no value there:

```go
func writeLinesAsCSV(w io.Writer, lines []*plotLineInfo) error {
	// Collect the sorted union of all Step values across all lines: they may
	// not be aligned (e.g. eval logged every 100 steps, train every step).
	stepSet := make(map[float64]bool)
	for _, line := range lines {
		for _, step := range line.steps {
			stepSet[step] = true
		}
	}
	steps := make([]float64, 0, len(stepSet))
	for step := range stepSet {
		steps = append(steps, step)
	}
	slices.Sort(steps)

	// Per-line lookup: step -> value.
	lineStepValue := make([]map[float64]float64, len(lines))
	for i, line := range lines {
		m := make(map[float64]float64, len(line.steps))
		for j, step := range line.steps {
			m[step] = line.values[j]
		}
		lineStepValue[i] = m
	}

	cw := csv.NewWriter(w)
	header := make([]string, 1+len(lines))
	header[0] = "step"
	for i := range lines {
		header[i+1] = fmt.Sprintf("col%d", i)
	}
	// ... write header, then one row per step, blank cell if a line has no value there.
}
```

`vizb` is expected to skip a blank numeric cell for that series/point, rather than treating it as
zero — this is the property `TestRunVizbLine_MisalignedStepsOmitMissingSeries` locks in against the
real binary (§7), not just this CSV-writing step in isolation.

### `runVizbLine`: solo mode, then a real correction to group mode

This function had two genuinely different implementations over the life of this work, and the
*reason* for the change is one of the more interesting parts of the whole project.

**First version (solo mode)**, from the initial PR:
```go
args := []string{"line", csvPath, "-o", jsonPath, "-n", metricType}
for i, line := range lines {
	args = append(args, "--select", fmt.Sprintf("step,col%d{%s}", i, line.short))
}
```
This groups every line for one metric type into a single DataSet JSON — confirmed correct at the
time. What wasn't confirmed (and turned out wrong) was an assumption about *how vizb renders* that
JSON: `vizb ui -c line` turned out to render **one chart per series**, not one chart with every
series overlaid on a shared, clickable legend. Confirmed by rendering real 19-series data and
looking at it in an actual browser, not by inspecting the JSON alone (the JSON genuinely does look
"merged" — the rendering choice is a separate layer on top of it).

Asking `vizb`'s own maintainers what was intended (rather than guessing) reframed the problem
entirely: solo mode's one-chart-per-series behavior was described as *intentional*, and the actual
path to an overlaid multi-line chart with a shared legend is vizb's **group** mode, putting series
on the Y axis:

```go
// cmd/gomlx_checkpoints/vizb.go (current)
func runVizbLine(tmpDir, metricType string, lines []*plotLineInfo, lineColors []string) (jsonPath string, err error) {
	// ... write CSV (unchanged, same writeLinesAsCSV) ...

	jsonPath = filepath.Join(tmpDir, metricType+".json")
	selectCols := make([]string, len(lines))
	for i, line := range lines {
		selectCols[i] = fmt.Sprintf("col%d{%s}", i, line.short)
	}
	args := []string{
		"line", csvPath, "-o", jsonPath, "-n", metricType,
		"-g", "step", "-p", "x", "--col-axis", "y",
		"--select", strings.Join(selectCols, ","),
		"--smooth",
	}
	if len(lineColors) > 0 {
		args = append(args, "--theme", strings.Join(lineColors, ","))
	}

	cmd := exec.Command("vizb", args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", errors.Wrapf(err, "vizb line failed for metric type %q:\n%s", metricType, string(output))
	}
	return jsonPath, nil
}
```

Two flag changes carry the whole fix: `-g step -p x` says "step is the X dimension," and
`--col-axis y` says "treat the CSV's column *names* as Y-axis series" — i.e. each `colN` becomes
its own line on one shared chart, instead of `--select step,colN{label}` (repeated per column)
producing what turned out to be independent, separately-charted series. `--smooth` (round-2
feedback) renders curved rather than piecewise-linear segments — confirmed via the live JSON
output (`"smooth":true`), not just the `--help` text.

`lineColors`, when non-empty, is passed straight to `--theme` — confirmed directly (not assumed)
that `vizb line` embeds a bare comma-separated hex list into the DataSet JSON's `themes` field and
applies it to the chart in that exact order, including repeating a hex at two positions when two
series should share a color. This is what makes the sidebar's swatch colors and the chart's actual
line colors match (§5) — found by re-reading `vizb line --help` a second time, after an earlier,
wrong conclusion that chart colors couldn't be influenced from outside at all.

**Requires vizb >= v0.18.0.** Earlier versions rounded every value to a fixed 2 decimal places
during CSV→JSON conversion regardless of magnitude — silently flattening closely-spaced values
(common on converging training curves) into misleading flat line segments. Found via a maintainer
report ("it seems to round the values... in the original plot there were no horizontal straight
lines"), traced precisely (confirmed full precision at every stage of *our* pipeline, then confirmed
via direct JSON inspection that `vizb line` itself was the source), worked around for a while
(scale values up before `vizb line` sees them, scale the resulting JSON back down before
rendering), then the workaround was deleted entirely once `fahimfaisaal` shipped a real fix in
v0.18.0 — reconfirmed the fix concretely (re-ran the exact original repro against the new version)
before removing anything, rather than trusting a version bump alone.

**Why this was safe to change, verified rather than assumed**: group mode's defining feature is
that it *aggregates* rows sharing the same key (summing them) — which is exactly what the original
solo-mode choice had been trying to avoid, since summing would silently corrupt a time series.
Turns out that only matters if rows *actually* collide on a key; with one value per (step, series),
which this data always has, there's nothing to sum, and `vizb` confirms this itself at generation
time (`"N grouped rows -- all unique, no duplicates to sum"`). Verified concretely, not just taken
on faith, against real production data:
- 19 series, some steps with missing values for some series (the `writeLinesAsCSV` blank-cell
  case) — confirmed each step's JSON entry only includes the series that actually had a value
  there, nothing zero-filled, nothing lost.
- Multiple metric types, still merged via `vizb merge` into one file with one panel per type —
  unaffected by the mode switch, still exactly two datasets (`"loss"`, `"img_loss"`) in the merged
  output for the real fixture.

One dead-end checked and ruled out *before* asking anyone: `vizb line`'s `--stack` flag looked
plausible from `--help` alone, but running it against this exact data shape reported
`flag "stack" skipped: requires axis "y" (present: [x])` — it only applies to the grouped axis
mode this code wasn't yet using, and (confirmed later, from vizb's own maintainers) even in group
mode it renders a *stacked area* chart, not a multi-line overlay — a different visual entirely.

### `mergeAndRenderVizb`: unchanged by any of the above

```go
func mergeAndRenderVizb(tmpDir string, jsonPaths []string, outputPath string) error {
	if len(jsonPaths) == 0 {
		return errors.New("mergeAndRenderVizb: no JSON datasets to render")
	}
	renderInput := jsonPaths[0]
	if len(jsonPaths) > 1 {
		mergedPath := filepath.Join(tmpDir, "merged.json")
		args := append([]string{"merge"}, jsonPaths...)
		args = append(args, "-o", mergedPath)
		// exec "vizb" args...
		renderInput = mergedPath
	}
	// exec "vizb" "ui" renderInput "-o" outputPath "-c" "line"
	return nil
}
```
Notably, this function never needed to change across the solo→group mode switch: it operates on
whatever DataSet JSON `runVizbLine` handed it, agnostic to which mode produced it. That's the
value of the CSV→JSON→merge→render pipeline being cleanly staged — the fix for the rendering bug
was entirely contained in one function's flag choices.

## 3. `PlotBuilder` and the `-loop` mechanic (`cmd/gomlx_checkpoints/plots.go`)

`--plot -loop 30s` re-runs the tool on a timer to watch training live. That needs state that
survives *across* repeated runs within one process: don't open a second browser tab every tick,
don't pick a new output file every tick, don't re-invoke the whole `vizb` subprocess chain if
nothing changed since the last render.

```go
type PlotBuilder struct {
	// Configuration, fixed for the lifetime of this PlotBuilder.
	openBrowser    bool
	outputOverride string        // -plot_output; if empty, a random path is picked once and reused.
	loopPeriod     time.Duration // -loop; > 0 enables the auto-refresh <meta> tag.
	plotTitle      string        // -plot_title; if empty, no title is added.

	// Session state, mutated across repeated Plot calls.
	browserOpenedOnce bool
	cachedOutputPath  string
	hasRenderedOnce   bool
	lastRenderModTime time.Time
}
```

This used to be four package-level variables instead of a struct — changed following real PR
review feedback (not a stylistic preference asserted in the abstract):

> Instead of using global variables, what about creating a small type `PlotBuilder` struct...
> This makes it easier to test: no fear of forgetting if some global was properly set, and allows
> tests to run in parallel.

The general reasoning, not vizb-specific: a free function reading/mutating package-level state
*looks* like a pure function of its arguments but secretly isn't — nothing in its signature reveals
what it depends on. Every test touching that state needs an explicit reset ritual to avoid leaking
into the next test (miss one, and the failure is order-dependent — often won't reproduce when you
rerun just the failing test alone). And `t.Parallel()` becomes unsafe. None of that is true once
the same state lives on a constructed value. `main()` now builds **one** `PlotBuilder` before the
`-loop` `for {}` starts (not inside it — recreating it every tick would lose exactly the state that
makes any of this work) and threads it down through `Reports` → `metrics` → `PlotBuilder.Plot`.

The actual `-loop` mechanics, inside `Plot`:
```go
outputFilePath := pb.resolveOutputFilePath(checkpointPaths) // same path every call
currentModTime := latestMetricsModTime(checkpointPaths)
if pb.hasRenderedOnce && !currentModTime.After(pb.lastRenderModTime) {
	// Nothing new: skip the expensive vizb subprocess chain entirely.
	fmt.Printf("\nNo new metrics since last update. Plot at:\t%s\n\n", outputFilePath)
	pb.openBrowserOnce(outputFilePath)
	return
}
// ... run the whole vizb.go pipeline (§2) ...
if pb.loopPeriod > 0 {
	injectAutoRefresh(outputFilePath, pb.loopPeriod) // <meta http-equiv="refresh">
}
pb.openBrowserOnce(outputFilePath)
```

`resolveOutputFilePath` picks a random temp path *once* and caches it:
```go
func (pb *PlotBuilder) resolveOutputFilePath(checkpointPaths []string) string {
	outputFilePath := pb.outputOverride
	if outputFilePath != "" {
		// -plot_output given explicitly: resolve relative to the first checkpoint path.
		return outputFilePath
	}
	if pb.cachedOutputPath == "" {
		tmpFile, _ := os.CreateTemp("", "gomlx-plots-*.html")
		pb.cachedOutputPath = tmpFile.Name()
	}
	return pb.cachedOutputPath
}
```
Deliberately *not* a deterministic path derived from the checkpoint directory — an earlier version
did exactly that, and the maintainer caught a real problem with it: two unrelated, concurrent
invocations of the tool against the *same* checkpoint directory would collide on the same computed
filename. Random-once-per-process avoids that entirely, and turned out simpler besides.

`injectAutoRefresh` is the first of three places this codebase edits vizb's *generated* HTML after
the fact, rather than trying to configure vizb into producing it directly — a pattern that recurs
in §5:
```go
func injectAutoRefresh(htmlPath string, period time.Duration) error {
	content, _ := os.ReadFile(htmlPath)
	seconds := max(1, int(period.Seconds()))
	metaTag := fmt.Sprintf(`<meta http-equiv="refresh" content="%d">`, seconds)
	updated := bytes.Replace(content, []byte("<head>"), []byte("<head>"+metaTag), 1)
	if bytes.Equal(updated, content) {
		return errors.Errorf("could not find a <head> tag to inject auto-refresh into")
	}
	return os.WriteFile(htmlPath, updated, 0644)
}
```
An exact-string `bytes.Replace` on `<head>` — no HTML parser, no dependency on anything about
vizb's *internal* structure, only that `<head>` (a basic, universal tag) exists somewhere in its
output. This exact-anchor-string approach is the template every later injection function follows.

## 4. From points to lines, and the label-collision bug (`plots.go`)

```go
type plotLineInfo struct {
	metricType             string
	short, coreShort, desc string
	modelIdx               int
	steps, values          []float64
}
```
`short` becomes the series label handed to `vizb` (§2), model-prefixed with `modelIndexMarker` in
multi-model mode (`❶`/`❷`, not `#1`/`#2` — round-2 feedback, since `#` already means something else
in these metric names, "average," vs `~` for "moving average"). `coreShort` is the *same* label but
*never* model-prefixed, added later specifically to let the sidebar collapse "❶ T/~loss" and
"❷ T/~loss" into one glossary entry (§5), since they describe the same underlying metric. `desc` is
the long-form description; per round-2 feedback it no longer repeats which model it's for (the
"Models compared" box already establishes that). `metricType` (added for §5's per-chart grouping)
records which chart this line belongs to. `modelIdx` records which compared model produced it.

`createPlotLines` groups a metric type's raw `[]plot.Point` into one `plotLineInfo` per (model,
metric name):
```go
info, exists := metricPoints[pt.MetricName]
if !exists {
	info = &plotLineInfo{metricType: metricType, modelIdx: modelIdx, coreShort: pt.Short, desc: pt.MetricName}
	if len(modelNames) == 1 {
		info.short = pt.Short
	} else {
		info.short = fmt.Sprintf("%s %s", modelIndexMarker(modelNum), pt.Short)
	}
}
```

**The bug, found using real maintainer-provided data, not guessed**: `vizb`'s chart renderer groups
series by `short` — its `"type"` field in the DataSet JSON, documented as each series' identity
key. Dumping the intermediate JSON for the real fixture showed duplicate `"type"` values within a
single dataset. Confirmed with a minimal isolated repro directly against `vizb line` (two
`--select` columns given the identical `{label}` → the duplicate `"type"` is preserved as-is in the
output, `vizb` doesn't dedupe it for you). Traced upstream: `pt.Short` (built in `ui/plot/plot.go`,
outside this codebase's boundary — see §1) appends a per-dataset suffix that came out *empty* for
this fixture's eval datasets, so "Mean Loss on train" and "Mean Loss on validation" both reduced to
the identical `"#loss()"`.

**The fix**, deliberately scoped entirely inside `cmd/gomlx_checkpoints` rather than "properly"
fixed upstream in `ui/plot` — that's separate PR territory, and this is the only place uniqueness
of the label actually matters. Current version, after a round-2 usability finding (see below):
```go
func disambiguationSuffix(desc string) string {
	const marker = " on "
	idx := strings.LastIndex(desc, marker)
	if idx < 0 {
		return ""
	}
	dataset := strings.TrimSpace(desc[idx+len(marker):])
	if dataset == "" {
		return ""
	}
	return fmt.Sprintf(" (%s)", dataset)
}

func disambiguateShortLabels(lines []*plotLineInfo) {
	slices.SortFunc(lines, func(a, b *plotLineInfo) int { return strings.Compare(a.desc, b.desc) })

	counts := make(map[string]int, len(lines))
	for _, line := range lines {
		counts[line.short]++
	}
	seen := make(map[string]int, len(lines))
	usedSuffixes := make(map[string]map[string]bool, len(lines))
	for _, line := range lines {
		base := line.short
		if counts[base] <= 1 {
			continue // No collision: leave the common case untouched.
		}
		suffix := disambiguationSuffix(line.desc)
		if suffix == "" || usedSuffixes[base][suffix] {
			seen[base]++
			suffix = fmt.Sprintf(" (%d)", seen[base])
		}
		if usedSuffixes[base] == nil {
			usedSuffixes[base] = make(map[string]bool)
		}
		usedSuffixes[base][suffix] = true
		line.short += suffix
		line.coreShort += suffix
	}
}
```
Called once per metric type, right after `createPlotLines`, before the lines are handed to
`runVizbLine`. This single fix explained *both* of the maintainer's originally reported bugs
(same-metric-type lines not showing together; multi-checkpoint comparison "not handled correctly")
— the multi-checkpoint case had the identical collisions underneath, just compounded by more lines
competing for the same short labels once two models' worth of metrics are combined.

**Two things were added after the original fix, both found by scrutiny, not by a new bug report:**

1. **A real determinism bug**, found while designing cross-model glossary dedup (§5), before it ever
   manifested as an observed failure: `createPlotLines` builds `lines` by ranging over a Go map,
   whose iteration order is randomized per process. Without a deterministic sort first, two models'
   independent train/validation collisions on the same underlying metric could get mismatched
   suffixes purely by chance of map order — silently breaking the dedup, which merges by exact
   `(coreShort, desc)` match. Fixed by sorting `lines` by `desc` before disambiguating, so the same
   metric always gets the same suffix regardless of which model produced it. Locked in by
   `TestDisambiguateShortLabels_ConsistentAcrossModels`, using deliberately shuffled input order.
2. **A labeling-clarity fix**: the original bare `" (%d)"` counter produced labels like
   `"#loss() (1)"` / `"#loss() (2)"` — which, sitting next to the `❶`/`❷` model markers introduced
   for round-2 feedback, read too easily as *per-model* duplication rather than what it actually is
   (train vs. validation disambiguation *within* one model). Traced the actual upstream format —
   `ui/plot/plot.go`'s `AddTrainAndEvalMetrics` builds eval descriptions as `"<name> on <dataset>"`
   (e.g. `"Mean Loss on train"`) — and `disambiguationSuffix` now extracts that dataset name
   directly, giving `"#loss() (train)"` / `"#loss() (validation)"` instead. Falls back to the old
   numeric counter only when `desc` doesn't fit that shape.

## 5. The sidebar: title, models compared, metric legend (`plots.go`)

Three small maintainer feature requests, all solved the same way as `injectAutoRefresh` (§3):
editing vizb's generated HTML after the fact, anchored only to exact, stable strings.

The sidebar was redesigned substantially after the version described in earlier drafts of this
doc — first for visual polish (applying the repo's `dataviz` skill), then twice more for real
usability problems found by actually looking at it against real data. Current shape:

```go
type legendEntry struct {
	term, detail string
}

type legendSection struct {
	heading string
	entries []legendEntry
}
```
`term`/`detail` replaced one pre-concatenated `text` string — `term` is the short, scannable label
(model name, or a metric's short code), `detail` is the longer text underneath (checkpoint path, or
the metric's full description). `legendSection` groups entries under one heading — used for
"Metric labels — loss", "Metric labels — img_loss", etc. (see `buildDescriptionEntries` below).
Neither struct carries `modelIdx` anymore — an earlier version did, to select a swatch color, but
that concept was removed once color stopped meaning "which model" (see below).

**The layout problem, and why it's a sidebar, not stacked boxes**: an early version put the
models-list and metric-descriptions as full-width boxes stacked above the charts — with ~25 real
entries, that pushed the actual charts below the fold. Fixed as a fixed-position, independently
scrolling column, styled purely through `#gomlx-sidebar` and a margin on `#app`:
```go
const pageSidebarCSS = `
#gomlx-sidebar{
--gomlx-surface:#fcfcfb;--gomlx-text:#0b0b0b;--gomlx-text-secondary:#52514e;--gomlx-text-muted:#898781;
--gomlx-series-1:#2a78d6;--gomlx-series-2:#eb6834;--gomlx-series-3:#1baf7a;--gomlx-series-4:#eda100;
--gomlx-series-5:#e87ba4;--gomlx-series-6:#008300;--gomlx-series-7:#4a3aa7;--gomlx-series-8:#e34948;
position:fixed;top:0;left:0;width:300px;height:100vh;overflow-y:auto; ...}
#app{margin-left:300px}
@media (max-width:900px){ #gomlx-sidebar{position:static;...} #app{margin-left:0} }
@media (prefers-color-scheme:dark){ #gomlx-sidebar{ --gomlx-series-1:#3987e5; ... } }
`
```
The key insight making this safe: `#app` is styled purely by its own stable `id` attribute — never
by anything about its *internal* structure. Confirmed by inspecting the raw generated HTML: only
two `id` attributes exist statically anywhere in it, `gomlx-sidebar` (ours) and `app` (vizb's) —
every chart, series, and legend is built by vizb's own JS *after* the page loads, invisible in what
this tool generates. That's a load-bearing fact for every injection function in this file: never
assume anything about vizb's DOM beyond `<head>`, `<body>`, and `id="app"` existing.

**Design system, not ad hoc styling**: asked to make the sidebar "clean, standard, industry" —
applied the repo's `dataviz` skill (pick the form, assign color by the job it does, *validate* the
palette with a script rather than eyeballing it) instead of picking colors and fonts by feel. Term
rows use `<dl>`/`<dt>`/`<dd>` (the semantically correct element for label→description pairs, not
`<li>` bullets); metric short-codes render in `<code>` since they're literal identifiers, not
prose; colors are CSS custom properties that flip automatically under `prefers-color-scheme: dark`.
The categorical palette itself was actually run through the skill's `validate_palette.js` against
both a light and dark sidebar surface — not assumed to pass just because it's the documented
reference palette.

```go
var swatchPaletteHex = []string{
	"#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
}

func swatchColor(rowIdx int) string  { return fmt.Sprintf("var(--gomlx-series-%d)", rowIdx%8+1) }
func swatchColorHex(rowIdx int) string { return swatchPaletteHex[rowIdx%len(swatchPaletteHex)] }
```
**Color assignment went through two full redesigns.** First: one color per compared *model*
(matching each model's swatch across all its sidebar entries) — explicitly *not* attempting to
match vizb's own chart-line colors, believed impossible at the time (vizb has a live in-browser
theme dropdown, and its colors seemed internally assigned with no way to influence from outside).

Second, after being asked directly whether the sidebar and chart colors could actually match: that
belief was wrong, and re-checking (rather than repeating the earlier answer) found `vizb line
--theme` accepts a bare hex-list palette, embedded into the DataSet JSON and applied in series
order (§2). Given a real design tension this created — the chart plots one line *per model per
metric*, but the sidebar already collapses that to one glossary entry, so a single swatch can't
match two differently-colored chart lines — the explicit choice made was: force the same color
per *metric* across models (accepting that two models' lines for a shared metric become the same
chart color, relying on the `❶`/`❷` marker text to tell them apart instead of color). Color now
means "which metric" everywhere on the page, consistently, matching the chart exactly:
```go
// inside Plot, per metric type, before calling runVizbLine:
colorByCoreShort := make(map[string]string, len(lines))
for i, entry := range buildMetricTypeSection(lines) {
	colorByCoreShort[entry.term] = swatchColorHex(i)
}
lineColors := make([]string, len(lines))
for i, line := range lines {
	lineColors[i] = colorByCoreShort[line.coreShort]
}
jsonPath, err := runVizbLine(tmpDir, metricType, lines, lineColors)
```
Verified against the real fixture end-to-end, not just unit-tested: parsed the actual embedded
`themes` JSON from the rendered HTML and the actual sidebar CSS, cross-checked all 19 series —
zero mismatches.

**Then a second usability issue surfaced from the first**: reusing this same palette for "Models
compared" swatches meant a model's dot could coincidentally match an unrelated metric's color
elsewhere on the page, implying a connection that doesn't exist (since color now consistently means
"which metric," not "which model," everywhere else). Fixed by dropping the swatch from "Models
compared" entirely — the `❶`/`❷` marker already uniquely identifies each model, so the color was
never load-bearing there, only decorative, and had become actively misleading once color's meaning
changed everywhere else.

```go
func buildMetricTypeSection(lines []*plotLineInfo) []legendEntry {
	// dedupes by (coreShort, desc) within one metric type, sorted by term then detail --
	// factored out of buildDescriptionEntries so Plot can compute the exact same, deterministically
	// ordered list *before* calling runVizbLine, since that order assigns swatch colors.
}

func buildDescriptionEntries(allLines []*plotLineInfo) []legendSection {
	// groups allLines by metricType (added to plotLineInfo for exactly this), one legendSection
	// per type, in first-encountered order -- which Plot always populates in
	// createSortedMetricTypes' order, the same order the charts themselves render in.
}
```
**Per-metric-type grouping** (a maintainer request: "only display labels according to the current
[chart]") — the live-DOM-sync version of this (detecting which chart panel the viewer currently has
selected) wasn't pursued: unlike this file's other DOM investigations, it would depend on
understanding vizb's internal, closed rendering state, the same category of risk that killed the
hover-tooltip attempt (§6). The chosen alternative needs no vizb-side cooperation at all: split
"Metric labels" into one section per metric type ("Metric labels — loss", "Metric labels —
img_loss"), so a reader looking at one chart isn't scrolling past every other chart's labels, using
data this codebase already fully owns.

```go
func buildPageSidebarHTML(title string, modelEntries []legendEntry, descriptionSections []legendSection) string {
	// ... writeList("Models compared", modelEntries, false, true, false)   // no swatch
	//     for each section: writeList("Metric labels — "+heading, entries, true, false, true)  // swatch
}

func injectPageExtras(htmlPath, title string, modelEntries []legendEntry, descriptionSections []legendSection) error {
	content, _ := os.ReadFile(htmlPath)
	if title != "" {
		// Best-effort: <title>Vizb</title> → <title>{title}</title>. If vizb's default text
		// ever changes, just skip this -- the visible <h1> in the sidebar is what matters most.
		content = bytes.Replace(content, []byte("<title>Vizb</title>"), []byte("<title>"+html.EscapeString(title)+"</title>"), 1)
	}
	sidebar := buildPageSidebarHTML(title, modelEntries, descriptionSections)
	if sidebar == "" {
		return os.WriteFile(htmlPath, content, 0644)
	}
	withCSS := bytes.Replace(content, []byte("<head>"), []byte("<head><style>"+pageSidebarCSS+"</style>"), 1)
	updated := bytes.Replace(withCSS, []byte("<body>"), []byte("<body>"+sidebar), 1)
	return os.WriteFile(htmlPath, updated, 0644)
}
```
The sidebar `<div>` lands right after `<body>`, *before* vizb's `<div id="app">` — since vizb's JS
only ever touches `#app` itself, content placed as `#app`'s preceding sibling survives untouched
once the page hydrates. This part is unchanged since the very first sidebar version.

## 6. What was tried and reverted: the hover-tooltip script

Worth documenting in full, code included, precisely *because* it was removed — the investigation
that killed it is exactly the kind of thing worth not re-discovering the hard way.

**The ask**: a maintainer follow-up wondering whether a tooltip showing each metric's *full* name
(not just its short legend label) could be added on hover, specifically by post-processing the
generated HTML ourselves (not asking vizb to build it).

**What got built** — a `<script>` mapping every known short label to its full description, using a
`MutationObserver` (since vizb's legend doesn't exist in the DOM until its own JS finishes) to find
matching text and set the native `title` attribute:
```go
func buildLegendTooltipScript(descriptionEntries []legendEntry) string {
	tooltips := make(map[string]string, len(descriptionEntries))
	for _, e := range descriptionEntries {
		if e.short != "" {
			tooltips[e.short] = e.desc
		}
	}
	if len(tooltips) == 0 {
		return ""
	}
	data, _ := json.Marshal(tooltips) // HTML-escapes '<'/'>'/'&' by default -- safe to embed as-is.

	return `<script>(function(){
var tooltips = ` + string(data) + `;
var tagged = new WeakSet();
function tag(el) {
  if (!el || tagged.has(el) || el.title) return;
  var text = el.textContent.trim();
  if (Object.prototype.hasOwnProperty.call(tooltips, text)) {
    el.title = tooltips[text];
    tagged.add(el);
  }
}
function scanTextNodes(root) {
  var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  var node;
  while ((node = walker.nextNode())) { tag(node.parentElement); }
}
scanTextNodes(document.body);
new MutationObserver(function(mutations) {
  mutations.forEach(function(m) {
    m.addedNodes.forEach(function(n) {
      if (n.nodeType === 1) scanTextNodes(n);
      else if (n.nodeType === 3) tag(n.parentElement);
    });
  });
}).observe(document.body, {childList: true, subtree: true});
})();</script>`
}
```
Exact text match (`===`, not a substring check) was deliberate from the start — an early
*investigation-only* probe script used a looser substring check and immediately produced a false
positive: `"#loss() (1)"` is a literal substring of `"#2 #loss() (1)"`, a different line's label
entirely. Exact match also happens to guarantee this script could never tag the sidebar's *own*
list entries (their text is always `"short: full description"` together, never equal to just the
short label alone).

**Why it doesn't exist anymore**: live testing (devtools, a real browser, real data) showed the
tagging logic genuinely worked — the right `<span>` elements ended up with the right `title`
attributes, confirmed directly by inspecting `el.title` in the console. The tooltip still never
appeared on hover. Right-clicking → "Inspect" directly on the visible legend text (not the
console — the browser's own element inspector) revealed why: vizb renders its legend onto a
`<canvas>`, not real interactive DOM. A native tooltip only fires when the browser hit-tests an
actual hovered *element* — canvas pixels are never that, no matter how correctly a `title`
attribute is set on some other, parallel (likely accessibility-tree) DOM node that happens to
contain matching text. Checked for an escape hatch before giving up: no `--renderer`/`svg`/`canvas`
flag exists anywhere in `vizb`'s CLI (`vizb ui --help`, `vizb line --help`, `vizb --help`, checked
directly) — no way to force a DOM-based legend instead of canvas from outside.

Shipping this anyway would mean every generated HTML file carries a page-wide `MutationObserver`
that silently tags elements no user can ever actually hover — real complexity for zero real
benefit. Reverted, with the finding preserved as a comment on `injectPageExtras` instead of in
removed code nobody will read again. The sidebar's "Metric labels" list remains the reliable,
already-working answer to "see the full name" — exactly the fallback the maintainer had offered as
acceptable from the start.

## 7. Testing strategy, and why each test lives where it does

- **Pure-logic unit tests**, no `vizb` binary needed: `writeLinesAsCSV` (including a specific
  `TestWriteLinesAsCSV_MisalignedSteps` case), `disambiguateShortLabels` (collision and
  no-collision cases), the HTML-injection functions (`injectAutoRefresh`,
  `injectPageExtras`/`buildPageSidebarHTML`), `resolveOutputFilePath`, `latestMetricsModTime`.
- **Real-subprocess integration tests**, calling the actual `vizb` binary: `TestRunVizbLine_*`,
  `TestMergeAndRenderVizb_*`. These `t.Skip()` gracefully if `vizb` isn't installed — mirroring the
  tool's own soft-dependency design in the tests themselves, not just the production code.
  `TestRunVizbLine_SeriesOnYAxis` and `TestRunVizbLine_MisalignedStepsOmitMissingSeries`
  specifically lock in the group-mode switch (§2) against the real binary, not just against
  `writeLinesAsCSV` in isolation.
- **Characterization tests written first**, against the *old* code, before a refactor (the
  `PlotBuilder` extraction, §3) — locking in existing behavior so the refactor could be verified
  behavior-preserving rather than just "looks right."
- **Real acceptance testing throughout**, beyond any unit test: live MNIST training runs through
  the actual `train.Loop` wiring; multi-iteration `-loop` runs against an actively-training job;
  and, for the bug fix and the group-mode switch specifically, the maintainer's own real
  FlowMatching fixture data — inspecting the actual intermediate JSON `vizb` produces to confirm
  zero duplicate labels and correct blank-cell handling, not just trusting the code "looks
  correct." The hover-tooltip investigation (§6) was verified the same way, in a real browser with
  devtools, right up to the point that verification was *what proved it couldn't work*.
- **Added since**: `TestRunVizbLine_PreservesNarrowRangePrecision` (locks in the value-rounding
  fix, using the exact values that first exposed the bug); `TestRunVizbLine_AppliesLineColorsAsTheme`
  (locks in the `--theme` wiring by parsing the real embedded `themes` JSON, not just checking the
  call succeeded); `TestDisambiguateShortLabels_ConsistentAcrossModels` (the determinism
  regression test, §4); `TestBuildDescriptionEntries_GroupsByMetricType` and
  `TestBuildPageSidebarHTML_GroupsDescriptionsByMetricType` (per-chart sidebar sections, §5);
  `TestBuildPageSidebarHTML_ModelsHaveNoSwatch` (locks in the swatch-collision fix, §5); and
  `TestPlot_SidebarReflectsCurrentMetricsAcrossRepeatedCalls`, a `Plot()`-level integration test —
  the first test in this file exercising the whole pipeline rather than one function in isolation —
  calling the same `PlotBuilder` twice (mirroring real `-loop` usage) with a new metric type
  appearing between calls, confirming the sidebar genuinely regenerates rather than staying stale.
  Verified this test could actually fail (not a tautology) by temporarily disabling its mtime bump
  and confirming it caught `Plot`'s own caching guard skipping the second render.

## 8. Two upstream findings investigated the same way, neither blocking a merge

Two more round-2 items (§10 of the narrative doc) turned out to be genuine vizb limitations, not
anything fixable in this codebase — found by the same discipline as everything above: verify
hands-on, don't assume, don't guess a diagnosis without evidence.

**Y-axis log-scale toggle silently no-ops on narrow-value-range data.** Isolated via direct
comparison (a ~59×-range repro reshapes correctly when toggled; a ~2.5×-range repro shows the
toggle as selected but the chart never changes) — filed as
[goptics/vizb#345](https://github.com/goptics/vizb/issues/345).

**X-axis is always categorical, even for purely numeric data**, which also means log-scale on X is
structurally impossible (a categorical axis holds only ordering, no numbers to take a logarithm
of) — not a bug, a vizb implementation choice, confirmed by building a real, minimal, standalone
ECharts page (outside vizb entirely) with the same data rendered both as `xAxis.type: "category"`
and `xAxis.type: "log"`, and screenshotting the result: the log version correctly space-by-ratio
positions `1, 10, 100, 1,000, 10,000, 100,000`. Proves the underlying library supports it, so the
gap is specifically in how vizb's line chart uses it. A short feature-request draft exists,
not yet filed.

Both are tracked as **follow-up work, not blockers**: "We could merge your PR as is for now, and do
another PR for when vizb fixes [the Y-axis bug] (and maybe add support for [continuous X-axis])."
