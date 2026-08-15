# Phase 0 explained: extracting `ui/plot.Config`

This is a learning doc, not part of the PR — kept untracked on `dev/plot-config-refactor`,
same as the other personal notes from Phase 1.

## 1. The problem, from first principles

`ui/gonb/plotly.PlotConfig` and `ui/gonb/margaid.Plots` each do two jobs that have nothing to
do with each other:

1. **Bookkeeping**: which datasets to evaluate, when to sample the training loop, whether a
   point is valid, how many points have been collected, and — optionally — persisting every
   point to `training_plot_points.json` on disk so a separate process (`gomlx_checkpoints
   --plot`) can read it back later.
2. **Rendering**: turning those points into a `go-plotly` figure (`plotly`) or an SVG built with
   the Margaid library (`margaid`), and pushing that into a Jupyter/GoNB notebook cell.

Job 1 is identical in *shape* between the two packages — both need "add this point, unless it's
NaN/Inf" and "call this every N steps" — but it was implemented twice, by hand, with small
accidental differences (one had a guard against double-registering a callback, the other didn't).
Job 2 is genuinely different between the two (plotly figures vs. Margaid SVG) and has to stay
that way.

The maintainer's ask: pull job 1 out into one shared type, `ui/plot.Config`, and have both
`PlotConfig` and `Plots` become thin wrappers around it.

## 2. Why "just embed it" doesn't work

The obvious Go move is struct embedding:

```go
type PlotConfig struct {
    plot.Config // embed it, get all its methods "for free"
    figs []*grob.Fig
    // ...
}
```

Two things break immediately.

**Chaining breaks.** `plot.Config.WithDatasets(...)` has to return `*plot.Config` — it doesn't
know it's embedded inside something else. So:

```go
plotly.New().WithDatasets(ds).Dynamic()
//            ^ this now returns *plot.Config, which has no Dynamic() method
```

The whole point of `PlotConfig`'s API is the fluent chain (`New().WithCheckpoint(...).Dynamic()
.WithDatasets(...).ScheduleExponential(...)`), used exactly this way in 7 real training scripts
in this repo. Naive embedding silently breaks that the moment a Config-owned method sits in the
middle of a chain.

**Virtual dispatch doesn't exist.** Suppose `Config` needs to call `AddPoint` as part of loading
points from a file (it does — see §4). If `Config.AddPoint` existed, it could only ever run
`Config`'s own logic — it has no way to know it's embedded inside a `PlotConfig` and "reach
upward" into `PlotConfig.AddPoint`'s figure-building logic. Go has no virtual methods. Embedding
gives you delegation, not inheritance.

## 3. The fix: an explicit callback + selective overriding

**For virtual dispatch**: `Config` doesn't try to do rendering-shaped work itself. Whenever it
needs to (e.g. "a point was loaded from a file, please render it"), it calls back into an
`owner Plotter` that was handed to it at construction time:

```go
type Config struct {
    owner Plotter // = the *PlotConfig or *Plots that embeds this Config
    // ...
}

func NewConfig(owner Plotter, onEndName string, finalize func()) *Config {
    return &Config{owner: owner, onEndName: onEndName, finalize: finalize}
}
```

`Plotter` is just `AddPoint(Point)` + `DynamicSampleDone(bool)` — it already existed in
`ui/plot/plot.go` before this refactor, used the same way by the pre-existing
`AddTrainAndEvalMetrics` helper. Nothing new conceptually, just reused for a second purpose.

Each `New()` constructor wires the self-reference *after* allocating itself, so the pointer is
valid:

```go
func New() *PlotConfig {
    pc := &PlotConfig{metricsTypesToFig: make(map[string]int)}
    pc.Config = plot.NewConfig(pc, "plotly.DynamicPlot", func() {
        if pc.gonbId != "" && !pc.finalPlot {
            pc.DynamicPlot(true)
            pc.finalPlot = true
        }
    })
    return pc
}
```

`pc` closes over itself in that `finalize` closure — when `Config` decides "training's done, time
for the final render," it calls `finalize()`, which reaches back into `PlotConfig`'s own
`DynamicPlot`. That's how the "virtual call" actually happens: not through the type system, but
through an explicit function value handed over at construction.

**For chaining**: embed `*plot.Config` *anonymously* (not as a named field), which gets you two
things for free —

- Every exported **field** on `Config` (like `EvalDatasets`) is promoted onto `PlotConfig`
  automatically. This mattered concretely: two real call sites
  (`examples/oxfordflowers102/diffusion/train.go`, `examples/FlowMatching/train.go`) read
  `plotter.EvalDatasets` as a bare field, not through a method. Anonymous embedding is what keeps
  that compiling without any change on their end.
- Every exported **method** is also promoted — but only the ones that don't need to return
  `*PlotConfig` are left alone. Anywhere the original API returned `*PlotConfig` for chaining, a
  one-line override shadows the promoted version:

```go
func (pc *PlotConfig) WithDatasets(datasets ...train.Dataset) *PlotConfig {
    pc.Config.WithDatasets(datasets...) // delegate the actual work
    return pc                            // but return the outer type, so the chain keeps working
}
```

Go's rule is simple here: a method defined directly on the outer type always wins over a promoted
one with the same name — no ambiguity, no error, just a clean override. So `PlotConfig` ends up
with ~6 of these one-liners (`WithDatasets`, `WithBatchNormalizationAveragesUpdate`,
`WithCustomMetricFn`, the three `Schedule*` methods), and everything else — getters, the
promoted `EvalDatasets` field — needs no boilerplate at all.

## 4. Walking the actual data flow

**Adding a point** (`AddPoint`, called by both live training and by loading from a file):

```go
func (pc *PlotConfig) AddPoint(pt plot.Point) {
    if !pc.Config.FilterAndWrite(pt) { // NaN/Inf check + async write-to-file, if a writer is open
        return
    }
    // ... build/extend the go-plotly figure and trace — this part is 100% plotly-specific
}
```

`FilterAndWrite` is the one piece of logic that was byte-for-byte identical in both packages
before this refactor (NaN/Inf guard, then `if fileWriter != nil { fileWriter <- pt }`) — an actual
duplication, now genuinely shared.

**Scheduling collection against a training loop** (`ScheduleExponential` etc.):

```go
func (c *Config) ScheduleExponential(loop *train.Loop, startStep int, stepFactor float64, name string) *Config {
    train.ExponentialCallback(loop, startStep, stepFactor, true, name, 0, c.addMetrics)
    c.attachOnEnd(loop)
    return c
}
```

`c.addMetrics` dedups by `loop.LoopStep` (in case scheduling was set up two different ways),
optionally runs a custom-metric hook, then calls the *existing* `AddTrainAndEvalMetrics(c.owner,
...)` — which itself calls `c.owner.AddPoint(...)` per metric and `c.owner.DynamicSampleDone(...)`
at the end. This is where the callback-to-owner pattern actually pays for itself: the collection
loop is 100% generic, but every point it produces still goes through the *owner's* rendering.

`attachOnEnd` registers `loop.OnEnd` exactly once (guarded by `scheduledFinalPlot`), no matter how
many `Schedule*` methods get called — it runs `finalize()` (the owner's closure from §3), then
closes the file writer.

**Loading points back from disk** (`LoadCheckpointData` / `PreloadFile`): this one deliberately
did **not** get unified. `plotly.LoadCheckpointData` counts "points added" as the number of
*distinct steps* seen while looping over the file; `margaid.PreloadFile` instead recomputes it
after the fact as the *minimum series length* across all loaded metrics. These aren't the same
number in general (staggered eval/train frequencies), and unifying them wasn't something the
maintainer asked for — so each package still owns its own counting loop, and `Config` just
exposes two low-level primitives (`IncrementPointsAdded()`, `SetPointsAddedIfGreater(n)`) for them
to drive the *same underlying counter* their own way.

## 5. Was this "standard"?

Yes, for Go specifically. Go has no class inheritance, no covariant return types, and no way to
override a promoted method's return type — so "shared base + explicit forwarding methods for
whichever calls need to keep the derived type in their signature" is the idiomatic way this
problem gets solved in Go codebases (it shows up under names like "type embedding with method
overriding" or people just call it "the promotion problem"). The alternatives are worse fits here:

- **Code generation** (`go:generate` a wrapper): more machinery than 6 one-line methods justify.
- **A generic self-type parameter** (`Config[Self Plotter]`): technically possible in modern Go,
  but it would infect every signature that touches `Config` with a type parameter, for a benefit
  (saving ~20 lines total) that doesn't clear the complexity bar.
- **Just leaving the duplication**: what existed before — the thing being fixed.

So: standard for the language, not a bespoke trick.

## 6. Was this efficient?

Runtime cost is negligible and was never really the design constraint here (this is training
infrastructure that runs at most every few hundred training steps, not a hot loop):

- One extra pointer indirection per call (`pc.Config.WithDatasets` vs `pc.WithDatasets`) — the
  forwarding methods are small enough the Go compiler can usually inline them.
- `owner Plotter` is a plain interface value (two words), not reflection — the "virtual dispatch"
  is a direct interface method call, same cost as any other Go interface call.
- No new allocations beyond what already existed: `Config` replaces fields that used to live
  directly on `PlotConfig`/`Plots`, it doesn't add a second copy of them.

Where it's *not* maximally DRY, on purpose: `plotly` and `margaid` still each carry their own
`PointsAdded` bookkeeping convention (§4) and their own literal names passed into
`train.Loop.OnEnd` (`"plotly.DynamicPlot"` vs `"margaid plots"` vs `"margaid.Plot"` — the latter
two were already inconsistent with each other *before* this refactor). Both of those were
conscious calls to preserve exact current behavior rather than silently "improving" it as a side
effect of a refactor that was explicitly scoped as behavior-preserving. If you wanted to spend
more here, the next real efficiency/consistency gain wouldn't be in `Config` — it'd be picking one
`PointsAdded` heuristic and one `OnEnd` naming convention and applying it everywhere, which is a
deliberate behavior change, not a refactor, and wasn't asked for.
