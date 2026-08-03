# `gomlx_checkpoints`

A command-line inspector for GoMLX checkpoint directories: model size, hyperparameters, variable
shapes, and training metrics (as tables or plots) — all read from files already written by your
training program. It doesn't train anything itself.

## Install

```bash
go install github.com/gomlx/gomlx/cmd/gomlx_checkpoints@latest
```

## Basic usage

```bash
gomlx_checkpoints [flags...] <checkpoint_path> [<checkpoint_path2> ...]
```

`<checkpoint_path>` is a directory a GoMLX model was checkpointed to (see
[`ml/model/checkpoint`](../../ml/model/checkpoint)). Some flags accept more than one path, to compare
models side by side.

```bash
# Model size, hyperparameters, and variable shapes:
gomlx_checkpoints -summary -params -vars ~/my_model/checkpoint

# Everything at once:
gomlx_checkpoints -all ~/my_model/checkpoint

# Compare two checkpoints:
gomlx_checkpoints -summary -params ~/model_a/checkpoint ~/model_b/checkpoint
```

## Plotting metrics (`--plot`)

If your training program records metrics (via `ui/gonb/plotly` or `ui/gonb/margaid` — see
[`ui/plot`](../../ui/plot)), `training_plot_points.json` lands in the checkpoint directory. Two ways to
look at it:

```bash
# As a table in the terminal -- no extra tools needed:
gomlx_checkpoints -metrics ~/my_model/checkpoint

# As an interactive chart in your browser:
gomlx_checkpoints -plot ~/my_model/checkpoint
```

`--plot` needs the [`vizb`](https://vizb.goptics.org/) CLI tool installed and on your `PATH` — it's not
a Go dependency of `gomlx_checkpoints` at all, just an external program this flag shells out to. Every
other flag works with or without it. If it's missing, `--plot` fails with an error that includes the
exact install command:

```bash
go install github.com/goptics/vizb@latest
```

One HTML file is generated (a temp file by default, or `-plot_output <path>` for a fixed location), with
one chart panel per metric type (e.g. "loss", "accuracy") and one line per metric within it (e.g. train
vs. eval) — all metrics of the same type share a panel, so you can see e.g. train and validation loss
diverge on the same chart. It opens in your default browser unless `-browser=false`.

Pass more than one `<checkpoint_path>` to compare models side by side: each metric line is prefixed with
`#1`, `#2`, ... and a persistent sidebar on the left lists the models being compared and every metric's
short label (e.g. `T/~loss`) next to its full name (e.g. "Train: Moving Average Loss") — each entry gets
a color-coded dot matching its model, so it's easy to tell at a glance which sidebar entries belong to
which run. Add `-plot_title "<text>"` to title the page — handy when sharing the generated HTML file
with others.

Narrow down what gets plotted (or listed with `-metrics`) with `-metrics_names <regex>` (matches metric
name or short name) and `-metrics_types <type1,type2,...>`.

## Watching a live training run (`-loop`)

`-loop <duration>` re-runs the report on a timer — pair it with `-metrics` for a live-updating terminal
table, or with `-plot` for a live-updating browser tab:

```bash
gomlx_checkpoints -plot -loop=30s ~/my_model/checkpoint
```

This opens **one** browser tab, which keeps itself current automatically (it's the same file each time,
with an auto-refresh tag matching your `-loop` period) — it does not open a new tab on every tick, and it
skips re-generating the plot entirely on ticks where nothing new has actually been recorded, so it's
cheap to leave running for the duration of a training job.

## Other flags

- `-scope <path>` (default `/`): restrict `--summary`/`--vars` to a sub-scope of the model — useful to
  exclude optimizer state or other support variables you don't care about.
- `-backup`: copy the most recent checkpoint into a `backup/` subdirectory.
- `-delete_vars <scope1,scope2,...>`: delete variables under the given scope(s) — e.g. to strip
  optimizer state before distributing a model for inference.
- `-perturb <x>`: multiply trainable float variables by `1.0+(RandomUniform(-1,1)*x)` — useful for
  testing checkpoint-loading/recovery code. Pair with `-delete_vars` to also clear any stale optimizer
  moving averages.

Run `gomlx_checkpoints -help` for the full, current flag reference.
