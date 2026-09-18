# Parameter-efficient fine-tuning (PEFT)

Package [`ml/layers/peft`](../ml/layers/peft) adds native LoRA and NF4 QLoRA
layers to GoMLX. It provides frozen base projections, trainable low-rank
adapters, named adapter selection, and exact optimizer-variable sets.

## Features

- LoRA for existing GoMLX dense projections.
- NF4 QLoRA with optional double quantization of block scales.
- Exact or dot-separated-suffix target matching, such as `q_proj` matching
  `layers.0.q_proj`.
- Multiple named adapters per projection, sharing one frozen base.
- Runtime adapter selection: one adapter, multiple additive adapters, or no
  active adapter.
- Adapter-only trainable-variable selection for GoMLX optimizers.
- Adapter dropout active only in training graphs.

## LoRA

LoRA adds a low-rank update to a frozen projection:

```text
output = base(input) + (alpha / rank) × input × A × B
```

`A` has shape `[input, rank]` and `B` has shape `[rank, output]`. The base
weight is marked non-trainable; A and B are the trainable adapter variables.

Create a layer around a projection:

```go
config := peft.Config{
	Rank:          16,
	Alpha:         32,
	Dropout:       0.05,
	TargetModules: []string{"q_proj", "v_proj"},
	Bias:          peft.BiasNone,
	WeightLayout:   compute.DenseLayoutOutputsInput,
}

layer, err := peft.NewLinear(
	"layers.0.q_proj",
	projectionScope,
	projectionWeight,
	projectionBias,
	config,
	rand.New(rand.NewSource(seed)),
)
if err != nil {
	return err
}

output := layer.Apply(scope, hiddenStates)
```

`Config.WeightLayout` is `compute.DenseLayoutInputOutputs` by default for a
base weight shaped `[input, output]`. Use
`compute.DenseLayoutOutputsInput` for a base weight shaped `[output, input]`.

## Injecting adapters into a model

Architectures expose replaceable projections through `peft.Model`:

```go
type Model interface {
	LinearModules() ([]peft.Module, error)
	ReplaceLoRALinearModules([]peft.Replacement) error
}
```

`LinearModules` returns named projections and their scopes. `Inject` matches
the configured targets, creates each LoRA layer, freezes the base weights, and
passes replacements to the architecture.

```go
adapter, err := peft.Inject("support-v1", model, config, rng)
if err != nil {
	return err
}

trainable := adapter.TrainableVariables()
```

Use `trainable` when constructing the training graph or optimizer so that only
the adapter variables (and a configured bias, when applicable) are updated.

## Named adapters

A `Linear` can hold several adapters without copying the base weight. The
first injection replaces the projection. For later injections, the host returns
the installed layer in `Module.Layer`.

```go
support, err := peft.Inject("support", model, config, rng)
if err != nil {
	return err
}
billing, err := peft.Inject("billing", model, config, rng)
if err != nil {
	return err
}

layer.SetActiveAdapters("support")
layer.SetActiveAdapters("billing")
layer.SetActiveAdapters("support", "billing") // additive updates
layer.SetActiveAdapters()                      // base projection only

_ = support.TrainableVariables()
_ = billing.TrainableVariables()
```

Each returned `Adapter` exposes only the variables belonging to that adapter.

## NF4 QLoRA

QLoRA uses a packed NF4 base projection and F32 LoRA A/B matrices. Quantize a
base weight, create the QLoRA layer, then use `Apply` as the projection:

```go
base, err := peft.QuantizeNF4Double(
	inputFeatures, outputFeatures,
	64, 256, fullPrecisionWeights,
)
if err != nil {
	return err
}

config := peft.QLoRAConfig{
	LoRA: peft.Config{
		Rank:          16,
		Alpha:         32,
		Dropout:       0.05,
		TargetModules: []string{"q_proj", "v_proj"},
	},
	BlockSize:      64,
	DoubleQuant:    true,
	ScaleBlockSize: 256,
}

layer, err := peft.NewNF4Linear(
	"layers.0.q_proj", projectionScope, base, projectionBias, config, rng,
)
if err != nil {
	return err
}
output := layer.Apply(scope, hiddenStates)
```

`NF4Weight` uses `[input, output]` storage and an even output width. Use
`InjectNF4` with an `NF4Model` to apply the same target-based replacement flow
to an architecture with NF4 module definitions.

`NF4Linear` supports named adapters exactly like `Linear`: return an existing
layer in `NF4Module.Layer` for later injections, then select updates with
`SetActiveAdapters`.

## Bias modes

`Config.Bias` controls selected module bias variables:

| Value | Behavior |
| --- | --- |
| `peft.BiasNone` | Freeze the bias. |
| `peft.BiasAll` | Train the bias with the adapter. |
| `peft.BiasLoRAOnly` | Train the bias for selected PEFT projections. |

## Validation

Run the focused package suite:

```bash
go test ./ml/layers/peft
```

It covers LoRA and QLoRA forward paths, adapter-only gradients, named adapter
selection, target injection, and rollback on replacement failures.
