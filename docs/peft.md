# Parameter-efficient fine-tuning (PEFT)

Package [`ml/layers/peft`](../ml/layers/peft) provides native LoRA and NF4
QLoRA projections for GoMLX. A PEFT projection is built directly in the model
graph: its base, A, and B variables all belong to the supplied `model.Scope`
and `model.Store`.

## LoRA

LoRA adds a low-rank update to a frozen dense projection:

```text
output = base(input) + (alpha / rank) × input × A × B
```

Configure the projection scope before building the model graph. `peft_rank=0`
is the default and leaves the original projection unchanged; a positive rank
freezes this base weight and creates `lora/A` and `lora/B`. To fine-tune
adapters only, the architecture must likewise mark its other base variables
non-trainable.

```go
modelScope := store.RootScope().In("model")
modelScope.SetParams(map[string]any{
	peft.ParamRank:    16,
	peft.ParamAlpha:   float32(32),
	peft.ParamDropout: float32(0.05),
	peft.ParamBias:    peft.BiasNone,
})

// Inside the model graph function:
projectionScope := scope.In("model").In("q_proj")
weight := projectionScope.VariableWithShape("weight", shapes.Make(dtypes.Float32, inputDim, outputDim))
bias := projectionScope.VariableWithShape("bias", shapes.Make(dtypes.Float32, outputDim))
output := peft.New(projectionScope, input, weight, bias).Done()
```

`peft.New(scope, input, weight, bias).Done()` replaces the corresponding
`nn.Dense(input, weight.NodeValue(input), bias.NodeValue(input), ...)` call for
one projection. Use it only for the projections being fine-tuned, such as a
transformer's query and value projections. `WithWeightLayout` supports an
existing `[output, input]` base weight; `[input, output]` is the default.

`ParamAlpha` defaults to the rank and `ParamDropout` to zero. `BiasNone` is
the default; `BiasAll` and `BiasLoRAOnly` both make the bias trainable for this
directly constructed projection.

The adapter A initializer uses the model store RNG. Set `model.ParamInitialSeed`
on the store for reproducible initialization. B starts at zero.

Invalid graph-building configuration or projection shapes raise GoMLX
exceptions with a stack trace. NF4 quantization is preprocessing and therefore
returns regular Go errors.

## NF4 QLoRA

Quantize a base matrix outside the graph, then construct the projection inside
the graph with the same scope configuration:

```go
base, err := peft.QuantizeNF4Double(inputDim, outputDim, 64, 256, weights)
if err != nil {
	return err
}

// Inside the model graph function:
output := peft.NewNF4(scope, input, base, bias).Done()
```

`NewNF4` stores the packed NF4 weight and scale data as frozen variables under
`scope/qlora`, while its F32 A/B matrices live under `scope/qlora/lora`.
NF4 output widths must be even.

## Validation

```bash
go test ./ml/layers/peft
```
