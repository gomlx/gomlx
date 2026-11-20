# GoMLX Training Loop, Checkpointing & PEFT/LoRA Reference

This reference covers model training with `ml/train`, checkpoint persistence with `ml/model/checkpoint`, and Parameter-Efficient Fine-Tuning (PEFT / LoRA) with `ml/layers/peft`.

---

## 1. Training Loop (`github.com/gomlx/gomlx/ml/train`)

- **`train.Trainer`**: Orchestrates the model function, loss function (`ml/train/loss`, e.g., `loss.BinaryCrossentropyLogits`, `loss.SparseCategoricalCrossEntropyLogits`, `loss.MeanSquaredError`), and optimizer (`ml/train/optimizer`, e.g., `optimizer.Adam`, `optimizer.AdamW`, `optimizer.StochasticGradientDescent`, or `optimizer.FromStore(store)`).
- **Metrics (`ml/train/metric`)**: Passed as two slices to `train.NewTrainer` — one for training metrics and one for evaluation metrics (e.g., `metric.NewMeanBinaryLogitsAccuracy`, `metric.NewMovingAverageBinaryLogitsAccuracy`, `metric.NewSparseCategoricalAccuracy`).
- **`train.Loop`**: Drives iteration over `train.Dataset`s, invokes the `Trainer`, and runs lifecycle hooks:
  - `loop.OnStart(name, priority, fn)`
  - `loop.OnFirstStep(name, priority, fn)` (auto-deregisters after the first step)
  - `loop.OnStep(name, priority, fn)`
  - `loop.OnEnd(name, priority, fn)`

### Example Training Pipeline

See `examples/adult/demo` for a full end-to-end example.

```go
// Create an empty store for variables and hyperparameters.
store := model.NewStore()
store.SetParam("learning_rate", *flagLearningRate)

// Optional: attach a checkpoint handler to save/restore weights and hyperparameters.
checkpointHandler, err := checkpoint.Build(store).
	Dir(*flagCheckpointDir).
	Keep(3).
	Done()
if err != nil {
	return err
}

// Create dataset.
trainDS := CreateDataset(...)

// Define training and evaluation metrics.
meanAccuracyMetric := metric.NewMeanBinaryLogitsAccuracy("Mean Accuracy", "#acc")
movingAccuracyMetric := metric.NewMovingAverageBinaryLogitsAccuracy("Moving Average Accuracy", "~acc", 0.01)

// Create a train.Trainer: orchestrates running the model, computing gradients, applying the optimizer, and tracking metrics.
trainer := train.NewTrainer(
	backend, store, Model, loss.BinaryCrossentropyLogits,
	optimizer.FromStore(store),
	[]metric.Interface{movingAccuracyMetric}, // trainMetrics
	[]metric.Interface{meanAccuracyMetric},   // evalMetrics
)

// Create a standard training loop and attach callbacks.
loop := train.NewLoop(trainer)
commandline.AttachProgressBar(loop)
loophelpers.PeriodicallyCall(loop, 100, func() error {
	return checkpointHandler.Save()
})

// Run training up to the target global step.
trainSteps := model.GetRootParamOr(store, "train_steps", 1000)
_, err = loop.RunToGlobalStep(trainDS, trainSteps)
if err != nil {
	return err
}
```

### Checkpointing (`github.com/gomlx/gomlx/ml/model/checkpoint`)
- Save and restore all variables and hyperparameters in a `model.Store` with `checkpoint.Build(store).Dir(dir).Done()`.
- Also supports loading from embedded files (`FromEmbed(jsonStr, binBytes)`) for single-binary inference deployments.
- Inspect saved checkpoints and plot metrics from the CLI using `gomlx_checkpoints` (`cmd/gomlx_checkpoints`).

---

## 2. Parameter-Efficient Fine-Tuning — PEFT / LoRA (`github.com/gomlx/gomlx/ml/layers/peft`)

GoMLX supports zero-code-change LoRA fine-tuning via post-linear hooks:

1. **Enable via blank import**:
   ```go
   import _ "github.com/gomlx/gomlx/ml/layers/peft"
   ```
   Importing `peft` auto-registers a post-linear hook (`layers.RegisterPostLinearHook`) that intercepts linear projections in `layers.Dense`, `fnn.Done()`, and `attention.MultiHeadAttention`.
2. **Configure via `store.SetParams` or CLI `-set` hyperparameters**:
   - `peft_adapter`: adapter sub-scope name (e.g., `"my_lora"`). Automatically freezes base weights and creates trainable low-rank `A` and `B` adapter variables under `<scope>/<adapter>/`. Setting `"off"` freezes base weights without adding adapters.
   - `peft_rank`: LoRA rank $r$.
   - `peft_alpha`: scaling factor ($\alpha / r$).
   - `peft_target_modules`: comma-separated string or slice of target module substrings (e.g., `"query,value"`). Unmatched modules are frozen without adapters.
3. **Multi-adapter support**:
   - Multiple named adapters can coexist in the same `model.Store` and checkpoint; switching adapters only requires changing `peft_adapter`.
4. **Manual / low-level construction**:
   - Use `peft.New(scope, x, weight, bias).Done()` to wrap custom linear weights directly.
