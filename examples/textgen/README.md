# Text Generation Example

Demonstrates autoregressive text generation with a simple character-level transformer.

## Quick Start

Train and generate text:

```bash
go run main.go
```

With more training steps:

```bash
go run main.go -steps 1000
```

Try different sampling strategies:

```bash
go run main.go -strategy temperature -temperature 1.5
```

## Available Flags

```bash
-strategy string      Sampling strategy: greedy|temperature (default "greedy")
-temperature float    Sampling temperature (default 1.0)
-max_length int       Max generation length (default 50)
-prompt string        Generation prompt (default "The quick")
-steps int            Training steps (default 200)
-lr float             Learning rate (default 0.01)
-use_cache bool       Use KV cache for generation (default false)
```

## Sampling Strategies

- **greedy**: Always picks the most likely token (deterministic)
- **temperature**: Samples with temperature scaling (higher = more random)

## Implementation

The generation code is in `pkg/ml/layers/generation/`:
- `decode.go` - Main generation loop
- `sampling.go` - Sampling strategies
- `beamsearch.go` - Beam search (planned)

KV caching support in `pkg/ml/layers/attention/`
