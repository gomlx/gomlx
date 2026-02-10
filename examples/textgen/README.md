# Text Generation Example

Demonstrates autoregressive text generation with a simple character-level transformer.

## Quick Start

Train and generate text:

```bash
go run main.go
```

With more training steps:

```bash
go run main.go -set="train_steps=1000"
```

Try different sampling strategies:

```bash
# For multiple context parameters, enclose them in quotes and separate with semicolons:
go run main.go -set="decode_strategy=temperature;decode_temperature=1.5"
```

## Available Flags

| Flag | Type | Description |
|---|---|---|
| `-prompt` | `string` | Generation prompt (default "The quick") |
| `-set` | `string` | Set context parameters defining the model |

> **Note**
> For the full list of available context parameters (passed via `-set`), please run:
> ```bash
> go run . -h
> ```

## Sampling Strategies

- **greedy**: Always picks the most likely token (deterministic)
- **temperature**: Samples with temperature scaling (higher = more random)

## Implementation

The generation code is in `pkg/ml/layers/generation/`:
- `decode.go` - Main generation loop
- `sampling.go` - Sampling strategies
- `beamsearch.go` - Beam search (planned)

KV caching support in `pkg/ml/layers/attention/`
