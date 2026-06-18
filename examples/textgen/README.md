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
# For multiple scope parameters, enclose them in quotes and separate with semicolons:
go run main.go -set="generate_strategy=temperature;generate_temperature=1.5"
```

## Available Flags

| Flag | Type | Description |
|---|---|---|
| `-prompt` | `string` | Generation prompt (default "The quick") |
| `-set` | `string` | Set scope parameters defining the model |

> **Note**
> For the full list of available scope parameters (passed via `-set`), please run:
> ```bash
> go run . -h
> ```

## Sampling Strategies

- **greedy**: Always picks the most likely token (deterministic)
- **temperature**: Samples with temperature scaling (higher = more random)

## Implementation

The generation code is in `ml/zoo/transformer/generate/`:
- `generate.go` - Main generation loop
- `sample/sample.go` - Sampling strategies
- `sample/beamsearch.go` - Beam search

KV caching support in `ml/layers/attention/`
