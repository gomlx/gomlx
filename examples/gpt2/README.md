# GPT-2 Text Generation

A complete GPT-2 implementation using GoMLX with safetensors weight loading and BPE tokenization.

## Usage

```bash
go build
./gpt2 --prompt "Once upon a time" --max-tokens 100 --temperature 0.8
```

If downloads fail, consider using a mirror by setting `HF_ENDPOINT`.

## Features

- Loads GPT-2 weights from Hugging Face (safetensors format)
- BPE tokenizer (vocab.json + merges.txt)
- It uses the `github.com/gomlx/gomlx/ml/zoo/transformer/generate` library to generate text with KV caching for efficiency.
- Achieves ~50 tokens/s on CPU (77% of PyTorch performance)

## Options

- `--prompt`: Input text (default: "Once upon a time")
- `--max-tokens`: Maximum tokens to generate (default: 50)
- `--temperature`: Sampling temperature (default: 0.8)
- `--checkpoint`: Model checkpoint directory

First run downloads ~500MB of weights to `~/.cache/gomlx/gpt2/`.
