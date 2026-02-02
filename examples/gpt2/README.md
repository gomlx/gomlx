# GPT-2 Text Generation

A complete GPT-2 implementation using GoMLX with safetensors weight loading and BPE tokenization.

## Usage

```bash
go build
# If downloads fail, set HF_ENDPOINT=https://hf-mirror.com then retry:
./gpt2 --prompt "Once upon a time" --max-tokens 100 --temperature 0.8
```

## Features

- Loads GPT-2 weights from Hugging Face (safetensors format)
- BPE tokenizer (vocab.json + merges.txt)
- KV caching for efficient generation
- Achieves ~50 tokens/s on CPU (77% of PyTorch performance)

## Options

- `--prompt`: Input text (default: "Once upon a time")
- `--max-tokens`: Maximum tokens to generate (default: 50)
- `--temperature`: Sampling temperature (default: 0.8)
- `--checkpoint`: Model checkpoint directory

First run downloads ~500MB of weights to `~/.cache/gomlx/gpt2/`.
