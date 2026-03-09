# gemma3: ONNX Text Generation

Demonstrates ONNX-based text generation using the [onnx-community/gemma-3-270m-it-ONNX](https://huggingface.co/onnx-community/gemma-3-270m-it-ONNX) model with GoMLX.

Given a user message, the model generates a response using Gemma3's chat template and autoregressive decoding.

## Usage

```bash
go run gemma3.go
```

With a custom prompt:

```bash
go run gemma3.go --prompt="What is Go?"
```

With different generation parameters:

```bash
go run gemma3.go --max-tokens=50 --temperature=0.6 --top-k=32
```

Use the fp16 model variant (570MB instead of 1.14GB):

```bash
go run gemma3.go --fp16
```

## Options

- `--prompt`: User message for chat generation (default: "Write a short poem about the sea.")
- `--max-tokens`: Maximum number of tokens to generate (default: 100)
- `--max-seq-len`: Maximum total sequence length including prompt (default: 256)
- `--temperature`: Sampling temperature; 0 = greedy (default: 0.8)
- `--top-k`: Top-k sampling; 0 = disabled (default: 64)
- `--fp16`: Use fp16 model variant (default: false)
- `--backend`: Backend to use (default: auto-detect)

The first run downloads the ONNX model and tokenizer from HuggingFace to `~/.cache/huggingface/`.

If downloads fail, consider using a mirror by setting `HF_ENDPOINT`.

## How It Works

The example loads the Gemma3 270M instruction-tuned model in ONNX format and runs inference via `model.CallGraph()`. Each generation step runs the full model on the entire sequence (prompt + generated tokens so far), padded to the next power of 2 to limit JIT recompilations. KV cache inputs are provided as zero-sequence-length tensors. Sampling is performed on the CPU with temperature scaling and top-k filtering.
