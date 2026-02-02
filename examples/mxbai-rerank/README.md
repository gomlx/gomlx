# mxbai-rerank: Cross-Encoder Reranking

Demonstrates cross-encoder reranking using the [mixedbread-ai/mxbai-rerank-base-v1](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v1) model with GoMLX.

Given a query and a set of documents, the model scores each document by relevance and returns them ranked.

Rerankers are commonly used as a second stage in [RAG (Retrieval-Augmented Generation) pipelines](https://www.mixedbread.com/blog/mxbai-rerank-v1): a fast retriever (keyword or embedding search) fetches candidate documents, then a cross-encoder reranker re-scores them for higher relevance accuracy.

## Usage

```bash
go run mxbai-rerank.go
```

With a custom query:

```bash
go run mxbai-rerank.go --query="What is deep learning?"
```

With a different max sequence length:

```bash
go run mxbai-rerank.go --max_length=256
```

## Options

- `--query`: Query text to rerank documents against (default: "What is machine learning?")
- `--max_length`: Maximum sequence length for tokenization (default: 512)

The first run downloads the ONNX model and tokenizer from HuggingFace to `~/.cache/huggingface/`.

If downloads fail, consider using a mirror by setting `HF_ENDPOINT`.
