# Go-HuggingFace Reference (`github.com/gomlx/go-huggingface`)

`go-huggingface` provides modular, Go-native tools for working with [HuggingFace](https://huggingface.co) models, tokenizers, and datasets, and converting HuggingFace models on-the-fly into GoMLX computation graphs (supporting SafeTensors, GGUF, and — together with `onnx-gomlx` — ONNX models).

```bash
go get -u github.com/gomlx/go-huggingface
```

### Modularity & Dependencies

Packages are strictly decoupled so you only pull in what you use:
- `hub` has **no** dependency on GoMLX or Parquet.
- `tokenizers` has **no** dependency on GoMLX, Parquet, or CGo/Rust (pure Go).
- `datasets` depends on `hub` and `parquet-go` (no GoMLX dependency).
- `models/*` integrate with `github.com/gomlx/gomlx` and `github.com/gomlx/compute`.

### Package Map

| Package | Purpose |
|---|---|
| `hub` | Download, cache, inspect, save, and read HuggingFace repos (remote Hub, local directory, or `//go:embed`). |
| `tokenizers` | High-level `Tokenizer` factory (`tokenizers.New(repo)`) and config reader (`GetConfig`). |
| `tokenizers/api` | `Tokenizer` interface, `SpecialToken`, `TokenSpan`, `AnnotatedEncoding`, `Config`. |
| `tokenizers/hftokenizer` | Pure-Go engine parsing `tokenizer.json` (BPE, WordPiece, Unigram) and GGUF embedded `tokenizer.ggml.*` metadata. |
| `tokenizers/sentencepiece` | Pure-Go SentencePiece tokenizer support (e.g., Gemma `.model` files). |
| `tokenizers/bucket` | Concurrent streaming sentence tokenizer + length bucketing (`ByTwoBitBucket`, `ByPower`, `Unpadded`) to minimize padding. |
| `models/safetensors` | Memory-mapped single/sharded SafeTensors reader and parallel tensor streaming to GoMLX backends. |
| `models/gguf` | GGUF v2/v3 parser, multi-quant file selection, pure-Go dequantization, raw GGML block loading, and embedded tokenizer extraction. |
| `models/transformer` | On-the-fly HuggingFace transformer & sentence-embedding graph builder for GoMLX (SafeTensors & GGUF) with self-contained checkpoint export/reload. |
| `models/sam2` | Segment Anything 2 (SAM2) image segmentation model (high-level `Segmenter` + low-level `Forward` graph API). |
| `models/timestral` | Mistral AI's Timestral probabilistic time-series forecasting models (high-level `Forecaster` + low-level `BuildGraph`). |
| `models/image` | Image preprocessing graph helpers (`PreprocessGraph`). |
| `datasets` | HuggingFace datasets metadata inspection, on-demand Parquet shard downloading, streaming iteration (`IterParquetFromDataset[T]`), and Go struct codegen. |
| `cmd/hubinfo`, `cmd/dataset_download`, `cmd/generate_dataset_structs` | CLI utilities for inspecting/saving repos, managing dataset downloads, and generating Parquet record structs. |

---

## 1. Downloading & Reading Repositories (`hub`)

`hub.Repo` shares the exact cache layout as Python's `huggingface_hub` (`~/.cache/huggingface/hub/` or `$XDG_CACHE_HOME/huggingface/hub/`), using inter-process file locking for safe concurrent downloads. Environment variables respected automatically: `HF_TOKEN`, `HF_ENDPOINT`, `XDG_CACHE_HOME`.

### Three Repository Modes

All functions accepting `*hub.Repo` work identically across all three modes:

1. **Remote Hub**: `hub.New("Qwen/Qwen3-0.6B")` (chain `.WithAuth(os.Getenv("HF_TOKEN"))`, `.WithRevision("main")`, `.WithType(hub.RepoTypeDataset)`, `.WithCacheDir(dir)`, `.WithProgressBar(false)`).
2. **Local directory (offline)**: `hub.NewLocal("/path/to/local/model")` — reads directly from disk with zero network calls (automatically skips `.git`, `.cache`, `.huggingface`, `.ipynb_checkpoints`).
3. **Embedded in binary (`//go:embed`)**: `hub.NewEmbed(embeddedFS, "my_model")` — reads directly from an `io/fs.FS` in memory.

### Listing, Reading & Saving Files

- **Prefer `repo.Open(name)` (`fs.File`) or `repo.ReadFile(name)` (`[]byte`)** over `DownloadFile` so embedded repos stay in-memory without temp-dir extraction.
- Use `repo.FetchFiles(names...)` to pre-cache files without requesting disk paths, or `repo.DownloadFile(name)` / `repo.DownloadFiles(names...)` when an OS file path is required.
- Export a remote repo to a self-contained local folder with `repo.Save(dirPath, linkOnly)` (pass `linkOnly=true` for hard links on the same filesystem), and clean cache with `repo.DeleteCache()`.

```go
repo := hub.New("sentence-transformers/all-MiniLM-L6-v2").WithAuth(os.Getenv("HF_TOKEN"))

// List repository files and sizes:
for info, err := range repo.IterFileInfos() {
	if err != nil {
		panic(err)
	}
	fmt.Printf("%s (%d bytes)\n", info.Name, info.Size)
}

// Read a file directly (works for remote, local, and embedded repos):
cfgBytes, err := repo.ReadFile("config.json")
```

---

## 2. Tokenizers & Streaming Bucketing (`tokenizers`, `tokenizers/bucket`)

### Standard Tokenizer Usage

`tokenizers.New(repo)` inspects `tokenizer_config.json` / `tokenizer.json` / SentencePiece `.model` and instantiates the matching pure-Go tokenizer implementing `api.Tokenizer`:

- `Encode(text string) []int`
- `EncodeWithAnnotations(text string) api.AnnotatedEncoding` — includes byte-offset `TokenSpan`s (`text[span.Start:span.End]`) and special-token masks.
- `Decode(tokens []int) string`
- `SpecialTokenID(tok api.SpecialToken) (int, error)` — standardized lookup for `api.TokBeginningOfSentence`, `api.TokEndOfSentence`, `api.TokPad`, `api.TokUnknown`, `api.TokMask`, `api.TokClassification`, etc.

```go
repo := hub.New("google/gemma-2-2b-it").WithAuth(os.Getenv("HF_TOKEN"))
tok, err := tokenizers.New(repo)
if err != nil {
	panic(err)
}
tokens := tok.Encode("The book is on the table.")
padID, _ := tok.SpecialTokenID(tokenizers.TokPad)
```

For GGUF-only repositories that lack `tokenizer.json`, `hftokenizer.NewFromGGUF` (called automatically by `transformer.Model.GetTokenizer()` and `gguf.File.LoadTokenizer`) builds a pure-Go WordPiece, ByteLevel BPE, or SentencePiece Unigram tokenizer directly from embedded `tokenizer.ggml.*` metadata.

> **Alternative**: If you need a tokenizer feature not yet supported by `hftokenizer` or want Rust-based bindings, you can read `tokenizer.json` with `repo.ReadFile("tokenizer.json")` and pass the bytes to `github.com/daulet/tokenizers`.

### Streaming Bucketing (`tokenizers/bucket`)

Groups streamed texts into discrete sequence-length buckets to minimize padding overhead and JIT recompilations (crucial for static-shape backends like XLA):

- `ByTwoBitBucket(batchSize, minLen)` / `ByTwoBitBucketBudget(tokensBudget, minLen)`: buckets lengths to 2-bit-representable values (`1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64...`), capping padding overhead to ~20%.
- `ByPower(batchSize, minLen, base)` / `ByPowerBudget(tokensBudget, minLen, base)`: power-of-`base` bucket sizes.
- `Unpadded()`: zero-copy pass-through emitting each sentence unpadded (`BatchSize = 1`) — ideal for dynamic-shape backends (`GOMLX_BACKEND=go` or `onnx`).
- `WithMaxDelay(d, useBatchPadding)`: flushes partial buckets after a latency bound for online serving.
- `WithMaxParallelization(n)`: controls parallel tokenization workers (`-1` for all CPUs).

```go
var wg sync.WaitGroup
inCh := make(chan bucket.SentenceRef, 64)
outCh := make(chan bucket.Bucket, 10)

bkt := bucket.New(tok).
	ByTwoBitBucketBudget(8*1024, 16). // ~8K tokens/batch, min length 16
	WithMaxDelay(50*time.Millisecond, true).
	WithMaxParallelization(-1)

wg.Go(func() { bkt.Run(inCh, outCh) })
// Send bucket.SentenceRef{Sentence: s, Ref: id} to inCh, close(inCh), and range over outCh.
```

---

## 3. On-the-Fly HuggingFace Model Conversion (`models/*`)

### Transformers & Sentence Embeddings (`models/transformer`)

`models/transformer` reads HuggingFace model configs (`config.json`, `config_sentence_transformers.json`, `modules.json`, `1_Pooling/config.json`) — or synthesizes them from GGUF metadata — loads weights into a GoMLX `model.Store`, and constructs GoMLX computation graphs via `github.com/gomlx/gomlx/ml/zoo/transformer`.

Supports both **SafeTensors** and **GGUF** repos transparently across encoder (BERT, Nomic-BERT, Jina-BERT) and decoder (Gemma / Gemma 2 / Gemma 3 / Gemma 4, LLaMA, Qwen, KaLM) architectures.

```go
repo := hub.New("tencent/KaLM-Embedding-Gemma3-12B-2511").WithAuth(os.Getenv("HF_TOKEN"))
hfModel, err := transformer.LoadModel(repo)
if err != nil {
	panic(err)
}
tok, err := hfModel.GetTokenizer() // Uses tokenizer.json or embedded GGUF tokenizer
if err != nil {
	panic(err)
}

backend := compute.MustNew()
defer backend.Finalize()

store := model.NewStore()
if err := hfModel.LoadStore(backend, store); err != nil {
	panic(err)
}

embedExec, err := model.NewExec(backend, store, func(scope *model.Scope, tokens *graph.Node) *graph.Node {
	x := hfModel.SentenceEmbeddingGraph(scope, tokens, nil)
	return graph.ConvertDType(x, dtypes.Float32)
})
if err != nil {
	panic(err)
}
defer embedExec.Finalize()

tokens := [][]int32{toInt32(tok.Encode("What is GoMLX?"))}
embeddings := embedExec.MustCall1(tokens)
defer embeddings.FinalizeAll()
```

#### Key `transformer.Model` Methods & Options
- **Graph building**:
  - `Forward(scope, tokenIDs, positionIDs, kvCache...)`: autoregressive forward pass returning `(logits, updatedKVCache)`.
  - `ForwardGraph(scope, tokens, seqLen)` / `AllLayers(scope, tokens, seqLen)`: hidden-state representations.
  - `SentenceEmbeddingGraph(scope, tokens, seqLen)` / `SentenceEmbeddingFromLastHiddenState(hidden, seqLen)` / `ApplySentencePooling(hidden, seqLen)` / `Similarity(queries, docs)`: full sentence-transformer pipeline (pooling + L2 normalization).
  - `CreateGoMLXModel(scope)`: returns the underlying `*mltransformer.Model`.
- **Configuration options**:
  - `WithCausalMask(bool)` / `UseCausalMask()`: auto-detected from model architecture (`true` for decoders, `false` for encoders); override explicitly if needed.
  - `WithGGUFFile(filename)`: selects a specific `.gguf` file in multi-quant repositories.
  - `WithApproximateGelu(bool)`: defaults to `true` (matches `llama.cpp`'s `tanh`-approximate GELU); set `false` for exact `erf` GELU matching PyTorch/ONNX.
  - `WithForceFloat32(bool)`, `WithMMap(bool)`.
- **Weight loading**: `LoadStore(backend, store)` (skips variables already restored from a checkpoint), `LoadStoreFiltered(backend, store, filterFn)`.
- **Self-contained fine-tuning checkpoints & offline/embedded distribution**:
  - Save config + tokenizer sidecars into `checkpointDir/hf/`: `hfModel.SaveToCheckpoint(checkpointDir)` (or `SaveConfigs(dir)`).
  - Reload offline without touching HuggingFace Hub: `transformer.LoadModelFromCheckpoint(checkpointDir)` + `checkpoint.Build(store).Dir(checkpointDir).Done()`.
  - Load from an embedded or `.zip` `fs.FS`: `transformer.LoadModelFromFS(fsys, subDir)`.

---

### GGUF Models & Native Quantized Execution (`models/gguf`)

`models/gguf` parses GGUF v2/v3 binaries (local files or HuggingFace repos) with both pure-Go dequantization (`ReadTensor`) and zero-expansion raw GGML block loading (`ReadTensorRaw` + `ToGGMLQuantType()`):

- **Supported quantized block formats**: `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `IQ4_NL`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K` (plus `F32`, `F16`, `BF16`).
- **Native GoMLX quantized dispatch**: when loaded via `transformer.Model.LoadStore`, 2D GGML quantized weights stay packed in `Uint8` with `model.Quantization{Scheme: compute.QuantGGML}` metadata so GoMLX dispatches `nn.QuantizedDense` and `nn.QuantizedGather` directly (with automatic decomposed graph fallback on backends without native GGML kernels).
- **Multi-GGUF file selection**: when a repo has multiple `.gguf` variants, `gguf.SelectGGUFFile` automatically picks the best default (`Q4_K_M` -> `Q4_K_S` -> `Q5_K_M` -> `Q8_0` -> `Q6_K` -> ...). Override with `ggufModel.WithFile(name)` or `hfModel.WithGGUFFile(name)`.
- **Pure-GGUF repositories**: even without `config.json`, `tokenizer.json`, or `modules.json`, `models/transformer` synthesizes architecture config, pooling (`<arch>.pooling_type`), L2 normalization, and the embedded tokenizer directly from GGUF header metadata.

```go
// Direct GGUF tensor inspection/loading:
gModel, err := gguf.NewFromFile("/path/to/model-Q4_K_M.gguf")
// Or from HuggingFace Hub:
// gModel, err := gguf.New(hub.New("second-state/All-MiniLM-L6-v2-Embedding-GGUF"))
defer gModel.Close()

for tn, err := range gModel.IterTensors(backend) {
	if err != nil {
		panic(err)
	}
	fmt.Printf("%s: %s\n", tn.Name, tn.Tensor.Shape())
}
```

---

### SafeTensors Weight Loading (`models/safetensors`)

Reads single-file or sharded (`model.safetensors.index.json`) SafeTensors weights with `mmap` and parallel streaming directly into a `compute.Backend`:

```go
repo := hub.New(modelID).WithAuth(os.Getenv("HF_TOKEN"))
for tn, err := range safetensors.IterTensorsFromRepo(backend, repo) {
	if err != nil {
		panic(err)
	}
	fmt.Printf("%s: %s\n", tn.Name, tn.Tensor.Shape())
}
```

---

### ONNX Models on HuggingFace

Many HuggingFace repositories provide `.onnx` files (e.g. `model.onnx` or `onnx/model.onnx`). Combine `go-huggingface/hub` (+ `tokenizers` / `bucket`) with `github.com/gomlx/onnx-gomlx/onnx/parser` to download, tokenize, convert on-the-fly to a GoMLX graph, and run inference or fine-tuning:

```go
repo := hub.New("sentence-transformers/all-MiniLM-L6-v2").WithAuth(os.Getenv("HF_TOKEN"))

// Download (or repo.Open + parser.ParseReader for embedded/in-memory repos):
localPath, err := repo.DownloadFile("onnx/model.onnx")
if err != nil {
	panic(err)
}
onnxModel, err := parser.ParseFile(localPath)
if err != nil {
	panic(err)
}
defer onnxModel.Close()

store := model.NewStore()
if err := onnxModel.VariablesToScope(store.RootScope()); err != nil {
	panic(err)
}

tok, err := tokenizers.New(repo)
// Build a GoMLX model.Exec calling onnxModel.CallGraph(scope, g, onnxInputs).
// For sentence-transformer ONNX models outputting 3D hidden states [B, S, D],
// you can also apply hfModel.SentenceEmbeddingFromLastHiddenState(hidden, seqLen).
```

See the [`onnx-gomlx` reference](./onnx-gomlx.md) for full details on dynamic shapes, external data files, fused-op matching, and fine-tuning / re-exporting `.onnx` weights.

---

### Other Built-in Model Families

- **SAM2 (`models/sam2`)**: Segment Anything 2 image segmentation (Hiera backbone, FPN neck, prompt encoder, mask decoder):
  ```go
  repo := hub.New("facebook/sam2-hiera-base-plus")
  samModel, err := sam2.LoadModel(repo)
  segmenter, err := sam2.NewSegmenter(backend, samModel)
  results, err := segmenter.Segment(img, &sam2.SegmentationOptions{
      Points: []sam2.PromptPoint{{X: 512, Y: 512, Label: sam2.LabelForeground}},
  })
  ```
  Also exposes low-level `sam2.Forward(...)` and `InternalStates` for custom GoMLX graphs and training loops.
- **Timestral (`models/timestral`)**: Mistral AI's probabilistic time-series foundation models (`RepoTimestral5M`, `RepoTimestral28M`, `RepoTimestral123M`, `RepoTimestral209M`) supporting univariate/multivariate context, `NaN` missing values, and numerical/categorical past & known-future covariates:
  ```go
  repo := hub.New(timestral.RepoTimestral28M)
  tmModel, err := timestral.LoadModel(repo)
  fc, err := timestral.NewForecaster(backend, tmModel)
  res, err := fc.Forecast(contextSeries, 24).Done() // res.Median(), res.Quantile(0.9)
  ```
  Also exposes low-level `tmModel.LoadStore` + `tmModel.BuildGraph(scope, x, horizon).Done()`.
- **Image helpers (`models/image`)**: `image.PreprocessGraph(x, targetHeight, targetWidth, mean, std)` for graph-level image resizing and normalization.

---

## 4. Iterating HuggingFace Datasets (`datasets`)

`datasets` queries the HuggingFace datasets server, inspects configurations/splits/features, downloads Parquet shards on demand (one shard at a time during iteration so multi-terabyte datasets don't fill your disk), and handles non-standard Parquet 3-level list schemas automatically via `ParquetFixListSchema[T]`.

### 1. Inspect Dataset & Generate Go Record Struct

```bash
# Inspect configs/splits and download shards:
go run github.com/gomlx/go-huggingface/cmd/dataset_download microsoft/ms_marco

# Generate a Go struct with json & parquet tags from the Parquet schema:
go run github.com/gomlx/go-huggingface/cmd/generate_dataset_structs -dataset HuggingFaceFW/fineweb
```

Or programmatically: `ds := datasets.New("HuggingFaceFW/fineweb").WithAuth(token)` -> `ds.Info()`, `ds.String()`, `ds.ListFiles(config, split)`, `ds.GenerateGoStruct(config, split)`.

### 2. Stream Records from Parquet

```go
type FinewebRecord struct {
	Text          string  `json:"text" parquet:"text"`
	ID            string  `json:"id" parquet:"id"`
	URL           string  `json:"url" parquet:"url"`
	LanguageScore float64 `json:"language_score" parquet:"language_score"`
	TokenCount    int64   `json:"token_count" parquet:"token_count"`
}

ds := datasets.New("HuggingFaceFW/fineweb").WithAuth(os.Getenv("HF_TOKEN"))
count := 0
for row, err := range datasets.IterParquetFromDataset[FinewebRecord](ds, "sample-10BT", "train") {
	if err != nil {
		panic(err)
	}
	fmt.Printf("#%d (score=%.2f): %.60q\n", count, row.LanguageScore, row.Text)
	count++
	if count >= 10 {
		break // Only downloads the first Parquet shard!
	}
}
```

Other iteration helpers:
- `datasets.IterParquetFromDatasetAt[T](ds, config, split, startRow)`: resume iteration from an arbitrary row offset.
- `datasets.IterParquetFromFile[T](localParquetPath)`: iterate over a local `.parquet` file directly.
- `ds.DownloadAll(ctx, config, split)` / `ds.ListDownloadedFiles(config, split)`: pre-download or inspect cached shards.

---

## 5. CLI Tools

| Tool | Example Usage |
|---|---|
| `cmd/hubinfo` | `go run github.com/gomlx/go-huggingface/cmd/hubinfo -save=/path/to/local Qwen/Qwen3-0.6B` (supports `-link_only`, `-delete_cache`, `-local`) |
| `cmd/dataset_download` | `go run github.com/gomlx/go-huggingface/cmd/dataset_download -config v2.1 -split validation microsoft/ms_marco` (supports `-list`, `-delete`) |
| `cmd/generate_dataset_structs` | `go run github.com/gomlx/go-huggingface/cmd/generate_dataset_structs -dataset HuggingFaceFW/fineweb` |
