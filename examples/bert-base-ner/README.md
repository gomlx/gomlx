# BERT-base-NER Example

Named Entity Recognition (NER) using the [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER) model from HuggingFace.

This example demonstrates loading and running an ONNX model with GoMLX to extract named entities from text.

## Entity Types

The model identifies four types of entities using BIO tagging:

- **PER** - Person names
- **ORG** - Organizations
- **LOC** - Locations
- **MISC** - Miscellaneous entities

## Prerequisites

- Go 1.24.3 or later
- (Optional) HuggingFace token in `HF_TOKEN` environment variable if the model requires authentication

## Building

From the `examples/bert-base-ner` directory:

```bash
go build -o bert-base-ner .
```

Or from the repository root:

```bash
go build -o /tmp/bert-base-ner ./examples/bert-base-ner
```

## Usage

```bash
./bert-base-ner --text="The Winter Olympics are happening in Italy"
```

## Example Output

```
Loading model dslim/bert-base-NER...

Input: The Winter Olympics are happening in Italy 

Found 2 entities:
  Winter Olympics => MISC (confidence: 0.98)
  Italy => LOC (confidence: 1.00)
```

## Implementation Details

The example uses:
- `go-huggingface` for downloading the model from HuggingFace Hub
- `onnx-gomlx` for loading and executing the ONNX model
- A simple inline WordPiece tokenizer for BERT tokenization
