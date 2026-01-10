package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"unsafe"

	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gopjrt/dtypes"
)

// SafetensorsHeader describes the metadata for tensors in a safetensors file.
type SafetensorsHeader struct {
	Tensors map[string]TensorInfo `json:"-"`
}

// TensorInfo contains metadata about a single tensor.
type TensorInfo struct {
	DType  string   `json:"dtype"`
	Shape  []int    `json:"shape"`
	Offset [2]int64 `json:"data_offsets"`
}

// LoadSafetensors loads tensors from a safetensors file into the context.
func LoadSafetensors(ctx *context.Context, filepath string) error {
	// Read entire file into memory first
	fileData, err := os.ReadFile(filepath)
	if err != nil {
		return fmt.Errorf("failed to read file: %w", err)
	}

	// Read header size (first 8 bytes, little-endian)
	if len(fileData) < 8 {
		return fmt.Errorf("file too small")
	}
	headerSize := int64(binary.LittleEndian.Uint64(fileData[0:8]))

	// Read header JSON
	if len(fileData) < int(8+headerSize) {
		return fmt.Errorf("file too small for header")
	}
	headerBytes := fileData[8 : 8+headerSize]

	// Parse header
	var rawHeader map[string]interface{}
	if err := json.Unmarshal(headerBytes, &rawHeader); err != nil {
		return fmt.Errorf("failed to parse header: %w", err)
	}

	// Data section starts after header
	dataSectionOffset := 8 + headerSize

	// Load each tensor
	fmt.Printf("Found %d items in safetensors header\n", len(rawHeader))
	count := 0
	skipped := 0

	for name, value := range rawHeader {
		if name == "__metadata__" {
			continue
		}

		infoMap, ok := value.(map[string]interface{})
		if !ok {
			continue
		}

		// Parse tensor info
		dtypeStr, _ := infoMap["dtype"].(string)
		shapeRaw, _ := infoMap["shape"].([]interface{})
		offsetsRaw, _ := infoMap["data_offsets"].([]interface{})

		if len(offsetsRaw) != 2 {
			fmt.Printf("Warning: invalid offsets for %s\n", name)
			continue
		}

		shape := make([]int, len(shapeRaw))
		for i, v := range shapeRaw {
			shape[i] = int(v.(float64))
		}

		// Offsets are relative to start of data section (after header)
		dataStart := int64(offsetsRaw[0].(float64))
		dataEnd := int64(offsetsRaw[1].(float64))

		// Convert dtype string to dtypes.DType
		dtype := parseDType(dtypeStr)

		// Map tensor name to GoMLX structure
		scopePath, varName, ok := mapTensorName(name)
		if !ok {
			skipped++
			continue // Skip unmapped tensors
		}

		// Read tensor data from memory
		// dataStart and dataEnd are relative to the beginning of the data section
		tensorDataStart := dataSectionOffset + dataStart
		tensorDataEnd := dataSectionOffset + dataEnd
		if int64(len(fileData)) < tensorDataEnd {
			fmt.Printf("Warning: file too small for %s (need %d, have %d)\n", name, tensorDataEnd, len(fileData))
			continue
		}

		tensorData := fileData[tensorDataStart:tensorDataEnd]

		// Set variable in context
		scopePathStr := strings.Join(scopePath, "/")
		if err := setContextVariable(ctx, scopePath, varName, shape, dtype, tensorData); err != nil {
			fmt.Printf("Warning: failed to load %s -> %s/%s: %v\n", name, scopePathStr, varName, err)
			continue
		}

		// Handle weight tying: GPT-2 shares token embeddings with output projection
		if name == "transformer.wte.weight" {
			// Copy to output layer - Dense expects [hidden, vocab] so transpose [vocab, hidden] -> [hidden, vocab]
			transposedData := transposeWeights(tensorData, shape, dtype)

			if err := setContextVariable(ctx, []string{"output", "dense"}, "weights", []int{shape[1], shape[0]}, dtype, transposedData); err != nil {
				fmt.Printf("Warning: failed to set tied output weights: %v\n", err)
			}
		}

		count++
	}

	if count > 0 {
		fmt.Printf("✓ Successfully loaded %d/%d tensors from checkpoint\n", count, count+skipped)
	} else {
		fmt.Printf("Warning: No tensors were loaded (%d skipped)\n", skipped)
	}
	return nil
}

// parseDType converts safetensors dtype string to dtypes.DType
func parseDType(s string) dtypes.DType {
	switch s {
	case "F32":
		return dtypes.Float32
	case "F64":
		return dtypes.Float64
	case "F16":
		return dtypes.Float16
	case "BF16":
		return dtypes.BFloat16
	case "I32":
		return dtypes.Int32
	case "I64":
		return dtypes.Int64
	default:
		return dtypes.Float32
	}
}

// mapTensorName maps safetensors tensor names to GoMLX context variable names
// DistilGPT-2/GPT-2 format: transformer.wte.weight, transformer.h.{N}.attn.c_attn.weight, etc.
// GoMLX format: token_embed/embeddings, layer_{N}/attn/..., etc.
func mapTensorName(safetensorsName string) (scopePath []string, varName string, ok bool) {
	switch {
	case safetensorsName == "transformer.wte.weight":
		return []string{"token_embed"}, "embeddings", true
	case safetensorsName == "transformer.wpe.weight":
		return []string{"pos_embed"}, "embeddings", true
	case safetensorsName == "transformer.ln_f.weight":
		return []string{"final_norm", "layer_normalization"}, "gain", true
	case safetensorsName == "transformer.ln_f.bias":
		return []string{"final_norm", "layer_normalization"}, "offset", true
	}

	// Layer-specific weights: transformer.h.{N}.{component}.{param}
	var layerNum int
	var component string

	// Parse layer number and component
	if n, err := fmt.Sscanf(safetensorsName, "transformer.h.%d.%s", &layerNum, &component); n == 2 && err == nil {
		layerScope := fmt.Sprintf("layer_%d", layerNum)

		// Parse component and parameter
		switch {
		// Attention weights - need special handling for fused QKV
		case component == "attn.c_attn.weight":
			// This returns a marker - we'll handle splitting in the loading code
			return []string{layerScope, "attn", "_fused_qkv"}, "weights", true
		case component == "attn.c_attn.bias":
			return []string{layerScope, "attn", "_fused_qkv"}, "biases", true
		case component == "attn.c_proj.weight":
			return []string{layerScope, "attn", "MultiHeadAttention", "output", "dense"}, "weights", true
		case component == "attn.c_proj.bias":
			return []string{layerScope, "attn", "MultiHeadAttention", "output", "dense"}, "biases", true

		// Layer norm 1 (pre-attention)
		case component == "ln_1.weight":
			return []string{layerScope, "norm1", "layer_normalization"}, "gain", true
		case component == "ln_1.bias":
			return []string{layerScope, "norm1", "layer_normalization"}, "offset", true

		// Feed-forward network
		case component == "mlp.c_fc.weight":
			return []string{layerScope, "ff1", "dense"}, "weights", true
		case component == "mlp.c_fc.bias":
			return []string{layerScope, "ff1", "dense"}, "biases", true
		case component == "mlp.c_proj.weight":
			return []string{layerScope, "ff2", "dense"}, "weights", true
		case component == "mlp.c_proj.bias":
			return []string{layerScope, "ff2", "dense"}, "biases", true

		// Layer norm 2 (pre-FFN)
		case component == "ln_2.weight":
			return []string{layerScope, "norm2", "layer_normalization"}, "gain", true
		case component == "ln_2.bias":
			return []string{layerScope, "norm2", "layer_normalization"}, "offset", true

		// Attention bias (for masking, not used in GoMLX standard attention)
		case component == "attn.bias":
			return nil, "", false // Skip attention bias
		}
	}

	return nil, "", false
}

// setContextVariable sets a variable in the context from raw bytes
// setContextVariable sets a variable in the context from raw bytes
func setContextVariable(ctx *context.Context, scopePath []string, varName string, shape []int, dtype dtypes.DType, data []byte) error {
	// Check if we need to transpose (GPT-2's Conv1D stores as [out, in])
	needsTranspose := false
	if len(scopePath) > 0 && scopePath[len(scopePath)-1] == "_transpose" {
		needsTranspose = true
		scopePath = scopePath[:len(scopePath)-1]
	}

	// Check if this is a fused QKV weight that needs splitting
	if len(scopePath) >= 3 && scopePath[len(scopePath)-1] == "_fused_qkv" {
		// This is transformer.h.N.attn.c_attn - needs to be split into Q, K, V
		// shape is [hidden_size, 3*hidden_size] or [3*hidden_size] for bias
		// Split into 3 equal parts along the last dimension

		// Get the actual scope without "_fused_qkv"
		baseScopePath := scopePath[:len(scopePath)-1]
		baseCtx := ctx
		for _, scope := range baseScopePath {
			baseCtx = baseCtx.In(scope)
		}
		baseCtx = baseCtx.In("MultiHeadAttention")

		// Convert to float32 first (we need flat data for splitting)
		numElements := 1
		for _, dim := range shape {
			numElements *= dim
		}
		flatData := make([]float32, numElements)

		for i := 0; i < numElements; i++ {
			switch dtype {
			case dtypes.Float32:
				flatData[i] = float32FromBytes(data[i*4 : (i+1)*4])
			case dtypes.Float16:
				flatData[i] = float16ToFloat32(data[i*2 : (i+1)*2])
			case dtypes.BFloat16:
				flatData[i] = bfloat16ToFloat32(data[i*2 : (i+1)*2])
			default:
				return fmt.Errorf("unsupported dtype: %s", dtype)
			}
		}

		if len(shape) == 2 {
			// Weight matrix: Conv1D format is already [768, 2304] = [in, out]
			// Split into Q, K, V: each [768, 768]
			// Then reshape to GoMLX format: [768, 12, 64]
			hiddenSize := shape[0]
			totalSize := shape[1]
			if totalSize%3 != 0 {
				return fmt.Errorf("expected fused QKV size to be divisible by 3, got %d", totalSize)
			}
			singleSize := totalSize / 3 // This is hidden_size again (768)

			// We need to find numHeads and headDim
			// For DistilGPT-2: 12 heads, 64 dim each = 768
			// We'll assume numHeads = 12 and headDim = singleSize/12
			numHeads := 12
			headDim := singleSize / numHeads

			// Create Q, K, V tensors with shape [hiddenSize, numHeads, headDim]
			qData := make([]float32, hiddenSize*numHeads*headDim)
			kData := make([]float32, hiddenSize*numHeads*headDim)
			vData := make([]float32, hiddenSize*numHeads*headDim)

			// Split along the last dimension: first 768 cols are Q, next 768 are K, last 768 are V
			// Then reshape each from [768, 768] to [768, 12, 64]
			for i := 0; i < hiddenSize; i++ {
				for h := 0; h < numHeads; h++ {
					for d := 0; d < headDim; d++ {
						flatIdx := h*headDim + d
						qData[i*numHeads*headDim+h*headDim+d] = flatData[i*totalSize+flatIdx]
						kData[i*numHeads*headDim+h*headDim+d] = flatData[i*totalSize+singleSize+flatIdx]
						vData[i*numHeads*headDim+h*headDim+d] = flatData[i*totalSize+2*singleSize+flatIdx]
					}
				}
			}

			baseCtx.In("query").In("dense").VariableWithValue(varName, tensors.FromFlatDataAndDimensions(qData, hiddenSize, numHeads, headDim))
			baseCtx.In("key").In("dense").VariableWithValue(varName, tensors.FromFlatDataAndDimensions(kData, hiddenSize, numHeads, headDim))
			baseCtx.In("value").In("dense").VariableWithValue(varName, tensors.FromFlatDataAndDimensions(vData, hiddenSize, numHeads, headDim))
		} else if len(shape) == 1 {
			// Bias vector: [3*hidden_size]
			totalSize := shape[0]
			if totalSize%3 != 0 {
				return fmt.Errorf("expected fused QKV bias size to be divisible by 3, got %d", totalSize)
			}
			singleSize := totalSize / 3

			// Reshape to [numHeads, headDim]
			numHeads := 12
			headDim := singleSize / numHeads

			qData := make([]float32, numHeads*headDim)
			kData := make([]float32, numHeads*headDim)
			vData := make([]float32, numHeads*headDim)

			for h := 0; h < numHeads; h++ {
				for d := 0; d < headDim; d++ {
					flatIdx := h*headDim + d
					qData[h*headDim+d] = flatData[flatIdx]
					kData[h*headDim+d] = flatData[singleSize+flatIdx]
					vData[h*headDim+d] = flatData[2*singleSize+flatIdx]
				}
			}

			baseCtx.In("query").In("dense").VariableWithValue(varName, tensors.FromFlatDataAndDimensions(qData, numHeads, headDim))
			baseCtx.In("key").In("dense").VariableWithValue(varName, tensors.FromFlatDataAndDimensions(kData, numHeads, headDim))
			baseCtx.In("value").In("dense").VariableWithValue(varName, tensors.FromFlatDataAndDimensions(vData, numHeads, headDim))
		} else {
			return fmt.Errorf("unexpected shape for fused QKV: %v", shape)
		}

		return nil
	}

	// Normal case: set single variable
	scopeCtx := ctx
	for _, scope := range scopePath {
		scopeCtx = scopeCtx.In(scope)
	}

	// Transpose data before creating tensor if needed (for Conv1D weights)
	if needsTranspose && len(shape) == 2 {
		// Convert to float32 first
		numElements := shape[0] * shape[1]
		flatData := make([]float32, numElements)
		for i := 0; i < numElements; i++ {
			switch dtype {
			case dtypes.Float32:
				flatData[i] = float32FromBytes(data[i*4 : (i+1)*4])
			case dtypes.Float16:
				flatData[i] = float16ToFloat32(data[i*2 : (i+1)*2])
			case dtypes.BFloat16:
				flatData[i] = bfloat16ToFloat32(data[i*2 : (i+1)*2])
			default:
				return fmt.Errorf("unsupported dtype for transpose: %s", dtype)
			}
		}

		// Transpose from [shape[0], shape[1]] to [shape[1], shape[0]]
		transposed := make([]float32, numElements)
		for i := 0; i < shape[0]; i++ {
			for j := 0; j < shape[1]; j++ {
				transposed[j*shape[0]+i] = flatData[i*shape[1]+j]
			}
		}

		// Create tensor with transposed shape
		tensor := tensors.FromFlatDataAndDimensions(transposed, shape[1], shape[0])
		scopeCtx.VariableWithValue(varName, tensor)
		return nil
	}

	tensor, err := bytesToTensor(data, shape, dtype)
	if err != nil {
		return fmt.Errorf("failed to convert bytes to tensor: %w", err)
	}

	scopeCtx.VariableWithValue(varName, tensor)
	return nil
}

// bytesToTensor converts raw bytes to a tensor with the given shape and dtype
func bytesToTensor(data []byte, shape []int, dtype dtypes.DType) (*tensors.Tensor, error) {
	// Calculate expected size
	elementSize := dtype.Size()
	numElements := 1
	for _, dim := range shape {
		numElements *= dim
	}
	expectedSize := numElements * elementSize

	if len(data) != expectedSize {
		return nil, fmt.Errorf("data size mismatch: got %d bytes, expected %d for shape %v and dtype %s",
			len(data), expectedSize, shape, dtype)
	}

	// Convert bytes to appropriate slice type
	switch dtype {
	case dtypes.Float32:
		values := make([]float32, numElements)
		for i := 0; i < numElements; i++ {
			values[i] = float32FromBytes(data[i*4 : (i+1)*4])
		}
		return tensors.FromFlatDataAndDimensions(values, shape...), nil

	case dtypes.Float16:
		// Convert float16 to float32 for GoMLX
		values := make([]float32, numElements)
		for i := 0; i < numElements; i++ {
			values[i] = float16ToFloat32(data[i*2 : (i+1)*2])
		}
		return tensors.FromFlatDataAndDimensions(values, shape...), nil

	case dtypes.BFloat16:
		// Convert bfloat16 to float32 for GoMLX
		values := make([]float32, numElements)
		for i := 0; i < numElements; i++ {
			values[i] = bfloat16ToFloat32(data[i*2 : (i+1)*2])
		}
		return tensors.FromFlatDataAndDimensions(values, shape...), nil

	default:
		return nil, fmt.Errorf("unsupported dtype: %s", dtype)
	}
}

// float32FromBytes converts 4 bytes (little-endian) to float32
func float32FromBytes(b []byte) float32 {
	bits := binary.LittleEndian.Uint32(b)
	return *(*float32)(unsafe.Pointer(&bits))
}

// float16ToFloat32 converts IEEE 754 float16 to float32
func float16ToFloat32(b []byte) float32 {
	h := binary.LittleEndian.Uint16(b)
	sign := uint32(h&0x8000) << 16
	exponent := uint32(h&0x7C00) >> 10
	mantissa := uint32(h & 0x03FF)

	if exponent == 0 {
		if mantissa == 0 {
			// Zero
			bits := sign
			return *(*float32)(unsafe.Pointer(&bits))
		}
		// Subnormal
		exponent = 1
		for (mantissa & 0x0400) == 0 {
			mantissa <<= 1
			exponent--
		}
		mantissa &= 0x03FF
	}

	if exponent == 0x1F {
		// Inf/NaN
		bits := sign | 0x7F800000 | (mantissa << 13)
		return *(*float32)(unsafe.Pointer(&bits))
	}

	// Normalized
	exponent = exponent + (127 - 15)
	mantissa = mantissa << 13
	bits := sign | (exponent << 23) | mantissa
	return *(*float32)(unsafe.Pointer(&bits))
}

// bfloat16ToFloat32 converts bfloat16 to float32
func bfloat16ToFloat32(b []byte) float32 {
	// BFloat16 is just the top 16 bits of float32
	bits := uint32(binary.LittleEndian.Uint16(b)) << 16
	return *(*float32)(unsafe.Pointer(&bits))
}

// transposeWeights transposes a 2D weight matrix from [A, B] to [B, A]
func transposeWeights(data []byte, shape []int, dtype dtypes.DType) []byte {
	if len(shape) != 2 {
		return data // Only transpose 2D matrices
	}

	rows := shape[0]
	cols := shape[1]
	elementSize := dtype.Size()

	result := make([]byte, len(data))

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			srcOffset := (i*cols + j) * elementSize
			dstOffset := (j*rows + i) * elementSize
			copy(result[dstOffset:dstOffset+elementSize], data[srcOffset:srcOffset+elementSize])
		}
	}

	return result
}
