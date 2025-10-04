package huggingface

import (
	"encoding/binary"
	"encoding/json"
	"io"
	"iter"
	"slices"

	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
)

// NamedTensor represents a tensor and its name in a ".safetensors" file.
type NamedTensor struct {
	Name   string
	Tensor *tensors.Tensor
}

const safetensorsMetadataKey = "__metadata__" //

type tensorMetadata struct {
	// Format is only present for the safetensorsMetadataKey ("__metadata__").
	Format string `json:"format"`

	DTypeName  string   `json:"dtype"`
	Dimensions []int    `json:"shape"`
	Offsets    []uint64 `json:"data_offsets"`

	// Name is filled later, with the key to the tensor.
	Name string `json:"-"`
}

// DType parses the dtype name into an actual dtype.
func (t *tensorMetadata) DType() dtypes.DType {
	dtype, found := dtypes.MapOfNames[t.DTypeName]
	if !found {
		dtype = dtypes.InvalidDType
	}
	return dtype
}

func (t *tensorMetadata) Shape() shapes.Shape {
	return shapes.Make(t.DType(), t.Dimensions...)
}

func scanSafetensorsFile(r io.Reader) iter.Seq2[*NamedTensor, error] {
	return func(yield func(*NamedTensor, error) bool) {
		var metadataLenBuf [8]byte
		n, err := r.Read(metadataLenBuf[:])
		if err != nil {
			yield(nil, errors.Wrapf(err, "failed to read metadata length"))
			return
		}
		if n != 8 {
			yield(nil, errors.Errorf("failed to read metadata length -- got only %d bytes", n))
			return
		}
		var metadataLen uint64
		n, err = binary.Decode(metadataLenBuf[:], binary.LittleEndian, &metadataLen)
		if err != nil {
			yield(nil, errors.Wrapf(err, "failed to parse metadata length"))
			return
		}
		metadataBuf := make([]byte, metadataLen)
		_, err = io.ReadFull(r, metadataBuf)
		if err != nil {
			yield(nil, errors.Wrapf(err, "failed to read metadata"))
			return
		}
		var metadata map[string]*tensorMetadata
		err = json.Unmarshal(metadataBuf, &metadata)
		if err != nil {
			yield(nil, errors.Wrapf(err, "failed to parse json from metadata"))
			return
		}

		globalMetadata, found := metadata[safetensorsMetadataKey]
		if !found {
			yield(nil, errors.Errorf("unknown tensor format, expected in metadata[%q][\"format\"]", safetensorsMetadataKey))
			return
		}
		if globalMetadata.Format != "pt" {
			yield(nil, errors.Errorf("unsupported tensor format %q set in metadata[%q][\"format\"], only "+
				"supported format is \"pt\" (PyTorch)", globalMetadata.Format, safetensorsMetadataKey))
			return
		}

		// Sort metadata by their offsets -- and strip the global metadata.
		sortedMetadata := make([]*tensorMetadata, 0, len(metadata)-1)
		for tName, tData := range metadata {
			if tName == safetensorsMetadataKey {
				continue
			}
			tData.Name = tName
			if len(tData.Offsets) != 2 || tData.Offsets[1] <= tData.Offsets[0] {
				yield(nil, errors.Errorf("offset metadata[%q][\"data_offsets\"] invalid, "+
					"expected [start, end] but got %v instead", tData.Name, tData.Offsets))
				return
			}
			if tData.DType() == dtypes.InvalidDType {
				yield(nil, errors.Errorf("unsupported dtype %q in metadata[%q][\"dtype\"]",
					tData.DTypeName, tData.Name))
				return
			}
			size := uintptr(tData.Offsets[1] - tData.Offsets[0])
			if size != tData.Shape().Memory() {
				yield(nil, errors.Errorf("tensor shape %s is expected to require %d bytes, but metadata[%q][\"data_offset\"] "+
					"reserves only %d bytes",
					tData.Shape(), tData.Shape().Memory(), tData.Name, size))
			}
			sortedMetadata = append(sortedMetadata, tData)
		}
		slices.SortFunc(sortedMetadata, func(a, b *tensorMetadata) int {
			if a.Offsets[0] < b.Offsets[0] {
				return -1
			}
			return 1
		})
		if len(sortedMetadata) == 0 {
			// Empty safetensors!?
			yield(nil, errors.New(".safetensors file seems not to hold any tensors!? metadata was empty"))
			return
		}

		// Makes sure data is contiguous.
		var lastOffset uint64
		for _, tData := range sortedMetadata {
			if tData.Offsets[0] != lastOffset {
				yield(nil, errors.Errorf("offset for metadata[%q][\"data_offsets\"] not starting at 0 or not contiguous: extected %d, got %d"+
					tData.Name, lastOffset, tData.Offsets[0]))
				return
			}
			lastOffset = tData.Offsets[1]
		}

		// Read and yield tensors.
		for _, tData := range sortedMetadata {
			t := tensors.FromShape(tData.Shape())
			t.MutableBytes(func(data []byte) {
				_, err := io.ReadFull(r, data)
				if err != nil {
					yield(nil, errors.Errorf("tensor %q: failed to read %d bytes from .safetensors file", tData.Name, len(data)))
					return
				}
			})
			if !yield(&NamedTensor{tData.Name, t}, nil) {
				// Caller interrupted iterator.
				return
			}
		}
	}
}
