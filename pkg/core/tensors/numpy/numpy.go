// Package numpy allows one to read/write tensors to Python's NumPy npy and npz file formats.
//
// Experimental: this package (numpy) is fresh from the oven and not well tested. Please open an issue if you find any bugs.
package numpy

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path"
	"regexp"
	"strconv"
	"strings"

	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// FromNpyFile reads a .npy file and returns a tensors.Tensor.
func FromNpyFile(filePath string) (*tensors.Tensor, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to open .npy file %q", filePath)
	}
	defer func() { _ = file.Close() }()
	return FromNpyReader(file)
}

// FromNpyReader reads a .npy file from an io.Reader and returns a tensors.Tensor.
func FromNpyReader(r io.Reader) (*tensors.Tensor, error) {
	// Read and validate the magic string.
	magic := make([]byte, 6)
	if _, err := io.ReadFull(r, magic); err != nil {
		return nil, errors.Wrapf(err, "failed to read magic string")
	}
	if string(magic) != "\x93NUMPY" {
		return nil, errors.Errorf("invalid .npy file format: magic string mismatch")
	}

	// Read version.
	version := make([]byte, 2)
	if _, err := io.ReadFull(r, version); err != nil {
		return nil, errors.Wrapf(err, "failed to read version")
	}
	// major := version[0]
	// minor := version[1]
	// We are not using major/minor yet, but they are available here.

	// Read header length.
	var headerLen uint16
	if version[0] == 1 { // Version 1.0
		lenBytes := make([]byte, 2)
		if _, err := io.ReadFull(r, lenBytes); err != nil {
			return nil, errors.Wrapf(err, "failed to read header length (v1.0)")
		}
		headerLen = binary.LittleEndian.Uint16(lenBytes)
	} else if version[0] >= 2 { // Version 2.0 and above
		lenBytes := make([]byte, 4)
		if _, err := io.ReadFull(r, lenBytes); err != nil {
			return nil, errors.Wrapf(err, "failed to read header length (v2.0+)")
		}
		headerLen32 := binary.LittleEndian.Uint32(lenBytes)
		if headerLen32 > 0xFFFF {
			return nil, errors.Errorf("header length %d exceeds uint16 max", headerLen32)
		}
		headerLen = uint16(headerLen32)
	} else {
		return nil, errors.Errorf("unsupported .npy version: %d.%d", version[0], version[1])
	}

	// Read the header.
	headerBytes := make([]byte, headerLen)
	if _, err := io.ReadFull(r, headerBytes); err != nil {
		return nil, errors.Wrapf(err, "failed to read header")
	}
	header := string(headerBytes)

	// Parse header (this is a simplified parser).
	// Example: "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 2, 3), }"
	dtypeStr, shapeInts, fortranOrder, err := parseNpyHeader(header)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to parse .npy header")
	}

	// Create the tensor shape.
	dtype, err := npyDTypeToGomlx(dtypeStr)
	if err != nil {
		return nil, err
	}
	shape := shapes.Make(dtype, shapeInts...)

	// Create the tensor and read data into it.
	tensor := tensors.FromShape(shape)
	accessErr := tensor.MutableBytes(func(data []byte) {
		if !fortranOrder || shape.Rank() <= 1 {
			// Row-major (C-Order) order used by GoMLX: we just copy over the data.
			_, err = io.ReadFull(r, data)
			if err != nil {
				err = errors.Wrapf(err, "failed to read tensor data (expected %d bytes)", len(data))
			}

		} else {
			fortranData := make([]byte, len(data))
			_, err = io.ReadFull(r, fortranData)
			if err != nil {
				err = errors.Wrapf(err, "failed to read tensor data (expected %d bytes)", len(data))
			}

			fortranStrides := make([]int, shape.Rank())
			{
				stride := 1
				for axis, dim := range shape.Dimensions {
					fortranStrides[axis] = stride
					stride *= dim
				}
			}

			cOrderIdx := 0
			dtypeSize := dtype.Size()
			var fortranOrderIdx int
			for _, indices := range shape.Iter() {
				fortranOrderIdx = 0
				for axis, axisIdx := range indices {
					fortranOrderIdx += axisIdx * fortranStrides[axis]
				}
				fortranOrderIdx *= dtypeSize

				// Copy the data from Fortran order to C-order.
				copy(data[cOrderIdx:cOrderIdx+dtypeSize], fortranData[fortranOrderIdx:fortranOrderIdx+dtypeSize])

				// cOrder is being scanned linearly, so just increment it by one DType.
				cOrderIdx += dtypeSize
			}
		}
	})
	if accessErr != nil {
		return nil, accessErr
	}
	if err != nil {
		err2 := tensor.FinalizeAll()
		if err2 != nil {
			klog.Errorf("Failed to finalize tensor while handling error: %+v", err2)
		}
		return nil, err
	}

	// Check endianness of the data:
	// NumPy's '<' prefix in descr indicates little-endian, '>' big-endian, '|' not applicable.
	if strings.HasPrefix(dtypeStr, ">") {
		// Need to byte swap for big-endian data if the target is little-endian.
		// This is a complex topic and depends on the specific dtype.
		// Not supported yet.
		return nil, errors.Errorf(
			"big-endian .npy files ('%s') are not fully supported for all dtypes yet -- open a GoMLX issue if you need support for this",
			dtypeStr,
		)
	}
	return tensor, nil
}

func FortranToCLayout(dtypeSize int, dims []int, fortranData []byte, cData []byte) error {
	if dtypeSize <= 0 {
		return fmt.Errorf("dtypeSize must be positive, got %d", dtypeSize)
	}

	// Calculate the total number of elements and validate buffer sizes.
	totalElements := 1
	for _, d := range dims {
		if d == 0 {
			// If any dimension is zero, the tensor is empty.
			totalElements = 0
			break
		}
		totalElements *= d
	}

	expectedBytes := totalElements * dtypeSize
	if len(fortranData) != expectedBytes {
		return fmt.Errorf("fortranData has incorrect size: got %d bytes, want %d", len(fortranData), expectedBytes)
	}
	if len(cData) != expectedBytes {
		return fmt.Errorf("cData has incorrect size: got %d bytes, want %d", len(cData), expectedBytes)
	}
	if totalElements == 0 {
		return nil // Nothing to do for an empty tensor.
	}

	// coordinates will hold the N-dimensional index of an element, e.g., (row, col)
	coordinates := make([]int, len(dims))

	// Loop through the destination C-order array sequentially.
	for cIndex := 0; cIndex < totalElements; cIndex++ {
		// --- Step 1: Calculate the N-dimensional coordinate from the C-order index ---
		// This converts a flat row-major index back to its tensor coordinates.
		tempIndex := cIndex
		for i := len(dims) - 1; i >= 0; i-- {
			dim := dims[i]
			coordinates[i] = tempIndex % dim
			tempIndex /= dim
		}

		// --- Step 2: Calculate the Fortran-order index from the N-D coordinate ---
		// This converts tensor coordinates to a flat column-major index.
		fortranIndex := 0
		multiplier := 1
		for i := 0; i < len(dims); i++ {
			fortranIndex += coordinates[i] * multiplier
			multiplier *= dims[i]
		}

		// --- Step 3: Copy the element bytes ---
		srcOffset := fortranIndex * dtypeSize
		dstOffset := cIndex * dtypeSize
		copy(cData[dstOffset:dstOffset+dtypeSize], fortranData[srcOffset:srcOffset+dtypeSize])
	}

	return nil
}

// parseNpyHeader extracts dtype, shape, and fortran_order from the .npy header string.
// This is a very simplified parser and not robust for all .npy header variations.
func parseNpyHeader(header string) (dtype string, shape []int, fortranOrder bool, err error) {
	// descr
	reDescr := regexp.MustCompile(`'descr'\s*:\s*'([^']*)'`)
	mDescr := reDescr.FindStringSubmatch(header)
	if len(mDescr) < 2 {
		err = errors.Errorf("could not find 'descr' in header: %q", header)
		return
	}
	dtype = mDescr[1]

	// fortran_order
	reFortran := regexp.MustCompile(`'fortran_order'\s*:\s*(True|False)`)
	mFortran := reFortran.FindStringSubmatch(header)
	if len(mFortran) < 2 {
		err = errors.Errorf("could not find 'fortran_order' in header: %q", header)
		return
	}
	fortranOrder = mFortran[1] == "True"

	// shape
	reShape := regexp.MustCompile(`'shape'\s*:\s*\(([^)]*)\)`)
	mShape := reShape.FindStringSubmatch(header)
	if len(mShape) < 2 {
		err = errors.Errorf("could not find 'shape' in header: %q", header)
		return
	}
	shapeStr := strings.TrimSpace(mShape[1])
	if shapeStr == "" { // Scalar or 0-dim array
		shape = []int{}
	} else {
		parts := strings.Split(shapeStr, ",")
		shape = make([]int, 0, len(parts))
		for _, p := range parts {
			p = strings.TrimSpace(p)
			if p == "" { // Handles trailing comma like (10,)
				continue
			}
			val, pErr := strconv.Atoi(p)
			if pErr != nil {
				err = errors.Wrapf(pErr, "invalid shape value %q in header", p)
				return
			}
			shape = append(shape, val)
		}
	}
	// If the shape has only one empty element and the original string was like "()", it means scalar.
	// If the original string was like "(N,)", len(parts) might be 2 with the last empty.
	// The current logic with `if p == ""` handles trailing commas correctly for non-empty dtypes.
	// For scalar `()`, `shapeStr` will be empty, leading to `shape = []int{}`.
	// For 1D array `(N,)`, `shapeStr` will be `N,`, split into `["N", ""]`, `shape = []int{N}`.
	return
}

// npyDTypeToGomlx converts a NumPy dtype string to a dtypes.DType.
func npyDTypeToGomlx(npyType string) (dtypes.DType, error) {
	// Normalize: remove the byte order prefix if present for basic types.
	// E.g., '<f4' -> 'f4', '>i8' -> 'i8'.
	// More complex types might need the byte order info.
	if len(npyType) > 0 && (npyType[0] == '<' || npyType[0] == '>' || npyType[0] == '=') {
		// Keep byte order for multi-byte types if needed for swapping later
		// For now, we simplify, assuming GoMLX expects little-endian for its numeric types
	}

	switch {
	case npyType == "|b1", npyType == "?", npyType == "b1": // boolean
		return dtypes.Bool, nil
	case strings.HasSuffix(npyType, "i1"): // int8
		return dtypes.Int8, nil
	case strings.HasSuffix(npyType, "u1"): // uint8
		return dtypes.Uint8, nil
	case strings.HasSuffix(npyType, "i2"): // int16
		return dtypes.Int16, nil
	case strings.HasSuffix(npyType, "u2"): // uint16
		return dtypes.Uint16, nil
	case strings.HasSuffix(npyType, "i4"): // int32
		return dtypes.Int32, nil
	case strings.HasSuffix(npyType, "u4"): // uint32
		return dtypes.Uint32, nil
	case strings.HasSuffix(npyType, "i8"): // int64
		return dtypes.Int64, nil
	case strings.HasSuffix(npyType, "u8"): // uint64
		return dtypes.Uint64, nil
	case strings.HasSuffix(npyType, "f2"): // float16 (half)
		// NumPy's f2 is IEEE 754 half-precision.
		return dtypes.Float16, nil
	case strings.HasSuffix(npyType, "f4"): // float32
		return dtypes.F32, nil
	case strings.HasSuffix(npyType, "f8"): // float64
		return dtypes.F64, nil
	case strings.HasSuffix(npyType, "c8"): // complex64 (two float32s)
		return dtypes.C64, nil
	case strings.HasSuffix(npyType, "c16"): // complex128 (two float64s)
		return dtypes.C128, nil
	// Add other types like strings, unicode, void (structured) as needed.
	// Example: S (bytes), U (unicode)
	// case strings.HasPrefix(npyType, "S"): // Byte string
	//	return dtypes.ByteStr, nil // Assuming GoMLX has a ByteStr DType or similar
	default:
		return dtypes.InvalidDType, errors.Errorf("unsupported NumPy dtype: %s", npyType)
	}
}

// FromNpzFile reads a .npz file and returns a map of tensor names to tensors.Tensor.
func FromNpzFile(filePath string) (map[string]*tensors.Tensor, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to open .npz file %q", filePath)
	}
	defer func() { _ = file.Close() }()

	// Need file info for zip.NewReader, which requires a ReaderAt and size.
	info, err := file.Stat()
	if err != nil {
		return nil, errors.Wrapf(err, "failed to stat .npz file %q", filePath)
	}

	return FromNpzReader(file, info.Size())
}

// FromNpzReader reads a .npz file from an io.ReaderAt and size,
// returning a map of tensor names to tensors.Tensor.
// .npz files are zip archives, so we need io.ReaderAt.
func FromNpzReader(r io.ReaderAt, size int64) (map[string]*tensors.Tensor, error) {
	zipReader, err := zip.NewReader(r, size)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to create zip reader for `.npz`")
	}

	results := make(map[string]*tensors.Tensor)
	for _, f := range zipReader.File {
		// For extra safety.
		cleanPath := path.Clean(f.Name)
		if path.IsAbs(cleanPath) || strings.HasPrefix(cleanPath, "..") {
			return nil, errors.Errorf(
				"invalid (malicious?) path in .npz archive: %q (normalized to %q)",
				f.Name,
				cleanPath,
			)
		}

		if !strings.HasSuffix(f.Name, ".npy") {
			// .npz might contain other metadata files, skip non .npy files.
			// Or, log a warning if unexpected.
			continue
		}

		rc, err := f.Open()
		if err != nil {
			return nil, errors.Wrapf(err, "failed to open %q within .npz", f.Name)
		}

		tensor, err := FromNpyReader(rc)
		_ = rc.Close() // Close the individual file reader.
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to read tensor %q from .npz", f.Name)
		}

		// The name in the map should be the filename without the .npy extension.
		name := strings.TrimSuffix(f.Name, ".npy")
		results[name] = tensor
	}

	return results, nil
}

// ToNpyWriter serializes a tensors.Tensor to an io.Writer in .npy format.
func ToNpyWriter(tensor *tensors.Tensor, w io.Writer) error {
	shape := tensor.Shape()
	dtype, err := gomlxDTypeToNpy(shape.DType)
	if err != nil {
		return err
	}

	// Construct header dictionary string.
	// {'descr': '...', 'fortran_order': False, 'shape': (...), }
	// Note the trailing comma in shape tuple for 1D arrays, and no comma for 0D.
	var shapeTuple string
	if shape.Rank() == 0 {
		shapeTuple = "()"
	} else if shape.Rank() == 1 {
		shapeTuple = fmt.Sprintf("(%d,)", shape.Dimensions[0])
	} else {
		dimsStr := make([]string, shape.Rank())
		for i, dim := range shape.Dimensions {
			dimsStr[i] = strconv.Itoa(dim)
		}
		shapeTuple = fmt.Sprintf("(%s)", strings.Join(dimsStr, ", "))
	}

	headerDict := fmt.Sprintf("{'descr': '%s', 'fortran_order': False, 'shape': %s, }",
		dtype, shapeTuple)

	// Calculate padding for the header.
	// The header must be padded with spaces to be a multiple of 64 bytes for version 2.0,
	// or 16 bytes for version 1.0, and terminate with a newline.
	// We'll use version 1.0 for simplicity here. Pad to 16 bytes.
	// The header length field itself and the magic string + version also take space.
	// Magic (6) + Version (2) + HeaderLen (2) = 10 bytes for v1.0 preamble.
	// Let's target a specific preamble + header length that's a multiple of 16.
	var headerBuf bytes.Buffer
	headerBuf.WriteString(headerDict)
	// Pad with spaces. Minimum one newline.
	for (10+headerBuf.Len()+1)%16 != 0 { // +1 for the newline
		headerBuf.WriteByte(' ')
	}
	headerBuf.WriteByte('\n') // Terminating newline.

	headerBytes := headerBuf.Bytes()
	headerLen := uint16(len(headerBytes))

	// Write magic string and version (1.0).
	if _, err := w.Write([]byte("\x93NUMPY")); err != nil {
		return errors.Wrapf(err, "failed to write magic string")
	}
	if _, err := w.Write([]byte{1, 0}); err != nil { // Version 1.0
		return errors.Wrapf(err, "failed to write version")
	}

	// Write header length.
	headerLenBytes := make([]byte, 2)
	binary.LittleEndian.PutUint16(headerLenBytes, headerLen)
	if _, err := w.Write(headerLenBytes); err != nil {
		return errors.Wrapf(err, "failed to write header length")
	}

	// Write header.
	if _, err := w.Write(headerBytes); err != nil {
		return errors.Wrapf(err, "failed to write header")
	}

	// Write tensor data.
	var writeErr error
	err = tensor.ConstBytes(func(data []byte) {
		_, writeErr = w.Write(data)
		if writeErr != nil {
			writeErr = errors.Wrapf(err, "failed to write tensor data")
		}
	})
	if err != nil {
		return err
	}
	return writeErr
}

// gomlxDTypeToNpy converts a dtypes.DType to a NumPy dtype string.
// It assumes little-endian ('<') for multi-byte types.
func gomlxDTypeToNpy(dtype dtypes.DType) (string, error) {
	switch dtype {
	case dtypes.Bool:
		return "?", nil // or "|b1"
	case dtypes.Int8:
		return "<i1", nil
	case dtypes.Uint8:
		return "<u1", nil
	case dtypes.Int16:
		return "<i2", nil
	case dtypes.Uint16:
		return "<u2", nil
	case dtypes.Int32:
		return "<i4", nil
	case dtypes.Uint32:
		return "<u4", nil
	case dtypes.Int64:
		return "<i8", nil
	case dtypes.Uint64:
		return "<u8", nil
	case dtypes.F32:
		return "<f4", nil
	case dtypes.F64:
		return "<f8", nil
	case dtypes.C64:
		return "<c8", nil
	case dtypes.C128:
		return "<c16", nil
	case dtypes.Float16:
		// NumPy's f2 is IEEE 754 half-precision.
		return "<f2", nil
	case dtypes.BFloat16:
		// NumPy doesn't have a standard bfloat16 dtype string recognized by all tools.
		// Sometimes 'V2' (void type of 2 bytes) with custom interpretation or
		// a structured dtype like ('<V2', {'names':['bfloat16'], 'formats':['<u2'] ...})
		// For simplicity, we'll error or use a convention that might not be universally portable.
		return "", errors.Errorf("BF16 to .npy dtype string is non-standard; direct mapping not straightforward")
	default:
		return "", errors.Errorf("unsupported GoMLX DType for .npy: %s", dtype.String())
	}
}

// ToNpyFile serializes a tensors.Tensor to a .npy file.
func ToNpyFile(tensor *tensors.Tensor, filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return errors.Wrapf(err, "failed to create .npy file")
	}
	defer func() { _ = file.Close() }()
	return ToNpyWriter(tensor, file)
}

// ToNpzFile serializes a map of tensors to a .npz file.
func ToNpzFile(tensorsMap map[string]*tensors.Tensor, filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return errors.Wrapf(err, "failed to create .npz file")
	}
	defer func() { _ = file.Close() }()

	zipWriter := zip.NewWriter(file)
	defer func() { _ = zipWriter.Close() }()

	for name, tensor := range tensorsMap {
		npyName := name + ".npy"
		fileWriter, err := zipWriter.Create(npyName)
		if err != nil {
			return errors.Wrapf(err, "failed to create '%s' in .npz archive", npyName)
		}
		if err := ToNpyWriter(tensor, fileWriter); err != nil {
			return errors.WithMessagef(err, "failed to write tensor '%s' to .npz archive", name)
		}
	}
	return nil
}

// ToNpzWriter serializes a map of tensors to an io.Writer as a .npz archive.
// Note: Creating a valid zip archive generally requires io.WriterAt for the central directory,
// which is written at the end. A simple io.Writer is not enough for zip.NewWriter directly
// if the output needs to be streamed without seeking (e.g., network stream).
// However, zip.Writer itself buffers and then writes. If 'w' is an os.File, it works.
// For a generic io.Writer that doesn't support Seek, this would be more complex, potentially
// requiring writing the entire NPZ to a bytes.Buffer first, then to the writer.
func ToNpzWriter(tensorsMap map[string]*tensors.Tensor, w io.Writer) error {
	zipWriter := zip.NewWriter(w)
	for name, tensor := range tensorsMap {
		npyName := name + ".npy"
		fileWriter, err := zipWriter.Create(npyName)
		if err != nil {
			return errors.Wrapf(err, "failed to create %q in .npz archive", npyName)
		}
		if err := ToNpyWriter(tensor, fileWriter); err != nil {
			return errors.WithMessagef(err, "failed to write tensor %q to .npz archive", name)
		}
	}
	err := zipWriter.Close()
	if err != nil {
		return errors.Wrapf(err, "failed to close zip archive")
	}
	return nil
}
