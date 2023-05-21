// Package hdf5 provides a trivial API to access HDF5 file contents.
//
// It requires the `hdf5-tools` (a deb package) installed in the system, more specifically the
// `h5dump` binary.
//
// It is basic but provides the necessary functionality to list the contents and extract
// the binary contents.
package hdf5

import (
	"bytes"
	"fmt"
	"github.com/golang/glog"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/pkg/errors"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
)

// Hdf5Contents is a map of all the datasets present in the HDF5 file. The key is the path
// built from the concatenation of the "group" (how HDF5 calls directories or folders) with
// the dataset name, separated by a "/" character.
type Hdf5Contents map[string]*Hdf5Dataset

// Hdf5Dataset has (some of) the metadata about a dataset (but not the data itself). The
// dataset "DATATYPE" and "DATASPACE" fields are converted to the equivalent GoMLX `shapes.Shape`.
type Hdf5Dataset struct {
	FilePath, GroupPath, RawHeader string
	DType                          shapes.DType
	Shape                          shapes.Shape
}

const H5DumpBinary = "h5dump"

// ParseFile in filePath as an HDF5 file and returns map of contents.
//
// It requires the `hdf5-tools` (a deb package) installed in the system, more specifically the
// `h5dump` binary.
func ParseFile(filePath string) (contents Hdf5Contents, err error) {
	// Check whether the file exists.
	_, err = os.Stat(filePath)
	if err != nil {
		err = errors.Wrapf(err, "cannot access HDF5 file in path %q", filePath)
		return
	}

	// List the contents of the filePath.
	contentsBytes, err := execH5Dump("--contents", filePath)
	if err != nil {
		return
	}
	matches := regexpH5Datasets.FindAllStringSubmatch(string(contentsBytes), -1)
	contents = make(Hdf5Contents, len(matches))
	for _, match := range matches {
		contents[match[1]] = &Hdf5Dataset{
			FilePath:  filePath,
			GroupPath: match[1],
		}
	}

	// Read header for datasets.
	headerArgs := make([]string, 0, len(contents)+2)
	headerArgs = append(headerArgs, "--header")
	for key := range contents {
		headerArgs = append(headerArgs, "--dataset="+key)
	}
	headerArgs = append(headerArgs, filePath)
	headerBytes, err := execH5Dump(headerArgs...)
	rawDatasetHeaders := strings.Split(string(headerBytes), "DATASET")
	if len(rawDatasetHeaders)-1 != len(contents) {
		err = errors.Errorf("failed to parse dataset headers for %q: expected %d DATASET, got %d",
			filePath, len(contents), len(rawDatasetHeaders)-1)
		return
	}
datasetHeaders:
	for _, part := range rawDatasetHeaders[1:] {
		matches := regexpH5DatasetHeaderName.FindStringSubmatch(part)
		if len(matches) != 2 {
			err = errors.Errorf("failed to parse dataset headers for %q: got %q", filePath, part)
			return
		}
		key := matches[1]
		ds, found := contents[key]
		if !found {
			err = errors.Errorf("unknown headers for %q: got %q", filePath, part)
			return
		}
		ds.RawHeader = "DATASET" + part

		// Parse data type.
		matches = regexpH5DatasetHeaderDataType.FindStringSubmatch(part)
		if len(matches) != 2 {
			// DType not parseable.
			continue
		}
		ds.DType = DtypeForH5T(matches[1])
		if ds.DType == shapes.InvalidDType {
			continue datasetHeaders
		}

		// Parse DATASPACE
		matches = regexpH5DatasetHeaderDataSpace.FindStringSubmatch(part)
		if len(matches) != 4 {
			// Shape not parseable.
			fmt.Printf("DATASPACE nor parsed: %s\n", part)
			continue datasetHeaders
		}
		switch matches[1] {
		case "SCALAR":
			ds.Shape = shapes.Make(ds.DType)
		case "SIMPLE":
			dimsStr := matches[3]
			dimsParts := strings.Split(dimsStr, ",")
			dims := make([]int, 0, len(dimsParts))
			for _, dimStr := range dimsParts {
				dimStr = strings.TrimSpace(dimStr)
				dim, numErr := strconv.Atoi(dimStr)
				if numErr != nil {
					fmt.Printf("failed to parse dimension in DATASPACE: %q", part)
					continue datasetHeaders
				}
				dims = append(dims, dim)
			}
			ds.Shape = shapes.Make(ds.DType, dims...)

		default:
			// Shape isn't supported:
			fmt.Printf("DATASPACE type unknown: %s\n", part)
			continue datasetHeaders
		}
	}
	return
}

var (
	regexpH5Datasets               = regexp.MustCompile(`\s+dataset\s+(/.*)\n`)
	regexpH5DatasetHeaderName      = regexp.MustCompile(`\s+"(.*?)" \{\n`)
	regexpH5DatasetHeaderDataType  = regexp.MustCompile(`\s+DATATYPE\s+(\w.*?)\n`)
	regexpH5DatasetHeaderDataSpace = regexp.MustCompile(`\s+DATASPACE\s+(\w+)(\s+\{\s+\((.*?)\).*?)?\n`)
)

// DtypeForH5T returns the DType corresponding to known HDF5 types. If not know/supported, returns
// invalid dtype.
func DtypeForH5T(h5type string) (dtype shapes.DType) {
	switch h5type {
	case "H5T_IEEE_F32LE", "H5T_IEEE_F32BE":
		return shapes.F32
	case "H5T_IEEE_F64LE", "H5T_IEEE_F64BE":
		return shapes.F64
	case "H5T_STD_I32LE", "H5T_STD_I32BE":
		return shapes.I32
	case "H5T_STD_I64LE", "H5T_STD_I64BE":
		return shapes.I64
	}
	return shapes.InvalidDType
}

// execH5Dump executes `h5dump`, and handles errors.
func execH5Dump(args ...string) (output []byte, err error) {
	binPath, err := findBinPath()
	if err != nil {
		return
	}
	cmd := exec.Command(binPath, args...)
	if cmd.Err != nil {
		err = errors.Wrapf(err, "cannot execute %q required to access HDF5 file", cmd)
		return
	}
	var stdoutBuf, stderrBuf bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdoutBuf, &stderrBuf
	err = cmd.Run()
	if err != nil {
		err = errors.Wrapf(err, "failed executing %q to access HDF5 file", cmd)
		err = errors.WithMessagef(err, "STDERR captured:\n%s\n", string(stderrBuf.Bytes()))
		return
	}
	output = stdoutBuf.Bytes()
	return
}

func findBinPath() (binPath string, err error) {
	// Find h5dump.
	binPath, err = exec.LookPath(H5DumpBinary)
	if err != nil {
		err = errors.Wrapf(err, "cannot find `h5dump` binary in PATH, needed to parse HDF5 "+
			"format files \"(extension \".h5\") -- please install package hdf5-tools, which usually "+
			"holds `h5dump`")
		return
	}
	glog.V(2).Infof("using h5dump from %q", binPath)
	return
}

func (ds *Hdf5Dataset) Load() (rawContent []byte, err error) {
	tmpFile, err := os.CreateTemp("", "hdf5_dataset")
	if err == nil {
		err = tmpFile.Close()
	}
	if err != nil {
		err = errors.Wrapf(err, "failed to create temporary file to extract HDF5 dataset")
		return
	}
	_, err = execH5Dump("--dataset="+ds.GroupPath, "--binary=NATIVE", "--output="+tmpFile.Name(), ds.FilePath)
	if err != nil {
		return
	}
	rawContent, err = os.ReadFile(tmpFile.Name())
	if err != nil {
		err = errors.Wrapf(err, "failed to read from temporary file %q to extract HDF5 dataset", tmpFile.Name())
		return
	}
	if newErr := os.Remove(tmpFile.Name()); newErr != nil {
		glog.Warningf("Failed to remove temporary file %q used to extract HDF5 dataset: %+v", tmpFile.Name(), newErr)
	}
	return
}
