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
	"os"
	"os/exec"
	"path"
	"regexp"
	"strconv"
	"strings"

	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/support/fsutil"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
	"github.com/schollz/progressbar/v3"
	_ "github.com/schollz/progressbar/v3"
	"k8s.io/klog/v2"
)

// Hdf5Contents is a map of all the datasets present in the HDF5 file. The key is the path
// built from the concatenation of the "group" (how HDF5 calls directories or folders) with
// the dataset name, separated by a "/" character.
type Hdf5Contents map[string]*Hdf5Dataset

// Hdf5Dataset has (some of) the metadata about a dataset (but not the data itself). The
// dataset "DATATYPE" and "DATASPACE" fields are converted to the equivalent GoMLX `shapes.Shape`.
type Hdf5Dataset struct {
	FilePath, GroupPath, RawHeader string
	DType                          dtypes.DType
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
		groupPath := match[1]
		//in case someone inserted args into dataset name('--help', etc)
		if strings.HasPrefix(groupPath, "-") {
			return nil, errors.Errorf("invalid dataset name starting with '-': %q", groupPath)
		}
		contents[groupPath] = &Hdf5Dataset{
			FilePath:  filePath,
			GroupPath: groupPath,
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
		if ds.DType == dtypes.InvalidDType {
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
func DtypeForH5T(h5type string) (dtype dtypes.DType) {
	switch h5type {
	case "H5T_IEEE_F32LE", "H5T_IEEE_F32BE":
		return dtypes.Float32
	case "H5T_IEEE_F64LE", "H5T_IEEE_F64BE":
		return dtypes.Float64
	case "H5T_STD_I32LE", "H5T_STD_I32BE":
		return dtypes.Int32
	case "H5T_STD_I64LE", "H5T_STD_I64BE":
		return dtypes.Int64
	}
	return dtypes.InvalidDType
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
	klog.V(2).Infof("using h5dump from %q", binPath)
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
		klog.Warningf("Failed to remove temporary file %q used to extract HDF5 dataset: %+v", tmpFile.Name(), newErr)
	}
	return
}

// ToTensor reads the HDF5 dataset into GoMLX's tensors.Tensor.
func (ds *Hdf5Dataset) ToTensor() (tensor *tensors.Tensor, err error) {
	if !ds.Shape.Ok() {
		return nil, errors.Errorf("no shape information from HDF5 dataset, can't convert to tensor")
	}
	loadedData, err := ds.Load()
	if err != nil {
		return
	}
	tensor = tensors.FromShape(ds.Shape)
	accessErr := tensor.MutableBytes(func(localData []byte) {
		if len(loadedData) != len(localData) {
			err = errors.Errorf("for shape %s: loaded %d bytes, but tensor uses %d bytes -- not sure how to load it!?",
				ds.Shape, len(loadedData), len(localData))
			return
		}
		copy(localData, loadedData)
	})
	if accessErr != nil {
		return nil, err
	}
	if err != nil {
		return nil, err
	}
	return tensor, nil
}

// UnpackToTensorsConfig holds the configuration created by UnpackToTensors, to unpack HDF5 files into a directory
// structure with the individual tensors saved in GoMLX format.
//
// The targetDir must not yet exist.
type UnpackToTensorsConfig struct {
	h5Path, targetDir, tmpDir string
	showProgressBar           bool
	dirPermissions            os.FileMode
	keepTemporary             bool
}

// UnpackToTensors unpacks tensors from an HDF5 file (typically with an '.h5' extension). It will generate
// one file per tensor, in subdirectories under `targetDirectory` mimicking the groups ("folder" or "directory"
// in HDF5 lingo) structure within the HDF5 file.
//
// UnpackToTensors returns a configuration structure, that can be further configured. Once done configuring, call
// Done, and it will do the unpacking.
//
// Tensors are serialized using `tensor.Local.Save`, and can be read with `tensor.Local.Load`.
//
// Example: unpack `weights.h5` file into `/my/target/directory`.
//
//	err := UnpackToTensors("weights.h5", "/my/target/directory").ProgressBar().Done()
func UnpackToTensors(targetDir, h5Path string) *UnpackToTensorsConfig {
	return &UnpackToTensorsConfig{
		h5Path:         h5Path,
		targetDir:      targetDir,
		dirPermissions: 0755,
	}
}

// ProgressBar configures a progressbar to be displayed during the unpacking.
//
// It modifies the configuration and returns itself, so configuration calls can be cascaded.
func (c *UnpackToTensorsConfig) ProgressBar() *UnpackToTensorsConfig {
	c.showProgressBar = true
	return c
}

// FilePermissions configures the file permissions used for the creation of the directories and files.
// Default is `os.FileMode(0755)`.
//
// It modifies the configuration and returns itself, so configuration calls can
// be cascaded.
func (c *UnpackToTensorsConfig) FilePermissions(perm os.FileMode) *UnpackToTensorsConfig {
	c.dirPermissions = perm
	return c
}

// KeepTemporary configures unpacking to keep the temporary directory with
// (potentially partially) unpackaged files.
//
// The default behavior is, if an error occurs, remove all temporary files.
//
// It modifies the configuration and returns itself, so configuration calls can
// be cascaded.
func (c *UnpackToTensorsConfig) KeepTemporary() *UnpackToTensorsConfig {
	c.keepTemporary = true
	return c
}

// Done actually does the unpacking according to the configuration. See details in UnpackToTensor.
//
// It unpacks first to a temporary directory and renames it at the very end if the unpackaging was
// successful.
//
// If an error occurs, by default, it will remove the temporary directory with unpackaged
// files generated thus far. You can change this behavior with KeepTemporary.
func (c *UnpackToTensorsConfig) Done() (err error) {
	// Sanity checking.
	if fsutil.MustFileExists(c.targetDir) {
		err = errors.Errorf("target directory %q already exists -- remove it or move it away first ?", c.targetDir)
		return
	}

	// Parse H5 file.
	h5, err := ParseFile(c.h5Path)
	if err != nil {
		return
	}

	// Create temporary directory.
	baseDir := path.Dir(c.targetDir)
	err = os.MkdirAll(baseDir, c.dirPermissions)
	if err != nil {
		err = errors.Wrapf(err, "can't create base directory %q where to unpack the HDF5 file to", baseDir)
		return
	}
	tmpDir, err := os.MkdirTemp(baseDir, path.Base(c.targetDir)+".")
	if err != nil {
		err = errors.Wrapf(err, "can't create temporary directory under %q to unpack the HDF5 file to", baseDir)
		return
	}

	// ProgressBar: collect total size.
	var bar *progressbar.ProgressBar
	if c.showProgressBar {
		var totalSize uintptr
		for _, ds := range h5 {
			if !ds.Shape.Ok() {
				continue
			}
			totalSize += ds.Shape.Memory()
		}
		bar = progressbar.DefaultBytesSilent(int64(totalSize), "")
	}

	// Prepare clean up in case of error.
	defer func() {
		if bar != nil {
			_ = bar.Finish()

		}
		if tmpDir != "" {
			newErr := os.RemoveAll(tmpDir)
			if newErr != nil {
				klog.Errorf(
					"UnpackToTensors(%q, %q): error while cleaning up temporary directory %q: %v",
					c.targetDir,
					c.h5Path,
					tmpDir,
					newErr,
				)
			}
		}
	}()

	// Loop over HDF5 "datasets", converting each to a corresponding file with the GoMLX tensor.
	for key, ds := range h5 {
		if !ds.Shape.Ok() {
			// This hdf5 dataset is not compatible with a tensor, skipping.
			klog.Infof("UnpackToTensors(%q, %q): skipping dataset %q not parsed as tensor", c.targetDir, c.h5Path, key)
			continue
		}
		local, newErr := ds.ToTensor()
		err = newErr
		if err != nil {
			return
		}

		dsPath := path.Join(tmpDir, key)
		dsDir := path.Dir(dsPath)
		err = os.MkdirAll(dsDir, c.dirPermissions)
		if err != nil {
			err = errors.Wrapf(
				err,
				"UnpackToTensors(%q, %q): can't create sub-directory %q for UnpackToTensors",
				c.targetDir,
				c.h5Path,
				dsDir,
			)
			return
		}
		err = local.Save(dsPath)
		if err != nil {
			err = errors.WithMessagef(err, "UnpackToTensors(%q, %q)", c.targetDir, c.h5Path)
			return
		}

		if bar != nil {
			_ = bar.Add64(int64(ds.Shape.Memory()))
			fmt.Printf("\r%s", bar.String())
		}
	}
	if bar != nil {
		_ = bar.Finish()
		bar = nil
	}

	// Move temporary directory to target dir
	err = os.Rename(tmpDir, c.targetDir)
	if err != nil {
		err = errors.Wrapf(
			err,
			"UnpackToTensors(%q, %q): failed to rename temporary dir %q with unpacked tensors to target %q",
			c.targetDir,
			c.h5Path,
			c.tmpDir,
			c.targetDir,
		)
		return
	}
	tmpDir = "" // Indicates to deferred function above there is no need for clean up.
	return
}
