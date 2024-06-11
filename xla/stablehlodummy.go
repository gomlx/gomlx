//go:build google3

package xla

import (
	"os"

	"github.com/pkg/errors"
)

// Dummy package for StableHLO interface when using "google3" build tag: this allows the dependency to the corresponding
// C++ code to be dropped.

// StableHLO is a dummy structure.
type StableHLO struct {
}

// Finalize implements Finalizer.
func (shlo *StableHLO) Finalize() {
}

// IsNil checks whether the computation is nil or it's C underlying object.
func (shlo *StableHLO) IsNil() bool {
	return true
}

// ToStableHLO returning a holder of the C++ object representing the StableHLO.
func (comp *Computation) ToStableHLO() (*StableHLO, error) {
	return nil, errors.Errorf("stablehlo disabled on google3")
}

// String generates a human-readable version of the StableHLO.
func (shlo *StableHLO) String() string {
	return "StableHLO dummy"
}

// StableHLOCurrentVersion returns the current version for the StableHLO library.
func StableHLOCurrentVersion() string {
	return "vDummy"
}

// SerializeWithVersion to bytecode that can presumably be used by PjRT and IREE, as well as
// embedded in one of the TensorFlow SavedModel formats.(??)
//
// It serializes to the given file descriptor, presumable a file opened for writing.
//
// For version, the usual is to use the value returned by StableHLOCurrentVersion.
func (shlo *StableHLO) SerializeWithVersion(fileDescriptor uintptr, version string) error {
	return errors.Errorf("stablehlo disabled on google3")
}

// Serialize to bytecode that can presumably be used by PjRT and IREE, as well as
// embedded in one of the TensorFlow SavedModel formats.(??)
//
// It serializes to the given file path.
func (shlo *StableHLO) Serialize(filePath string) error {
	f, err := os.Create(filePath)
	if err != nil {
		return errors.Wrapf(err, "Cannot create file %q to save StableHLO", filePath)
	}
	return shlo.SerializeWithVersion(f.Fd(), StableHLOCurrentVersion())
}

// NewStableHLOFromSerialized
func NewStableHLOFromSerialized(serialized []byte) (*StableHLO, error) {
	return nil, errors.Errorf("stablehlo disabled on google3")
}
