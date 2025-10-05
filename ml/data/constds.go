package data

import (
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/pkg/core/tensors"
)

// NewConstantDataset returns a dataset that yields always the scalar 0.
//
// This is useful when training something that generates its own inputs and labels -- like trying to
// approximate a function with another function.
//
// It loops indefinitely.
func NewConstantDataset() train.Dataset {
	return &constDataset{}
}

// constDataset is a dataset that yields always the scalar 0.
//
// This is useful when training something that generates its own inputs and labels -- like trying to
// approximate a function with another function.
type constDataset struct{}

var _ train.Dataset = &constDataset{}

func (ds *constDataset) Name() string {
	return "constDataset"
}

func (ds *constDataset) Reset() {}

func (ds *constDataset) Yield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {
	spec = ds
	inputs = []*tensors.Tensor{tensors.FromScalar(int32(0.0))}
	labels = []*tensors.Tensor{tensors.FromScalar(int32(0.0))}
	return
}
