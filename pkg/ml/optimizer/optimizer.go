package optimizer

import (
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/support/polymorphicjson"
)

// Interface implemented by optimizers.
//
// Use the concrete wrapper Optimizer for a serializable generic Optimizer that implements this interface.
type Interface interface {
	polymorphicjson.JSONIdentifiable // To make it serializable when using the Optimizer object.

	// UpdateGraph calculates the updates to the variables (weights) of the model needed for one training step.
	//
	// The model is a user-defined object, usually a pointer to a structure. See package model for more details of
	// valid model structures.
	//
	// The loss must be a scalar value.
	UpdateGraph(model any, loss *Node)

	// Clear deletes all temporary variables used by the optimizer.
	//
	// This may be used for a model to be used by inference to save space, or if the training should be reset
	// for some other reason.
	Clear()
}

// Optimizer wraps an optimizer.Interface and provides (de-)serialization.
// It implements itself the optimizer.Interface by calling the wrapped interface.
type Optimizer struct {
	polymorphicjson.Wrapper[Interface]
}

// JSONTags implements the polymorphicjson.JSONIdentifiable interface, used for (de-)serialization.
func (o *Optimizer) JSONTags() (interfaceName, concreteType string) {
	return o.Wrapper.Value.JSONTags()
}

// UpdateGraph calculates the updates to the variables (weights) of the model needed for one training step.
func (o *Optimizer) UpdateGraph(model any, loss *Node) {
	o.Wrapper.Value.UpdateGraph(model, loss)
}

// Clear deletes all temporary variables used by the optimizer.
//
// This may be used for a model to be used by inference to save space, or if the training should be reset
// for some other reason.
func (o *Optimizer) Clear() {
	o.Wrapper.Value.Clear()
}
