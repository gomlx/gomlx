// Package fnn implements a generic FNN (Feedforward Neural Network) with various configurations.
// It should suffice for the common cases and can be extended as needed.
//
// It also provides support for various hyperparameter configuration -- so the defaults can be given by
// the context parameters.
//
// E.g: A FNN for a multi-class classification model with NumClasses classes.
//
//	func MyMode(ctx *context.Context, inputs []*Node) (outputs []*Node) {
//		x := inputs[0]
//		logits := fnn.New(ctx.In("model"), x, NumClasses).
//			NumHiddenLayers(3).
//			Activation("swish").
//			Dropout(0.3).
//			UseResidual(true).
//			Done()
//		return []*Node{logits}
//	}
package fnn

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers/regularizers"
)

// ParamNumHiddenLayers is the hyperparameter that defines the default number of hidden layers.
// The default is 0, so no hidden layers.
ParamNumHiddenLayers = "fnn_num_hidden_layers"

// ParamNumHiddenNodes is the hyperparameter that defines the default number of hidden nodes for KAN hidden layers.
// Default is 10.
ParamNumHiddenNodes = "fnn_num_hidden_nodes"

// Config is created with New and can be configured with its methods, or simply setting the corresponding
// hyperparameters in the context.
type Config struct {
	ctx                             *context.Context
	input                           *Node
	numOutputNodes                  int
	numHiddenLayers, numHiddenNodes int
	activation, normalization       string
	useBias, useResidual            bool

	regularizer, biasRegulariser regularizers.Regularizer

	err error
}

func New() *Config {

}
