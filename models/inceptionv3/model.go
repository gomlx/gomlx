package inceptionv3

import (
	"fmt"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/pkg/errors"
	"os"
	"path"
)

// ModelGraph implements the pre-trained inception model.
//
// Before calling this, call DownloadAndUnpackWeights to make sure the weights are there.
//
// Parameters:
//   - ctx: context.Context where variables are created and loaded. Variables
//     will be re-used if they were already created before in the current scope.
//     That means one can call ModelGraph more than once, and have the same
//     model be used for more than one input -- for instance for 2-tower models.
//     To instantiate more than one model with different weights, just use
//     the context in different scope.
//   - baseDir: Directory where the weights for the model were downloaded and unpacked.
//     Same value passed to DownloadAndUnpackWeights.
func ModelGraph(ctx *context.Context, baseDir string, image *Node) *Node {
	// conv2DWithBatchNorm(ctx, kw, img_input, 32, 3, 3, strides=(2, 2), padding="valid"):
	kw := &kerasWeights{
		baseDir:      data.ReplaceTildeInDir(baseDir),
		channelsAxis: image.Rank() - 1, // Only supported value for now (`ChannelsAfter` configuration).
	}

	// The implementation follows closely the definition in
	// https://github.com/keras-team/keras/blob/v2.12.0/keras/applications/inception_v3.py
	x := image
	x = conv2DWithBatchNorm(ctx, kw, x, 32, 3, 3, []int{2, 2}, false)
	x = conv2DWithBatchNorm(ctx, kw, x, 32, 3, 3, nil, false)
	x = conv2DWithBatchNorm(ctx, kw, x, 64, 3, 3, nil, true)
	x = MaxPool(x).ChannelsAfter().Window(3).Strides(2).NoPadding().Done()

	x = conv2DWithBatchNorm(ctx, kw, x, 80, 1, 1, nil, false)
	x = conv2DWithBatchNorm(ctx, kw, x, 192, 3, 3, nil, false)
	x = MaxPool(x).ChannelsAfter().Window(3).Strides(2).NoPadding().Done()

	// Mixed sizes convolutions 0: 35x35x256 or 7x7x256 (depending on image size)
	branch1x1 := conv2DWithBatchNorm(ctx, kw, x, 64, 1, 1, nil, true)
	_ = branch1x1

	branch5x5 := conv2DWithBatchNorm(ctx, kw, x, 48, 1, 1, nil, true)
	branch5x5 = conv2DWithBatchNorm(ctx, kw, branch5x5, 64, 5, 5, nil, true)

	branch3x3Dbl := conv2DWithBatchNorm(ctx, kw, x, 64, 1, 1, nil, true)
	branch3x3Dbl = conv2DWithBatchNorm(ctx, kw, branch3x3Dbl, 96, 3, 3, nil, true)
	branch3x3Dbl = conv2DWithBatchNorm(ctx, kw, branch3x3Dbl, 96, 3, 3, nil, true)

	branchPool := MeanPool(x).Window(3).Strides(1).PadSame().Done()
	branchPool = conv2DWithBatchNorm(ctx, kw, branchPool, 32, 1, 1, nil, true)
	x = Concatenate([]*Node{branch1x1, branch5x5, branch3x3Dbl, branchPool}, kw.channelsAxis)

	// Mixed convolutions 1: 35x35x288 or 7x7x288
	branch1x1 = conv2DWithBatchNorm(ctx, kw, x, 64, 1, 1, nil, true)

	branch5x5 = conv2DWithBatchNorm(ctx, kw, x, 48, 1, 1, nil, true)
	branch5x5 = conv2DWithBatchNorm(ctx, kw, branch5x5, 64, 5, 5, nil, true)

	branch3x3Dbl = conv2DWithBatchNorm(ctx, kw, x, 64, 1, 1, nil, true)
	branch3x3Dbl = conv2DWithBatchNorm(ctx, kw, branch3x3Dbl, 96, 3, 3, nil, true)
	branch3x3Dbl = conv2DWithBatchNorm(ctx, kw, branch3x3Dbl, 96, 3, 3, nil, true)

	branchPool = MeanPool(x).Window(3).Strides(1).PadSame().Done()
	branchPool = conv2DWithBatchNorm(ctx, kw, branchPool, 64, 1, 1, nil, true)
	x = Concatenate([]*Node{branch1x1, branch5x5, branch3x3Dbl, branchPool}, kw.channelsAxis)

	// Mixed convolutions 2: 35x35x288 or 7x7x288
	branch1x1 = conv2DWithBatchNorm(ctx, kw, x, 64, 1, 1, nil, true)

	branch5x5 = conv2DWithBatchNorm(ctx, kw, x, 48, 1, 1, nil, true)
	branch5x5 = conv2DWithBatchNorm(ctx, kw, branch5x5, 64, 5, 5, nil, true)

	branch3x3Dbl = conv2DWithBatchNorm(ctx, kw, x, 64, 1, 1, nil, true)
	branch3x3Dbl = conv2DWithBatchNorm(ctx, kw, branch3x3Dbl, 96, 3, 3, nil, true)
	branch3x3Dbl = conv2DWithBatchNorm(ctx, kw, branch3x3Dbl, 96, 3, 3, nil, true)

	branchPool = MeanPool(x).Window(3).Strides(1).PadSame().Done()
	branchPool = conv2DWithBatchNorm(ctx, kw, branchPool, 64, 1, 1, nil, true)
	x = Concatenate([]*Node{branch1x1, branch5x5, branch3x3Dbl, branchPool}, kw.channelsAxis)

	// Mixed convolutions 3:
	branch3x3 := conv2DWithBatchNorm(ctx, kw, x, 384, 3, 3, []int{2, 2}, false)

	branch3x3Dbl = conv2DWithBatchNorm(ctx, kw, x, 64, 1, 1, nil, true)
	branch3x3Dbl = conv2DWithBatchNorm(ctx, kw, branch3x3Dbl, 96, 3, 3, nil, true)
	branch3x3Dbl = conv2DWithBatchNorm(ctx, kw, branch3x3Dbl, 96, 3, 3, []int{2, 2}, false)

	branchPool = MaxPool(x).ChannelsAfter().Window(3).Strides(2).NoPadding().Done()
	x = Concatenate([]*Node{branch3x3, branch3x3Dbl, branchPool}, kw.channelsAxis)

	// Mixed convolutions 4: 768 channels
	branch1x1 = conv2DWithBatchNorm(ctx, kw, x, 192, 1, 1, nil, true)

	branch7x7 := conv2DWithBatchNorm(ctx, kw, x, 128, 1, 1, nil, true)
	branch7x7 = conv2DWithBatchNorm(ctx, kw, branch7x7, 128, 1, 7, nil, true)
	branch7x7 = conv2DWithBatchNorm(ctx, kw, branch7x7, 192, 7, 1, nil, true)

	branch7x7Dbl := conv2DWithBatchNorm(ctx, kw, x, 128, 1, 1, nil, true)
	branch7x7Dbl = conv2DWithBatchNorm(ctx, kw, branch7x7Dbl, 128, 7, 1, nil, true)
	branch7x7Dbl = conv2DWithBatchNorm(ctx, kw, branch7x7Dbl, 128, 1, 7, nil, true)
	branch7x7Dbl = conv2DWithBatchNorm(ctx, kw, branch7x7Dbl, 128, 7, 1, nil, true)
	branch7x7Dbl = conv2DWithBatchNorm(ctx, kw, branch7x7Dbl, 192, 1, 7, nil, true)

	branchPool = MeanPool(x).Window(3).Strides(1).PadSame().Done()
	branchPool = conv2DWithBatchNorm(ctx, kw, branchPool, 192, 1, 1, nil, true)

	x = Concatenate([]*Node{branch1x1, branch7x7, branch7x7Dbl, branchPool}, kw.channelsAxis)

	// Mixed convolutions 5 & 6: 768 channels
	for ii := 0; ii < 2; ii++ {
		branch1x1 = conv2DWithBatchNorm(ctx, kw, x, 192, 1, 1, nil, true)

		branch7x7 = conv2DWithBatchNorm(ctx, kw, x, 160, 1, 1, nil, true)
		branch7x7 = conv2DWithBatchNorm(ctx, kw, branch7x7, 160, 1, 7, nil, true)
		branch7x7 = conv2DWithBatchNorm(ctx, kw, branch7x7, 192, 7, 1, nil, true)

		branch7x7Dbl = conv2DWithBatchNorm(ctx, kw, x, 160, 1, 1, nil, true)
		branch7x7Dbl = conv2DWithBatchNorm(ctx, kw, branch7x7Dbl, 160, 7, 1, nil, true)
		branch7x7Dbl = conv2DWithBatchNorm(ctx, kw, branch7x7Dbl, 160, 1, 7, nil, true)
		branch7x7Dbl = conv2DWithBatchNorm(ctx, kw, branch7x7Dbl, 160, 7, 1, nil, true)
		branch7x7Dbl = conv2DWithBatchNorm(ctx, kw, branch7x7Dbl, 192, 1, 7, nil, true)

		branchPool = MeanPool(x).Window(3).Strides(1).PadSame().Done()
		branchPool = conv2DWithBatchNorm(ctx, kw, branchPool, 192, 1, 1, nil, true)
		x = Concatenate([]*Node{branch1x1, branch7x7, branch7x7Dbl, branchPool}, kw.channelsAxis)
	}

	// Mixed convolutions 7: 768 channels
	branch1x1 = conv2DWithBatchNorm(ctx, kw, x, 192, 1, 1, nil, true)

	branch7x7 = conv2DWithBatchNorm(ctx, kw, x, 192, 1, 1, nil, true)
	branch7x7 = conv2DWithBatchNorm(ctx, kw, branch7x7, 192, 1, 7, nil, true)
	branch7x7 = conv2DWithBatchNorm(ctx, kw, branch7x7, 192, 7, 1, nil, true)

	branch7x7Dbl = conv2DWithBatchNorm(ctx, kw, x, 192, 1, 1, nil, true)
	branch7x7Dbl = conv2DWithBatchNorm(ctx, kw, branch7x7Dbl, 192, 7, 1, nil, true)
	branch7x7Dbl = conv2DWithBatchNorm(ctx, kw, branch7x7Dbl, 192, 1, 7, nil, true)
	branch7x7Dbl = conv2DWithBatchNorm(ctx, kw, branch7x7Dbl, 192, 7, 1, nil, true)
	branch7x7Dbl = conv2DWithBatchNorm(ctx, kw, branch7x7Dbl, 192, 1, 7, nil, true)

	branchPool = MeanPool(x).Window(3).Strides(1).PadSame().Done()
	branchPool = conv2DWithBatchNorm(ctx, kw, branchPool, 192, 1, 1, nil, true)
	x = Concatenate([]*Node{branch1x1, branch7x7, branch7x7Dbl, branchPool}, kw.channelsAxis)

	// Mixed convolutions 8: 768 channels
	branch3x3 = conv2DWithBatchNorm(ctx, kw, x, 192, 1, 1, nil, true)
	branch3x3 = conv2DWithBatchNorm(ctx, kw, branch3x3, 320, 3, 3, []int{2, 2}, false)

	branch7x7x3 := conv2DWithBatchNorm(ctx, kw, x, 192, 1, 1, nil, true)
	branch7x7x3 = conv2DWithBatchNorm(ctx, kw, branch7x7x3, 192, 1, 7, nil, true)
	branch7x7x3 = conv2DWithBatchNorm(ctx, kw, branch7x7x3, 192, 7, 1, nil, true)
	branch7x7x3 = conv2DWithBatchNorm(ctx, kw, branch7x7x3, 192, 3, 3, []int{2, 2}, false)

	branchPool = MaxPool(x).Window(3).Strides(2).NoPadding().Done()
	x = Concatenate([]*Node{branch3x3, branch7x7x3, branchPool}, kw.channelsAxis)

	// Mixed convolutions 9 & 10: 2048 channels
	for ii := 0; ii < 2; ii++ {
		branch1x1 = conv2DWithBatchNorm(ctx, kw, x, 320, 1, 1, nil, true)

		branch3x3 = conv2DWithBatchNorm(ctx, kw, x, 384, 1, 1, nil, true)
		branch3x3Branch1 := conv2DWithBatchNorm(ctx, kw, branch3x3, 384, 1, 3, nil, true)
		branch3x3Branch2 := conv2DWithBatchNorm(ctx, kw, branch3x3, 384, 3, 1, nil, true)
		branch3x3 = Concatenate([]*Node{branch3x3Branch1, branch3x3Branch2}, kw.channelsAxis)

		branch3x3Dbl = conv2DWithBatchNorm(ctx, kw, x, 448, 1, 1, nil, true)
		branch3x3Dbl = conv2DWithBatchNorm(ctx, kw, branch3x3Dbl, 384, 3, 3, nil, true)
		branch3x3DblBranch1 := conv2DWithBatchNorm(ctx, kw, branch3x3Dbl, 384, 1, 3, nil, true)
		branch3x3DblBranch2 := conv2DWithBatchNorm(ctx, kw, branch3x3Dbl, 384, 3, 1, nil, true)
		branch3x3Dbl = Concatenate([]*Node{branch3x3DblBranch1, branch3x3DblBranch2}, kw.channelsAxis)

		branchPool = MeanPool(x).Window(3).Strides(1).PadSame().Done()
		branchPool = conv2DWithBatchNorm(ctx, kw, branchPool, 192, 1, 1, nil, true)
		x = Concatenate([]*Node{branch1x1, branch3x3, branch3x3Dbl, branchPool}, kw.channelsAxis)
	}

	return x
}

// conv2DWithBatchNorm adds a 2D convolution, followed by batch normalization and an activation. In addition,
// it reads the weights for the layers (convolution and batch normalization) from the downloaded `.h5` file
// with the InceptionV3 pre-trained model.
func conv2DWithBatchNorm(ctx *context.Context, kw *kerasWeights, x *Node, kernelFilters, kernelHeight, kernelWidth int,
	strides []int, padding bool) (output *Node) {
	g := x.Graph()
	output = g.InvalidNode()
	if !g.Ok() {
		return
	}

	// 2D Convolution:
	ctxWithWeights := kw.ReadNextConv2D(ctx, g) // Create a new context scope and read weights from `.h5` file.
	convCfg := layers.Convolution(ctxWithWeights, x).CurrentScope().ChannelsAfter().
		Filters(kernelFilters).UseBias(false).KernelSizePerDim(kernelHeight, kernelWidth)
	if len(strides) > 0 {
		convCfg = convCfg.StridePerDim(strides...)
	}
	if padding {
		convCfg = convCfg.PadSame()
	} else {
		convCfg = convCfg.NoPadding()
	}
	x = convCfg.Done()
	if !g.Ok() {
		return
	}

	// Batch Normalization:
	ctxWithWeights = kw.ReadNextBatchNormalization(ctx, g) // Create a new context scope and read weights from `.h5` file.
	x = layers.BatchNormalization(ctxWithWeights, x, kw.channelsAxis).CurrentScope().Scale(false).Done()
	if !g.Ok() {
		return
	}

	// Activation:
	x = layers.Relu(x)

	output = x
	return
}

// kerasWeights manages the retrieval of keras weights.
//
// It understands and translates the naming used by keras and maps to the
// correct unpacked tensors. It requires layers to be "read" in the exact
// same order as they were created in Keras.
//
// See ReadNextConv2D, ReadNextBatchNorm
type kerasWeights struct {
	baseDir                     string
	channelsAxis                int
	conv2dCount, batchNormCount int
}

// LoadTensorToVariable loads the tensor from a file named tensorFileName, under the unpacking directory and
// moves contents to a variable named `variableName`.
//
// Any errors are set in the graph.
func (kw *kerasWeights) LoadTensorToVariable(ctx *context.Context, graph *Graph, tensorFileName, variableName string) {
	//fmt.Printf("Loading %s to %s::%s\n", tensorFileName, ctx.Scope(), variableName)
	if ctx.InspectVariable(ctx.Scope(), variableName) != nil {
		// Assume it's already correctly loaded.
		return
	}
	tensorPath := path.Join(kw.baseDir, UnpackedWeightsName, tensorFileName)
	f, err := os.Open(tensorPath)
	if err != nil {
		graph.SetError(errors.Wrapf(err, "inceptionv3.ModelGraph(): failed to read weights from %q", tensorPath))
		return
	}
	local, err := tensor.Load(f)
	if err != nil {
		graph.SetError(errors.Wrapf(err, "inceptionv3.ModelGraph(): failed to read weights from %q", tensorPath))
		return
	}

	// We don't need the value, since the layer will re-load it.
	_ = ctx.VariableWithValue(variableName, local)
}

// ReadNextConv2D enters a new scope and initializes it with the pre-trained weights for the next Conv2D layer.
//
// It returns the modified scope to be used in `layers.Convolution`.
func (kw *kerasWeights) ReadNextConv2D(ctx *context.Context, graph *Graph) (ctxInScope *context.Context) {
	ctxInScope = ctx
	if !graph.Ok() {
		return
	}

	// Set scope name to something similar to the original model layer names (cosmetic only).
	if kw.conv2dCount == 0 {
		ctxInScope = ctx.In("conv2d")
	} else {
		ctxInScope = ctx.In(fmt.Sprintf("conv2d_%d", kw.conv2dCount))
	}
	kw.conv2dCount += 1

	// h5 names start with 1 instead of 0 (!!)
	h5Name := fmt.Sprintf("conv2d_%d/conv2d_%d/kernel:0", kw.conv2dCount, kw.conv2dCount)
	kw.LoadTensorToVariable(ctxInScope, graph, h5Name, "weights")
	if !graph.Ok() {
		return
	}

	// Context has variable set already, mark context for reuse.
	ctxInScope = ctxInScope.Reuse()
	return
}

// ReadNextBatchNormalization enters a new scope and initializes it with the pre-trained weights for the next
// batch normalization layer.
//
// It returns the modified scope to use for `BatchNormalization`.
func (kw *kerasWeights) ReadNextBatchNormalization(ctx *context.Context, graph *Graph) (ctxInScope *context.Context) {
	ctxInScope = ctx
	if !graph.Ok() {
		return
	}

	// Set scope name to something similar to the original model layer names (cosmetic only).
	if kw.batchNormCount == 0 {
		ctxInScope = ctx.In("batch_normalization")
	} else {
		ctxInScope = ctx.In(fmt.Sprintf("batch_normalization_%d", kw.batchNormCount))
	}
	kw.batchNormCount += 1

	// h5 names start with 1 instead of 0 (!!)
	h5Group := fmt.Sprintf("batch_normalization_%d/batch_normalization_%d/", kw.conv2dCount, kw.conv2dCount)
	kw.LoadTensorToVariable(ctxInScope, graph, h5Group+"moving_mean:0", "mean")
	if !graph.Ok() {
		return
	}
	kw.LoadTensorToVariable(ctxInScope, graph, h5Group+"moving_variance:0", "variance")
	if !graph.Ok() {
		return
	}
	kw.LoadTensorToVariable(ctxInScope, graph, h5Group+"beta:0", "offset")
	if !graph.Ok() {
		return
	}

	// Context will have mixed usage: some variables will be reused, some (like the "avg_weight") will be dynamically
	// created. So we mark context as unchecked.
	ctxInScope = ctxInScope.Checked(false)
	return
}
