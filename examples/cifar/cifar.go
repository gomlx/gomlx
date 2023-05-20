/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

// Package cifar provides a library of tools to download and manipulate Cifar-10 dataset.
// Information about it in https://www.cs.toronto.edu/~kriz/cifar.html
package cifar

import (
	"fmt"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/pkg/errors"
	"image"
	"io"
	"math/rand"
	"os"
	"path"
	"reflect"
)

const (
	C10Url     = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
	C10TarName = "cifar-10-binary.tar.gz"
	C10SubDir  = "cifar-10-batches-bin"

	C100Url     = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"
	C100TarName = "cifar-100-binary.tar.gz"
	C100SubDir  = "cifar-100-binary"

	// NumExamples is the total number of examples, including training and testing.
	// The value is the same for both, Cifar-10 and Cifar-100.
	NumExamples = 60000

	// NumTrainExamples is the number of examples reserved for training, the starting ones.
	// The value is the same for both, Cifar-10 and Cifar-100.
	NumTrainExamples = 50000

	// NumTestExamples is the number of examples reserved for testing, the last ones.
	// The value is the same for both, Cifar-10 and Cifar-100.
	NumTestExamples = 10000
)

// Width, Height and Depth are the dimensions of the images, the same
// for Cifar-10 and Cifar-100.
const (
	Width  int = 32
	Height int = 32
	Depth  int = 3
)

func DownloadCifar10(baseDir string) error {
	return data.DownloadAndUntarIfMissing(C10Url, baseDir, C10TarName, C10SubDir,
		"c4a38c50a1bc5f3a1c5537f2155ab9d68f9f25eb1ed8d9ddda3db29a59bca1dd")
}

func DownloadCifar100(baseDir string) error {
	return data.DownloadAndUntarIfMissing(C100Url, baseDir, C100TarName, C100SubDir,
		"58a81ae192c23a4be8b1804d68e518ed807d710a4eb253b1f2a199162a40d8ec")
}

var (
	C10Labels = [10]string{"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"}

	C100CoarseLabels = [20]string{"aquatic_mammals", "fish", "flowers", "food_containers", "fruit_and_vegetables",
		"household_electrical_devices", "household_furniture", "insects", "large_carnivores",
		"large_man-made_outdoor_things", "large_natural_outdoor_scenes", "large_omnivores_and_herbivores",
		"medium_mammals", "non-insect_invertebrates", "people", "reptiles", "small_mammals", "trees", "vehicles_1",
		"vehicles_2"}
	C100FineLabels = [100]string{"apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle",
		"bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
		"chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
		"dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard", "lamp",
		"lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
		"mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain",
		"plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal",
		"shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower",
		"sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
		"turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"}
)

const C10ExamplesPerFile = 10000
const imageSizeBytes = Height * Width * Depth

func convertBytesToTensor[T shapes.GoFloat](image []byte, imagesT *tensor.Local, exampleNum int) error {
	var t T
	if shapes.DTypeForType(reflect.TypeOf(t)) != imagesT.DType() {
		return errors.Errorf("trying to convert to dtype %s from go type %t", imagesT.DType(), any(t))
	}
	tensorData := imagesT.Data().([]T)
	tensorPos := exampleNum * imageSizeBytes
	for h := 0; h < Height; h++ {
		for w := 0; w < Width; w++ {
			for d := 0; d < Depth; d++ {
				value := T(image[d*(Height*Width)+h*(Width)+w]) / T(255)
				tensorData[tensorPos] = value
				tensorPos++
			}
		}
	}
	return nil
}

// LoadCifar10 into 2 tensors of the given DType: images with given dtype and shaped
// [NumExamples=60000, Height=32, Width=32, Depth=3], and labels shaped
// [NumExamples=60000, 1] of Int64.
// The first 50k examples are for training, and the last 10k for testing.
// Only Float32 and Float64 dtypes are supported for now.
func LoadCifar10(baseDir string, dtype shapes.DType) (images, labels *tensor.Local, err error) {
	baseDir = data.ReplaceTildeInDir(baseDir)

	// Allocate the tensor.
	images = tensor.FromShape(shapes.Make(dtype, NumExamples, Height, Width, Depth))
	labels = tensor.FromShape(shapes.Make(shapes.I64, NumExamples, 1))
	labelsData := labels.Data().([]int)

	var labelImageBytes [imageSizeBytes + 1]byte
	for fileIdx := 0; fileIdx < 6; fileIdx++ {
		dataFile := path.Join(baseDir, C10SubDir, fmt.Sprintf("data_batch_%d.bin", fileIdx+1))
		if fileIdx == 5 {
			dataFile = path.Join(baseDir, C10SubDir, "test_batch.bin")
		}
		f, err := os.Open(dataFile)
		if err != nil {
			return nil, nil, errors.Wrapf(err, "opening data file %q", dataFile)
		}
		fileStart := fileIdx * C10ExamplesPerFile
		for inFileIdx := 0; inFileIdx < C10ExamplesPerFile; inFileIdx++ {
			exampleIdx := fileStart + inFileIdx
			bytesRead, err := f.Read(labelImageBytes[:])
			if err != nil {
				return nil, nil, errors.Wrapf(err, "reading example %d (out of %d) from %q", inFileIdx, C10ExamplesPerFile, dataFile)
			}
			if bytesRead != len(labelImageBytes) {
				return nil, nil, errors.Errorf("read only %d bytes reading example %d (out of %d) from %q, wanted %d bytes", bytesRead, inFileIdx, C10ExamplesPerFile, dataFile, len(labelImageBytes))
			}
			switch dtype {
			case shapes.Float64:
				err = convertBytesToTensor[float64](labelImageBytes[1:], images, exampleIdx)
			case shapes.Float32:
				err = convertBytesToTensor[float32](labelImageBytes[1:], images, exampleIdx)
			default:
				return nil, nil, errors.Errorf("DType %s not supported", dtype)
			}
			if err != nil {
				return nil, nil, errors.WithMessagef(err, "failed converting bytes to tensor of %s", dtype)
			}
			labelsData[exampleIdx] = int(labelImageBytes[0])
		}
	}
	return
}

// LoadCifar100 into 2 tensors of the given DType: images with given dtype and shaped
// [NumExamples=60000, Height=32, Width=32, Depth=3], and labels shaped
// [NumExamples=60000, 1] of Int64.
// The first 50k examples are for training, and the last 10k for testing.
// Only Float32 and Float64 dtypes are supported for now.
func LoadCifar100(baseDir string, dtype shapes.DType) (images, labels *tensor.Local, err error) {
	baseDir = data.ReplaceTildeInDir(baseDir)

	// Allocate the tensor.
	images = tensor.FromShape(shapes.Make(dtype, NumExamples, Height, Width, Depth))
	labels = tensor.FromShape(shapes.Make(shapes.I64, NumExamples, 1))
	labelsData := labels.Data().([]int)

	var labelImageBytes [imageSizeBytes + 2]byte
	dataFiles := []string{"train.bin", "test.bin"}
	exampleIdx := 0
	for _, fileName := range dataFiles {
		dataFile := path.Join(baseDir, C100SubDir, fileName)
		f, err := os.Open(dataFile)
		if err != nil {
			return nil, nil, errors.Wrapf(err, "opening data file %q", dataFile)
		}
		for inFileIdx := 0; true; inFileIdx++ {
			bytesRead, err := f.Read(labelImageBytes[:])
			if bytesRead == 0 && err == io.EOF {
				break
			}
			if err != nil {
				return nil, nil, errors.WithMessagef(err, "reading example %d from %q", inFileIdx, dataFile)
			}
			if bytesRead != len(labelImageBytes) {
				return nil, nil, errors.Errorf("read only %d bytes reading example %d from %q, wanted %d bytes", bytesRead, inFileIdx, dataFile, len(labelImageBytes))
			}
			switch dtype {
			case shapes.Float64:
				err = convertBytesToTensor[float64](labelImageBytes[2:], images, exampleIdx)
			case shapes.Float32:
				err = convertBytesToTensor[float32](labelImageBytes[2:], images, exampleIdx)
			default:
				return nil, nil, errors.Errorf("DType %s not supported", dtype)
			}
			if err != nil {
				return nil, nil, errors.Wrapf(err, "failed converting bytes to tensor of %s", dtype)
			}
			labelsData[exampleIdx] = int(labelImageBytes[1]) // Take the fine-label (and discard the coarse-label).
			exampleIdx++
		}
	}
	return
}

func ConvertToGoImage(images *tensor.Local, exampleNum int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, Width, Height))
	tensorData := reflect.ValueOf(images.Data())
	tensorPos := exampleNum * imageSizeBytes
	floatT := reflect.TypeOf(float64(0))
	for h := 0; h < Height; h++ {
		for w := 0; w < Width; w++ {
			for d := 0; d < Depth; d++ {
				v := tensorData.Index(tensorPos)
				f := v.Convert(floatT).Interface().(float64)
				tensorPos++
				img.Pix[h*img.Stride+w*4+d] = uint8(f * 255)
			}
			img.Pix[h*img.Stride+w*4+3] = uint8(255) // Alpha channel.
		}
	}
	return img
}

// Dataset provides a stream of data for Train/Test partitions. It implements `gomlx/ml/train.Dataset`
//
// It yields indices and labels. Indices have to be converted to images
// using Dataset.GatherImagesGraph -- the images themselves are stored in device
// tensor, presumably faster.
//
// If eval is set, it will yield batches sequentially (not-shuffled) until the end
// of the partition (Train/Test), and then yield io.EOF.
//
// If eval is false, it will yield batches of random samples (with replacement)
// indefinitely.
//
// See example usage in the demo/ subpackage.
type Dataset struct {
	name           string     // Dataset name
	source         DataSource // C10 or C100
	scopeName      string     // Scope of the variable used to store values.
	images, labels tensor.Tensor
	batchSize      int

	start, end int  // Range of examples being considered
	random     bool // Whether to yield infinitely random examples.
	position   int  // Position to yield from, if not random.

	// indicesShape
	indicesShape shapes.Shape
}

func (ds *Dataset) Name() string {
	return ds.name
}

// Shapes implements train.Dataset. It returns the shape of the batch of indices.
func (ds *Dataset) Shapes() (inputs []shapes.Shape, labels shapes.Shape) {
	return []shapes.Shape{ds.indicesShape}, ds.indicesShape
}

// Yield implements train.Dataset. It returns a pointer to the Dataset itself as `spec`.
func (ds *Dataset) Yield() (spec any, inputs, labels []tensor.Tensor, err error) {
	indicesT := tensor.FromShape(ds.indicesShape)
	indicesRef := indicesT.AcquireData()
	defer indicesRef.Release()
	indices := tensor.FlatFromRef[int](indicesRef)
	if ds.random {
		// Training, always use random indices.
		for ii := range indices {
			indices[ii] = rand.Intn(ds.end-ds.start) + ds.start
		}
	} else {
		// Testing, generate indices sequentially.
		if ds.position+ds.batchSize > ds.end {
			// Past beyond end, simply issues an io.EOF.
			return nil, nil, nil, io.EOF
		}
		for ii := 0; ii < ds.batchSize; ii++ {
			indices[ii] = ds.position + ii
		}
		ds.position += ds.batchSize
	}

	// Gather labels.
	allLabelsRef := ds.labels.Local().AcquireData()
	defer allLabelsRef.Release()
	allLabelsData := tensor.FlatFromRef[int](allLabelsRef)

	batchLabelsT := tensor.FromShape(ds.indicesShape)
	batchLabelsRef := batchLabelsT.AcquireData()
	defer batchLabelsRef.Release()
	batchLabelsData := tensor.FlatFromRef[int](batchLabelsRef)
	for ii, index := range indices {
		batchLabelsData[ii] = allLabelsData[index]
	}

	return ds, []tensor.Tensor{indicesT}, []tensor.Tensor{batchLabelsT}, nil
}

// Reset implements train.Dataset and, for an evaluation dataset, restarts it.
func (ds *Dataset) Reset() {
	// Test examples start just after the train examples.
	ds.position = ds.start
}

// GatherImagesGraph converts a batch of indices to a batch of images
// (shape=[batchSize, 32, 32, 3]).
// Since datasets hold all the Cifar (10 or 100) data, a train or test dataset will work for
// indices from either.
func (ds *Dataset) GatherImagesGraph(ctx *context.Context, batchIndices *Node) (batchImages *Node) {
	g := batchIndices.Graph()
	if !g.Ok() {
		return g.InvalidNode()
	}
	if !ctx.Ok() {
		g.SetError(ctx.Error())
		return g.InvalidNode()
	}

	// Create constant with whole dataset. Since this will be called more than once within the model, as well
	// as if multiple Dataset objects are using the same data, we mark context as unchecked (for reuse).
	// This relies on all dataset sharing the same large images tensor, that is, `ds.images`
	// are all the same tensor. In the future if this is not the case, one should create
	// a context scope per underlying images tensor.
	ctx = ctx.Checked(false).In(ds.scopeName)
	allImages := ctx.VariableWithValue(fmt.Sprintf("dataset_images_%d", ds.source), ds.images).SetTrainable(false)
	imagesNode := allImages.ValueGraph(g)
	batchImages = Gather(imagesNode, batchIndices)
	return
}

// DataSource refers to Cifar-10 (C10) or Cifar-100 (C100).
type DataSource int

const (
	C10 DataSource = iota
	C100
)

// Partition refers to the train or test partitions of the datasets.
type Partition int

const (
	Train Partition = iota
	Test
)

type ImagesAndLabels struct {
	images, labels *tensor.Local
}

var (
	// Cache of loaded data: one per DataSource, per DType.
	imagesAndLabelsCache = [2]map[shapes.DType]ImagesAndLabels{make(map[shapes.DType]ImagesAndLabels), make(map[shapes.DType]ImagesAndLabels)}

	// How do download & load data: one per DataSource.
	downloadFunctions = [2]func(baseDir string) error{DownloadCifar10, DownloadCifar100}
	loadFunctions     = [2]func(baseDir string, dType shapes.DType) (images, labels *tensor.Local, err error){LoadCifar10, LoadCifar100}
	scopeNames        = [2]string{"c10dataset", "c100dataset"}
)

func ResetCache() {
	imagesAndLabelsCache = [2]map[shapes.DType]ImagesAndLabels{make(map[shapes.DType]ImagesAndLabels), make(map[shapes.DType]ImagesAndLabels)}
}

// NewDataset returns a Dataset for the training data, which implements train.Dataset and hence can be used
// by train.Trainer methods.
//
// It automatically downloads the data from the web, and then loads the data into memory, if it hasn't been
// loaded yet. It caches the result, so multiple Dataset's can be created without any extra costs in time/memory.
func NewDataset(name, baseDir string, source DataSource, dtype shapes.DType, partition Partition, batchSize int, eval bool) (*Dataset, error) {
	if source > C100 {
		return nil, errors.Errorf("Invalid source value %d, only C10 or C100 accepted", source)
	}
	imagesAndLabels, found := imagesAndLabelsCache[source][dtype]
	if !found {
		err := downloadFunctions[source](baseDir)
		if err != nil {
			return nil, errors.WithMessagef(err, "Creating a new Dataset")
		}
		imagesAndLabels.images, imagesAndLabels.labels, err = loadFunctions[source](baseDir, dtype)
		if err != nil {
			return nil, errors.WithMessagef(err, "Creating a new Dataset")
		}
		imagesAndLabelsCache[source][dtype] = imagesAndLabels
	}
	ds := &Dataset{
		name:         name,
		source:       source,
		images:       imagesAndLabels.images,
		labels:       imagesAndLabels.labels,
		batchSize:    batchSize,
		scopeName:    scopeNames[source],
		random:       !eval,
		indicesShape: shapes.Make(shapes.Int64, batchSize, 1),
	}

	switch partition {
	case Train:
		ds.start, ds.end = 0, NumTrainExamples
	case Test:
		ds.start, ds.end = NumTrainExamples, NumExamples
	default:
		return nil, errors.Errorf("Invalid partition value %d, only Train or Test accepted", source)
	}
	ds.Reset()
	return ds, nil
}
