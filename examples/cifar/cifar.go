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
	"image"
	"io"
	"os"
	"path"
	"reflect"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/examples/downloader"
	. "github.com/gomlx/gomlx/internal/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/datasets"
	"github.com/gomlx/gomlx/pkg/support/fsutil"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
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
	return downloader.DownloadAndUntarIfMissing(C10Url, baseDir, C10TarName, C10SubDir,
		"c4a38c50a1bc5f3a1c5537f2155ab9d68f9f25eb1ed8d9ddda3db29a59bca1dd")
}

func DownloadCifar100(baseDir string) error {
	return downloader.DownloadAndUntarIfMissing(C100Url, baseDir, C100TarName, C100SubDir,
		"58a81ae192c23a4be8b1804d68e518ed807d710a4eb253b1f2a199162a40d8ec")
}

var (
	C10Labels = []string{"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"}

	C100CoarseLabels = []string{"aquatic_mammals", "fish", "flowers", "food_containers", "fruit_and_vegetables",
		"household_electrical_devices", "household_furniture", "insects", "large_carnivores",
		"large_man-made_outdoor_things", "large_natural_outdoor_scenes", "large_omnivores_and_herbivores",
		"medium_mammals", "non-insect_invertebrates", "people", "reptiles", "small_mammals", "trees", "vehicles_1",
		"vehicles_2"}
	C100FineLabels = []string{"apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle",
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

func convertBytesToTensor[T dtypes.GoFloat](image []byte, imagesT *tensors.Tensor, exampleNum int) error {
	var t T
	if dtypes.FromGoType(reflect.TypeOf(t)) != imagesT.DType() {
		return errors.Errorf("trying to convert to dtype %s from go type %t", imagesT.DType(), any(t))
	}
	tensors.MustMutableFlatData[T](imagesT, func(tensorData []T) {
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
	})
	return nil
}

// LoadCifar10 into 2 tensors of the given DType: images with given dtype and shaped
// [NumExamples=60000, Height=32, Width=32, Depth=3], and labels shaped
// [NumExamples=60000, 1] of Int64.
// The first 50k examples are for training, and the last 10k for testing.
// Only Float32 and Float64 dtypes are supported for now.
func LoadCifar10(
	backend backends.Backend,
	baseDir string,
	dtype dtypes.DType,
) (partitioned PartitionedImagesAndLabels) {
	baseDir = fsutil.MustReplaceTildeInDir(baseDir)

	// Allocate the tensor.
	images := tensors.FromShape(shapes.Make(dtype, NumExamples, Height, Width, Depth))
	labels := tensors.FromShape(shapes.Make(dtypes.Int64, NumExamples, 1))
	defer func() {
		// Free images and labels resources in accelerator (GPU) immediately (don't wait for GC).
		images.MustFinalizeAll()
		labels.MustFinalizeAll()
	}()
	tensors.MustMutableFlatData[int64](labels, func(labelsData []int64) {
		var labelImageBytes [imageSizeBytes + 1]byte
		for fileIdx := 0; fileIdx < 6; fileIdx++ {
			dataFile := path.Join(baseDir, C10SubDir, fmt.Sprintf("data_batch_%d.bin", fileIdx+1))
			if fileIdx == 5 {
				dataFile = path.Join(baseDir, C10SubDir, "test_batch.bin")
			}
			f, err := os.Open(dataFile)
			if err != nil {
				panic(errors.Wrapf(err, "opening data file %q", dataFile))
			}
			fileStart := fileIdx * C10ExamplesPerFile
			for inFileIdx := 0; inFileIdx < C10ExamplesPerFile; inFileIdx++ {
				exampleIdx := fileStart + inFileIdx
				bytesRead, err := f.Read(labelImageBytes[:])
				if err != nil {
					panic(errors.Wrapf(err, "reading example %d (out of %d) from %q",
						inFileIdx, C10ExamplesPerFile, dataFile))
				}
				if bytesRead != len(labelImageBytes) {
					Panicf("read only %d bytes reading example %d (out of %d) from %q, wanted %d bytes",
						bytesRead, inFileIdx, C10ExamplesPerFile, dataFile, len(labelImageBytes))
				}
				switch dtype {
				case dtypes.Float64:
					err = convertBytesToTensor[float64](labelImageBytes[1:], images, exampleIdx)
				case dtypes.Float32:
					err = convertBytesToTensor[float32](labelImageBytes[1:], images, exampleIdx)
				default:
					panic(errors.Errorf("DType %s not supported", dtype))
				}
				if err != nil {
					panic(errors.WithMessagef(err, "failed converting bytes to tensor of %s", dtype))
				}
				labelsData[exampleIdx] = int64(labelImageBytes[0])
			}
		}
	})
	return partitionImagesAndLabels(backend, images, labels)
}

// LoadCifar100 into 2 tensors of the given DType: images with given dtype and shaped
// [NumExamples=60000, Height=32, Width=32, Depth=3], and labels shaped
// [NumExamples=60000, 1] of Int64.
// The first 50k examples are for training, and the last 10k for testing.
// Only Float32 and Float64 dtypes are supported for now.
func LoadCifar100(
	backend backends.Backend,
	baseDir string,
	dtype dtypes.DType,
) (partitioned PartitionedImagesAndLabels) {
	baseDir = fsutil.MustReplaceTildeInDir(baseDir)

	// Allocate the tensor.
	images := tensors.FromShape(shapes.Make(dtype, NumExamples, Height, Width, Depth))
	labels := tensors.FromShape(shapes.Make(dtypes.Int64, NumExamples, 1))
	tensors.MustMutableFlatData[int64](labels, func(labelsData []int64) {
		var labelImageBytes [imageSizeBytes + 2]byte
		dataFiles := []string{"train.bin", "test.bin"}
		exampleIdx := 0
		for _, fileName := range dataFiles {
			dataFile := path.Join(baseDir, C100SubDir, fileName)
			f, err := os.Open(dataFile)
			if err != nil {
				panic(errors.Wrapf(err, "opening data file %q", dataFile))
			}
			for inFileIdx := 0; true; inFileIdx++ {
				bytesRead, err := f.Read(labelImageBytes[:])
				if bytesRead == 0 && err == io.EOF {
					break
				}
				if err != nil {
					panic(errors.WithMessagef(err, "reading example %d from %q", inFileIdx, dataFile))
				}
				if bytesRead != len(labelImageBytes) {
					panic(errors.Errorf("read only %d bytes reading example %d from %q, wanted %d bytes",
						bytesRead, inFileIdx, dataFile, len(labelImageBytes)))
				}
				switch dtype {
				case dtypes.Float64:
					err = convertBytesToTensor[float64](labelImageBytes[2:], images, exampleIdx)
				case dtypes.Float32:
					err = convertBytesToTensor[float32](labelImageBytes[2:], images, exampleIdx)
				default:
					Panicf("DType %s not supported", dtype)
				}
				if err != nil {
					panic(errors.Wrapf(err, "failed converting bytes to tensor of %s", dtype))
				}
				labelsData[exampleIdx] = int64(
					labelImageBytes[1],
				) // Take the fine-label (and discard the coarse-label).
				exampleIdx++
			}
		}
	})
	return partitionImagesAndLabels(backend, images, labels)
}

func ConvertToGoImage(images *tensors.Tensor, exampleNum int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, Width, Height))
	images.MustConstFlatData(func(flatAny any) {
		tensorData := reflect.ValueOf(flatAny)
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
	})
	return img
}

// partitionImagesAndLabels into train and test partitions.
func partitionImagesAndLabels(
	backend backends.Backend,
	images, labels *tensors.Tensor,
) (partitioned PartitionedImagesAndLabels) {
	parts := MustExecOnceN(backend, func(images, labels *Node) []*Node {
		imagesTrain := Slice(images, AxisRange(0, NumTrainExamples))
		labelsTrain := Slice(labels, AxisRange(0, NumTrainExamples))
		imagesTest := Slice(images, AxisRange(NumTrainExamples))
		labelsTest := Slice(labels, AxisRange(NumTrainExamples))
		return []*Node{imagesTrain, labelsTrain, imagesTest, labelsTest}
	}, images, labels)
	partitioned[0].images = parts[0]
	partitioned[0].labels = parts[1]
	partitioned[1].images = parts[2]
	partitioned[1].labels = parts[3]
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
	images, labels *tensors.Tensor
}

// PartitionedImagesAndLabels holds for each partition (Train, Test), one set of Images and Labels.
type PartitionedImagesAndLabels [2]ImagesAndLabels

var (
	// Cache of loaded data: one per DataSource, per DType, per Partition(Train/Test).
	imagesAndLabelsCache [2]map[dtypes.DType]PartitionedImagesAndLabels
)

func ResetCache() {
	imagesAndLabelsCache = [2]map[dtypes.DType]PartitionedImagesAndLabels{
		make(map[dtypes.DType]PartitionedImagesAndLabels), // Cifar10
		make(map[dtypes.DType]PartitionedImagesAndLabels), // Cifar100
	}
}

func init() {
	ResetCache()
}

// NewDataset returns a Dataset for the training data, which implements train.Dataset and hence can be used
// by train.Trainer methods.
//
// It automatically downloads the data from the web, and then loads the data into memory if it hasn't been
// loaded yet.
// It caches the result, so multiple Datasets can be created without any extra costs in time/memory.
func NewDataset(
	backend backends.Backend,
	name, baseDir string,
	source DataSource,
	dtype dtypes.DType,
	partition Partition,
) *datasets.InMemoryDataset {
	if source > C100 {
		Panicf("Invalid source value %d, only C10 or C100 accepted", source)
	}
	partitioned, found := imagesAndLabelsCache[source][dtype]
	if !found {
		// How do download & load data: one per DataSource.
		downloadFunctions := [2]func(baseDir string) error{
			DownloadCifar10, DownloadCifar100}
		loadFunctions := [2]func(backend backends.Backend, baseDir string, dType dtypes.DType) PartitionedImagesAndLabels{
			LoadCifar10,
			LoadCifar100,
		}

		err := downloadFunctions[source](baseDir)
		if err != nil {
			panic(errors.WithMessagef(err, "Creating a new Dataset"))
		}
		partitioned = loadFunctions[source](backend, baseDir, dtype)
		imagesAndLabelsCache[source][dtype] = partitioned
	}
	imagesAndLabels := partitioned[partition]
	ds, err := datasets.InMemoryFromData(backend, name, []any{imagesAndLabels.images}, []any{imagesAndLabels.labels})
	if err != nil {
		panic(err)
	}
	return ds
}
