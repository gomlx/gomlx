// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dogsvscats

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"iter"
	"log"
	"math/rand"
	"os"
	"path"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/tensors"
	timage "github.com/gomlx/gomlx/core/tensors/images"
	"github.com/gomlx/gomlx/examples/downloader"
	"github.com/gomlx/gomlx/ml/dataset"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/pkg/errors"
	"github.com/schollz/progressbar/v3"

	"github.com/disintegration/imaging"
)

const (
	DownloadURL   = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
	LocalZipFile  = "kagglecatsanddogs_5340.zip"
	LocalZipDir   = "PetImages"
	InvalidSubDir = "invalid"

	DownloadChecksum = "b7974bd00a84a99921f36ee4403f089853777b5ae8d151c76a86e64900334af9"
)

type DogOrCat int8

const (
	Dog DogOrCat = iota
	Cat
)

func (t DogOrCat) String() string {
	switch t {
	case Dog:
		return "Dog"
	case Cat:
		return "Cat"
	}
	return "Unknown"
}

var (
	ImgSubDirs   = []string{"Dog", "Cat"}
	BadDogImages = map[int]bool{11233: true, 11702: true, 11912: true, 2317: true, 9500: true}
	BadCatImages = map[int]bool{10404: true, 11095: true, 12080: true, 5370: true, 6435: true, 666: true}
	BadImages    = [2]map[int]bool{BadDogImages, BadCatImages}

	MaxCount = 12500
	NumDogs  = MaxCount - len(BadDogImages)
	NumCats  = MaxCount - len(BadCatImages)
	NumValid = [2]int{NumDogs, NumCats}
	_        = NumValid
)

// Download Dogs vs Cats Dataset to baseDir, unzips it, and checks for mal-formed files (there are a few).
func Download(baseDir string) error {
	zipFilePath := path.Join(baseDir, LocalZipFile)
	targetZipPath := path.Join(baseDir, LocalZipDir)
	if err := downloader.DownloadAndUnzipIfMissing(DownloadURL, zipFilePath, baseDir, targetZipPath, DownloadChecksum); err != nil {
		return err
	}
	return PrefilterValidImages(baseDir)
}

// FilterValidImages tries to open every image, and moves invalid (that are not readable) to a separate directory.
// One should use it only if the database of images change, otherwise use the PrefilterValidImages, which uses
// the static list of invalid images.
func FilterValidImages(baseDir string) error {
	invalidDir := path.Join(baseDir, InvalidSubDir)

	// Check if it has already been filtered.
	_, err := os.Stat(invalidDir)
	if err == nil {
		// Assume they have already been filtered, return immediately.
		return nil
	} else if !os.IsNotExist(err) {
		return err
	}

	// Create subdirectories for invalid files.
	if err = os.Mkdir(invalidDir, 0755); err != nil {
		return err
	}
	for _, subDir := range ImgSubDirs {
		dir := path.Join(invalidDir, subDir)
		if err = os.Mkdir(dir, 0755); err != nil {
			return err
		}
	}

	for _, subDir := range ImgSubDirs {
		dir := path.Join(baseDir, LocalZipDir, subDir)
		invalidSubDir := path.Join(invalidDir, subDir)
		entries, err := os.ReadDir(dir)
		if err != nil {
			return err
		}
		for _, entry := range entries {
			if !strings.HasSuffix(entry.Name(), ".jpg") {
				continue
			}
			imgPath := path.Join(dir, entry.Name())
			_, err := GetImageFromFilePath(imgPath)
			if err != nil {
				fmt.Printf("Failed to read %s: %v  ---> moving to %s\n", imgPath, err, invalidSubDir)
				if err = os.Rename(imgPath, path.Join(invalidSubDir, entry.Name())); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// PrefilterValidImages is like FilterValidImages, but uses pre-generated list of images known to be invalid.
func PrefilterValidImages(baseDir string) error {
	// Check if it has already been filtered.
	invalidDir := path.Join(baseDir, InvalidSubDir)
	_, err := os.Stat(invalidDir)
	if err == nil {
		// Assume they have already been filtered, return immediately.
		return nil
	} else if !os.IsNotExist(err) {
		return err
	}

	// Create subdirectories for invalid files.
	if err = os.Mkdir(invalidDir, 0755); err != nil {
		return err
	}
	for _, subDir := range ImgSubDirs {
		dir := path.Join(invalidDir, subDir)
		if err = os.Mkdir(dir, 0755); err != nil {
			return err
		}
	}

	for classIdx, subDir := range ImgSubDirs {
		dir := path.Join(baseDir, LocalZipDir, subDir)
		invalidSubDir := path.Join(invalidDir, subDir)
		var invalidList map[int]bool
		if classIdx == 0 {
			invalidList = BadDogImages
		} else {
			invalidList = BadCatImages
		}
		for imgIdx := range invalidList {
			name := fmt.Sprintf("%d.jpg", imgIdx)
			imgPath := path.Join(dir, name)
			fmt.Printf("Moving %s to %s: image known to have invalid format\n", imgPath, invalidSubDir)
			if err = os.Rename(imgPath, path.Join(invalidSubDir, name)); err != nil {
				return err
			}
		}
	}
	return nil
}

// Dataset implements train.Dataset so it can be used by a train.Loop object to train/evaluate, and offers a few
// more functionality for sampling images (as opposed to tensors).
type Dataset struct {
	name    string
	BaseDir string

	// Image transformation.
	width, height int
	angleStdDev   float64
	flipRandomly  bool
	rng           *rand.Rand
	dtype         dtypes.DType
	toTensor      *timage.ToTensorConfig
	parallelism   int

	// Dataset sampling strategy.
	batchSize            int
	infinite, yieldPairs bool

	// Folds.
	numFolds  int
	folds     []int
	foldsSeed int32

	shuffle *rand.Rand
}

var (
	AssertDatasetIsTrainDataset *Dataset
	_                           train.Dataset = AssertDatasetIsTrainDataset
)

// NewDataset creates a train.Dataset that yields images from Dogs vs Cats Dataset.
//
// It takes the following arguments:
//
//   - batchSize: how many images are returned by each Yield call.
//   - infinite: if it is set keeps looping, never ends. Typically used for training with `train.Loop.RunSteps()`.
//     Set this to false for evaluation datasets, or if training with `train.Loop.RunEpochs()`.
//   - shuffle: if set (not nil) use this `*rand.Rand` object to shuffle. If infinite it samples with
//     replacement.
//   - numFolds, folds and foldsSeed: splits whole data into numFolds folds (using foldsSeed) and take
//     only `folds` in this Dataset. Can be used to split train/test/validation datasets or run a cross-validation
//     scheme.
//   - width, height: resize images to this size. It will not distort the scale, and extra padding will
//     be included with 0s, including on the alpha channel (so padding is transparent).
//   - angleStdDev and flipRandomly: if set to true, it will randomly transform the images, augmenting
//     so to say the Dataset. It serves as a type of regularization. Set angleStdDev to 0 and flipRandomly
//     to false not to include any augmentation.
func NewDataset(name, baseDir string, batchSize int, infinite bool, shuffle *rand.Rand,
	numFolds int, folds []int, foldsSeed int32,
	width, height int, angleStdDev float64, flipRandomly bool, dtype dtypes.DType) *Dataset {
	ds := &Dataset{
		name:         name,
		BaseDir:      baseDir,
		width:        width,
		height:       height,
		angleStdDev:  angleStdDev,
		flipRandomly: flipRandomly,
		rng:          rand.New(rand.NewSource(time.Now().UTC().UnixNano())),
		dtype:        dtype,
		toTensor:     timage.ToTensor(dtype).WithAlpha(),
		parallelism:  1,

		batchSize: batchSize,
		infinite:  infinite,
		shuffle:   shuffle,
		numFolds:  numFolds,
		folds:     folds,
		foldsSeed: foldsSeed,
	}
	return ds
}

// Name implements train.Dataset.
func (ds *Dataset) Name() string { return ds.name }

// WithImagePairs configures the dataset to yield image pairs: the same image with different augmentation.
// Used for BYOL training.
//
// Returns itself, to allow chain of method calls.
func (ds *Dataset) WithImagePairs(yieldPairs bool) *Dataset {
	ds.yieldPairs = yieldPairs
	return ds
}

// WithParallelism sets the number of parallel workers used to load and preprocess images.
// If set to > 1, it will parallelize host-side image loading and resizing.
func (ds *Dataset) WithParallelism(parallelism int) *Dataset {
	ds.parallelism = parallelism
	return ds
}

// checkFolds checks whether the folds configuration is valid.
func (ds *Dataset) checkFolds() error {
	if ds.numFolds > 0 && len(ds.folds) == 0 {
		return errors.Errorf("dataset with %d folds, but none selected for this dataset", ds.numFolds)
	}
	for _, foldNum := range ds.folds {
		if foldNum < 0 || foldNum >= ds.numFolds {
			return errors.Errorf(
				"fold %d invalid for dataset with %d folds (folds selection is %v)",
				foldNum,
				ds.numFolds,
				ds.folds,
			)
		}
	}
	return nil
}

// inFold checks whether the image index imgIdx is in the Dataset fold selection and if it's not in the bad list.
func (ds *Dataset) inFold(imgType DogOrCat, imgIdx int) bool {
	if BadImages[imgType][imgIdx] {
		// We selected a bad image.
		return false
	}
	if ds.numFolds == 0 {
		return true
	}
	buffer := bytes.NewBuffer(make([]byte, 8))
	err := binary.Write(buffer, binary.LittleEndian, ds.foldsSeed)
	if err != nil {
		log.Fatalf("Failed to write to generate hash: %+v", err)
	}
	err = binary.Write(buffer, binary.LittleEndian, int32(imgIdx))
	if err != nil {
		log.Fatalf("Failed to write to generate hash: %+v", err)
	}
	hashValue := crc32.ChecksumIEEE(buffer.Bytes())
	fold := int(hashValue % uint32(ds.numFolds))
	return slices.Contains(ds.folds, fold)
}

// yieldIndices deals with the selection of the images to yield for each batch.
// It only returns the image indices. The actual loading/transformation of the images
// happens in YieldImages below.
// ImagesBatch represents a batch of raw images and labels.
type ImagesBatch struct {
	Images  []image.Image
	Labels  []DogOrCat
	Indices []int
}

type imageTask struct {
	dogsAndCats [2][]int
}

// IterImagesTasks returns an iterator over the tasks of selecting which dog/cat indices to load next.
// It is sequential and concurrency-free.
func (ds *Dataset) IterImagesTasks() iter.Seq2[imageTask, error] {
	return func(yield func(imageTask, error) bool) {
		if err := ds.checkFolds(); err != nil {
			yield(imageTask{}, err)
			return
		}

		var selection [2][]int
		if !ds.infinite && ds.shuffle != nil {
			adjustFolds := func(num int) int {
				if ds.numFolds == 0 {
					return num
				}
				num = num * ds.numFolds / len(ds.folds)
				num = num + num/20
				return num
			}
			numDogsFold := adjustFolds(NumDogs)
			numCatsFold := adjustFolds(NumCats)
			selection[Dog] = make([]int, 0, numDogsFold)
			selection[Cat] = make([]int, 0, numCatsFold)

			for imgType := range 2 {
				for imgIdx := 0; imgIdx < MaxCount; imgIdx++ {
					if !ds.inFold(DogOrCat(imgType), imgIdx) {
						continue
					}
					selection[imgType] = append(selection[imgType], imgIdx)
				}
			}

			for imgType := range 2 {
				ds.shuffle.Shuffle(len(selection[imgType]), func(i, j int) {
					selection[imgType][i], selection[imgType][j] = selection[imgType][j], selection[imgType][i]
				})
			}
		}

		counters := [2]int{0, 0}
		rng := rand.New(rand.NewSource(time.Now().UTC().UnixNano()))

		for {
			var dogsAndCats [2][]int
			numYield := [2]int{(ds.batchSize + 1) / 2, ds.batchSize / 2}
			for imgType := range 2 {
				dogsAndCats[imgType] = make([]int, 0, numYield[imgType])
			}
			var eof bool
			for imgType := range 2 {
				for ii := 0; ii < numYield[imgType]; ii++ {
					for {
						var imgIdx int
						if ds.infinite {
							if ds.shuffle != nil {
								imgIdx = rng.Intn(MaxCount)
							} else {
								imgIdx = counters[imgType]
								counters[imgType] = (counters[imgType] + 1) % MaxCount
							}
						} else {
							if ds.shuffle != nil {
								selectionIdx := counters[imgType]
								if selectionIdx >= len(selection[imgType]) {
									eof = true
									break
								}
								imgIdx = selection[imgType][selectionIdx]
								counters[imgType]++
							} else {
								imgIdx = counters[imgType]
								if imgIdx >= MaxCount {
									eof = true
									break
								}
								counters[imgType]++
							}
						}
						if !ds.inFold(DogOrCat(imgType), imgIdx) {
							continue
						}
						dogsAndCats[imgType] = append(dogsAndCats[imgType], imgIdx)
						break
					}
					if eof {
						break
					}
				}
				if eof {
					break
				}
			}
			if eof {
				return
			}

			if !yield(imageTask{dogsAndCats: dogsAndCats}, nil) {
				return
			}
		}
	}
}

// // tasksDataset implements train.Dataset to yield task batches where Spec is imageTask.
type tasksDataset struct {
	name string
	seq  iter.Seq2[imageTask, error]
}

func (td *tasksDataset) Name() string { return td.name }

func (td *tasksDataset) Iter() iter.Seq2[train.Batch, error] {
	return func(yield func(train.Batch, error) bool) {
		for task, err := range td.seq {
			if err != nil {
				yield(train.Batch{}, err)
				return
			}
			// We put the task metadata inside Batch.Spec.
			if !yield(train.Batch{Spec: task}, nil) {
				return
			}
		}
	}
}

// IterImages returns an iterator over raw images. Used by Save() and Iter().
func (ds *Dataset) IterImages() iter.Seq2[ImagesBatch, error] {
	seq := ds.IterImagesTasks()
	// Use a seed source to create thread-safe worker RNGs.
	rngSeed := rand.New(rand.NewSource(time.Now().UTC().UnixNano()))

	taskDS := &tasksDataset{
		name: ds.Name(),
		seq:  seq,
	}

	mapFn := func(batch train.Batch) train.Batch {
		// Spec holds the imageTask metadata.
		t := batch.Spec.(imageTask)
		numExamples := len(t.dogsAndCats[0]) + len(t.dogsAndCats[1])
		var images []image.Image
		if ds.yieldPairs {
			images = make([]image.Image, 2*numExamples)
		} else {
			images = make([]image.Image, numExamples)
		}
		labels := make([]DogOrCat, 0, numExamples)
		indices := make([]int, 0, numExamples)
		count := 0

		workerRng := rand.New(rand.NewSource(time.Now().UTC().UnixNano() + rngSeed.Int63()))

		for ii, imgSet := range t.dogsAndCats {
			subDir := ImgSubDirs[ii]
			for _, imgIdx := range imgSet {
				indices = append(indices, imgIdx)
				labels = append(labels, DogOrCat(ii))
				imgPath := path.Join(ds.BaseDir, LocalZipDir, subDir, fmt.Sprintf("%d.jpg", imgIdx))
				originalImg, err := GetImageFromFilePath(imgPath)
				if err != nil {
					// We return the error wrapped as an error inside Spec so the main loop can yield it.
					return train.Batch{Spec: errors.Wrapf(err, "while reading %s image %d", subDir, imgIdx)}
				}
				img := ds.Augment(workerRng, originalImg)
				img = ResizeWithPadding(img, ds.width, ds.height)
				images[count] = img
				if ds.yieldPairs {
					img = ds.Augment(workerRng, originalImg)
					img = ResizeWithPadding(img, ds.width, ds.height)
					images[count+numExamples] = img
				}
				count++
			}
		}

		// Pack the loaded ImagesBatch inside the Spec field.
		return train.Batch{
			Spec: ImagesBatch{
				Images:  images,
				Labels:  labels,
				Indices: indices,
			},
		}
	}

	pds := dataset.ParallelMapOnHost(taskDS, mapFn).WithParallelism(ds.parallelism)

	return func(yield func(ImagesBatch, error) bool) {
		for batch, err := range pds.Iter() {
			if err != nil {
				yield(ImagesBatch{}, err)
				return
			}
			// If mapFn returned an error inside Spec, yield it.
			if errSpec, ok := batch.Spec.(error); ok {
				yield(ImagesBatch{}, errSpec)
				return
			}
			imgBatch := batch.Spec.(ImagesBatch)
			if !yield(imgBatch, nil) {
				return
			}
		}
	}
}

// Augment image according to specification of the Dataset.
func (ds *Dataset) Augment(rng *rand.Rand, img image.Image) image.Image {
	if ds.angleStdDev > 0 {
		img = imaging.Rotate(img, rng.NormFloat64()*ds.angleStdDev, color.RGBA{R: 0, G: 0, B: 0, A: 255})
	}
	if ds.flipRandomly && rng.Intn(2) == 1 {
		img = imaging.FlipH(img)
	}
	return img
}

// Iter implements `train.Dataset`.
func (ds *Dataset) Iter() iter.Seq2[train.Batch, error] {
	return func(yield func(train.Batch, error) bool) {
		for imgBatch, err := range ds.IterImages() {
			if err != nil {
				yield(train.Batch{}, err)
				return
			}
			var inputs, labels []*tensors.Tensor
			numExamples := len(imgBatch.Indices)
			if ds.yieldPairs {
				firstHalf, secondHalf := ds.toTensor.Batch(imgBatch.Images[:numExamples]), ds.toTensor.Batch(imgBatch.Images[numExamples:])
				inputs = []*tensors.Tensor{
					firstHalf,
					tensors.FromValue(imgBatch.Indices),
					secondHalf,
				}
			} else {
				inputs = []*tensors.Tensor{ds.toTensor.Batch(imgBatch.Images), tensors.FromValue(imgBatch.Indices)}
			}
			labels = []*tensors.Tensor{tensors.MustFromAnyValue(shapes.CastAsDType(imgBatch.Labels, ds.dtype))}
			batch := train.Batch{
				Spec:   ds,
				Inputs: inputs,
				Labels: labels,
			}
			if !yield(batch, nil) {
				return
			}
		}
	}
}

func GetImageFromFilePath(imagePath string) (image.Image, error) {
	f, err := os.Open(imagePath)
	if err != nil {
		return nil, err
	}
	img, _, err := image.Decode(f)
	return img, err
}

// ResizeWithPadding will
func ResizeWithPadding(img image.Image, width, height int) image.Image {
	imgSize := img.Bounds().Size()
	wRatio := float64(width) / float64(imgSize.X)
	hRatio := float64(height) / float64(imgSize.Y)

	adjustedWidth, adjustedHeight := width, height
	if wRatio < hRatio {
		adjustedHeight = int(wRatio * float64(imgSize.Y))
	} else if hRatio < wRatio {
		adjustedWidth = int(hRatio * float64(imgSize.X))
	}
	img = imaging.Resize(img, adjustedWidth, adjustedHeight, imaging.Lanczos)
	if adjustedWidth != width || adjustedHeight != height {
		bgImg := image.NewRGBA(image.Rect(0, 0, width, height))
		img = imaging.PasteCenter(bgImg, img)
	}
	return img
}

// Save will generate numEpochs of the dataset, with configured augmentations and resizing, and saves to
// the given file(s).
//
// If more than one file is given, the same image but with a different augmentations is saved in each file.
//
// If dataset is set to infinite it fails.
//
// If verbose is set to true, it will output a progress bar.
func (ds *Dataset) Save(numEpochs int, verbose bool, writers ...io.Writer) error {
	if ds.infinite {
		return errors.Errorf("cannot Dataset.Save %d epochs if dataset it configure to loop infinitely", numEpochs)
	}
	if ds.batchSize%2 != 0 {
		return errors.Errorf(
			"batch size %d is not even, and will lead to more dogs than cats, please choose something divided by 2",
			ds.batchSize,
		)
	}
	if ds.yieldPairs && len(writers) != 2 {
		return errors.Errorf(
			"dataset %q configured to yield pairs, 2 writers required but %d writers given",
			ds.Name(),
			len(writers),
		)
	}
	if !ds.yieldPairs && len(writers) != 1 {
		return errors.Errorf(
			"dataset %q configured to single images, only 1 writer required but %d writers given",
			ds.Name(),
			len(writers),
		)
	}

	numSteps := numEpochs * (NumDogs + NumCats)
	if ds.numFolds != 0 {
		numSteps = numSteps * len(ds.folds) / ds.numFolds
	}
	numBatches := numSteps / ds.batchSize
	numSteps = numBatches * ds.batchSize // Multiple of ds.batchSize.
	var pBar *progressbar.ProgressBar
	maxFrequencyPBar := 250 * time.Millisecond
	nextPBarUpdate := time.Now() // When pBar is ready for another update.
	addToPBar := 0               // Amount to add to pBar
	if verbose {
		pBar = progressbar.NewOptions(numSteps,
			progressbar.OptionSetDescription("Pre-generating"),
			progressbar.OptionUseANSICodes(true),
			progressbar.OptionEnableColorCodes(true),
			progressbar.OptionShowIts(),
			progressbar.OptionSetItsString("images"),
			progressbar.OptionSetTheme(progressbar.ThemeUnicode),
		)
	}

	var err error
	parallelism := runtime.NumCPU() + 1
	fmt.Printf("\tParallelism: %d\n", parallelism-1)
	for range numEpochs {
		var wg sync.WaitGroup
		var muWrite sync.Mutex
		errChan := make(chan error, parallelism)
		batchesChan := make(chan ImagesBatch, parallelism)

		go func() {
			defer close(batchesChan)
			for imgBatch, err := range ds.IterImages() {
				if err != nil {
					errChan <- err
					return
				}
				batchesChan <- imgBatch
			}
		}()

		// Start parallelism goroutines that read images and write them.
		for range parallelism {
			wg.Add(1)
			go func() {
				defer wg.Done()
				var err error
			loopImageWrite:
				for imgBatch := range batchesChan {
					var buffers [2][][]byte
					for writerIdx := range writers {
						buffers[writerIdx] = make([][]byte, ds.batchSize)
						for imgIdx := 0; imgIdx < ds.batchSize; imgIdx++ {
							buffer := make([]byte, ds.width*ds.height*4+1)
							buffers[writerIdx][imgIdx] = buffer
							pos := 0
							buffer[pos] = (byte)(imgBatch.Labels[imgIdx])
							pos += 1
							img := imgBatch.Images[imgIdx+writerIdx*ds.batchSize]
							for y := 0; y < ds.height; y++ {
								for x := 0; x < ds.width; x++ {
									r, g, b, a := img.At(x, y).RGBA()
									for _, channel := range []uint32{r, g, b, a} {
										buffer[pos] = byte(channel >> 8)
										pos += 1
									}
								}
							}
						}
					}
					muWrite.Lock()
					for writerIdx, writer := range writers {
						for imgIdx := 0; imgIdx < ds.batchSize; imgIdx++ {
							_, err = writer.Write(buffers[writerIdx][imgIdx])
							if err != nil {
								break loopImageWrite
							}
						}
					}
					if verbose {
						addToPBar += ds.batchSize
						now := time.Now()
						if now.After(nextPBarUpdate) {
							_ = pBar.Add(addToPBar)
							fmt.Printf("      ")
							addToPBar = 0
							nextPBarUpdate = now.Add(maxFrequencyPBar)
						}
					}
					muWrite.Unlock()
				}
				if err != nil && err != io.EOF {
					errChan <- err
				}
			}()
		}
		wg.Wait()
		if addToPBar > 0 {
			// Add final steps to progress-bar.
			_ = pBar.Add(addToPBar)
		}

		close(errChan)
		for err = range errChan {
			fmt.Printf("Error: %v", err)
		}
		if err != nil {
			return err
		}
	}
	if verbose {
		_ = pBar.Close()
		fmt.Println()
	}
	return nil
}

// PreGeneratedDataset implements train.Dataset by reading the images from the pre-generated (scaled and
// optionally augmented) images data. See Dataset.Save for saving these pre-generated files.
type PreGeneratedDataset struct {
	name                   string
	filePath, pairFilePath string
	dtype                  dtypes.DType
	batchSize              int
	width, height          int
	infinite               bool
	yieldPairs             bool
	maxSteps               int
}

// Assert PreGeneratedDataset is a train.Dataset.
var _ train.Dataset = &PreGeneratedDataset{}

// NewPreGeneratedDataset creates a PreGeneratedDataset that yields dogsvscats images and labels.
func NewPreGeneratedDataset(
	name, filePath string,
	batchSize int,
	infinite bool,
	width, height int,
	dtype dtypes.DType,
) *PreGeneratedDataset {
	pds := &PreGeneratedDataset{
		name:      name,
		filePath:  filePath,
		dtype:     dtype,
		batchSize: batchSize,
		width:     width,
		height:    height,
		infinite:  infinite,
	}
	return pds
}

// Name implements train.Dataset.
func (pds *PreGeneratedDataset) Name() string { return pds.name }

// WithMaxSteps configures the dataset to exhaust after those many steps, returning `io.EOF`.
//
// This is useful for testing.
func (pds *PreGeneratedDataset) WithMaxSteps(numSteps int) *PreGeneratedDataset {
	pds.maxSteps = numSteps
	return pds
}

// WithImagePairs configures the dataset to yield image pairs (with the different augmentation).
//
// It takes a second file path `pairFilePath` that points to the pair images.
// If `pairFilePath` is empty, it disables yielding image pairs.
func (pds *PreGeneratedDataset) WithImagePairs(pairFilePath string) *PreGeneratedDataset {
	pds.yieldPairs = pairFilePath != ""
	pds.pairFilePath = pairFilePath
	return pds
}

func (pds *PreGeneratedDataset) entrySize() int {
	return 1 + 4*pds.width*pds.height
}

// Yield implements train.Dataset.
// Iter implements `train.Dataset`.
func (pds *PreGeneratedDataset) Iter() iter.Seq2[train.Batch, error] {
	return func(yield func(train.Batch, error) bool) {
		var openedFile, openedPairFile *os.File
		var steps int
		var err error

		defer func() {
			if openedFile != nil {
				_ = openedFile.Close()
			}
			if openedPairFile != nil {
				_ = openedPairFile.Close()
			}
		}()

		reset := func() error {
			if openedFile != nil {
				_ = openedFile.Close()
				openedFile = nil
			}
			openedFile, err = os.Open(pds.filePath)
			if err != nil {
				return errors.Wrapf(err, "failed to open file %q", pds.filePath)
			}

			if pds.yieldPairs {
				if openedPairFile != nil {
					_ = openedPairFile.Close()
					openedPairFile = nil
				}
				openedPairFile, err = os.Open(pds.pairFilePath)
				if err != nil {
					return errors.Wrapf(err, "failed to open file %q", pds.pairFilePath)
				}
			}
			return nil
		}

		if err = reset(); err != nil {
			yield(train.Batch{}, err)
			return
		}

		entrySize := pds.entrySize()
		batchBytes := pds.batchSize * entrySize
		buffer := make([]byte, batchBytes)
		var pairBuffer []byte
		if pds.yieldPairs {
			pairBuffer = make([]byte, batchBytes)
		}
		labelsAsTypes := make([]DogOrCat, pds.batchSize)

		retries := 0
		for {
			steps++
			if pds.maxSteps > 0 && steps >= pds.maxSteps {
				return
			}

			for {
				if openedFile == nil {
					yield(train.Batch{}, errors.Errorf("PreGeneratedDataset for file %q not opened, invalid state", pds.filePath))
					return
				}
				n, readErr := openedFile.Read(buffer)
				if readErr == io.EOF || n < len(buffer) {
					if !pds.infinite {
						return
					}
					if retries != 0 {
						yield(train.Batch{}, errors.Errorf(
							"not enough data for %d batches in PreGeneratedDataset for file %q, maybe it failed during generation of the file?",
							pds.batchSize,
							pds.filePath,
						))
						return
					}
					retries++
					if err = reset(); err != nil {
						yield(train.Batch{}, err)
						return
					}
					continue
				}
				if readErr != nil {
					yield(train.Batch{}, errors.Wrapf(readErr, "failed reading PreGeneratedDataset from file %q ", pds.filePath))
					return
				}
				if pds.yieldPairs {
					_, readErr = openedPairFile.Read(pairBuffer)
					if readErr != nil {
						yield(train.Batch{}, errors.Wrapf(readErr, "failed reading PreGeneratedDataset from file %q ", pds.pairFilePath))
						return
					}
				}

				for ii := 0; ii < pds.batchSize; ii++ {
					labelsAsTypes[ii] = DogOrCat(buffer[ii*entrySize])
				}
				labels := []*tensors.Tensor{tensors.MustFromAnyValue(shapes.CastAsDType(labelsAsTypes, pds.dtype))}
				var t, pairT *tensors.Tensor
				switch pds.dtype {
				case dtypes.Float32:
					t = BytesToTensor[float32](buffer, pds.batchSize, pds.width, pds.height)
					if pds.yieldPairs {
						pairT = BytesToTensor[float32](pairBuffer, pds.batchSize, pds.width, pds.height)
					}
				case dtypes.Float64:
					t = BytesToTensor[float64](buffer, pds.batchSize, pds.width, pds.height)
					if pds.yieldPairs {
						pairT = BytesToTensor[float64](pairBuffer, pds.batchSize, pds.width, pds.height)
					}
				default:
					yield(train.Batch{}, errors.Errorf("PreGeneratedDataset with dtype=%q not supported", pds.dtype))
					return
				}
				var inputs []*tensors.Tensor
				if pds.yieldPairs {
					inputs = []*tensors.Tensor{t, pairT}
				} else {
					inputs = []*tensors.Tensor{t}
				}

				batch := train.Batch{
					Spec:   pds,
					Inputs: inputs,
					Labels: labels,
				}
				if !yield(batch, nil) {
					return
				}
				break
			}
		}
	}
}

// BytesToTensor converts a batch of saved images as bytes to a tensor.Local with 4 channels: R,G,B and A. It assumes
// all images have the exact same size. There should be one byte with the label before each image.
func BytesToTensor[T interface {
	float32 | float64 | int | int16 | int32 | int64 | uint16 | uint32 | uint64
}](
	buffer []byte, numImages, width, height int) (t *tensors.Tensor) {
	var zero T
	t = tensors.FromShape(shapes.Make(dtypes.FromGoType(reflect.TypeOf(zero)), numImages, height, width, 4))
	t.MustMutableFlatData(func(flatAny any) {
		tensorData := flatAny.([]T)
		dataPos := 0
		bufferPos := 0
		convertToDType := func(val byte) T {
			return T(val) / T(0xFF)
		}
		for range numImages {
			bufferPos += 1 // Skip the label.
			for range height {
				for range width {
					// Channel varies through RGBA (4)
					for range 4 {
						tensorData[dataPos] = convertToDType(buffer[bufferPos])
						dataPos++
						bufferPos++
					}
				}
			}
		}
	})
	return
}
