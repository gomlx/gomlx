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
	"log"
	"math/rand"
	"os"
	"path"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/gomlx/gomlx/examples/downloader"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	timage "github.com/gomlx/gomlx/pkg/core/tensors/images"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gopjrt/dtypes"
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

	// Dataset sampling strategy.
	batchSize            int
	infinite, yieldPairs bool

	// Folds.
	numFolds  int
	folds     []int
	foldsSeed int32

	// muCountersSelection protects shuffle, counters and selection.
	muCountersSelection sync.Mutex

	// Index of current position in whole data.
	counters  [2]int
	selection [2][]int // Pre-generated shuffle selection of the data.
	shuffle   *rand.Rand
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

		batchSize: batchSize,
		infinite:  infinite,
		shuffle:   shuffle,
		numFolds:  numFolds,
		folds:     folds,
		foldsSeed: foldsSeed,
	}
	ds.Reset() // Create first shuffle, if needed.
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
	for _, included := range ds.folds {
		if fold == included {
			return true
		}
	}
	return false
}

// yieldIndices deals with the selection of the images to yield for each batch.
// It only returns the image indices. The actual loading/transformation of the images
// happens in YieldImages below.
func (ds *Dataset) yieldIndices() (dogsAndCats [2][]int, err error) {
	if err = ds.checkFolds(); err != nil {
		return
	}
	ds.muCountersSelection.Lock()
	defer ds.muCountersSelection.Unlock()

	numYield := [2]int{(ds.batchSize + 1) / 2, ds.batchSize / 2}
	for imgType := 0; imgType < 2; imgType++ {
		dogsAndCats[imgType] = make([]int, 0, numYield[imgType])
	}
	for imgType := 0; imgType < 2; imgType++ {
		for ii := 0; ii < numYield[imgType]; ii++ {
			for {
				var imgIdx int
				if ds.infinite {
					// Infinite: loop indefinitely.
					if ds.shuffle != nil {
						// Continuously sample randomly with replacement.
						imgIdx = ds.shuffle.Intn(MaxCount)
					} else {
						// Continuously go over examples, looping around.
						imgIdx = ds.counters[imgType]
						ds.counters[imgType] = (ds.counters[imgType] + 1) % MaxCount
					}
				} else {
					if ds.shuffle != nil {
						selectionIdx := ds.counters[imgType]
						if selectionIdx >= len(ds.selection[imgType]) {
							return dogsAndCats, io.EOF
						}
						imgIdx = ds.selection[imgType][selectionIdx]
						ds.counters[imgType] = ds.counters[imgType] + 1
					} else {
						imgIdx = ds.counters[imgType]
						if imgIdx >= MaxCount {
							return dogsAndCats, io.EOF
						}
						ds.counters[imgType] = ds.counters[imgType] + 1
					}
				}
				if !ds.inFold(DogOrCat(imgType), imgIdx) {
					continue
				}
				dogsAndCats[imgType] = append(dogsAndCats[imgType], imgIdx)
				break
			}
		}
	}
	return
}

// YieldImages yields a batch of images, their labels (Dog or Cat) and their indices. These are
// the raw images that can be used for displaying. See Yield below to get tensors
// that can be used for training.
//
// If WithImagePairs is set to true, it will return double the number of images: the second half is a repeat
// of the first, just with a different augmentation.
func (ds *Dataset) YieldImages() (images []image.Image, labels []DogOrCat, indices []int, err error) {
	// It uses Dataset.yieldIndices to select which images to return. This function then
	// reads / scales / augment the images.
	//fmt.Printf("YieldImages: yieldPairs=%v\n", ds.yieldPairs)
	dogsAndCats, err := ds.yieldIndices()
	if err != nil {
		return
	}
	numExamples := len(dogsAndCats[0]) + len(dogsAndCats[1])
	if ds.yieldPairs {
		images = make([]image.Image, 2*numExamples)
	} else {
		images = make([]image.Image, numExamples)
	}
	labels = make([]DogOrCat, 0, numExamples)
	indices = make([]int, 0, numExamples)
	count := 0
	for ii, imgSet := range dogsAndCats {
		subDir := ImgSubDirs[ii]
		for _, imgIdx := range imgSet {
			indices = append(indices, imgIdx)
			labels = append(labels, DogOrCat(ii))
			imgPath := path.Join(ds.BaseDir, LocalZipDir, subDir, fmt.Sprintf("%d.jpg", imgIdx))
			var originalImg image.Image
			originalImg, err = GetImageFromFilePath(imgPath)
			if err != nil {
				err = errors.Wrapf(err, "while reading %s image %d", subDir, imgIdx)
				return
			}
			img := ds.Augment(originalImg)
			img = ResizeWithPadding(img, ds.width, ds.height)
			images[count] = img
			if ds.yieldPairs {
				// Yield paired image augmentation on the second-half of the `images` slice.
				img = ds.Augment(originalImg)
				img = ResizeWithPadding(img, ds.width, ds.height)
				images[count+numExamples] = img
			}
			count++
		}
	}
	return
}

// Augment image according to specification of the Dataset.
func (ds *Dataset) Augment(img image.Image) image.Image {
	if ds.angleStdDev > 0 {
		img = imaging.Rotate(img, ds.rng.NormFloat64()*ds.angleStdDev, color.RGBA{R: 0, G: 0, B: 0, A: 255})
	}
	if ds.flipRandomly && ds.rng.Intn(2) == 1 {
		img = imaging.FlipH(img)
	}
	return img
}

var muT sync.Mutex

// Yield implements `train.Dataset`. It returns:
//
//   - spec: not used, left as nil.
//   - inputs: two tensors, the first is the images batch (shaped `[batch_size, height, width, depth==4]`) and
//     the second holds the indices of the images as int (I64), shaped `[batch_size]`.
func (ds *Dataset) Yield() (spec any, inputs, labels []*tensors.Tensor, err error) {
	muT.Lock()
	defer muT.Unlock()

	spec = ds // spec is a pointer to the Dataset.
	var images []image.Image
	var labelsAsTypes []DogOrCat
	var indices []int
	images, labelsAsTypes, indices, err = ds.YieldImages()
	if err != nil {
		return
	}
	if ds.yieldPairs {
		numExamples := len(indices)
		firstHalf, secondHalf := ds.toTensor.Batch(images[:numExamples]), ds.toTensor.Batch(images[numExamples:])
		inputs = []*tensors.Tensor{
			firstHalf,
			tensors.FromValue(indices),
			secondHalf,
		}
	} else {
		// No paired image.
		inputs = []*tensors.Tensor{ds.toTensor.Batch(images), tensors.FromValue(indices)}
	}
	labels = []*tensors.Tensor{tensors.FromAnyValue(shapes.CastAsDType(labelsAsTypes, ds.dtype))}
	return
}

// Reset restarts the Dataset from the beginning. Can be called after io.EOF is reached,
// for instance when running another evaluation on a test Dataset.
func (ds *Dataset) Reset() {
	ds.muCountersSelection.Lock()
	defer ds.muCountersSelection.Unlock()

	ds.counters = [2]int{0, 0}
	if ds.infinite || ds.shuffle == nil {
		return
	}

	// Create new shuffle of the Dataset.
	if len(ds.selection[Dog]) == 0 || len(ds.selection[Cat]) == 0 {
		// Create a slice of selections for dogs and cats that will fit the data, and
		// initialize the indices sequentially.
		adjustFolds := func(num int) int {
			if ds.numFolds == 0 {
				return num
			}
			num = num * ds.numFolds / len(ds.folds)
			// Add 5% margin because the folds split is not perfect.
			num = num + num/20
			return num
		}
		numDogsFold := adjustFolds(NumDogs)
		numCatsFold := adjustFolds(NumCats)
		ds.selection[Dog] = make([]int, 0, numDogsFold)
		ds.selection[Cat] = make([]int, 0, numCatsFold)

		for imgType := 0; imgType < 2; imgType++ {
			for imgIdx := 0; imgIdx < MaxCount; imgIdx++ {
				if !ds.inFold(DogOrCat(imgType), imgIdx) {
					continue
				}
				ds.selection[imgType] = append(ds.selection[imgType], imgIdx)
			}
		}
	}

	// (Re-)Shuffle the selections.
	for imgType := 0; imgType < 2; imgType++ {
		ds.shuffle.Shuffle(len(ds.selection[imgType]), func(i, j int) {
			ds.selection[imgType][i], ds.selection[imgType][j] = ds.selection[imgType][j], ds.selection[imgType][i]
		})
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
	for epoch := 0; epoch < numEpochs; epoch++ {
		var wg sync.WaitGroup
		var muWrite sync.Mutex
		errChan := make(chan error, parallelism)
		// Start parallelism goroutines that read images and write them.
		for ii := 0; ii < parallelism; ii++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				var err error
				var images []image.Image
				var labels []DogOrCat
			loopImageWrite:
				for {
					images, labels, _, err = ds.YieldImages()
					if err != nil {
						break
					}

					var buffers [2][][]byte
					for writerIdx := range writers {
						buffers[writerIdx] = make([][]byte, ds.batchSize)
						for imgIdx := 0; imgIdx < ds.batchSize; imgIdx++ {
							buffer := make([]byte, ds.width*ds.height*4+1)
							buffers[writerIdx][imgIdx] = buffer
							pos := 0
							buffer[pos] = (byte)(labels[imgIdx])
							pos += 1
							img := images[imgIdx+writerIdx*ds.batchSize]
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
		ds.Reset() // Reset dataset and start over.
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
	name                       string
	filePath, pairFilePath     string
	dtype                      dtypes.DType
	batchSize                  int
	width, height              int
	openedFile, openedPairFile *os.File
	infinite                   bool
	yieldPairs                 bool
	err                        error
	buffer, pairBuffer         []byte
	labelsAsTypes              []DogOrCat
	steps, maxSteps            int
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

	batchBytes := pds.batchSize * pds.entrySize()
	pds.buffer = make([]byte, batchBytes)
	pds.labelsAsTypes = make([]DogOrCat, pds.batchSize)
	pds.Reset() // Sets pds.openedFile with the opened filePath.
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
	pds.Reset()
	if pds.yieldPairs {
		batchBytes := pds.batchSize * pds.entrySize()
		pds.pairBuffer = make([]byte, batchBytes)
	} else {
		pds.pairBuffer = nil
	}
	return pds
}

func (pds *PreGeneratedDataset) entrySize() int {
	return 1 + 4*pds.width*pds.height
}

// Yield implements train.Dataset.
func (pds *PreGeneratedDataset) Yield() (spec any, inputs, labels []*tensors.Tensor, err error) {
	retries := 0

	// Check if maxSteps is reached.
	pds.steps++
	if pds.maxSteps > 0 && pds.steps >= pds.maxSteps {
		err = io.EOF
		return
	}

	spec = pds
	for { // Loop in case pds.infinite is true, and we retry at end of file.
		if pds.err != nil {
			return nil, nil, nil, pds.err
		}
		if pds.openedFile == nil {
			pds.err = errors.Errorf("PreGeneratedDataset for file %q not opened, invalid state", pds.filePath)
			return nil, nil, nil, pds.err
		}
		n, err := pds.openedFile.Read(pds.buffer)
		if err == io.EOF || n < len(pds.buffer) {
			if !pds.infinite {
				return nil, nil, nil, io.EOF
			}
			if retries != 0 {
				pds.err = errors.Errorf(
					"not enough data for %d batches in PreGeneratedDataset for file %q, maybe it failed during generation of the file?",
					pds.batchSize,
					pds.filePath,
				)
			}
			retries++
			pds.Reset()
			continue
		}
		if err != nil {
			pds.err = errors.Wrapf(err, "failed reading PreGeneratedDataset from file %q ", pds.filePath)
			return nil, nil, nil, pds.err
		}
		if pds.yieldPairs {
			_, err = pds.openedPairFile.Read(pds.pairBuffer)
			if err != nil {
				pds.err = errors.Wrapf(err, "failed reading PreGeneratedDataset from file %q ", pds.pairFilePath)
				return nil, nil, nil, pds.err
			}
		}

		entrySize := pds.entrySize()
		for ii := 0; ii < pds.batchSize; ii++ {
			pds.labelsAsTypes[ii] = DogOrCat(pds.buffer[ii*entrySize])
		}
		labels = []*tensors.Tensor{tensors.FromAnyValue(shapes.CastAsDType(pds.labelsAsTypes, pds.dtype))}
		var t, pairT *tensors.Tensor
		switch pds.dtype {
		case dtypes.Float32:
			t = BytesToTensor[float32](pds.buffer, pds.batchSize, pds.width, pds.height)
			if pds.yieldPairs {
				pairT = BytesToTensor[float32](pds.pairBuffer, pds.batchSize, pds.width, pds.height)
			}
		case dtypes.Float64:
			t = BytesToTensor[float64](pds.buffer, pds.batchSize, pds.width, pds.height)
			if pds.yieldPairs {
				pairT = BytesToTensor[float64](pds.pairBuffer, pds.batchSize, pds.width, pds.height)
			}
		default:
			pds.err = errors.Wrapf(err, "PreGeneratedDataset with dtype=%q not supported", pds.dtype)
			return nil, nil, nil, pds.err
		}
		if pds.yieldPairs {
			inputs = []*tensors.Tensor{t, pairT}
		} else {
			inputs = []*tensors.Tensor{t}
		}
		break
	}
	return
}

// Reset implements train.Dataset.
func (pds *PreGeneratedDataset) Reset() {
	pds.steps = 0
	if pds.openedFile != nil {
		_ = pds.openedFile.Close()
	}
	pds.openedFile, pds.err = os.Open(pds.filePath)
	if pds.err != nil {
		pds.err = errors.Wrapf(pds.err, "failed to open file %q", pds.filePath)
	}

	// Open image pairs file if requested:
	if pds.openedPairFile != nil {
		_ = pds.openedPairFile.Close()
	}
	if pds.yieldPairs {
		pds.openedPairFile, pds.err = os.Open(pds.pairFilePath)
		if pds.err != nil {
			pds.err = errors.Wrapf(pds.err, "failed to open file %q", pds.pairFilePath)
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
		for imgIdx := 0; imgIdx < numImages; imgIdx++ {
			bufferPos += 1 // Skip the label.
			for y := 0; y < height; y++ {
				for x := 0; x < width; x++ {
					// Channel varies through RGBA (4)
					for channel := 0; channel < 4; channel++ {
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
