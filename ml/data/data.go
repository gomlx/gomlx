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

// Package data is a collection of tools that facilitate data loading and preprocessing.
package data

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"github.com/schollz/progressbar/v3"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/user"
	"path"
	"runtime"
	"strings"
	"sync"

	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/pkg/errors"
)

// FileExists returns true if file or directory exists.
func FileExists(path string) bool {
	_, err := os.Stat(path)
	if err == nil {
		return true
	}
	if errors.Is(err, os.ErrNotExist) {
		return false
	}
	panic(err)
}

// ReplaceTildeInDir by the user's home directory. Returns dir if it doesn't start with "~".
func ReplaceTildeInDir(dir string) string {
	if len(dir) == 0 || dir[0] != '~' {
		return dir
	}
	usr, _ := user.Current()
	homeDir := usr.HomeDir
	return path.Join(homeDir, dir[1:])
}

// ValidateChecksum verifies that the checksum of the file in the given path matches the checksum
// given. If it fails, it will remove the file (!) and return and error.
func ValidateChecksum(path, checkHash string) error {
	hasher := sha256.New()
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer func() {
		_ = f.Close() // Discard reading error on Close.
	}()

	_, err = io.Copy(hasher, f)
	if err != nil {
		return err
	}
	fileHash := hex.EncodeToString(hasher.Sum(nil))
	if fileHash != strings.ToLower(checkHash) {
		err = errors.Errorf("file %q sha256 hash is %q, but expected %q, deleting file.",
			path, fileHash, checkHash)
		if e2 := os.Remove(path); e2 != nil {
			log.Printf("Failed to remove %q, which failed checksum test. Please remove it. %+v", path, e2)
		}
		return err
	}
	return nil
}

// ByteCountIEC converts a byte count to string using the appropriate unit (B, Kb, MiB, GiB, ...).
// It uses the binary prefix system from IEC -- so powers of 1024 (as opposed to powers 1000).
func ByteCountIEC(count int64) string {
	const unit = 1024
	if count < unit {
		return fmt.Sprintf("%d B", count)
	}
	div, exp := int64(unit), 0
	for n := count / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(count)/float64(div), "KMGTPE"[exp])
}

// copyBytesBar copies bytes from an io.Reader to an io.Writer while displaying a progressbar.
// It requires knowing the contentLength.
type copyBytesBar struct {
	w                             io.Writer
	bar                           *progressbar.ProgressBar
	contentLength, amountWritten  int64
	barUnit, numUnits, addedUnits int64
}

// newCopyBytesBar creates a new copyBytesBar. It requires knowing the contentLength.
func newCopyBytesBar(w io.Writer, contentLength int64) *copyBytesBar {
	bar := &copyBytesBar{w: w}
	bar.barUnit = 1
	for contentLength > bar.barUnit*1024*1024 {
		bar.barUnit *= 1024
	}
	bar.numUnits = (contentLength + bar.barUnit - 1) / bar.barUnit
	bar.bar = progressbar.NewOptions(int(bar.numUnits),
		progressbar.OptionSetDescription(fmt.Sprintf("%s", ByteCountIEC(contentLength))),
		progressbar.OptionUseANSICodes(true),
		progressbar.OptionEnableColorCodes(true),
		progressbar.OptionSetTheme(progressbar.Theme{
			Saucer:        "[green]=[reset]",
			SaucerHead:    "[green]>[reset]",
			SaucerPadding: ".",
			BarStart:      "[",
			BarEnd:        "]",
		}),
	)
	return bar
}

// Write implements io.Write, while updating the progress bar.
func (bar *copyBytesBar) Write(p []byte) (n int, err error) {
	n, err = bar.w.Write(p)
	bar.amountWritten += int64(n)
	toUnits := bar.amountWritten / bar.barUnit
	if toUnits > bar.addedUnits {
		_ = bar.bar.Add(int(toUnits - bar.addedUnits))
		bar.addedUnits = toUnits
	}
	return
}

// CopyWithProgressBar is similar to io.Copy, but updates the progress bar with the amount
// of data copied.
//
// It requires knowing the amount of data to copy up-front.
func CopyWithProgressBar(dst io.Writer, src io.Reader, contentLength int64) (n int64, err error) {
	bar := newCopyBytesBar(dst, contentLength)
	n, err = io.Copy(bar, src)
	if bar.addedUnits < bar.numUnits {
		_ = bar.bar.Add(int(bar.numUnits - bar.addedUnits))
	}
	_ = bar.bar.Close()
	fmt.Println()
	return
}

// Download file from url and save at given path.
// Optionally, use showProgressBar.
func Download(url, path string, showProgressBar bool) (size int64, err error) {
	var file *os.File
	file, err = os.Create(path)
	if err != nil {
		return 0, errors.Wrapf(err, "failed creating file %q", path)
	}
	client := http.Client{
		CheckRedirect: func(r *http.Request, via []*http.Request) error {
			r.URL.Opaque = r.URL.Path
			return nil
		},
	}
	var resp *http.Response
	resp, err = client.Get(url)
	if err != nil {
		return 0, errors.Wrapf(err, "failed downloading %q", url)
	}

	if showProgressBar {
		size, err = CopyWithProgressBar(file, resp.Body, resp.ContentLength)
	} else {
		size, err = io.Copy(file, resp.Body)
	}
	if err != nil {
		return 0, errors.Wrapf(err, "downloading %q to %q", url, path)
	}
	err = file.Close()
	if err != nil {
		return 0, errors.Wrapf(err, "failed closing %q", path)
	}
	err = resp.Body.Close()
	if err != nil {
		return 0, errors.Wrapf(err, "failed closing connection to %q", url)
	}
	return size, nil
}

// DownloadIfMissing will check if the path exists already, and if not it will download the file
// from the given URL.
//
// If checkHash is provided, it checks that the file has the hash or fail.
func DownloadIfMissing(url, path, checkHash string) error {
	if !FileExists(path) {
		// Download compressed file first.
		fmt.Printf("Downloading %s ...\n", url)
		_, err := Download(url, path, true)
		if err != nil {
			return err
		}
	}
	if checkHash == "" {
		return nil
	}
	return ValidateChecksum(path, checkHash)
}

// Untar file, using decompression flags according to suffix: .gz for gzip, bz2 for bzip2.
func Untar(baseDir, tarFile string) error {
	baseDir = ReplaceTildeInDir(baseDir)
	compressionFlag := ""
	if strings.HasSuffix(tarFile, ".gz") || strings.HasSuffix(tarFile, ".tgz") {
		compressionFlag = "z"
	} else if strings.HasSuffix(tarFile, ".bz2") {
		compressionFlag = "j"
	}
	cmd := exec.Command("tar", fmt.Sprintf("x%sf", compressionFlag), tarFile)
	cmd.Dir = baseDir
	err := cmd.Run()
	if err != nil {
		return errors.Wrapf(err, "failed to run %q", cmd)
	}
	return nil
}

// DownloadAndUntarIfMissing downloads tarFile from given url, if file not there yet, and then untar it
// if the target directory is missing.
//
// If checkHash is provided, it checks that the file has the hash or fail.
func DownloadAndUntarIfMissing(url, baseDir, tarFile, targetUntarDir, checkHash string) error {
	baseDir = ReplaceTildeInDir(baseDir)
	if !path.IsAbs(tarFile) {
		tarFile = path.Join(baseDir, tarFile)
	}
	if !path.IsAbs(targetUntarDir) {
		targetUntarDir = path.Join(baseDir, targetUntarDir)
	}
	if FileExists(targetUntarDir) {
		return nil
	}
	err := DownloadIfMissing(url, tarFile, checkHash)
	if err != nil {
		return err
	}
	err = Untar(baseDir, tarFile)
	if err != nil {
		return err
	}
	if !FileExists(targetUntarDir) {
		return errors.Errorf("downloaded from %q and untar'ed %q, but didn't get directory %q", url, tarFile, targetUntarDir)
	}
	return nil
}

// DownloadAndUnzipIfMissing downloads zipFile from given url, if file not there yet, and then unzip it
// under directory `unzipBaseDir`. If the target `targetUnzipDir` directory is missing.
//
// If checkHash is provided, it checks that the file has the hash or fail.
func DownloadAndUnzipIfMissing(url, zipFile, unzipBaseDir, targetUnzipDir, checkHash string) error {
	if FileExists(targetUnzipDir) {
		return nil
	}
	err := DownloadIfMissing(url, zipFile, checkHash)
	if err != nil {
		return err
	}
	err = Unzip(zipFile, unzipBaseDir)
	if err != nil {
		return err
	}
	if !FileExists(targetUnzipDir) {
		return errors.Errorf("downloaded from %q and unzip'ed %q, but didn't get directory %q", url, zipFile, targetUnzipDir)
	}
	return nil
}

// Unzip file, from the given zipBaseDir.
func Unzip(zipFile, zipBaseDir string) error {
	cmd := exec.Command("unzip", "-u", zipFile)
	cmd.Dir = zipBaseDir
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		//fmt.Printf("%v", cmd.Error())
		return errors.Wrapf(err, "failed to run %q", cmd)
	}
	return nil
}

// ParallelDataset is a wrapper around a `train.Dataset` that parallelize calls to Yield.
type ParallelDataset struct {
	Dataset train.Dataset

	// Parallelism is the number of goroutines started generating examples.
	Parallelism int

	// BufferSize is the size of the cache of pre-generated batches.
	BufferSize int

	// impl is the actual implementation.
	impl *parallelDatasetImpl

	// keepAlive is used only to keep ParallelDataset alive in the middle of long calls.
	keepAlive int64
}

type yieldUnit struct {
	spec   any
	inputs []tensor.Tensor
	labels []tensor.Tensor
}

// parallelDatasetImpl separates the implementation of ParallelDataset. It's important
// that it doesn't point back to the original ParallelDataset, so garbage collecting
// will also stop the goroutines.
type parallelDatasetImpl struct {
	config ParallelDataset // A copy of ht

	err   error
	muErr sync.Mutex

	cache                                 chan yieldUnit
	epochFinished, stopEpoch, stopDataset chan struct{}
}

// NewParallelDataset starts building a ParallelDataset that can be used to parallelize any
// train.Dataset, as long as the underlying dataset ds is thread-safe.
//
// parallelism is the number of goroutines to start, each calling `ds.Yield()` in parallel
// to accelerate the generation of batches. Set to 0, and it will use the number of cores in
// the system plus 1.
//
// bufferSize is the space reserved to keep pre-generated batches. This is on top of the batches
// being generated in each goroutine.
//
// To avoid leaking goroutines, call ParallelDataset.Cancel when exiting.
func NewParallelDataset(ds train.Dataset, parallelism, bufferSize int) *ParallelDataset {
	if parallelism == 0 {
		parallelism = runtime.NumCPU() + 1
	}
	impl := &parallelDatasetImpl{
		cache:       make(chan yieldUnit, bufferSize),
		stopDataset: make(chan struct{}),
	}
	pd := &ParallelDataset{
		Dataset:     ds,
		Parallelism: parallelism,
		BufferSize:  bufferSize,
	}
	impl.config = *pd
	pd.impl = impl
	// If the ParallelDataset is garbage collected, stop all parallel goroutines.
	runtime.SetFinalizer(pd, func(pd *ParallelDataset) {
		if pd.impl != nil {
			close(pd.impl.stopDataset)
			pd.impl = nil
		}
	})

	// Start goroutines
	impl.startGoRoutines()
	return pd
}

func (impl *parallelDatasetImpl) startGoRoutines() {
	impl.epochFinished = make(chan struct{})
	impl.stopEpoch = make(chan struct{})
	var wg sync.WaitGroup
	for ii := 0; ii < impl.config.Parallelism; ii++ {
		// Start all goroutines.
		wg.Add(1)
		go func(impl *parallelDatasetImpl) {
			defer wg.Done()
			for {
				select {
				case <-impl.stopEpoch:
					return
				case <-impl.stopDataset:
					return
				default:
					// Move forward and generate the next batch.
				}
				var unit yieldUnit
				var err error
				unit.spec, unit.inputs, unit.labels, err = impl.config.Dataset.Yield()
				if err == io.EOF {
					return
				}
				if err != nil {
					log.Printf("Error: %+v", err)
					// Fatal error, stop everything.
					impl.muErr.Lock()
					if impl.err != nil {
						impl.err = err
					}
					close(impl.stopEpoch)
					close(impl.stopDataset)
					impl.muErr.Unlock()
					return
				}
				select {
				case <-impl.stopEpoch:
					return
				case <-impl.stopDataset:
					return
				case impl.cache <- unit:
					// Batch generated and cached, move to next.
					continue
				}
			}
		}(impl)
	}

	// Start controller job.
	go func() {
		wg.Wait()
		impl.muErr.Lock()
		defer impl.muErr.Unlock()
		select {
		case <-impl.stopDataset:
			return
		default:
			//
		}
		close(impl.epochFinished)
	}()
}

// Name implements train.Dataset.
func (pd *ParallelDataset) Name() string {
	return pd.Dataset.Name()
}

// Reset implements train.Dataset.
func (pd *ParallelDataset) Reset() {
	impl := pd.impl
	impl.muErr.Lock()
	close(impl.stopEpoch) // Indicate to goroutines to stop generating batches.
	impl.muErr.Unlock()
	select {
	case <-impl.stopDataset:
		// Return immediately, do nothing.
		return
	case <-impl.cache:
		// Discard remaining entries in cache.
	case <-impl.epochFinished:
		// All finished, we can move on.
	}

	// Reset underlying dataset and start again.
	impl.config.Dataset.Reset()
	impl.startGoRoutines()

	// This no-op prevents `pd` from being garbage collected and the goroutines killed in the middle
	// of the Reset operation. Leave this at the end.
	pd.keepAlive++
}

// Yield implements train.Dataset.
func (pd *ParallelDataset) Yield() (spec any, inputs []tensor.Tensor, labels []tensor.Tensor, err error) {
	impl := pd.impl
	var unit yieldUnit
	select {
	case <-impl.stopDataset:
		// An error occurred, dataset is closed.
		impl.muErr.Lock()
		err = impl.err
		impl.muErr.Unlock()
		return
	case unit = <-impl.cache:
		// We got a new batch
	case <-impl.epochFinished:
		// No more records being produced (until Reset() is called), but we still need to exhaust the cache.
		select {
		case unit = <-impl.cache:
			// We got a new batch, simply continue.
		default:
			// Generation exhausted, and no more records in cache.
			err = io.EOF
			return
		}
	}
	spec, inputs, labels = unit.spec, unit.inputs, unit.labels

	// This no-op prevents `pd` from being garbage collected and the goroutines killed in the middle
	// of the Yield operation. Leave this at the end.
	pd.keepAlive++
	return
}
