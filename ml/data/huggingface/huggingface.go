// Package huggingface 🤗 provides functionality do download HuggingFace (HF) models and extract tensors
// stored in the ".safetensors" format.
//
// Example: Download (only the first time) and enumerate all the tensors from Google's Gemma v2 model:
//
//	import (
//		"github.com/janpfeifer/must"
//		hfd "github.com/gomlx/gomlx/ml/data"
//		hfd "github.com/gomlx/gomlx/ml/data/huggingface"
//	)
//
//	var (
//		hfModelId = "google/gemma-2-2b-it"
//		hfToken = "..."  // Create a read-only token for you in HuggingFace site.
//		flagDataDir = flag.String("data", "~/work/gemma", "Directory to cache downloaded and generated dataset files.")
//	)
//
//	func HuggingFaceDir() string {
//		dataDir := data.ReplaceTildeInDir(*flagDataDir)
//		return path.Join(dataDir, "huggingface")
//	}
//
//	func main() {
//		flag.Parse()
//		hfm := must.M1(hfd.New(hfModelId, hfToken, HuggingFaceDir()))
//		for e, err := range hfm.EnumerateTensors() {
//			must.M(err)
//			fmt.Printf("\t%s -> %s\n", e.Name, e.Tensor.Shape())
//		}
//	}
package huggingface

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/dustin/go-humanize"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/data/downloader"
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/xsync"
	"github.com/pkg/errors"
	"iter"
	"os"
	"path"
	"strings"
	"sync"
	"time"
)

type Model struct {
	// ID may include owner/model. E.g.: google/gemma-2-2b-it
	ID string

	// AuthToken is the HuggingFace authentication token to be used when downloading the files.
	AuthToken string

	// BaseDir is where the local copy of the model is stored.
	BaseDir string

	// Verbosity: 0 for quiet operation; 1 for information about progress; 2 and higher for debugging.
	Verbosity int

	// MaxParallelDownload indicates how many files to download at the same time. Default is 20.
	// If set to <= 0 it will download all files in parallel.
	// Set to 1 to make downloads sequential.
	MaxParallelDownload int

	// Info downloaded from model.
	// It is only available after DownloadInfo is called.
	Info *Info
}

// New creates a reference to a HuggingFace model given its id.
//
// The id typically include owner/model. E.g.: "google/gemma-2-2b-it"
//
// The authToken can be created in HuggingFace site, in the profile settings page. A "read-only" token will do for
// most models.
// Leave empty if not using one (but some models can't be downloaded without it).
//
// The baseDir is suffixed with the model's id (after converting "/" to "_").
// So the same baseDir can be used to hold different models.
func New(id string, authToken, baseDir string) (*Model, error) {
	baseDir = data.ReplaceTildeInDir(baseDir)
	if !path.IsAbs(baseDir) {
		workingDir, err := os.Getwd()
		if err != nil {
			return nil, errors.Wrapf(err, "cannot find current working dir for huggingface.New() baseDir")
		}
		baseDir = path.Join(workingDir, baseDir)
	}
	baseDir = path.Join(baseDir, strings.Replace(id, "/", "_", -1))
	err := os.MkdirAll(baseDir, 0755)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to create base directory for model %q in %q", id, baseDir)
	}
	return &Model{
		ID:                  id,
		AuthToken:           authToken,
		BaseDir:             baseDir,
		Verbosity:           1,
		MaxParallelDownload: 20, // At most 20 parallel downloads.
	}, nil
}

// InfoFile is the file with the information about the model.
// The info about a model is fetched once and cached on this file, to prevent going to the network.
const InfoFile = "_info_.json"

// Info holds information about a HuggingFace model, it is the json served when hitting the URL
// https://huggingface.co/api/models/<model_id>
type Info struct {
	ID          string          `json:"id"`
	ModelID     string          `json:"model_id"`
	Author      string          `json:"author"`
	SHA         string          `json:"sha"`
	Tags        []string        `json:"tags"`
	Siblings    []*FileInfo     `json:"siblings"`
	SafeTensors SafeTensorsInfo `json:"safetensors"`
}

// FileInfo represents one of the model file, in the Info structure.
type FileInfo struct {
	Name string `json:"rfilename"`
}

// SafeTensorsInfo holds counts on number of parameters of various types.
type SafeTensorsInfo struct {
	Total int

	// Parameters: maps dtype name to int.
	Parameters map[string]int
}

// DownloadInfo structure about the model -- or read it from disk if it is cached locally already.
// It sets Model.Info with the downloaded information if successful.
func (hfm *Model) DownloadInfo() error {
	if hfm.Info != nil {
		return nil
	}
	infoFilePath := path.Join(hfm.BaseDir, InfoFile)
	if !data.FileExists(infoFilePath) {
		// Download Model's info file from network.
		_, err := data.Download(hfm.infoURL(), infoFilePath, true)
		if err != nil {
			return errors.WithMessagef(err, "failed to download info for model from %q", hfm.infoURL())
		}
	}

	// Read _info_.json from disk.
	infoJson, err := os.ReadFile(infoFilePath)
	if err != nil {
		return errors.Wrapf(err, "failed to read info for model from disk in %q -- remove the file if you want to have it re-downloaded",
			infoFilePath)
	}

	decoder := json.NewDecoder(bytes.NewReader(infoJson))
	if err = decoder.Decode(&hfm.Info); err != nil {
		return errors.Wrapf(err, "failed to parse info for model in %q (downloaded from %q)",
			infoFilePath, hfm.infoURL())
	}
	return nil
}

// FileNameAndPath to a files for a model.
// Name is stored in the info "Siblings" field, and Path is the path in the local storage.
type FileNameAndPath struct {
	Name, Path string
}

// EnumerateFileNames loads the model info and lists the file names stored for the model.
// It doesn't download the files, only lists their relative name, and their local storage path.
//
// See Model.Download to actually download the files.
func (hfm *Model) EnumerateFileNames() iter.Seq2[FileNameAndPath, error] {
	// Download info and files.
	err := hfm.DownloadInfo()
	if err != nil {
		// Error downloading: yield error only.
		return func(yield func(FileNameAndPath, error) bool) {
			yield(FileNameAndPath{}, err)
			return
		}
	}
	return func(yield func(FileNameAndPath, error) bool) {
		for _, si := range hfm.Info.Siblings {
			fileName := si.Name
			if path.IsAbs(fileName) || strings.Index(fileName, "..") != -1 {
				yield(FileNameAndPath{}, errors.Errorf("model %q contains illegal file name %q -- it cannot be an absolute path, nor contain \"..\"",
					hfm.ID, fileName))
				return
			}
			filePath := path.Join(hfm.BaseDir, fileName)
			if !yield(FileNameAndPath{Name: fileName, Path: filePath}, nil) {
				return
			}
		}
		return
	}
}

// Download first download the info about the model, with the list of files associated with the model, and then all
// the model files.
//
// It then downloads any files not available locally yet -- files are downloaded to a ".downloading" suffix, and moved
// to the final destination once they finished to download.
func (hfm *Model) Download() error {
	requireDownload := types.MakeSet[string](10)
	for f, err := range hfm.EnumerateFileNames() {
		if err != nil {
			return err
		}
		if !data.FileExists(f.Path) {
			requireDownload.Insert(f.Name)
		}
	}

	mgr := downloader.New().WithAuthToken(hfm.AuthToken)
	type DownloadInfo struct {
		cancel *xsync.Latch
		bytes  int64
	}
	downloading := make(map[string]*DownloadInfo, len(requireDownload))
	var downloadingMu sync.Mutex
	var wg sync.WaitGroup
	var allFilesBytes uint64
	numDownloadedFiles := 0
	var firstError error
	busyLoop := `-\|/`
	busyLoopPos := 0
	lastPrintTime := time.Now()

	for fileName := range requireDownload {
		wg.Add(1)
		filePath := path.Join(hfm.BaseDir, fileName)
		downloadingMu.Lock()
		canceller := mgr.Download(hfm.urlForFile(fileName), filePath+".downloading", func(downloadedBytes, totalBytes int64, finished bool, err error) {
			// Execute at every report of download.
			downloadingMu.Lock()
			defer downloadingMu.Unlock()

			if err == nil {
				downloadInfo := downloading[fileName]
				if downloadInfo != nil {
					newBytes := downloadedBytes - downloadInfo.bytes
					allFilesBytes += uint64(newBytes)
					downloadInfo.bytes = downloadedBytes
				}
			}
			if err != nil {
				if firstError == nil {
					firstError = err
				}
				for _, di := range downloading {
					di.cancel.Trigger()
				}
			}
			if finished {
				delete(downloading, fileName)
				numDownloadedFiles++
			}
			ratePrint := finished || time.Since(lastPrintTime) > time.Second
			if ratePrint {
				if firstError == nil {
					fmt.Printf("\rDownloaded %d/%d files %c %s downloaded    ",
						numDownloadedFiles, len(requireDownload), busyLoop[busyLoopPos], humanize.Bytes(allFilesBytes))
				} else {
					fmt.Printf("\rDownloaded %d/%d files: error - %v     ", numDownloadedFiles, len(requireDownload), firstError)
				}
				busyLoopPos = (busyLoopPos + 1) % len(busyLoop)
				lastPrintTime = time.Now()
			}
			if finished {
				if err == nil {
					err = os.Rename(filePath+".downloading", filePath)
					if err != nil {
						if firstError == nil {
							firstError = errors.Wrapf(err, "failed to rename file %q", filePath)
							for _, di := range downloading {
								di.cancel.Trigger()
							}
						}
					}
				}
				wg.Done()
			}
		})
		downloading[fileName] = &DownloadInfo{canceller, 0}
		downloadingMu.Unlock()
	}
	wg.Wait()
	if len(requireDownload) > 0 {
		fmt.Println()
	}
	if firstError != nil {
		return firstError
	}
	return nil
}

// EnumerateTensors returns an iterator over all the tensors stored in ".safetensors" files,
// already converted to GoMLX *tensors.Tensor, with their associated names.
//
// It calls Download first, to make sure the files are already there.
func (hfm *Model) EnumerateTensors() iter.Seq2[*NamedTensor, error] {
	// Download info and files.
	err := hfm.Download()
	if err != nil {
		// Error downloading: yield error only.
		return func(yield func(*NamedTensor, error) bool) {
			yield(nil, err)
			return
		}
	}

	return func(yield func(*NamedTensor, error) bool) {
		for fInfo, err := range hfm.EnumerateFileNames() {
			if err != nil {
				yield(nil, err)
				return
			}
			if path.Ext(fInfo.Name) != ".safetensors" {
				continue
			}
			f, err := os.Open(fInfo.Path)
			if err != nil {
				err = errors.Wrapf(err, "failed to open %q", fInfo.Path)
				yield(nil, err)
				return
			}
			for tInfo, err := range scanSafetensorsFile(f) {
				if err != nil {
					yield(nil, err)
					return
				}
				if !yield(tInfo, nil) {
					return
				}
			}
		}
	}
}

func (hfm *Model) infoURL() string {
	return fmt.Sprintf("https://huggingface.co/api/models/%s", hfm.ID)
}

func (hfm *Model) urlForFile(fileName string) string {
	return fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", hfm.ID, fileName)
}
