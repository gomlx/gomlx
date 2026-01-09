// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package oxfordflowers102

import (
	"image"
	"image/jpeg"
	"log"
	"os"
	"path"
	"sort"
	"strings"

	"github.com/daniellowtw/matlab"
	"github.com/gomlx/gomlx/examples/downloader"
	"github.com/gomlx/gomlx/pkg/support/fsutil"
	"github.com/pkg/errors"
)

var (
	DownloadBaseURL           = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
	DownloadSubdir            = "downloads"
	DownloadFilesAndChecksums = []struct {
		File, Checksum, UntarDir string
	}{
		// Order matters below:
		{"102flowers.tgz", "", "jpg"}, // Always a different checksum :(
		{"imagelabels.mat", "4903e94206bac23bf772aadf06451916df56b58fc483a62db32a97b82656651d", ""},
		{"setid.mat", "46b8678f91fd95d3c8f4feab80d271a6c834a1dd896fe29fd3e6ad9ce5c8dccd", ""},
	}
)

// DownloadAndParse "Oxford Flowers 102" dataset files to baseDir and untar it.
// If files are already downloaded, their previous copy is used.
//
// After download, the contents of the files are parsed, and the global AllLabels is set.
func DownloadAndParse(baseDir string) error {
	baseDir = fsutil.MustReplaceTildeInDir(baseDir) // If baseDir starts with "~", it is replaced.
	downloadPath := path.Join(baseDir, DownloadSubdir)
	if err := os.MkdirAll(downloadPath, 0777); err != nil && !os.IsExist(err) {
		return errors.Wrapf(err, "Failed to create path for downloading %q", downloadPath)
	}

	// DownloadAndParse files.
	for _, file := range DownloadFilesAndChecksums {
		filePath := path.Join(downloadPath, file.File)
		url := DownloadBaseURL + file.File
		if file.UntarDir == "" {
			err := downloader.DownloadIfMissing(url, filePath, file.Checksum)
			if err != nil {
				return errors.Wrapf(err, "Failed to download %q from %q", file.File, url)
			}
		} else {
			err := downloader.DownloadAndUntarIfMissing(url, baseDir, filePath, file.UntarDir, file.Checksum)
			if err != nil {
				return errors.Wrapf(err, "Failed to download and untar %q from %q", file.File, url)
			}
		}
	}

	if err := ParseLabels(path.Join(downloadPath, DownloadFilesAndChecksums[1].File)); err != nil {
		return err
	}
	if err := ParseImages(path.Join(baseDir, DownloadFilesAndChecksums[0].UntarDir)); err != nil {
		return err
	}
	NumExamples = len(AllLabels)
	if len(AllLabels) != len(AllImages) {
		return errors.Errorf("after downloading got %d labels from %s, and %d images in %s, numbers don't match!",
			len(AllLabels), DownloadFilesAndChecksums[1].File, len(AllImages), ImagesDir)
	}
	return nil
}

func ParseImages(dirPath string) error {
	ImagesDir = dirPath
	entries, err := os.ReadDir(dirPath)
	if err != nil {
		return errors.Errorf("failed to scan images in directory %q", dirPath)
	}
	AllImages = make([]string, 0, len(entries))
	for _, entry := range entries {
		name := entry.Name()
		if !strings.HasSuffix(name, ".jpg") {
			log.Printf("Invalid image file found: %q", path.Join(ImagesDir, name))
			continue
		}
		AllImages = append(AllImages, entry.Name())
	}
	sort.Strings(AllImages)
	// We assume order of labels match sorted image names.
	// But just in case we check that the image numbers are increasing and not missing and in order.
	return nil
}

func ParseLabels(filePath string) error {
	f, err := os.Open(filePath)
	if err != nil {
		return errors.Wrapf(err, "failed to open labels file %q", filePath)
	}
	defer func() { _ = f.Close() }()

	matlabFile, err := matlab.NewFileFromReader(f)
	if err != nil {
		return errors.Wrapf(err, "failed to parse labels file %q", filePath)
	}
	matLabels, found := matlabFile.GetVar("labels")
	if !found {
		return errors.Errorf("failed to parse var \"labels\" in Matlab file %q", filePath)
	}

	values := matLabels.Value()
	AllLabels = make([]int32, len(values))
	for ii, value := range values {
		AllLabels[ii] = int32(value.(uint8)) - 1 // The original labels are 1-based.
	}
	return nil
}

// ReadExample reads an image for the example idx. The example idx must be between 0 and
// NumExamples.
func ReadExample(idx int) (img image.Image, label int32, err error) {
	if NumExamples == 0 {
		err = errors.Errorf("either oxfordflowers102.DownloadAndParse hasn't been called yet, or " +
			"it failed for some reason -- no examples available to read")
		return
	}
	if idx < 0 || idx >= NumExamples {
		err = errors.Errorf("examaple index %d invalid: there are only %d examples", idx, NumExamples)
		return
	}
	imagePath := path.Join(ImagesDir, AllImages[idx])
	f, err := os.Open(imagePath)
	if err != nil {
		err = errors.Wrapf(err, "failed to open image %d in %q", idx, imagePath)
		return
	}
	defer func() { _ = f.Close() }()
	img, err = jpeg.Decode(f)
	if err != nil {
		err = errors.Wrapf(err, "failed to parse JPEG for image %d in %q", idx, imagePath)
		return
	}
	label = AllLabels[idx]
	return
}
