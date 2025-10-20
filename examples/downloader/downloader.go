// Package downloader provides functions for downloading and extracting files.
package downloader

import (
	"compress/gzip"
	"encoding/csv"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path"
	"strings"

	"github.com/gomlx/gomlx/pkg/support/fsutil"
	"github.com/pkg/errors"
	"github.com/schollz/progressbar/v3"
)

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
		progressbar.OptionSetDescription(fmt.Sprintf("%s", fsutil.ByteCountIEC(contentLength))),
		progressbar.OptionUseANSICodes(true),
		progressbar.OptionEnableColorCodes(true),
		progressbar.OptionSetTheme(progressbar.ThemeUnicode),
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

// Download file from url and save it at the given path.
// It attempts to create the directory if it doesn't yet exist.
//
// Optionally, use showProgressBar.
func Download(url, filePath string, showProgressBar bool) (size int64, err error) {
	filePath = fsutil.MustReplaceTildeInDir(filePath)
	err = os.MkdirAll(path.Dir(filePath), 0777)
	if err != nil && !os.IsExist(err) {
		err = errors.Wrapf(err, "Failed to create the directory for the path: %q", path.Dir(filePath))
		return
	}
	var file *os.File
	file, err = os.Create(filePath)
	if err != nil {
		return 0, errors.Wrapf(err, "failed creating file %q", filePath)
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
		return 0, errors.Wrapf(err, "downloading %q to %q", url, filePath)
	}
	err = file.Close()
	if err != nil {
		return 0, errors.Wrapf(err, "failed closing %q", filePath)
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
func DownloadIfMissing(url, filePath, checkHash string) error {
	filePath = fsutil.MustReplaceTildeInDir(filePath)
	if !fsutil.MustFileExists(filePath) {
		// Download the compressed file first.
		fmt.Printf("Downloading %s ...\n", url)
		_, err := Download(url, filePath, true)
		if err != nil {
			return err
		}
	}
	if checkHash == "" {
		return nil
	}
	return fsutil.ValidateChecksum(filePath, checkHash)
}

// Untar file, using decompression flags according to suffix: .gz for gzip, bz2 for bzip2.
func Untar(baseDir, tarFile string) error {
	baseDir = fsutil.MustReplaceTildeInDir(baseDir)
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
	baseDir = fsutil.MustReplaceTildeInDir(baseDir)
	if !path.IsAbs(tarFile) {
		tarFile = path.Join(baseDir, tarFile)
	}
	if !path.IsAbs(targetUntarDir) {
		targetUntarDir = path.Join(baseDir, targetUntarDir)
	}
	if fsutil.MustFileExists(targetUntarDir) {
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
	if !fsutil.MustFileExists(targetUntarDir) {
		return errors.Errorf("downloaded from %q and untar'ed %q, but didn't get directory %q", url, tarFile, targetUntarDir)
	}
	return nil
}

// DownloadAndUnzipIfMissing downloads `zipFile` from given url, if file not there yet.
// And then unzip it under directory `unzipBaseDir`, if the target `targetUnzipDir` directory is missing.
//
// It's recommended that all paths be absolute.
//
// If checkHash is provided, it checks that the file has the hash or fail.
func DownloadAndUnzipIfMissing(url, zipFile, unzipBaseDir, targetUnzipDir, checkHash string) error {
	if fsutil.MustFileExists(targetUnzipDir) {
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
	if !fsutil.MustFileExists(targetUnzipDir) {
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

// ParseGzipCSVFile opens a `CSV.gz` file and iterates over each of its rows, calling `perRowFn`, with a slice
// of strings for each cell value in the row.
func ParseGzipCSVFile(filePath string, perRowFn func(row []string) error) error {
	f, err := os.Open(filePath)
	if err != nil {
		return errors.Wrapf(err, "failed to open file %q", filePath)
	}
	defer func() { _ = f.Close() }()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return errors.Wrapf(err, "failed to un-gzip file %q", filePath)
	}
	r := csv.NewReader(gz)
	var record []string
	for {
		record, err = r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return errors.Wrapf(err, "while reading gzip+csv %q", filePath)
		}
		err = perRowFn(record)
		if err != nil {
			return errors.WithMessagef(err, "while processing file %q", filePath)
		}
	}
	return nil
}
