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
	"github.com/pkg/errors"
	"github.com/schollz/progressbar/v3"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/user"
	"path"
	"strings"
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

// Download file from url and save at given path. Attempts to create directory
// if it doesn't yet exist.
//
// Optionally, use showProgressBar.
func Download(url, filePath string, showProgressBar bool) (size int64, err error) {
	filePath = ReplaceTildeInDir(filePath)
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
	filePath = ReplaceTildeInDir(filePath)
	if !FileExists(filePath) {
		// Download compressed file first.
		fmt.Printf("Downloading %s ...\n", url)
		_, err := Download(url, filePath, true)
		if err != nil {
			return err
		}
	}
	if checkHash == "" {
		return nil
	}
	return ValidateChecksum(filePath, checkHash)
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
