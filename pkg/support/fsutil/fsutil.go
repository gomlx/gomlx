// Package fsutil contains utilities for working with the file system.
package fsutil

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"log"
	"os"
	"os/user"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/pkg/errors"
)

// MustFileExists returns whether the file or directory exists.
// It panics on file system errors.
func MustFileExists(path string) bool {
	exists, err := FileExists(path)
	if err != nil {
		panic(err)
	}
	return exists
}

// FileExists returns whether the file or directory exists or an error if something went wrong in the filesystem.
func FileExists(path string) (bool, error) {
	_, err := os.Stat(path)
	if err == nil {
		return true, nil
	}
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	return false, errors.Wrapf(err, "failed to FileExists(%q)", path)
}

// MustReplaceTildeInDir by the user's home directory. Returns dir if it doesn't start with "~".
//
// It may panic with an error if `dir` has an unknown user (e.g: `~unknown/...`)
func MustReplaceTildeInDir(dir string) string {
	dir, err := ReplaceTildeInDir(dir)
	if err != nil {
		panic(err)
	}
	return dir
}

// ReplaceTildeInDir by the user's home directory. Returns dir if it doesn't start with "~".
//
// It returns an error if `dir` has an unknown user or some other filesystem error (e.g: `~unknown/...`)
func ReplaceTildeInDir(dir string) (string, error) {
	if len(dir) == 0 {
		return "", nil
	}
	if dir[0] != '~' {
		return dir, nil
	}

	// Accept either '/' or '\' as separator following the user name.
	sepIdx := -1
	if runtime.GOOS != "windows" {
		sepIdx = strings.IndexRune(dir, filepath.Separator)
	} else {
		// In windows we accept both: "/" and "\\".
		sepIdxUnix := strings.IndexRune(dir, '/')
		sepIdxWin := strings.IndexRune(dir, '\\')

		// Find the earliest separator (if any)
		if sepIdxUnix == -1 {
			sepIdx = sepIdxWin
		} else if sepIdxWin == -1 {
			sepIdx = sepIdxUnix
		} else if sepIdxUnix < sepIdxWin {
			sepIdx = sepIdxUnix
		} else {
			sepIdx = sepIdxWin
		}
	}

	// Find user name after the tilde, if one is given.
	var userName string
	if dir != "~" && sepIdx != 1 { // "~/" or "~\\"
		// Extract the username, whatever the first separator is
		if sepIdx == -1 {
			userName = dir[1:]
		} else {
			userName = dir[1:sepIdx]
		}
	}

	// Retrive user and their home directory.
	var usr *user.User
	var err error
	if userName == "" {
		usr, err = user.Current()
	} else {
		usr, err = user.Lookup(userName)
	}
	if err != nil {
		return "", errors.Wrapf(err, "failed to lookup home directory for user in path %q", dir)
	}
	homeDir := usr.HomeDir
	// Replace ~ or ~user with user home, preserve any following path, no matter the separator
	remaining := ""
	if userName == "" {
		remaining = dir[1:]
	} else {
		remaining = dir[1+len(userName):]
	}
	// If remaining starts with '/' or '\', remove it so Join works as expected.
	remaining = strings.TrimLeft(remaining, `/\`)

	return filepath.Join(homeDir, remaining), nil
}

// ReportedClose closes the closer object (a file?) and reports in case of error.
//
// It reports by setting reportErrPtr if it's set and not in use yet. Otherwise, it logs the error and continues.
//
// Use this in a defer statement.
func ReportedClose(closer io.Closer, name string, reportErrPtr *error) {
	err := closer.Close()
	if err == nil {
		return
	}
	if reportErrPtr == nil || *reportErrPtr != nil {
		// Simply report the closing error.
		return
	}
	// Set the returning error.
	*reportErrPtr = errors.Wrapf(err, "failed to close %s", name)
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
		_ = f.Close() // Discard the error on Close.
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
func ByteCountIEC[T interface {
	int | int64 | uint64 | uint | uintptr
}](count T) string {
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
