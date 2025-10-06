// Package fsutil contains utilities for working with the file system.
package fsutil

import (
	"io"
	"os"
	"os/user"
	"path"
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
		return dir, nil
	}
	if dir[0] != '~' {
		return dir, nil
	}
	var userName string
	if dir != "~" && !strings.HasPrefix(dir, "~/") {
		sepIdx := strings.IndexRune(dir, '/')
		if sepIdx == -1 {
			userName = dir[1:]
		} else {
			userName = dir[1:sepIdx]
		}
	}
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
	return path.Join(homeDir, dir[1+len(userName):]), nil
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
