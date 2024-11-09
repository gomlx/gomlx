// Package downloader implements download in parallel of various URLs, with various progress report callback.
package downloader

import (
	"context"
	"fmt"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/types/xsync"
	"github.com/pkg/errors"
	"io"
	"net/http"
	"os"
	"path"
)

// ProgressCallback is called as download progresses.
//   - totalBytes may be set to 0 if total size is not yet known.
type ProgressCallback func(downloadedBytes, totalBytes int64)

// Manager handles downloads, reporting back progress and errors.
type Manager struct {
	semaphore            *xsync.Semaphore
	authToken, userAgent string
}

// New creates a Manager that download files in parallel -- by default mostly 20 in parallel.
func New() *Manager {
	return &Manager{semaphore: xsync.NewSemaphore(20)}
}

// MaxParallel indicates how many files to download at the same time. Default is 20.
// If set to <= 0 it will download all files in parallel.
// Set to 1 to make downloads sequential.
func (m *Manager) MaxParallel(n int) *Manager {
	m.semaphore.Resize(n)
	return m
}

// WithAuthToken sets the authentication token to use in the requests.
// It is passed in the header "Authorization" and prefixed with "Bearer ".
func (m *Manager) WithAuthToken(authToken string) *Manager {
	m.authToken = authToken
	return m
}

// WithUserAgent sets the user agent to user.
func (m *Manager) WithUserAgent(userAgent string) *Manager {
	m.userAgent = userAgent
	return m
}

var CancellationError = errors.New("download cancelled")

// setRequestHeader with configured fields.
func (m *Manager) setRequestHeader(req *http.Request) {
	if m.authToken != "" {
		req.Header.Set("Authorization", "Bearer "+m.authToken)
	}
	if m.userAgent != "" {
		req.Header.Set("user-agent", m.userAgent)
	}
}

// Download downloads the given url to be downloaded to the given filePath.
// This may lock if it reached the maximum number of parallel downloads.
// Consider calling this on its own go-routine.
//
// Progress of download is reported back to the given callback, if not nil.
//
// The context ctx can be used to interrupt the downloading.
func (m *Manager) Download(ctx context.Context, url string, filePath string, callback ProgressCallback) error {
	m.semaphore.Acquire()
	defer m.semaphore.Release()

	client := &http.Client{
		CheckRedirect: func(r *http.Request, via []*http.Request) error {
			r.URL.Opaque = r.URL.Path
			return nil
		},
	}

	filePath = data.ReplaceTildeInDir(filePath)
	err := os.MkdirAll(path.Dir(filePath), 0777)
	if err != nil && !os.IsExist(err) {
		return errors.Wrapf(err, "Failed to create the directory for the path: %q", path.Dir(filePath))
	}
	var file *os.File
	file, err = os.Create(filePath)
	if err != nil {
		return errors.Wrapf(err, "failed creating file %q", filePath)
	}
	defer func() {
		if file != nil {
			_ = file.Close()
		}
	}()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return errors.Wrapf(err, "failed creating request for %q", url)
	}
	m.setRequestHeader(req)
	var resp *http.Response
	resp, err = client.Do(req)
	if err != nil {
		return errors.Wrapf(err, "failed downloading %q", url)
	}
	// _ = resp.Header.Write(os.Stdout)
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status code %d: %q", resp.StatusCode, resp.Header.Get("X-Error-Message"))
	}

	contentLength := resp.ContentLength
	if callback != nil {
		callback(0, contentLength)
	}
	const maxBufferSize = 1 * 1024 * 1024
	var buf [maxBufferSize]byte
	downloadedBytes := int64(0)
	for {
		if ctx.Err() != nil {
			return CancellationError
		}
		n, err := resp.Body.Read(buf[:])
		if err != nil && err != io.EOF {
			if ctx.Err() != nil {
				return CancellationError
			}
			return errors.Wrapf(err, "failed downloading %q", url)
		}
		if err == io.EOF {
			break
		}
		wn, err := file.Write(buf[:n])
		if err != nil && err != io.EOF {
			return errors.Wrapf(err, "failed writing %q to %q", url, filePath)
		}
		if wn != n {
			return errors.Wrapf(io.ErrShortWrite, "failed writing %q to %q: not enough bytes written (wanted %d, wrote only %d)",
				url, filePath, n, wn)
		}
		downloadedBytes += int64(n)
		if callback != nil {
			callback(downloadedBytes, contentLength)
		}
	}
	err = file.Close()
	file = nil
	if err != nil {
		return errors.Wrapf(err, "failed closing file %q", filePath)
	}
	if err = resp.Body.Close(); err != nil {
		return errors.Wrapf(err, "failed closing connection to %q", url)
	}
	return nil
}

// FetchHeader fetches the header of a URL (using HTTP method "HEAD").
//
// Notice it may lock on the maximum number of parallel requests, so consider calling this on a separate goroutine.
//
// The context ctx can be used to interrupt the downloading.
func (m *Manager) FetchHeader(ctx context.Context, url string) (header http.Header, contentLength int64, err error) {
	m.semaphore.Acquire()
	defer m.semaphore.Release()

	client := &http.Client{
		CheckRedirect: func(r *http.Request, via []*http.Request) error {
			r.URL.Opaque = r.URL.Path
			return nil
		},
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, url, nil)
	if err != nil {
		err = errors.Wrapf(err, "failed creating request for %q", url)
		return
	}
	m.setRequestHeader(req)
	req.Header.Set("Accept-Encoding", "identity")

	// Make the request and download the tokenizer.
	resp, err := client.Do(req)
	if err != nil {
		err = errors.Wrap(err, "failed request for metadata: ")
		return
	}

	// TODO: handle redirects.
	defer func() { _ = resp.Body.Close() }()
	_, err = io.ReadAll(resp.Body)
	if err != nil {
		err = errors.Wrapf(err, "failed reading response (%d) for metadata: ", resp.StatusCode)
		return
	}

	// Check status code.
	if resp.StatusCode != 200 {
		err = errors.Errorf("request for metadata from %q failed with the following message: %q",
			url, resp.Status)
		return
	}
	header = resp.Header
	contentLength = resp.ContentLength
	err = nil
	return
}
