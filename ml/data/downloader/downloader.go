// Package downloader implements download in parallel of various URLs, with various progress report callback.
package downloader

import (
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
//
// Args:
//   - totalBytes may be set to 0 if total size is not yet known.
//   - finished is set to true when download is finished. Indicates task is finished.
//   - err if there was an error, in which case the transfer was cancelled. In this case finished is also set to true.
type ProgressCallback func(downloadedBytes, totalBytes int64, finished bool, err error)

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

// Download enqueues the given url to be downloaded to the given filePath.
// Progress is reported back by the given callback.
//
// The returned latch can be used to cancel the download, in which case callback will be called with a CancellationError.
func (m *Manager) Download(url string, filePath string, callback ProgressCallback) *xsync.Latch {
	canceller := xsync.NewLatch()
	go func() {
		m.semaphore.Acquire()
		defer m.semaphore.Release()

		filePath = data.ReplaceTildeInDir(filePath)
		err := os.MkdirAll(path.Dir(filePath), 0777)
		if err != nil && !os.IsExist(err) {
			err = errors.Wrapf(err, "Failed to create the directory for the path: %q", path.Dir(filePath))
			return
		}
		var file *os.File
		file, err = os.Create(filePath)
		if err != nil {
			err = errors.Wrapf(err, "failed creating file %q", filePath)
			callback(0, 0, true, err)
			return
		}
		defer func() {
			if file != nil {
				_ = file.Close()
			}
		}()

		client := http.Client{
			CheckRedirect: func(r *http.Request, via []*http.Request) error {
				r.URL.Opaque = r.URL.Path
				return nil
			},
		}
		req, err := http.NewRequest("GET", url, nil)
		if err != nil {
			err = errors.Wrapf(err, "failed creating request for %q", url)
			callback(0, 0, true, err)
			return
		}
		if m.authToken != "" {
			req.Header.Set("Authorization", "Bearer "+m.authToken)
		}
		if m.userAgent != "" {
			req.Header.Set("user-agent", m.userAgent)
		}
		var resp *http.Response
		resp, err = client.Do(req)
		if err != nil {
			err = errors.Wrapf(err, "failed downloading %q", url)
			callback(0, 0, true, err)
			return
		}
		// _ = resp.Header.Write(os.Stdout)
		if resp.StatusCode != http.StatusOK {
			callback(0, 0, true, fmt.Errorf("bad status code %d: %q", resp.StatusCode,
				resp.Header.Get("X-Error-Message")))
			return
		}

		contentLength := resp.ContentLength
		callback(0, contentLength, false, nil)
		const maxBufferSize = 1 * 1024 * 1024
		var buf [maxBufferSize]byte
		downloadedBytes := int64(0)
		for {
			if canceller.Test() {
				callback(downloadedBytes, contentLength, true, CancellationError)
			}
			n, err := resp.Body.Read(buf[:])
			if err != nil && err != io.EOF {
				err = errors.Wrapf(err, "failed downloading %q", url)
				callback(downloadedBytes, contentLength, true, err)
				return
			}
			if err == io.EOF {
				break
			}
			wn, err := file.Write(buf[:n])
			if err != nil && err != io.EOF {
				err = errors.Wrapf(err, "failed writing %q to %q", url, filePath)
				callback(downloadedBytes, contentLength, true, err)
				return
			}
			if wn != n {
				err = errors.Errorf("failed writing %q to %q: not enough bytes written (wanted %d, wrote only %d)",
					url, filePath, n, wn)
				callback(downloadedBytes, contentLength, true, err)
				return
			}
			downloadedBytes += int64(n)
			callback(downloadedBytes, contentLength, false, nil)
		}
		err = file.Close()
		file = nil
		if err != nil {
			err = errors.Wrapf(err, "failed closing file %q", filePath)
			callback(downloadedBytes, contentLength, true, err)
			return
		}
		err = resp.Body.Close()
		if err != nil {
			err = errors.Wrapf(err, "failed closing connection to %q", url)
			callback(downloadedBytes, contentLength, true, err)
			return
		}
		callback(downloadedBytes, contentLength, true, nil)
	}()
	return canceller
}
