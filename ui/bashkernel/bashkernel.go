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

// Package bashkernel implements tools to output rich content to a Jupyter notebook running
// the bash_kernel (https://github.com/takluyver/bash_kernel).
//
// See usage example in gomlx/examples/cifar/demo.
package bashkernel

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"github.com/pkg/errors"
	"image"
	"image/png"
	"os"
)

const BashNotebookEnv = "NOTEBOOK_BASH_KERNEL_CAPABILITIES"

// IsBashNotebook returns true if NOTEBOOK_BASH_KERNEL_CAPABILITIES is set, which indicates it is running
// from within a Notebook.
func IsBashNotebook() bool {
	_, found := os.LookupEnv(BashNotebookEnv)
	return found
}

// EmbedImageSrc returns a string that can be used as in an HTML <img> tag, as its source (it's `src` field) -- without
// requiring separate files. It embeds it as a PNG file base64 encoded.
func EmbedImageSrc(img image.Image) (string, error) {
	buf := &bytes.Buffer{}
	err := png.Encode(buf, img)
	if err != nil {
		return "", errors.Wrapf(err, "failed to encode image as PNG")
	}
	encoded := base64.StdEncoding.EncodeToString(buf.Bytes())
	return fmt.Sprintf("data:image/png;base64,%s", encoded), nil
}

type BashKernelNotebookPrefix string

const (
	// HTMLPrefix indicates the file that follows holds HTML content to be displayed.
	HTMLPrefix BashKernelNotebookPrefix = "bash_kernel: saved html data to: "

	// JavascriptPrefix indicates the file that follows holds Javascript content to be executed.
	JavascriptPrefix = "bash_kernel: saved javascript data to: "
)

// OutputToPrefix saves the given content in a temporary file and outputs a prefixed line
// pointing to the file. If displayId != "", displays using that id.
func OutputToPrefix(prefix BashKernelNotebookPrefix, content string, displayId string) error {
	if !IsBashNotebook() {
		fmt.Printf("[Not displaying HTML content, since apparently not in a notebook -- %q env variable not set]\n",
			BashNotebookEnv)
		return nil
	}
	file, err := os.CreateTemp("", "bash_kernel.go_notebook.*")
	if err != nil {
		return errors.Wrap(err, "failed to create temporary file for OutputHTML")
	}
	fileName := file.Name()
	_, err = fmt.Fprint(file, content)
	if err != nil {
		return errors.Wrapf(err, "failed to write to temporary file %q for OutputHTML", fileName)
	}
	err = file.Close()
	if err != nil {
		return errors.Wrapf(err, "failed to close temporary file %q for OutputHTML", fileName)
	}
	if displayId == "" {
		fmt.Printf("%s%s\n", prefix, fileName)
	} else {
		fmt.Printf("%s(%s) %s\n", prefix, displayId, fileName)
	}
	return nil
}

// OutputHTML outputs the html content such that it gets displayed on a Jupyter(Lab) Notebook, using
// the bash_kernel.
//
// If displayId is set (displayId != ""), and a cell with the same displayId was created
// (and not deleted) earlier, it will update the contents of the same cell. Useful for updating
// dynamic content.
//
// Notice that if the cell is re-run the cell disappears and cannot be updated. Hence, a
// new unique displayId should always be created at the first use time. Also, displayId's
// shouldn't be shared with different types of content.
func OutputHTML(html string, displayId string) error {
	return OutputToPrefix(HTMLPrefix, html, displayId)
}

// OutputJavascript sends the javascript content to be executed on a Jupyter(Lab) Notebook, using
// the bash_kernel.
//
// If displayId is set (displayId != ""), and a cell with the same displayId was created
// (and not deleted) earlier, it will update the contents of the same cell. Useful for updating
// dynamic content.
//
// Notice that if the cell is re-run the cell disappears and cannot be updated. Hence, a
// new unique displayId should always be created at the first use time. Also, displayId's
// shouldn't be shared with different types of content.
func OutputJavascript(javascript, displayId string) error {
	return OutputToPrefix(JavascriptPrefix, javascript, displayId)
}
