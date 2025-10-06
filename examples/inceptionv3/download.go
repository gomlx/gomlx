package inceptionv3

import (
	"fmt"
	"path"

	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/data/hdf5"
	"github.com/gomlx/gomlx/pkg/support/fsutil"
)

const (
	// WeightsURL is the URL for the whole model, including the top layer, a 1000-classes linear layer on top.
	WeightsURL = "https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels.h5"

	// WeightsH5Checksum is the SHA256 checksum of the weights file.
	WeightsH5Checksum = "00c9ea4e4762f716ac4d300d6d9c2935639cc5e4d139b5790d765dcbeea539d0"

	// WeightsH5Name is the name of the local ".h5" file with the weights.
	WeightsH5Name = "weights.h5"

	// UnpackedWeightsName is the name of the subdirectory that will hold the unpacked weights.
	UnpackedWeightsName = "gomlx_weights"
)

// DownloadAndUnpackWeights to the given baseDir. It only does the work if the files are not there yet (downloaded
// and unpacked).
//
// It is verbose and uses a progressbar if downloading/unpacking. It is quiet if there is nothing to do, that is,
// if the files are already there.
func DownloadAndUnpackWeights(baseDir string) (err error) {
	return downloadAndUnpackWeightsImpl(baseDir, WeightsURL, WeightsH5Checksum, WeightsH5Name, UnpackedWeightsName)
}

func downloadAndUnpackWeightsImpl(baseDir, weightsURL, sha256Checksum, weightsH5Name, unpackedWeightsName string) (err error) {
	baseDir = fsutil.MustReplaceTildeInDir(baseDir)
	unpackedWeightPath := path.Join(baseDir, unpackedWeightsName)
	if fsutil.MustFileExists(unpackedWeightPath) {
		// Weights already unpacked, done.
		return
	}

	weightsH5Path := path.Join(baseDir, weightsH5Name)
	err = data.DownloadIfMissing(weightsURL, weightsH5Path, sha256Checksum)
	if err != nil {
		return err
	}

	fmt.Printf("Unpacking weights to %s:\n", unpackedWeightPath)
	err = hdf5.UnpackToTensors(unpackedWeightPath, weightsH5Path).ProgressBar().Done()
	return
}

// PathToTensor returns the path to tensorName (name within the h5 file).
func PathToTensor(baseDir, tensorName string) string {
	baseDir = fsutil.MustReplaceTildeInDir(baseDir)
	return path.Join(baseDir, UnpackedWeightsName, tensorName)
}

/*
 * Version of functions for the weights without the top-layer: since it is the same as with the top-layer, except
 * for the top, we just exclude it to make it simpler -- at the cost of a couple MB if one is not using the top
 * layer.
 */

/*
const (
	// WeightsNoTopURL is the URL for the model without the top layer. When using only the image embedding.
	WeightsNoTopURL = "https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

	// WeightsNoTopH5Checksum is the SHA256 checksum of the weights file, the version without the top layer.
	WeightsNoTopH5Checksum = "9617109a16463f250180008f9818336b767bdf5164315e8cd5761a8c34caa62a"

	// WeightsNoTopH5Name is the name of the local ".h5" file with the weights.
	WeightsNoTopH5Name = "weights_no_top.h5"

	// UnpackedWeightsNoTopName is the name of the subdirectory that will hold the unpacked weights without the top layer.
	UnpackedWeightsNoTopName = "gomlx_weights_no_top"
)

// DownloadAndUnpackWeightsNoTop is similar to DownloadAndUnpackWeights, but uses the version without the top layer.
func DownloadAndUnpackWeightsNoTop(baseDir string) (err error) {
	return downloadAndUnpackWeightsImpl(baseDir, WeightsNoTopURL, WeightsNoTopH5Checksum, WeightsNoTopH5Name, UnpackedWeightsNoTopName)
}

// PathToTensorNoTop returns the path to tensorName (name within the h5 file) for the version of weights without
// the top layer.
func PathToTensorNoTop(baseDir, tensorName string) string {
	baseDir = data.MustReplaceTildeInDir(baseDir)
	return path.Join(baseDir, UnpackedWeightsNoTopName, tensorName)
}

*/
