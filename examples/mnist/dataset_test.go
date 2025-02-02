package mnist

import (
	"math/rand"
	"testing"
	"time"

	"github.com/gomlx/gopjrt/dtypes"
)

func TestDataset(t *testing.T) {
	dataDir := "/home/rener/tmp/mnist"
	if err := Download(dataDir); err != nil {
		t.Errorf("Download: %v", err)
		return
	}

	batchSize := 1000
	shuffle := rand.New(rand.NewSource(time.Now().UTC().UnixNano()))

	trainDS, err := NewDataset("MNIST train", dataDir, "train", batchSize, shuffle, dtypes.Float32)
	if err != nil {
		t.Errorf("NewDataset: %v", err)
		return
	}
	testDS, err := NewDataset("MNIST test", dataDir, "test", batchSize, shuffle, dtypes.Float32)
	if err != nil {
		t.Errorf("NewDataset: %v", err)
		return
	}
	for _, ds := range []*Dataset{trainDS, testDS} {
		_, images, labels, err := ds.Yield()
		if err != nil {
			t.Errorf("Download: %v", err)
			return
		}
		images[0].AssertValid()
		images[0].Shape().AssertDims(batchSize, ds.width, ds.height, 3)

		labels[0].AssertValid()
		labels[0].Shape().AssertDims(batchSize)
	}

}
