package mnist

import "testing"

func TestDataset(t *testing.T) {
	dataDir := "/home/rener/tmp/mnist"
	if err := Download(dataDir); err != nil {
		t.Errorf("Download: %v", err)
		return
	}
	trainDS := NewDataset("MNIST train", dataDir, "train")
	testDS := NewDataset("MNIST test", dataDir, "test")
	for _, ds := range []*Dataset{trainDS, testDS} {
		_, images, labels, err := ds.Yield()
		if err != nil {
			t.Errorf("Download: %v", err)
			return
		}
		images[0].AssertValid()
		images[0].Shape().AssertDims(ds.size, ds.width, ds.height, 3)

		labels[0].AssertValid()
		labels[0].Shape().AssertDims(ds.size)
	}

}
