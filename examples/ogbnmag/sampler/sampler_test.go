package sampler

import (
	"fmt"
	mldata "github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"path"
	"testing"
)

func createTestSampler(t *testing.T) *Sampler {
	s := New()
	s.AddNodeType("papers", 5)
	s.AddNodeType("authors", 10)
	authorWritesPapers := tensor.FromValue([][]int32{
		{0, 2}, // Author 0 writes paper 2.
		{3, 2},
		{4, 2},
		{0, 3},
		{0, 4},
		{4, 4},
		{7, 4},
	})
	require.NoError(t, authorWritesPapers.Shape().Check(shapes.Int32, 7, 2))
	s.AddEdgeType("writes", "authors", "papers", authorWritesPapers, false)
	s.AddEdgeType("written_by", "authors", "papers", authorWritesPapers, true)
	return s
}

func TestSampler(t *testing.T) {
	s := createTestSampler(t)
	fmt.Printf("%s\n", s)

	// We create a checking function because we'll use it 2 times.
	checkSamplerFn := func() {
		assert.EqualValues(t, []int32{3, 3, 3, 4, 6, 6, 6, 7, 7, 7}, s.d.EdgeTypes["writes"].Starts)
		assert.EqualValues(t, []int32{2, 3, 4, 2, 2, 4, 4}, s.d.EdgeTypes["writes"].EdgeTargets)
		assert.EqualValues(t, []int32{2, 4}, s.d.EdgeTypes["writes"].EdgeTargetsForSourceIdx(4))
		assert.EqualValues(t, []int32{}, s.d.EdgeTypes["writes"].EdgeTargetsForSourceIdx(9))

		assert.EqualValues(t, []int32{0, 0, 3, 4, 7}, s.d.EdgeTypes["written_by"].Starts)
		assert.EqualValues(t, []int32{0, 3, 4, 0, 0, 4, 7}, s.d.EdgeTypes["written_by"].EdgeTargets)
		assert.EqualValues(t, []int32{0, 4, 7}, s.d.EdgeTypes["written_by"].EdgeTargetsForSourceIdx(4))
		assert.EqualValues(t, []int32{}, s.d.EdgeTypes["written_by"].EdgeTargetsForSourceIdx(0))
	}

	// Checks sampler s is correct.
	checkSamplerFn()

	// Save it, load it and check that we get the same results.
	filePath := path.Join(t.TempDir(), "test_sampler.bin")
	require.NoErrorf(t, s.Save(filePath), "Saving test sampler to %q", filePath)
	var err error
	s, err = Load(filePath)
	require.NoErrorf(t, err, "Loading test sampler from %q", filePath)
	checkSamplerFn()

	// Makes sure Sampler can't be modified after a Strategy is created.
	_ = s.NewStrategy()
	require.Panics(t, func() { s.AddNodeType("rogue", 10) }, "No changes to sampler allowed after frozen.")
}

func createTestStrategy(t *testing.T, s *Sampler) *Strategy {
	var strategy *Strategy
	require.NotPanics(t, func() {
		strategy = s.NewStrategy()
		seeds := strategy.NodesFromSet("seeds", "papers", 2, []int32{2, 3, 4})
		authors := seeds.FromEdges("authors", "written_by", 5)
		_ = authors.FromEdges("otherPapers", "writes", 3)

		seeds2 := strategy.Nodes("seeds2", "papers", 3)
		_ = seeds2.FromEdges("authors2", "written_by", 1)

	})
	return strategy
}

func TestStrategy(t *testing.T) {
	s := createTestSampler(t)
	strategy := createTestStrategy(t, s)
	fmt.Printf("\n%s\n\n", strategy)

	require.Equal(t, 5, len(strategy.rules))
	require.Len(t, strategy.seeds, 2)

	seeds, found := strategy.rules["seeds"]
	require.True(t, found)
	require.NoError(t, seeds.shape.Check(shapes.Int32, 2))
	assert.Equal(t, 1, len(seeds.dependents))

	seeds2, found := strategy.rules["seeds2"]
	require.True(t, found)
	require.NoError(t, seeds2.shape.Check(shapes.Int32, 3))
	assert.Equal(t, 0, len(seeds2.dependents))

	authors, found := strategy.rules["authors"]
	require.True(t, found)
	require.NoError(t, authors.shape.Check(shapes.Int32, 2, 5))
	assert.Equal(t, seeds, authors.sourceRule)
	assert.Equal(t, 1, len(authors.dependents))

	otherPapers, found := strategy.rules["otherPapers"]
	require.True(t, found)
	require.NoError(t, otherPapers.shape.Check(shapes.Int32, 2, 5, 3))
	assert.Equal(t, authors, otherPapers.sourceRule)
	assert.Equal(t, 0, len(otherPapers.dependents))

	// Checks that frozen strategies can no longer be modified.
	_ = strategy.NewDataset("test")
	require.Panics(t, func() { strategy.Nodes("other_seeds", "authors", 1) },
		"After creating a dataset, strategy is frozen and can't be modified")
	require.Panics(t, func() { strategy.NodesFromSet("other_seeds", "authors", 1, []int32{1}) },
		"After creating a dataset, strategy is frozen and can't be modified")
	require.Panics(t, func() { seeds.FromEdges("other_authors", "written_by", 1) },
		"After creating a dataset, strategy is frozen and can't be modified")
}

func TestDataset(t *testing.T) {
	s := createTestSampler(t)
	strategy := createTestStrategy(t, s)

	// checkInputsFn make automatic checks of expected dimensions and errors.
	checkInputsFn := func(t *testing.T, spec any, inputs, labels []tensor.Tensor, err error) map[string]ValueMask {
		require.NoError(t, err)
		require.Empty(t, labels)
		require.Equal(t, strategy, spec.(*Strategy))
		graphSample := strategy.MapInputs(inputs)
		for name, rule := range strategy.rules {
			require.Containsf(t, graphSample, name, "Missing input for rule %q", name)
			value, mask := graphSample[name].Value, graphSample[name].Mask
			require.True(t, value.Shape().Eq(rule.shape), "Mismatch of shapes for value of rule %q", name)
			require.NoErrorf(t, mask.Shape().Check(shapes.Bool, rule.shape.Dimensions...),
				"Mismatch of shapes for mask of rule %q", name)
		}
		return graphSample
	}

	ds := strategy.NewDataset("one_epoch_in_order").Epochs(1)
	ds.Epochs(1)
	{
		spec, inputs, labels, err := ds.Yield()
		_ = checkInputsFn(t, spec, inputs, labels, err)
		spec, inputs, labels, err = ds.Yield()
		_ = checkInputsFn(t, spec, inputs, labels, err)
	}
	_, _, _, err := ds.Yield()
	require.Error(t, err, "Dataset should have been exhausted.")

	ds = strategy.NewDataset("one_epoch_in_order").Infinite().Shuffle()
	parallelDS := mldata.Parallel(ds)
	for _ = range 100 { // Sample 100 using parallel datasets, and checks that it works ok.
		spec, inputs, labels, err := parallelDS.Yield()
		_ = checkInputsFn(t, spec, inputs, labels, err)
	}
}
