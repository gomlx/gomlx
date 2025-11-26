package sampler

import (
	"fmt"
	"path"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/tensors"
	mldata "github.com/gomlx/gomlx/pkg/ml/datasets"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/exp/constraints"

	_ "github.com/gomlx/gomlx/backends/default"
)

func createTestSampler(t *testing.T) *Sampler {
	s := New()
	s.AddNodeType("papers", 5)
	s.AddNodeType("authors", 10)
	authorWritesPapers := tensors.FromValue([][]int32{
		{0, 2}, // Author 0 writes paper 2.
		{3, 2},
		{4, 2},
		{0, 3},
		{0, 4},
		{4, 4},
		{7, 4},
	})
	require.NoError(t, authorWritesPapers.Shape().Check(dtypes.Int32, 7, 2))
	s.AddEdgeType("writes", "authors", "papers", authorWritesPapers, false)
	s.AddEdgeType("written_by", "authors", "papers", authorWritesPapers, true)
	return s
}

func TestSampler(t *testing.T) {
	s := createTestSampler(t)
	fmt.Printf("%s\n", s)

	// We create a checking function because we'll use it 2 times.
	checkSamplerFn := func() {
		assert.EqualValues(t, []int32{3, 3, 3, 4, 6, 6, 6, 7, 7, 7}, s.EdgeTypes["writes"].Starts)
		assert.EqualValues(t, []int32{2, 3, 4, 2, 2, 4, 4}, s.EdgeTypes["writes"].EdgeTargets)
		assert.EqualValues(t, []int32{2, 4}, s.EdgeTypes["writes"].EdgeTargetsForSourceIdx(4))
		assert.EqualValues(t, []int32{}, s.EdgeTypes["writes"].EdgeTargetsForSourceIdx(9))

		assert.EqualValues(t, []int32{0, 0, 3, 4, 7}, s.EdgeTypes["written_by"].Starts)
		assert.EqualValues(t, []int32{0, 3, 4, 0, 0, 4, 7}, s.EdgeTypes["written_by"].EdgeTargets)
		assert.EqualValues(t, []int32{0, 4, 7}, s.EdgeTypes["written_by"].EdgeTargetsForSourceIdx(4))
		assert.EqualValues(t, []int32{}, s.EdgeTypes["written_by"].EdgeTargetsForSourceIdx(0))
	}

	// Checks Sampler s is correct.
	checkSamplerFn()

	// Save it, load it and check that we get the same results.
	filePath := path.Join(t.TempDir(), "test_sampler.bin")
	require.NoErrorf(t, s.Save(filePath), "Saving test Sampler to %q", filePath)
	var err error
	s, err = Load(filePath)
	require.NoErrorf(t, err, "Loading test Sampler from %q", filePath)
	checkSamplerFn()

	// Makes sure Sampler can't be modified after a Strategy is created.
	_ = s.NewStrategy()
	require.Panics(t, func() { s.AddNodeType("rogue", 10) }, "No changes to Sampler allowed after frozen.")
}

func createTestStrategy(t *testing.T, s *Sampler) *Strategy {
	var strategy *Strategy
	require.NotPanics(t, func() {
		strategy = s.NewStrategy()
		seeds := strategy.NodesFromSet("Seeds", "papers", 2, []int32{2, 3, 4})
		_ = seeds.IdentitySubRule("SubSeeds")
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

	require.Equal(t, 6, len(strategy.Rules))
	require.Len(t, strategy.Seeds, 2)

	seeds, found := strategy.Rules["Seeds"]
	require.True(t, found)
	require.NoError(t, seeds.Shape.Check(dtypes.Int32, 2))
	assert.Equal(t, 2, len(seeds.Dependents))

	subSeeds, found := strategy.Rules["SubSeeds"]
	require.True(t, found)
	require.NoError(t, subSeeds.Shape.Check(dtypes.Int32, 2, 1))
	assert.Equal(t, 0, len(subSeeds.Dependents))
	assert.Equal(t, seeds, subSeeds.SourceRule)

	seeds2, found := strategy.Rules["seeds2"]
	require.True(t, found)
	require.NoError(t, seeds2.Shape.Check(dtypes.Int32, 3))
	assert.Equal(t, 1, len(seeds2.Dependents))

	authors, found := strategy.Rules["authors"]
	require.True(t, found)
	require.NoError(t, authors.Shape.Check(dtypes.Int32, 2, 5))
	assert.Equal(t, seeds, authors.SourceRule)
	assert.Equal(t, 1, len(authors.Dependents))

	otherPapers, found := strategy.Rules["otherPapers"]
	require.True(t, found)
	require.NoError(t, otherPapers.Shape.Check(dtypes.Int32, 2, 5, 3))
	assert.Equal(t, authors, otherPapers.SourceRule)
	assert.Equal(t, 0, len(otherPapers.Dependents))

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
	checkInputsFn := func(t *testing.T, spec any, inputs, labels []*tensors.Tensor, err error) map[string]*ValueMask[*tensors.Tensor] {
		require.NoError(t, err)
		require.Empty(t, labels)
		require.Equal(t, strategy, spec.(*Strategy))
		graphSample, remaining := MapInputsToStates[*tensors.Tensor](strategy, inputs)
		require.Empty(t, remaining)
		if strategy.KeepDegrees == true {
			seeds := tensors.MustCopyFlatData[int32](graphSample["Seeds"].Value)
			degrees := tensors.MustCopyFlatData[int32](
				graphSample[NameForNodeDependentDegree("Seeds", "authors")].Value,
			)
			for ii, paper := range seeds {
				var want int32
				switch paper {
				case 2:
					want = 3
				case 3:
					want = 1
				case 4:
					want = 3
				case 0:
					// Padding.
					want = 0
				default:
					require.Failf(t, "Invalid value for paper", "paper=%d!?", paper)
				}
				require.Equalf(t, want, degrees[ii], "Got degree %d for paper %d, wanted %d", degrees[ii], paper, want)
			}

			// Test degrees of identity.
			degrees = tensors.MustCopyFlatData[int32](
				graphSample[NameForNodeDependentDegree("Seeds", "SubSeeds")].Value,
			)
			seedsMask := tensors.MustCopyFlatData[bool](graphSample["Seeds"].Mask)
			for ii, paperMask := range seedsMask {
				if !paperMask {
					continue
				}
				require.Equal(
					t,
					int32(1),
					degrees[ii],
					"Got degree %d for paper %d, wanted %d",
					degrees[ii],
					seeds[ii],
					1,
				)
			}
		}
		for name, rule := range strategy.Rules {
			require.Containsf(t, graphSample, name, "Missing input for rule %q", name)
			value, mask := graphSample[name].Value, graphSample[name].Mask
			require.True(
				t,
				value.Shape().Equal(rule.Shape),
				"Mismatch of shapes for value of rule %q: value.Shape=%s, rule.Shape=%s",
				name,
				value.Shape(),
				rule.Shape,
			)
			require.NoErrorf(t, mask.Shape().Check(dtypes.Bool, rule.Shape.Dimensions...),
				"Mismatch of shapes for mask of rule %q", name)
			if rule.SourceRule != nil && strategy.KeepDegrees {
				degreeName := NameForNodeDependentDegree(rule.SourceRule.Name, rule.Name)
				degrees := graphSample[degreeName].Value
				wantShape := value.Shape().Clone()
				wantShape.Dimensions[wantShape.Rank()-1] = 1
				require.Truef(
					t,
					degrees.Shape().Equal(wantShape),
					"Mismatch degree shapes for %q: degree shape is %s, wanted %s",
					degreeName,
					degrees.Shape(),
					wantShape,
				)
			}
		}
		return graphSample
	}

	for _, strategy.KeepDegrees = range []bool{false, true} {
		fmt.Printf("> strategy.KeepDegrees=%v\n", strategy.KeepDegrees)
		ds := strategy.NewDataset("one_epoch_in_order").Epochs(1)
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
		for range 100 { // Sample 100 using parallel datasets, and checks that it works ok.
			spec, inputs, labels, err := parallelDS.Yield()
			_ = checkInputsFn(t, spec, inputs, labels, err)
		}
		parallelDS.Done()
	}
}

// TestSamplingRandomness checks that the distribution of seeds and edge sampling is homogeneous.
func TestSamplingRandomness(t *testing.T) {
	// We create a new Sampler+strategy for this.
	s := New()
	numPapers := 3
	numAuthors := numPapers*2 + 1 // One extra author that never wrote any paper.
	s.AddNodeType("papers", numPapers)
	s.AddNodeType("authors", numAuthors)
	var authorWritesPapers [][]int32
	for paper := range int32(numPapers) {
		for author := paper; author < (paper+1)*2; author++ {
			authorWritesPapers = append(authorWritesPapers, []int32{author, paper})
		}
	}
	authorWritesPapersT := tensors.FromValue(authorWritesPapers)
	require.NoError(t, authorWritesPapersT.Shape().Check(dtypes.Int32, 2+3+4, 2))
	s.AddEdgeType("written_by", "authors", "papers", authorWritesPapersT, true)

	strategy := s.NewStrategy()
	{
		seeds := strategy.NodesFromSet("seeds", "papers", 1, []int32{1, 2})
		_ = seeds.FromEdges("authors", "written_by", 2)
		_ = seeds.FromEdges("authors2", "written_by", 2)
	}

	// Sample
	numSamples := 10000
	dsNames := []string{"without_replacement", "with_replacement"}
	for dsIdx, ds := range []train.Dataset{
		strategy.NewDataset("infinite").Infinite().Shuffle(),
		strategy.NewDataset("infinite").WithReplacement(),
	} {
		parallelDS := mldata.Parallel(ds)

		// Keep counts.
		papersCounts := make([]int, numPapers)
		authorsPerPapersCounts := make([][]int, numPapers)
		for ii := range authorsPerPapersCounts {
			authorsPerPapersCounts[ii] = make([]int, numAuthors)
		}

		for range numSamples {
			_, inputs, _, err := parallelDS.Yield()
			require.NoErrorf(t, err, "while testing dataset %q", dsNames[dsIdx])
			graphSample, remaining := MapInputsToStates[*tensors.Tensor](strategy, inputs)
			require.Empty(t, remaining)

			require.NoErrorf(
				t,
				graphSample["seeds"].Value.Shape().CheckDims(1),
				"while testing dataset %q",
				dsNames[dsIdx],
			)
			sampledPaper := tensors.MustCopyFlatData[int32](graphSample["seeds"].Value)[0]
			papersCounts[sampledPaper]++

			require.NoErrorf(
				t,
				graphSample["authors"].Value.Shape().CheckDims(1, 2),
				"while testing dataset %q",
				dsNames[dsIdx],
			)
			require.NoErrorf(
				t,
				graphSample["authors"].Mask.Shape().CheckDims(1, 2),
				"while testing dataset %q",
				dsNames[dsIdx],
			)
			authors := tensors.MustCopyFlatData[int32](graphSample["authors"].Value)
			authorsMask := tensors.MustCopyFlatData[bool](graphSample["authors"].Mask)
			for ii, author := range authors {
				require.True(t, authorsMask[ii])
				authorsPerPapersCounts[sampledPaper][author]++
			}
		}

		fmt.Printf("ds=%s, papersCounts=%v\n", dsNames[dsIdx], papersCounts)
		assert.Equalf(t, numSamples, papersCounts[1]+papersCounts[2], "while testing dataset %q", dsNames[dsIdx])
		if dsIdx == 0 {
			require.Lessf(
				t,
				diff(papersCounts[1], papersCounts[2]), /* 2% */
				10,
				"while testing dataset %q",
				dsNames[dsIdx],
			)
		} else {
			require.Lessf(t, diff(papersCounts[1], papersCounts[2]) /* 10% */, 500, "while testing dataset %q", dsNames[dsIdx])
		}
		fmt.Printf("authorsPerPapersCounts=%v\n", authorsPerPapersCounts)
		for _, paper := range []int{1, 2} {
			authorsCounts := authorsPerPapersCounts[paper]
			for author := paper + 1; author < (paper+1)*2; author++ {
				// author loop starts one after the first author for the paper, so we can compare
				// the count of the author with the previous one. They should be similar.
				require.Lessf(
					t,
					diff(authorsCounts[author], authorsCounts[author-1]),
					200,
					"while testing dataset %q",
					dsNames[dsIdx],
				)
			}
		}
	}
}

func diff[T interface {
	constraints.Integer | constraints.Float
}](a, b T) T {
	r := a - b
	if r < 0 {
		r = -r
	}
	return r
}

func TestRandKOfN(t *testing.T) {
	names := []string{"randKOfNLinear", "randKOfNReservoir"}
	const numSamples = 5
	const n = 20
	const numRepeats = 5000
	sample := make([]int32, numSamples)
	for ii, fn := range []func([]int32, int){randKOfNLinear, randKOfNReservoir} {
		name := names[ii]
		counts := make([]int, n)
		for range numRepeats {
			fn(sample, n)
			for ii, x := range sample {
				for jj := 0; jj < ii; jj++ {
					require.NotEqualf(t, sample[jj], x, "Repeated value %d for %s", x, name)
				}
				counts[x]++
			}
		}
		fmt.Printf("%s: counts=%v\n", name, counts)
		for _, count := range counts {
			require.Lessf(t, diff(count, numRepeats*numSamples/n), 200, "Unbalanced sampling for %s: %v", name, counts)
		}
	}
}
