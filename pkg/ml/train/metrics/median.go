package metrics

import (
	"math/rand/v2"
	"slices"

	. "github.com/gomlx/gomlx/internal/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
)

// StreamingMedianMetric implements a metric that keeps an approximate median of a metric from a streaming
// input.
type StreamingMedianMetric struct {
	baseMetric
	maxNumSamples, samplesSeen int
	samples                    []float64
	rng                        *rand.Rand
}

// NewMedianMetric creates a streaming median metric from any BaseMetricGraph function.
//
// BaseMetricGraph must return a scalar value (one value is consumed at a time). If you are
// processing batches at a time (and not a batch of size 1), this will return a median of the
// batch means. This may be a reasonable approximation, but something to be mindful.
//
// `prettyPrintFn` can be left as nil, and a default will be used.
func NewMedianMetric(
	name, shortName, metricType string,
	metricFn BaseMetricGraph,
	prettyPrintFn PrettyPrintFn,
) *StreamingMedianMetric {
	return &StreamingMedianMetric{
		baseMetric: baseMetric{
			name:       name,
			shortName:  shortName,
			metricType: metricType,
			metricFn:   metricFn,
			pPrintFn:   prettyPrintFn,
		},
		maxNumSamples: 10_001,
	}
}

// WithSampleSize configures the default number of random samples to keep to estimate the median.
func (m *StreamingMedianMetric) WithSampleSize(n int) *StreamingMedianMetric {
	m.maxNumSamples = n
	return m
}

const streamingMedianScope = "streaming_median"

func (m *StreamingMedianMetric) UpdateGraph(ctx *context.Context, labels, predictions []*Node) (metric *Node) {
	var results *Node
	err := TryCatch[error](func() { results = m.metricFn(ctx, labels, predictions) })
	if err != nil {
		panic(errors.WithMessagef(err, "failed building computation graph for streaming median metric %q", m.Name()))
	}
	return ConvertDType(results, dtypes.Float64)
}

func (m *StreamingMedianMetric) UpdateGo(results *tensors.Tensor) {
	if m.samples == nil {
		m.samples = make([]float64, 0, m.maxNumSamples)
		m.samplesSeen = 0
		if m.rng == nil {
			m.rng = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
		}
	}
	tensors.MustConstFlatData(results, func(resultsFlat []float64) {
		for _, x := range resultsFlat {
			m.samplesSeen++

			// Simple case: we have space to simply store the new sampled x.
			if len(m.samples) < m.maxNumSamples {
				m.samples = append(m.samples, x)
				return
			}

			// We must decide whether to keep x:
			if m.rng.Float64() >= float64(m.maxNumSamples)/float64(m.samplesSeen) {
				// We don't add new sample.
				return
			}
			// We replace the new sampled x in a random position.
			pos := m.rng.IntN(m.maxNumSamples)
			m.samples[pos] = x
		}
	})
}

func (m *StreamingMedianMetric) ReadGo() *tensors.Tensor {
	if len(m.samples) == 0 {
		Panicf("streaming median metric %q has seen no samples to read", m.Name())
	}
	slices.Sort(m.samples)
	return tensors.FromScalar(m.samples[len(m.samples)/2])
}

// Reset will delete all related variables to the streaming median: they will be recreated again
// at the start of an update.
func (m *StreamingMedianMetric) Reset(ctx *context.Context) {
	m.samples = nil
	m.samplesSeen = 0
}
