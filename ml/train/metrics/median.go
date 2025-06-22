package metrics

import (
	. "github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
)

// StreamingMedianMetric implements a metric that keeps an approximate median of a metric from a streaming
// input.
type StreamingMedianMetric struct {
	baseMetric

	markers  [5]float64
	counters [5]int64
}

// NewMedianMetric creates a streaming median metric from any BaseMetricGraph function.
//
// BaseMetricGraph must return a scalar value (one value is consumed at a time). If you are
// processing batches at a time (and not a batch of size 1), this will return a median of the
// batch means. This may be a reasonable approximation, but something to be mindful.
//
// It uses the P^2 algorithm, described in the paper https://dl.acm.org/doi/abs/10.1145/4372.4378,
// and in a more friendly way in the post in: https://www.baeldung.com/cs/streaming-median
//
// `prettyPrintFn` can be left as nil, and a default will be used.
func NewMedianMetric(name, shortName, metricType string, metricFn BaseMetricGraph, prettyPrintFn PrettyPrintFn) *StreamingMedianMetric {
	return &StreamingMedianMetric{
		baseMetric: baseMetric{name: name, shortName: shortName, metricType: metricType, metricFn: metricFn, pPrintFn: prettyPrintFn},
	}
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

func (m *StreamingMedianMetric) UpdateGo(results *tensors.Tensor) *tensors.Tensor {
	tensors.ConstFlatData(results, func(resultsFlat []float64) {
		for _, x := range resultsFlat {
			if m.counters[4] == 0 {
				// This is the very first element:
				for i := range 5 {
					m.markers[i] = x
					if i > 0 {
						m.counters[i] = 1
					}
				}
				continue
			}

			// Update the first and last markers and counters:
			m.markers[0] = min(x, m.markers[0])
			m.markers[4] = max(x, m.markers[4])
			// m.counter[0] is always 0.
			m.counters[4]++ // Always incremented.
			for i := 1; i < 4; i++ {
				if x <= m.markers[i] {
					m.counters[i]++
				}
			}

			// Find inner ideal counters:
			var idealCounters [5]float64
			currentN := float64(m.counters[4])
			p2quantiles := [5]float64{0, 0.25, 0.5, 0.75, 1}
			for i := 1; i < 4; i++ {
				idealCounters[i] = p2quantiles[i] * (currentN - 1)
			}

			// Adjust counts and markers where needed:
			for i := 1; i < 4; i++ {
				d := idealCounters[i] - float64(m.counters[i])
				if d >= 1 {
					d = 1
					if m.counters[i] >= m.counters[i+1] || m.markers[i] >= m.markers[i+1] {
						// No margin to adjust markers[i] or counters[i].
						continue
					}
				} else if d <= -1 {
					d = -1
					if m.counters[i] <= m.counters[i-1] || m.markers[i] <= m.markers[i-1] {
						// No margin to adjust markers[i] or counters[i].
						continue
					}
				} else {
					// The difference is not large enough that we need to adjust counts.
					continue
				}

				// Update the counter by d_i
				n_current := float64(m.counters[i])
				n_previous := float64(m.counters[i-1])
				n_next := float64(m.counters[i+1])
				q_previous := m.markers[i-1]
				q_current := m.markers[i]
				q_next := m.markers[i+1]

				delta_n_previous := n_current - n_previous
				delta_n_next := n_next - n_current
				delta_n_outer := n_next - n_previous

				delta_q_previous := q_current - q_previous
				delta_q_next := q_next - q_current
				delta_q_outer := q_next - q_previous

				q_new := m.markers[i] // Default to no change if interpolation fails

				// Attempt Parabolic Interpolation
				if delta_n_previous > 0 && delta_n_next > 0 && delta_n_outer > 0 {
					adjustmentAmount := d / delta_n_outer
					term1 := (delta_n_previous + d) * delta_q_next / delta_n_next
					term2 := (delta_n_next - d) * delta_q_previous / delta_n_previous
					q_new = q_current + adjustmentAmount*(term1+term2)

				} else if delta_n_outer > 0 {
					// Linear interpolation between neighbor markers:
					q_new = q_previous + (delta_n_previous+d)*delta_q_outer/delta_n_outer

				} else {
					// All markers are at the same rank (clumped), cannot interpolate.
					q_new = m.markers[i]
				}
				m.markers[i] = q_new // Commit the new marker value
				m.counters[i] += int64(d)
			}
		}
	})
	return tensors.FromScalar(m.markers[2])
}

// Reset will delete all related variables to the streaming median: they will be recreated again
// at the start of an update.
func (m *StreamingMedianMetric) Reset(ctx *context.Context) {
	m.markers = [5]float64{0, 0, 0, 0, 0}
	m.counters = [5]int64{0, 0, 0, 0, 0}
}
