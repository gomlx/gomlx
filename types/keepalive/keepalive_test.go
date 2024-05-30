package keepalive

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestAcquire(t *testing.T) {
	// Check acquired references are reused.
	var someData float64
	acquired := make([]KeepAlive, 0, InitialFreeSlots)
	for ii := 0; ii < InitialFreeSlots; ii++ {
		acquired = append(acquired, Acquire(&someData))
	}
	require.Len(t, allRefs, InitialFreeSlots)
	for _, k := range acquired {
		k.Release()
	}
	acquired = acquired[:0]
	for ii := 0; ii < InitialFreeSlots; ii++ {
		acquired = append(acquired, Acquire(&someData))
	}
	require.Len(t, allRefs, InitialFreeSlots)

	// Check new allocations are also properly reused.
	for ii := 0; ii < InitialFreeSlots; ii++ {
		acquired = append(acquired, Acquire(&someData))
	}
	require.Len(t, allRefs, 2*InitialFreeSlots)
	for _, k := range acquired {
		k.Release()
	}
	acquired = acquired[:0]
	for ii := 0; ii < 2*InitialFreeSlots; ii++ {
		acquired = append(acquired, Acquire(&someData))
	}
	require.Len(t, allRefs, 2*InitialFreeSlots)
}
