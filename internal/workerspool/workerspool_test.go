package workerspool

import (
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gomlx/gomlx/pkg/support/xsync"
	"github.com/stretchr/testify/assert"
)

func TestPool_Saturate(t *testing.T) {
	// Test saturation.
	pool := New()
	wantTasks := 5
	pool.SetMaxParallelism(wantTasks)

	var count atomic.Int32
	doneNewTasks := xsync.NewLatch()
	doneTest := xsync.NewLatch()

	go func() {
		pool.Saturate(func() {
			got := count.Add(1)
			runtime.Gosched()
			if int(got) == wantTasks {
				doneNewTasks.Trigger()
				return
			}
			doneNewTasks.Wait()
		})
		doneTest.Trigger()
	}()

	select {
	case <-doneTest.WaitChan():
		// Success
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Timeout before all tasks were executed.")
	}
	if int(count.Load()) != wantTasks {
		t.Fatalf("Expected %d tasks, got %d", wantTasks, count.Load())
	}

	// Test No Parallelism
	pool.SetMaxParallelism(0)
	count.Store(0)
	pool.Saturate(func() { count.Add(1) })
	assert.Equal(t, int32(1), count.Load())

	// Test Unlimited
	pool.SetMaxParallelism(-1)
	count.Store(0)
	var started atomic.Int32
	pool.Saturate(func() {
		started.Add(1)
		runtime.Gosched()
		count.Add(1)
	})
	assert.Greater(t, int(started.Load()), 1)
	assert.Equal(t, count.Load(), started.Load())
}
