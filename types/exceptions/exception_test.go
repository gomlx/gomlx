package exceptions

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"testing"
)

func testCatch(fn func()) (eInt int, eErr error, eFloat float64) {
	defer Catch(func(x int) { eInt = x })
	defer Catch(func(x error) { eErr = x })
	defer Catch(func(x float64) { eFloat = x })
	fn()
	return
}

func TestCatch(t *testing.T) {
	// No throws.
	eInt, eErr, eFloat := testCatch(func() {})
	assert.Equal(t, 0, eInt)
	assert.Equal(t, nil, eErr)
	assert.Equal(t, 0.0, eFloat)

	// Throw an int.
	eInt, eErr, eFloat = testCatch(func() {
		Throw(7)
	})
	assert.Equal(t, 7, eInt)
	assert.Equal(t, nil, eErr)
	assert.Equal(t, 0.0, eFloat)

	// Throw an error.
	e := fmt.Errorf("blah")
	eInt, eErr, eFloat = testCatch(func() { Throw(e) })
	assert.Equal(t, 0, eInt)
	assert.Equal(t, e, eErr)
	assert.Equal(t, 0.0, eFloat)
}

func TestTry(t *testing.T) {
	assert.Equal(t, 3, Try(func() { Throw(3) }))
	assert.Equal(t, 7.0, Try(func() { Throw(7.0) }))
	assert.Equal(t, nil, Try(func() {}))
}

func TestTryFor(t *testing.T) {
	x := TryFor[int](func() {}) // No exceptions, should return 0.
	assert.Equal(t, 0, x)
	x = TryFor[int](func() { Throw(11) })
	assert.Equal(t, 11, x)
	x = 13
	assert.Panics(t, func() {
		// Throwing a float shouldn't be caught by a Try[int].
		x = TryFor[int](func() { Throw(1.0) })
	})
	assert.Equal(t, 13, x, "x shouldn't have changed value")
}
