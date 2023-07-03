package exceptions

import (
	"fmt"
	"github.com/pkg/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

func testCountExceptions(fn func()) (eInt int, eErr error, eFloat float64) {
	exception := Try(fn)
	if exception != nil {
		if e, ok := exception.(int); ok {
			eInt = e
		} else if e, ok := exception.(error); ok {
			eErr = e
		} else if e, ok := exception.(float64); ok {
			eFloat = e
		} else {
			panic(e)
		}
	}
	return
}

func TestTry(t *testing.T) {
	// No throws.
	eInt, eErr, eFloat := testCountExceptions(func() {})
	assert.Equal(t, 0, eInt)
	assert.Equal(t, nil, eErr)
	assert.Equal(t, 0.0, eFloat)

	// Panicf an int.
	eInt, eErr, eFloat = testCountExceptions(func() {
		panic(7)
	})
	assert.Equal(t, 7, eInt)
	assert.Equal(t, nil, eErr)
	assert.Equal(t, 0.0, eFloat)

	// Panicf an error.
	e := fmt.Errorf("blah")
	eInt, eErr, eFloat = testCountExceptions(func() { panic(e) })
	assert.Equal(t, 0, eInt)
	assert.Equal(t, e, eErr)
	assert.Equal(t, 0.0, eFloat)

	// Panicf something different.
	assert.Panics(t,
		func() {
			// Exception of type string is not caught.
			_, _, _ = testCountExceptions(func() { panic("some string") })
		})
}

func TestTryCatch(t *testing.T) {
	want := errors.New("test error")
	var err error
	require.NotPanics(t, func() { err = TryCatch[error](func() { panic(want) }) })
	require.EqualError(t, err, want.Error())
}

func TestThrow(t *testing.T) {
	err := TryCatch[error](func() { Panicf("2+3=%d", 2+3) })
	require.EqualError(t, err, "2+3=5")
}
