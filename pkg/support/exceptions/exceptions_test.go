// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package exceptions_test

import (
	"fmt"
	"testing"

	"github.com/gomlx/gomlx/pkg/support/exceptions"
	"github.com/pkg/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// testCatchExceptions is a helper function to catch different types of exceptions and return them accordingly.
func testCatchExceptions(fn func()) (int, float64, error) {
	var (
		eInt   int
		eErr   error
		eFloat float64
	)
	exception := exceptions.Try(fn)
	if exception != nil {
		switch e := exception.(type) {
		case int:
			eInt = e
		case error:
			eErr = e
		case float64:
			eFloat = e
		default:
			panic(e)
		}
	}
	return eInt, eFloat, eErr
}

//nolint:testifylint // The checks for errors and equality of floats don't apply here.
func TestTry(t *testing.T) {
	// No throws.
	eInt, eFloat, eErr := testCatchExceptions(func() {})
	assert.Equal(t, 0, eInt)
	require.NoError(t, eErr)
	assert.Equal(t, 0.0, eFloat)

	// Panicf an int.
	eInt, eFloat, eErr = testCatchExceptions(func() {
		panic(7)
	})
	assert.Equal(t, 7, eInt)
	assert.NoError(t, eErr)
	assert.Equal(t, 0.0, eFloat)

	// Panicf an error.
	e := errors.New("blah")
	eInt, eFloat, eErr = testCatchExceptions(func() { panic(e) })
	assert.Equal(t, 0, eInt)
	assert.ErrorIs(t, e, eErr)
	assert.Equal(t, 0.0, eFloat)

	// Panicf something different.
	assert.Panics(t,
		func() {
			// A string exception is not caught.
			_, _, _ = testCatchExceptions(func() { panic("some string") })
		})
}

func TestTryCatch(t *testing.T) {
	want := errors.New("test error")
	var err error
	require.NotPanics(t, func() { err = exceptions.TryCatch[error](func() { panic(want) }) })
	require.EqualError(t, err, want.Error())
}

func TestThrow(t *testing.T) {
	err := exceptions.TryCatch[error](func() { exceptions.Panicf("2+3=%d", 2+3) })
	require.EqualError(t, err, "2+3=5")
}

func TestRuntimeErrors(t *testing.T) {
	var x any = 0.0
	//nolint:forbidigo // fmt.Println is used to cause a panic here.
	err := exceptions.TryCatch[error](func() { fmt.Println(x.(string)) })
	require.ErrorContains(t, err, "interface conversion")
}
