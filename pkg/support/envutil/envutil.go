// Package envutil provides utility functions for working with environment variables.
package envutil

import (
	"os"
	"strconv"
	"strings"

	"github.com/gomlx/compute/support/xslices"
	"github.com/gomlx/gomlx/pkg/support/sets"
	"github.com/pkg/errors"
)

var (
	FalseValues = sets.MakeWith("0", "false", "f", "no", "off", "disabled")
	TrueValues  = sets.MakeWith("1", "true", "t", "yes", "on", "enabled")
)

// ReadBool reads the boolean value of the environment variable with the given name.
// If not set, it returns the defaultValue.
// It returns an error if the value is set but not a valid boolean value.
func ReadBool(name string, defaultValue bool) (bool, error) {
	val := os.Getenv(name)
	if val == "" {
		return defaultValue, nil
	}
	val = strings.ToLower(val)
	if TrueValues.Has(val) {
		return true, nil
	}
	if FalseValues.Has(val) {
		return false, nil
	}
	return defaultValue, errors.Errorf("invalid value %q for environment variable %q; valid values are %q",
		val, name, append(xslices.Keys(TrueValues), xslices.Keys(FalseValues)...))
}

// ReadInt reads the integer value of the environment variable with the given name.
// If not set, it returns the defaultValue.
// It returns an error if the value is set but not a valid integer value.
func ReadInt(name string, defaultValue int) (int, error) {
	val := os.Getenv(name)
	if val == "" {
		return defaultValue, nil
	}
	valInt, err := strconv.Atoi(val)
	if err != nil {
		return defaultValue, errors.Wrapf(err, "invalid value %q for environment variable %q; must be an integer", val, name)
	}
	return valInt, nil
}
