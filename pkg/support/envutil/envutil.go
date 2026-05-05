// Package envutil provides utility functions for working with environment variables.
package envutil

import (
	"os"
	"strings"

	"github.com/gomlx/compute/support/xslices"
	"github.com/gomlx/gomlx/pkg/support/sets"
	"github.com/pkg/errors"
)

var (
	FalseValues = sets.MakeWith("0", "false", "f", "no", "off", "disabled")
	TrueValues  = sets.MakeWith("1", "true", "t", "yes", "on", "enabled")
)

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
