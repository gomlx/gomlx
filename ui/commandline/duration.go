// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package commandline

import (
	"fmt"
	"regexp"
	"strconv"
	"time"
)

// FormatDuration pretty prints duration without a long list of decimal points.
func FormatDuration(d time.Duration) string {
	s := d.String()
	re := regexp.MustCompile(`(\d+\.?\d*)([µa-z]+)`)
	matches := re.FindStringSubmatch(s)
	if len(matches) != 3 {
		return s
	}
	num, err := strconv.ParseFloat(matches[1], 64)
	if err != nil {
		return s
	}
	return fmt.Sprintf("%.2f%s", num, matches[2])
}
