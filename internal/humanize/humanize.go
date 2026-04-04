package humanize

import (
	"fmt"
)

// Bytes returns the rendering of bytes aproximated to the nearest power of 1024 (Kb, Mb, Gb, Tb, etc.)
// with one decimal place.
func Bytes[T ~int64 | ~uint64 | ~int | ~uint | ~int32 | ~uint32 | ~int16 | ~uint16 | ~int8 | ~uint8 | ~uintptr](numAny T) string {
	num := int64(numAny)
	sign := ""
	if num < 0 {
		sign = "-"
		num = -num
	}
	const unit = 1024
	if num < unit {
		return fmt.Sprintf("%s%d B", sign, num)
	}
	div, exp := int64(unit), 0
	for n := num / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%s%.1f %ciB", sign, float64(num)/float64(div), "KMGTPE"[exp])
}
