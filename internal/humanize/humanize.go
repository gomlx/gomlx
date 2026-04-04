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

// Underscores returns the rendering of an integer with underscores as thousands separators.
func Underscores[T ~int64 | ~uint64 | ~int | ~uint | ~int32 | ~uint32 | ~int16 | ~uint16 | ~int8 | ~uint8 | ~uintptr](numAny T) string {
	s := fmt.Sprintf("%d", numAny)
	start := 0
	if s[0] == '-' {
		start = 1
	}
	var result []byte
	if start == 1 {
		result = append(result, '-')
	}
	for i := start; i < len(s); i++ {
		if i > start && (len(s)-i)%3 == 0 {
			result = append(result, '_')
		}
		result = append(result, s[i])
	}
	return string(result)
}
