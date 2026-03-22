package humanize

import "fmt"

// Bytes returns the rendering of bytes aproximated to the nearest power of 1024 (Kb, Mb, Gb, Tb, etc.)
// with one decimal place.
func Bytes(num int64) string {
	sign := ""
	if num < 0 {
		sign = "-"
		num = -num
	}
	if num < 1024 {
		return fmt.Sprintf("%s%d B", sign, num)
	}
	const unit = 1024
	div, exp := int64(unit), 0
	for n := num / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%s%.1f %cb", sign, float64(num)/float64(div), "KMGTPE"[exp])
}
