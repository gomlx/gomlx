//go:build darwin && arm64

package simplego

import (
	"syscall"
)

// detectFP16NEON checks if FP16 NEON instructions (FMLAL/FMLAL2) are available.
// These require ARMv8.2-A with FEAT_FHM (half-precision FP multiply-add).
func detectFP16NEON() bool {
	// macOS: Check for FP16 support via sysctl
	// Apple Silicon (M1+) supports FP16 instructions
	val, err := syscall.Sysctl("hw.optional.arm.FEAT_FHM")
	if err == nil && len(val) > 0 && val[0] != 0 {
		return true
	}
	// Fallback: Apple M1 and later always support FP16
	val, err = syscall.Sysctl("hw.optional.neon_fp16")
	if err == nil && len(val) > 0 && val[0] != 0 {
		return true
	}
	return false
}

// detectBF16NEON checks if BF16 NEON instructions (BFMLALB) are available.
// These require ARMv8.6-A with FEAT_BF16.
// Note: On Apple Silicon, BFMLALB computes both even and odd element products
// for each lane (i.e., full dot product per pair), so we only use BFMLALB
// without BFMLALT in our implementation.
func detectBF16NEON() bool {
	// macOS: Check for BF16 support via sysctl
	// Apple Silicon M3+ supports BF16 instructions
	val, err := syscall.Sysctl("hw.optional.arm.FEAT_BF16")
	if err == nil && len(val) > 0 && val[0] != 0 {
		return true
	}
	// Fallback: Check for Apple M3 or later (M1/M2 don't have BF16)
	// We can't reliably detect this, so conservative approach
	return false
}
