//go:build linux && arm64

package simplego

// detectFP16NEON checks if FP16 NEON instructions (FMLAL/FMLAL2) are available.
// These require ARMv8.2-A with FEAT_FHM (half-precision FP multiply-add).
func detectFP16NEON() bool {
	// NOTE: Even when /proc/cpuinfo reports "fphp" or "asimdhp", Apple Silicon
	// running in Docker does NOT actually support FMLAL/FMLAL2 instructions.
	// They cause SIGILL. Apple's ARM implementation doesn't include these.
	// Disable FP16 NEON for safety until we can verify on actual ARM Linux hardware
	// (e.g., AWS Graviton, Ampere Altra, or other server ARM64 chips).
	// TODO: Re-enable with proper hardware detection on real ARM64 Linux servers.
	return false
}

// detectBF16NEON checks if BF16 NEON instructions (BFMLALB/BFMLALT) are available.
// These require ARMv8.6-A with FEAT_BF16.
func detectBF16NEON() bool {
	// NOTE: Even when /proc/cpuinfo reports "bf16", Apple Silicon running in Docker
	// does NOT actually support BFMLALB/BFMLALT instructions - they cause SIGILL.
	// This appears to be a hardware limitation of Apple M-series chips.
	// Disable BF16 NEON for safety until we can verify on actual ARM Linux hardware.
	// TODO: Re-enable with proper hardware detection on real ARM64 Linux servers.
	return false
}
