package packgemm

/*
//  go:generate go tool github.com/ajroetker/go-highway/cmd/hwygen -input gemm_16reg_base.go -output_prefix=gen_gemm_16reg_impl -dispatch gen_gemm_16reg_dispatch -targets avx2,avx512,neon,fallback

func BaseGEMM16Registers[T hwy.Floats](
	alpha, beta float32, lhsFlat, rhsFlat []T, batchSize, lhsCrossSize, rhsCrossSize, contractingSize int, outputFlat []T,
	bufAllocFn BufAllocFn[T], bufReleaseFn BufReleaseFn, pool *workerspool.Pool) error {
	avx512WarningOnce.Do(func() {
		klog.Infof("AVX512 GEMM (General Matrix Multiplication) algorithm still experimental!")
	})

	// 1. Resolve Strides
	params := &avx512Float32Params
	lhsBatchStride := lhsCrossSize * contractingSize
	rhsBatchStride := contractingSize * rhsCrossSize
	outputBatchStride := lhsCrossSize * rhsCrossSize

	// Split work in reasonable number of "chunks".
	maxWorkers := 1
	if pool != nil {
		maxWorkers = pool.AdjustedMaxParallelism()
	}
	if maxWorkers <= 1 {
		// Do everything sequentially.
		packedLhsRef, packedLHS := bufAllocFn(params.LHSPanelCrossSize * params.PanelContractingSize)
		packedRhsRef, packedRHS := bufAllocFn(params.PanelContractingSize * params.RHSPanelCrossSize)
		packedOutRef, packedOutput := bufAllocFn(params.LHSPanelCrossSize * params.RHSPanelCrossSize)
		defer func() {
			bufReleaseFn(packedLhsRef)
			bufReleaseFn(packedRhsRef)
			bufReleaseFn(packedOutRef)
		}()
		for batchIdx := range batchSize {
			batchLhs := lhsFlat[batchIdx*lhsBatchStride : (batchIdx+1)*lhsBatchStride]
			batchRhs := rhsFlat[batchIdx*rhsBatchStride : (batchIdx+1)*rhsBatchStride]
			batchOutput := outputFlat[batchIdx*outputBatchStride : (batchIdx+1)*outputBatchStride]
			avx512Float32GemmChunk(
				alpha, beta,
				batchLhs, batchRhs, batchOutput,
				lhsCrossSize, rhsCrossSize, contractingSize,
				params, 0, lhsCrossSize, 0, rhsCrossSize,
				packedLHS, packedRHS, packedOutput,
			)
		}
		return nil
	}

	// 1. Split work in workItems.
	workChan := make(chan workItem, max(2000, 2*maxWorkers))
	go feedWorkItems(
		batchSize, lhsCrossSize, rhsCrossSize,
		params, maxWorkers, workChan)

	// 2. Saturate (fan-out workers) on workItems.
	pool.Saturate(func() {
		packedLhsRef, packedLHS := bufAllocFn(params.LHSPanelCrossSize * params.PanelContractingSize)
		packedRhsRef, packedRHS := bufAllocFn(params.PanelContractingSize * params.RHSPanelCrossSize)
		packedOutRef, packedOutput := bufAllocFn(params.LHSPanelCrossSize * params.RHSPanelCrossSize)
		defer func() {
			bufReleaseFn(packedLhsRef)
			bufReleaseFn(packedRhsRef)
			bufReleaseFn(packedOutRef)
		}()

		for item := range workChan {
			for batchIdx := item.batchStart; batchIdx < item.batchEnd; batchIdx++ {
				batchLhs := lhsFlat[batchIdx*lhsBatchStride : (batchIdx+1)*lhsBatchStride]
				batchRhs := rhsFlat[batchIdx*rhsBatchStride : (batchIdx+1)*rhsBatchStride]
				batchOutput := outputFlat[batchIdx*outputBatchStride : (batchIdx+1)*outputBatchStride]
				avx512Float32GemmChunk(
					alpha, beta,
					batchLhs, batchRhs, batchOutput,
					lhsCrossSize, rhsCrossSize, contractingSize,

					params, item.lhsRowStart, item.lhsRowEnd, item.rhsColStart, item.rhsColEnd,
					packedLHS, packedRHS, packedOutput,
				)
			}
		}
	})
	return nil
}
*/
