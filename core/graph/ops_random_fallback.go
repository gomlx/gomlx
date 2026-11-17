// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/support/exceptions"
)

// rngBitGeneratorFallback decomposes counter-based PRNG algorithms into standard GoMLX
// graph operations when the backend returns ErrNotImplemented for RNGBitGenerator.
func rngBitGeneratorFallback(rngState *Node, shape shapes.Shape, algo RNGAlgorithm) (newRNGState, values *Node) {
	expectedState := algo.StateShape()
	if !expectedState.Ok() {
		Panicf("RNGBitGenerator fallback: algorithm %s is not counter-based or has no canonical state shape", algo)
	}
	if !rngState.Shape().Equal(expectedState) {
		Panicf("RNGBitGenerator(%s): expected state shape %s, got %s", algo, expectedState, rngState.Shape())
	}
	if shape.IsDynamic() {
		Panicf("RNGBitGenerator(%s): decomposed fallback does not support dynamic output shapes (%s)", algo, shape)
	}
	if shape.Size() == 0 {
		return rngState, Zeros(rngState.Graph(), shape)
	}

	switch algo {
	case RNGPhilox:
		return rngPhiloxFallback(rngState, shape)
	case RNGThreeFry:
		return rngThreeFryFallback(rngState, shape)
	case RNGSplitMix64:
		return rngSplitMix64Fallback(rngState, shape)
	case RNGChaCha8:
		return rngChaCha8Fallback(rngState, shape)
	default:
		Panicf("RNGBitGenerator: algorithm %s does not support graph decomposition fallback", algo)
		return nil, nil
	}
}

func rngExtractScalar(state *Node, idx int) *Node {
	return Reshape(Slice(state, AxisRange(idx, idx+1)))
}

func rngSplitUint64ToPair32(v *Node) (lo, hi *Node) {
	lo = ConvertDType(v, dtypes.Uint32)
	hi = ConvertDType(BitwiseShiftRightLogicalScalar(v, 32), dtypes.Uint32)
	return lo, hi
}

func rngCombinePair32ToUint64(lo, hi *Node) *Node {
	lo64 := ConvertDType(lo, dtypes.Uint64)
	hi64 := BitwiseShiftLeftScalar(ConvertDType(hi, dtypes.Uint64), 32)
	return BitwiseOr(lo64, hi64)
}

func rngPack32BitWordsToOutput(words *Node, shape shapes.Shape) *Node {
	numElems := shape.Size()
	var flat *Node
	if shape.DType == dtypes.Uint64 {
		pairs := Reshape(words, numElems, 2)
		lo := Reshape(Slice(pairs, AxisRange(), AxisRange(0, 1)), numElems)
		hi := Reshape(Slice(pairs, AxisRange(), AxisRange(1, 2)), numElems)
		flat = rngCombinePair32ToUint64(lo, hi)
	} else if shape.DType != dtypes.Uint32 {
		flat = ConvertDType(words, shape.DType)
	} else {
		flat = words
	}
	if shape.Rank() == 0 {
		return Reshape(flat)
	}
	return Reshape(flat, shape.Dimensions...)
}

func rngRotateLeft32(x *Node, r int) *Node {
	return BitwiseOr(
		BitwiseShiftLeftScalar(x, r),
		BitwiseShiftRightLogicalScalar(x, 32-r),
	)
}

func rngSplitMix64Fallback(rngState *Node, shape shapes.Shape) (newRNGState, values *Node) {
	g := rngState.Graph()
	key := rngExtractScalar(rngState, 0)
	ctr := rngExtractScalar(rngState, 1)

	numElems := shape.Size()
	idx := Iota(g, shapes.Make(dtypes.Uint64, numElems), 0)
	gamma := Const(g, uint64(0x9e3779b97f4a7c15))
	c1 := Const(g, uint64(0xbf58476d1ce4e5b9))
	c2 := Const(g, uint64(0x94d049bb133111eb))

	z := Add(key, Mul(Add(Add(idx, ctr), Const(g, uint64(1))), gamma))
	z = Mul(BitwiseXor(z, BitwiseShiftRightLogicalScalar(z, 30)), c1)
	z = Mul(BitwiseXor(z, BitwiseShiftRightLogicalScalar(z, 27)), c2)
	z = BitwiseXor(z, BitwiseShiftRightLogicalScalar(z, 31))

	if shape.DType != dtypes.Uint64 {
		z = ConvertDType(z, shape.DType)
	}
	if shape.Rank() == 0 {
		values = Reshape(z)
	} else {
		values = Reshape(z, shape.Dimensions...)
	}

	newCtr := Add(ctr, Const(g, uint64(numElems)))
	newRNGState = Concatenate([]*Node{Reshape(key, 1), Reshape(newCtr, 1)}, 0)
	return newRNGState, values
}

func rngThreeFryFallback(rngState *Node, shape shapes.Shape) (newRNGState, values *Node) {
	g := rngState.Graph()
	key64 := rngExtractScalar(rngState, 0)
	ctr64 := rngExtractScalar(rngState, 1)

	numElems := shape.Size()
	wordsPerElem := 1
	if shape.DType == dtypes.Uint64 {
		wordsPerElem = 2
	}
	numWords := numElems * wordsPerElem
	numBlocks := (numWords + 1) / 2

	key0, key1 := rngSplitUint64ToPair32(key64)
	ctr0, ctr1 := rngSplitUint64ToPair32(ctr64)

	blockIota := Iota(g, shapes.Make(dtypes.Uint32, numBlocks), 0)
	c0 := Add(blockIota, ctr0)
	carry := ConvertDType(LessThan(c0, ctr0), dtypes.Uint32)
	c1 := Add(ctr1, carry)

	key2 := BitwiseXor(BitwiseXor(key0, key1), Const(g, uint32(0x1BD11BDA)))
	ks := [3]*Node{key0, key1, key2}

	x0 := Add(c0, ks[0])
	x1 := Add(c1, ks[1])

	rotations := [8]int{13, 15, 26, 6, 17, 29, 16, 24}
	for r := range 20 {
		x0 = Add(x0, x1)
		x1 = BitwiseXor(rngRotateLeft32(x1, rotations[r%8]), x0)
		if (r+1)%4 == 0 {
			inj := (r + 1) / 4
			x0 = Add(x0, ks[inj%3])
			x1 = Add(x1, Add(ks[(inj+1)%3], Scalar(g, dtypes.Uint32, uint32(inj))))
		}
	}

	paired := Concatenate([]*Node{Reshape(x0, numBlocks, 1), Reshape(x1, numBlocks, 1)}, 1)
	flatWords := Reshape(paired, numBlocks*2)
	if numWords < numBlocks*2 {
		flatWords = Slice(flatWords, AxisRange(0, numWords))
	}
	values = rngPack32BitWordsToOutput(flatWords, shape)

	newCtr64 := Add(ctr64, Scalar(g, dtypes.Uint64, uint64(numBlocks)))
	newRNGState = Concatenate([]*Node{Reshape(key64, 1), Reshape(newCtr64, 1)}, 0)
	return newRNGState, values
}

func rngPhiloxFallback(rngState *Node, shape shapes.Shape) (newRNGState, values *Node) {
	g := rngState.Graph()
	key64 := rngExtractScalar(rngState, 0)
	ctrLo := rngExtractScalar(rngState, 1)
	ctrHi := rngExtractScalar(rngState, 2)

	numElems := shape.Size()
	wordsPerElem := 1
	if shape.DType == dtypes.Uint64 {
		wordsPerElem = 2
	}
	numWords := numElems * wordsPerElem
	numBlocks := (numWords + 3) / 4

	key0, key1 := rngSplitUint64ToPair32(key64)
	blockIota64 := Iota(g, shapes.Make(dtypes.Uint64, numBlocks), 0)
	blockCtrLo := Add(ctrLo, blockIota64)
	blockCarry := ConvertDType(LessThan(blockCtrLo, ctrLo), dtypes.Uint64)
	blockCtrHi := Add(ctrHi, blockCarry)

	c0, c1 := rngSplitUint64ToPair32(blockCtrLo)
	c2, c3 := rngSplitUint64ToPair32(blockCtrHi)

	const (
		w0 uint32 = 0x9E3779B9
		w1 uint32 = 0xBB67AE85
		m0 uint64 = 0xD2511F53
		m1 uint64 = 0xCD9E8D57
	)
	m0Const := Const(g, m0)
	m1Const := Const(g, m1)
	k0 := key0
	k1 := key1
	for range 10 {
		p0 := Mul(m0Const, ConvertDType(c0, dtypes.Uint64))
		p1 := Mul(m1Const, ConvertDType(c2, dtypes.Uint64))
		lo0, hi0 := rngSplitUint64ToPair32(p0)
		lo1, hi1 := rngSplitUint64ToPair32(p1)
		c0 = BitwiseXor(BitwiseXor(hi1, c1), k0)
		c1 = lo1
		c2 = BitwiseXor(BitwiseXor(hi0, c3), k1)
		c3 = lo0
		k0 = AddScalar(k0, w0)
		k1 = AddScalar(k1, w1)
	}

	packed := Concatenate([]*Node{
		Reshape(c0, numBlocks, 1),
		Reshape(c1, numBlocks, 1),
		Reshape(c2, numBlocks, 1),
		Reshape(c3, numBlocks, 1),
	}, 1)
	flatWords := Reshape(packed, numBlocks*4)
	if numWords < numBlocks*4 {
		flatWords = Slice(flatWords, AxisRange(0, numWords))
	}
	values = rngPack32BitWordsToOutput(flatWords, shape)

	newLo := Add(ctrLo, Scalar(g, dtypes.Uint64, uint64(numBlocks)))
	carry := ConvertDType(LessThan(newLo, ctrLo), dtypes.Uint64)
	newHi := Add(ctrHi, carry)
	newRNGState = Concatenate([]*Node{Reshape(key64, 1), Reshape(newLo, 1), Reshape(newHi, 1)}, 0)
	return newRNGState, values
}

func rngChaCha8Fallback(rngState *Node, shape shapes.Shape) (newRNGState, values *Node) {
	g := rngState.Graph()
	numElems := shape.Size()
	wordsPerElem := 1
	if shape.DType == dtypes.Uint64 {
		wordsPerElem = 2
	}
	numWords := numElems * wordsPerElem
	numBlocks := (numWords + 15) / 16

	ctr64 := rngExtractScalar(rngState, 4)
	blockIota64 := Iota(g, shapes.Make(dtypes.Uint64, numBlocks), 0)
	blockCtr64 := Add(ctr64, blockIota64)
	ctrLo, ctrHi := rngSplitUint64ToPair32(blockCtr64)

	zeroVec := Zeros(g, shapes.Make(dtypes.Uint32, numBlocks))
	onesConst := func(v uint32) *Node {
		return BroadcastToDims(Const(g, v), numBlocks)
	}

	var init [16]*Node
	init[0] = onesConst(0x61707865)
	init[1] = onesConst(0x3320646e)
	init[2] = onesConst(0x79622d32)
	init[3] = onesConst(0x6b206574)
	for i := range 4 {
		k := rngExtractScalar(rngState, i)
		kLo, kHi := rngSplitUint64ToPair32(k)
		init[4+2*i] = BroadcastToDims(kLo, numBlocks)
		init[4+2*i+1] = BroadcastToDims(kHi, numBlocks)
	}
	init[12] = ctrLo
	init[13] = ctrHi
	init[14] = zeroVec
	init[15] = zeroVec

	var work [16]*Node
	copy(work[:], init[:])

	qr := func(a, b, c, d int) {
		work[a] = Add(work[a], work[b])
		work[d] = rngRotateLeft32(BitwiseXor(work[d], work[a]), 16)
		work[c] = Add(work[c], work[d])
		work[b] = rngRotateLeft32(BitwiseXor(work[b], work[c]), 12)
		work[a] = Add(work[a], work[b])
		work[d] = rngRotateLeft32(BitwiseXor(work[d], work[a]), 8)
		work[c] = Add(work[c], work[d])
		work[b] = rngRotateLeft32(BitwiseXor(work[b], work[c]), 7)
	}

	for range 4 {
		qr(0, 4, 8, 12)
		qr(1, 5, 9, 13)
		qr(2, 6, 10, 14)
		qr(3, 7, 11, 15)
		qr(0, 5, 10, 15)
		qr(1, 6, 11, 12)
		qr(2, 7, 8, 13)
		qr(3, 4, 9, 14)
	}

	cols := make([]*Node, 16)
	for i := range 16 {
		cols[i] = Reshape(Add(work[i], init[i]), numBlocks, 1)
	}
	flatWords := Reshape(Concatenate(cols, 1), numBlocks*16)
	if numWords < numBlocks*16 {
		flatWords = Slice(flatWords, AxisRange(0, numWords))
	}
	values = rngPack32BitWordsToOutput(flatWords, shape)

	newCtr64 := Add(ctr64, Scalar(g, dtypes.Uint64, uint64(numBlocks)))
	keyPrefix := Slice(rngState, AxisRange(0, 4))
	newRNGState = Concatenate([]*Node{keyPrefix, Reshape(newCtr64, 1)}, 0)
	return newRNGState, values
}
