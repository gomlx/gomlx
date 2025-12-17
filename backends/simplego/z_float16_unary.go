package simplego

// This file must come alphabetically after exec_unary.go to ensure its init() runs later.
// It adds Float16 support to unary operations.

import (
	"math"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/x448/float16"
)

// Float16 unary operation helpers

func execNegF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		outputs[ii] = float16.Fromfloat32(-input.Float32())
	}
}

func execAbsF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		f := input.Float32()
		if f < 0 {
			outputs[ii] = float16.Fromfloat32(-f)
		} else {
			outputs[ii] = input
		}
	}
}

func execSignF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		f := input.Float32()
		switch {
		case f < 0:
			outputs[ii] = float16.Fromfloat32(-1.0)
		case f > 0:
			outputs[ii] = float16.Fromfloat32(1.0)
		default:
			outputs[ii] = float16.Fromfloat32(0.0)
		}
	}
}

func execExpF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		outputs[ii] = float16.Fromfloat32(float32(math.Exp(float64(input.Float32()))))
	}
}

func execExpm1F16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		outputs[ii] = float16.Fromfloat32(float32(math.Expm1(float64(input.Float32()))))
	}
}

func execLogF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		outputs[ii] = float16.Fromfloat32(float32(math.Log(float64(input.Float32()))))
	}
}

func execLog1pF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		outputs[ii] = float16.Fromfloat32(float32(math.Log1p(float64(input.Float32()))))
	}
}

func execCeilF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		outputs[ii] = float16.Fromfloat32(float32(math.Ceil(float64(input.Float32()))))
	}
}

func execFloorF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		outputs[ii] = float16.Fromfloat32(float32(math.Floor(float64(input.Float32()))))
	}
}

func execRoundF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		outputs[ii] = float16.Fromfloat32(float32(math.Round(float64(input.Float32()))))
	}
}

func execRsqrtF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		outputs[ii] = float16.Fromfloat32(float32(1.0 / math.Sqrt(float64(input.Float32()))))
	}
}

func execCosF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		outputs[ii] = float16.Fromfloat32(float32(math.Cos(float64(input.Float32()))))
	}
}

func execSinF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		outputs[ii] = float16.Fromfloat32(float32(math.Sin(float64(input.Float32()))))
	}
}

func execTanhF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		outputs[ii] = float16.Fromfloat32(float32(math.Tanh(float64(input.Float32()))))
	}
}

func execLogisticF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		input64 := float64(input.Float32())
		var output64 float64
		if input64 >= 0 {
			output64 = 1.0 / (1.0 + math.Exp(-input64))
		} else {
			e_x := math.Exp(input64)
			output64 = e_x / (1.0 + e_x)
		}
		outputs[ii] = float16.Fromfloat32(float32(output64))
	}
}

func execIsFiniteF16(inputs []float16.Float16, outputs []bool) {
	for ii, input := range inputs {
		f := input.Float32()
		outputs[ii] = !math.IsInf(float64(f), 0) && !math.IsNaN(float64(f))
	}
}

func execErfF16(inputs, outputs []float16.Float16) {
	for ii, input := range inputs {
		outputs[ii] = float16.Fromfloat32(float32(math.Erf(float64(input.Float32()))))
	}
}

// Store original executors
var (
	origExecNeg      func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecAbs      func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecSign     func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecExp      func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecExpm1    func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecLog      func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecLog1p    func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecCeil     func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecFloor    func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecRound    func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecRsqrt    func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecCos      func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecSin      func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecTanh     func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecLogistic func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecIsFinite func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecErf      func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
)

func makeFloat16UnaryWrapper(
	origExec func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error),
	opFn func(inputs, outputs []float16.Float16),
) func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error) {
	return func(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
		if inputs[0].shape.DType != dtypes.Float16 {
			return origExec(backend, node, inputs, inputsOwned)
		}
		input, output := unaryOperandAndOutput(backend, inputs, inputsOwned)
		opFn(input.flat.([]float16.Float16), output.flat.([]float16.Float16))
		return output, nil
	}
}

func init() {
	// Save original executors
	origExecNeg = nodeExecutors[backends.OpTypeNeg]
	origExecAbs = nodeExecutors[backends.OpTypeAbs]
	origExecSign = nodeExecutors[backends.OpTypeSign]
	origExecExp = nodeExecutors[backends.OpTypeExp]
	origExecExpm1 = nodeExecutors[backends.OpTypeExpm1]
	origExecLog = nodeExecutors[backends.OpTypeLog]
	origExecLog1p = nodeExecutors[backends.OpTypeLog1p]
	origExecCeil = nodeExecutors[backends.OpTypeCeil]
	origExecFloor = nodeExecutors[backends.OpTypeFloor]
	origExecRound = nodeExecutors[backends.OpTypeRound]
	origExecRsqrt = nodeExecutors[backends.OpTypeRsqrt]
	origExecCos = nodeExecutors[backends.OpTypeCos]
	origExecSin = nodeExecutors[backends.OpTypeSin]
	origExecTanh = nodeExecutors[backends.OpTypeTanh]
	origExecLogistic = nodeExecutors[backends.OpTypeLogistic]
	origExecIsFinite = nodeExecutors[backends.OpTypeIsFinite]
	origExecErf = nodeExecutors[backends.OpTypeErf]

	// Wrap with Float16 support
	nodeExecutors[backends.OpTypeNeg] = makeFloat16UnaryWrapper(origExecNeg, execNegF16)
	nodeExecutors[backends.OpTypeAbs] = makeFloat16UnaryWrapper(origExecAbs, execAbsF16)
	nodeExecutors[backends.OpTypeSign] = makeFloat16UnaryWrapper(origExecSign, execSignF16)
	nodeExecutors[backends.OpTypeExp] = makeFloat16UnaryWrapper(origExecExp, execExpF16)
	nodeExecutors[backends.OpTypeExpm1] = makeFloat16UnaryWrapper(origExecExpm1, execExpm1F16)
	nodeExecutors[backends.OpTypeLog] = makeFloat16UnaryWrapper(origExecLog, execLogF16)
	nodeExecutors[backends.OpTypeLog1p] = makeFloat16UnaryWrapper(origExecLog1p, execLog1pF16)
	nodeExecutors[backends.OpTypeCeil] = makeFloat16UnaryWrapper(origExecCeil, execCeilF16)
	nodeExecutors[backends.OpTypeFloor] = makeFloat16UnaryWrapper(origExecFloor, execFloorF16)
	nodeExecutors[backends.OpTypeRound] = makeFloat16UnaryWrapper(origExecRound, execRoundF16)
	nodeExecutors[backends.OpTypeRsqrt] = makeFloat16UnaryWrapper(origExecRsqrt, execRsqrtF16)
	nodeExecutors[backends.OpTypeCos] = makeFloat16UnaryWrapper(origExecCos, execCosF16)
	nodeExecutors[backends.OpTypeSin] = makeFloat16UnaryWrapper(origExecSin, execSinF16)
	nodeExecutors[backends.OpTypeTanh] = makeFloat16UnaryWrapper(origExecTanh, execTanhF16)
	nodeExecutors[backends.OpTypeLogistic] = makeFloat16UnaryWrapper(origExecLogistic, execLogisticF16)
	nodeExecutors[backends.OpTypeErf] = makeFloat16UnaryWrapper(origExecErf, execErfF16)

	// IsFinite is special - returns bool
	nodeExecutors[backends.OpTypeIsFinite] = func(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
		if inputs[0].shape.DType != dtypes.Float16 {
			return origExecIsFinite(backend, node, inputs, inputsOwned)
		}
		input := inputs[0]
		output := backend.getBuffer(dtypes.Bool, input.shape.Size())
		output.shape = node.shape
		execIsFiniteF16(input.flat.([]float16.Float16), output.flat.([]bool))
		return output, nil
	}
}
