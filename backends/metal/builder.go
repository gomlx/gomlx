//go:build darwin && cgo

package metal

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/notimplemented"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/support/sets"
	"github.com/pkg/errors"
)

// Builder constructs a computation graph for the Metal backend.
type Builder struct {
	notimplemented.Builder

	name     string
	backend  *Backend
	mainFn   *Function
	compiled bool
}

var _ backends.Builder = (*Builder)(nil)

func newBuilder(b *Backend, name string) *Builder {
	bldr := &Builder{
		name:    name,
		backend: b,
	}

	bldr.Builder = notimplemented.Builder{
		ErrFn: notImplementedError,
	}

	bldr.mainFn = newFunction(bldr, backends.MainName, nil)

	return bldr
}

func notImplementedError(op backends.OpType) error {
	return errors.Wrapf(backends.ErrNotImplemented, "metal backend: op %s", op)
}

func (b *Builder) Name() string            { return b.name }
func (b *Builder) Main() backends.Function { return b.mainFn }

func (b *Builder) NewFunction(name string) (backends.Function, error) {
	if b.compiled {
		return nil, errors.New("builder already compiled")
	}

	if name == "" {
		return nil, errors.New("function name cannot be empty")
	}

	return newFunction(b, name, nil), nil
}

func (b *Builder) checkValues(opName string, values ...backends.Value) ([]*Node, error) {
	if b == nil {
		return nil, errors.Errorf("%s: Builder is nil", opName)
	}

	if b.compiled {
		return nil, errors.Errorf("cannot add op %s: builder already compiled", opName)
	}

	nodes := make([]*Node, len(values))

	for i, v := range values {
		if v == nil {
			return nil, errors.Errorf("%s: input #%d is nil", opName, i)
		}

		node, ok := v.(*Node)

		if !ok {
			return nil, errors.Errorf("%s: input #%d is not a metal node", opName, i)
		}

		if node.builder != b {
			return nil, errors.Errorf("%s: input #%d is from a different builder", opName, i)
		}

		nodes[i] = node
	}

	return nodes, nil
}

func (b *Builder) OpShape(op backends.Value) (shapes.Shape, error) {
	node, ok := op.(*Node)

	if !ok {
		return shapes.Invalid(), errors.New("not a metal node")
	}

	return node.shape, nil
}

func (b *Builder) Compile() (backends.Executable, error) {
	if !b.mainFn.returned {
		return nil, errors.New("main function must have Return() called before Compile()")
	}

	outputs := b.mainFn.outputs
	seen := sets.Make[*Node]()

	for i, node := range outputs {
		if seen.Has(node) {
			idVal, err := b.mainFn.Identity(node)

			if err != nil {
				return nil, errors.WithMessagef(err, "duplicate output at index %d", i)
			}

			idNode, ok := idVal.(*Node)

			if !ok {
				return nil, errors.Errorf("Identity returned unexpected type at output %d", i)
			}

			outputs[i] = idNode
		} else {
			seen.Insert(node)
		}
	}

	for _, node := range outputs {
		if len(node.multiOutputsShapes) > 0 {
			return nil, errors.Errorf("output cannot be internal multi-output node %s", node.opType)
		}
	}

	b.mainFn.outputs = outputs
	mainFe, err := newFunctionExecutable(b.mainFn)

	if err != nil {
		return nil, err
	}

	b.mainFn.compiled = mainFe
	b.compiled = true

	return &Executable{
		backend: b.backend,
		builder: b,
	}, nil
}

// Node in the Metal computation graph.
type Node struct {
	idx      int
	opType   backends.OpType
	shape    shapes.Shape
	inputs   []*Node
	function *Function
	builder  *Builder
	data     any // op-specific data

	// Multi-output support
	multiOutputsShapes []shapes.Shape
	multiOutputsNodes  []*Node
	isNodeSelectOutput bool
	selectOutputIdx    int

	// capturedInputs groups parent-scope nodes used by closures (per closure index).
	capturedInputs [][]*Node
}

// IsMultiOutputs reports whether this node represents a multi-output op head.
func (n *Node) IsMultiOutputs() bool {
	return len(n.multiOutputsShapes) > 0
}

// MultiOutputValues returns per-output values for a multi-output op.
func (node *Node) MultiOutputValues() []backends.Value {
	v := make([]backends.Value, len(node.multiOutputsNodes))

	for i, n := range node.multiOutputsNodes {
		v[i] = n
	}

	return v
}
