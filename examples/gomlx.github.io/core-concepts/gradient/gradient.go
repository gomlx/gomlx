package gradient

import (
	. "github.com/gomlx/gomlx/core/graph"
)

//md_start:grad_fn

func GradFn(x, y *Node) (loss, gradX, gradY *Node) {
	// f(x, y) = x^2 + xy
	loss = Add(Square(x), Mul(x, y))

	// Reduce if inputs are not scalars, as Gradient requires a scalar loss
	scalarLoss := ReduceAllSum(loss)

	// Calculate gradients df/dx and df/dy symbolically
	grads := Gradient(scalarLoss, x, y)
	return loss, grads[0], grads[1]
}

//md_end:grad_fn
