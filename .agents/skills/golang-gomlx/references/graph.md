# GoMLX `graph` Package Reference

This table maps common functions from the `pkg/core/graph` package to their PyTorch equivalents, to help AI agents understand their purpose.

**Note**: It is assumed that the `graph` package is dot-imported (`import . "github.com/gomlx/gomlx/pkg/core/graph"`), so these functions can be called directly without the `graph.` prefix.

| GoMLX `graph` Function | Description | PyTorch Equivalent |
|---|---|---|
| **Element-wise Math** | | |
| `Add(a, b)` | Adds two nodes element-wise. Supports broadcasting. | `a + b` |
| `Sub(a, b)` | Subtracts `b` from `a` element-wise. | `a - b` |
| `Mul(a, b)` | Multiplies two nodes element-wise. | `a * b` |
| `Div(a, b)` | Divides `a` by `b` element-wise. | `a / b` |
| `Abs(x)` | Element-wise absolute value. | `torch.abs(x)` |
| `Exp(x)` | Element-wise exponential. | `torch.exp(x)` |
| `Log(x)` | Element-wise natural logarithm. | `torch.log(x)` |
| `Log1p(x)` | Element-wise `log(1 + x)`. | `torch.log1p(x)` |
| `Sqrt(x)` | Element-wise square root. | `torch.sqrt(x)` |
| `Square(x)` | Element-wise square. | `torch.square(x)` |
| `Pow(x, y)` | Element-wise power `x^y`. | `torch.pow(x, y)` |
| `Max(a, b)` | Element-wise maximum of two nodes. | `torch.maximum(a, b)` |
| `Min(a, b)` | Element-wise minimum of two nodes. | `torch.minimum(a, b)` |
| `Sin(x)`, `Cos(x)`, `Tanh(x)` | Element-wise trigonometric functions. | `torch.sin(x)`, `torch.cos(x)`, `torch.tanh(x)` |
| **Shape & Tensor Manipulation** | | |
| `Reshape(x, dims...)` | Reshapes a node to the given dimensions. | `torch.reshape(x, dims)` or `x.view(dims)` |
| `TransposeAllDims(x)` | Reverses the order of all dimensions (similar to `.T` in 2D). | `x.T` (for 2D) |
| `Transpose(x, axes...)` | Permutes the dimensions according to the given axes. | `torch.permute(x, axes)` |
| `ExpandDims(x, axes...)` | Inserts dimensions of size 1 at the specified axes. | `torch.unsqueeze(x, axis)` |
| `Squeeze(x, axes...)` | Removes dimensions of size 1 at the specified axes. | `torch.squeeze(x, axis)` |
| `Slice(x, start, end)` | Slices a tensor along dimensions. | `x[start:end]` |
| `Concat(axis, nodes...)` | Concatenates a list of nodes along the specified axis. | `torch.cat(nodes, dim=axis)` |
| `Stack(axis, nodes...)` | Stacks a list of nodes along a new axis. | `torch.stack(nodes, dim=axis)` |
| `BroadcastToShape(x, shape)` | Broadcasts a node to a target shape. | `x.expand(shape)` |
| `BroadcastToDims(x, dims...)` | Broadcasts a node to target dimensions. | `x.expand(dims...)` |
| `Pad(x, padding...)` | Pads a tensor. | `torch.nn.functional.pad(x, ...)` |
| **Reduction** | | |
| `ReduceAllSum(x)` | Computes the sum of all elements. | `torch.sum(x)` |
| `ReduceSum(x, axes...)` | Computes the sum along the specified axes. | `torch.sum(x, dim=axes)` |
| `ReduceAllMean(x)` | Computes the mean of all elements. | `torch.mean(x)` |
| `ReduceMean(x, axes...)` | Computes the mean along the specified axes. | `torch.mean(x, dim=axes)` |
| `ReduceAllMax(x)` | Computes the maximum of all elements. | `torch.max(x)` |
| `ReduceMax(x, axes...)` | Computes the maximum along the specified axes. | `torch.amax(x, dim=axes)` |
| `ReduceAllMin(x)` | Computes the minimum of all elements. | `torch.min(x)` |
| `ReduceMin(x, axes...)` | Computes the minimum along the specified axes. | `torch.amin(x, dim=axes)` |
| `ArgMax(x, axis)` | Returns the index of the maximum value along an axis. | `torch.argmax(x, dim=axis)` |
| **Linear Algebra** | | |
| `Dot(a, b)` | Matrix multiplication (or dot product for 1D). | `torch.matmul(a, b)` or `a @ b` |
| `DotGeneral(a, axesA, b, axesB)` | Generalized dot product / tensor contraction. | `torch.tensordot(a, b, dims)` |
| `Einsum(equation, operands...)` | Einstein summation convention. | `torch.einsum(equation, operands)` |
| **Constants & Generation** | | |
| `Scalar(g, dtype, value)` | Creates a scalar node in the graph `g`. | `torch.tensor(value, dtype=dtype)` |
| `Zeros(g, shape)` | Creates a tensor of zeros. | `torch.zeros(shape)` |
| `Ones(g, shape)` | Creates a tensor of ones. | `torch.ones(shape)` |
| `ZerosLike(x)` | Creates a tensor of zeros with the same shape and dtype as `x`. | `torch.zeros_like(x)` |
| `OnesLike(x)` | Creates a tensor of ones with the same shape and dtype as `x`. | `torch.ones_like(x)` |
| `Iota(g, shape, axis)` | Creates a sequence along `axis`. | `torch.arange(...)` or `torch.linspace(...)` |
| **Control Flow & Selection** | | |
| `Where(condition, x, y)` | Selects elements from `x` or `y` based on `condition`. | `torch.where(condition, x, y)` |
| `Equal(a, b)` | Element-wise equality check. | `a == b` |
| `NotEqual(a, b)` | Element-wise inequality check. | `a != b` |
| `GreaterThan(a, b)` | Element-wise greater than check. | `a > b` |
| `LessThan(a, b)` | Element-wise less than check. | `a < b` |
| `LogicalAnd(a, b)` | Element-wise logical AND. | `torch.logical_and(a, b)` |
| `LogicalOr(a, b)` | Element-wise logical OR. | `torch.logical_or(a, b)` |
| `LogicalNot(x)` | Element-wise logical NOT. | `torch.logical_not(x)` |
| `StopGradient(x)` | Prevents gradients from flowing through `x` during backpropagation. | `x.detach()` |
