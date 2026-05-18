# GoMLX `layers` Package Reference

This table maps common machine learning layers and functions from `ml/layers` and its sub-packages to their PyTorch equivalents.

| GoMLX Function/Object | Description | PyTorch Equivalent |
|---|---|---|
| **Feed-Forward / Linear Layers** (`pkg/ml/layers/fnn`) | | |
| `fnn.New(scope, x, hiddenSizes...)` | Creates a multi-layer perceptron (feed-forward neural network) with the given hidden sizes. Often includes optional batch normalization, dropout, etc. depending on the scope settings. | `torch.nn.Sequential(torch.nn.Linear(...), ...)` |
| `layers.Dense(scope, x, true, outputDim)` | A single densely connected layer (linear transformation). The boolean flag dictates whether a bias term is used. | `torch.nn.Linear(in_features, out_features, bias=True)` |
| **Activations** (`pkg/ml/layers/activations`) | | |
| `activations.Apply(type, x)` | Applies an activation function specified by an `activations.Type`. | `torch.nn.functional.<activation>(x)` |
| `activations.Relu(x)` | Applies Rectified Linear Unit activation. | `torch.nn.ReLU()` or `torch.nn.functional.relu(x)` |
| `activations.Sigmoid(x)` | Applies Sigmoid activation. | `torch.nn.Sigmoid()` or `torch.nn.functional.sigmoid(x)` |
| `activations.Tanh(x)` | Applies Hyperbolic Tangent activation. | `torch.nn.Tanh()` or `torch.nn.functional.tanh(x)` |
| `activations.Swish(x)` | Applies Swish activation (`x * sigmoid(x)`). | `torch.nn.SiLU()` or `torch.nn.functional.silu(x)` |
| `activations.Gelu(x)` | Applies Gaussian Error Linear Unit activation. | `torch.nn.GELU()` or `torch.nn.functional.gelu(x)` |
| `activations.Softmax(x, axis)` | Applies Softmax activation along the specified axis. | `torch.nn.Softmax(dim=axis)` |
| **Convolutional Layers** (`pkg/ml/layers`) | | |
| `layers.Convolution(scope, x)` | Builder for a convolutional layer (1D, 2D, or 3D inferred from input shape). Use chained methods (e.g., `Filters(n)`, `KernelSize(k)`) and call `Done()` to build. | `torch.nn.Conv1d`, `torch.nn.Conv2d`, `torch.nn.Conv3d` |
| `layers.Conv2D(scope, x, filters, kernelSize, strides, padding)` | Shorthand for a 2D convolution. | `torch.nn.Conv2d` |
| **Normalization Layers** (`pkg/ml/layers/batchnorm`, `pkg/ml/layers/layernorm`) | | |
| `batchnorm.New(scope, x, axis)` | Batch Normalization. Use chained methods and `Done()`. The `axis` specifies the feature dimension. | `torch.nn.BatchNorm1d`, `torch.nn.BatchNorm2d`, `torch.nn.BatchNorm3d` |
| `layernorm.New(scope, x, axes...)` | Layer Normalization applied over the given axes. Use chained methods and `Done()`. | `torch.nn.LayerNorm` |
| `rmsnorm.New(scope, x, axes...)` | Root Mean Square Normalization. | `torch.nn.RMSNorm` |
| **Regularization / Dropout** (`pkg/ml/layers/regularizers`) | | |
| `regularizers.Dropout(scope, x, rate)` | Applies dropout to the input with the given rate (probability of dropping an element). Behavior depends on the scope's phase (training vs inference). | `torch.nn.Dropout(p=rate)` |
| `regularizers.L2(scope, name, weight)` | Adds an L2 regularization penalty to a variable (weight decay). | `weight_decay` parameter in `torch.optim` |
| **Other Advanced Layers** (`pkg/ml/layers/kan`, `pkg/ml/layers/attention`) | | |
| `kan.New(scope, x, hiddenSizes...)` | Kolmogorov-Arnold Network block. Uses spline-based learned activation functions instead of fixed ones. | (No direct equivalent, custom implementation required) |
| `attention.MultiHeadAttention(scope, query, key, value)` | Standard Multi-Head Attention layer. Builder pattern, finish with `Done()`. | `torch.nn.MultiheadAttention` |
| **Losses** (`pkg/ml/train/losses`) | | |
| `losses.MeanSquaredError(labels, predictions)` | Computes the mean squared error loss. | `torch.nn.MSELoss()` |
| `losses.MeanAbsoluteError(labels, predictions)` | Computes the mean absolute error (L1) loss. | `torch.nn.L1Loss()` |
| `losses.BinaryCrossentropyLogits(labels, logits)` | Computes binary cross-entropy loss from logits. | `torch.nn.BCEWithLogitsLoss()` |
| `losses.SparseCategoricalCrossEntropyLogits(labels, logits)` | Computes categorical cross-entropy where labels are sparse integers. | `torch.nn.CrossEntropyLoss()` |
| `losses.CategoricalCrossEntropyLogits(labels, logits)` | Computes categorical cross-entropy where labels are one-hot encoded. | `torch.nn.CrossEntropyLoss()` |
| **Optimizers** (`pkg/ml/train/optimizers`) | | |
| `optimizers.StochasticGradientDescent(scope)` | Simple SGD optimizer. Configured via scope hyperparameters. | `torch.optim.SGD` |
| `optimizers.Adam(scope)` | Adam optimizer. Configured via scope hyperparameters. | `torch.optim.Adam` |
| `optimizers.AdamW(scope)` | Adam optimizer with weight decay. | `torch.optim.AdamW` |
