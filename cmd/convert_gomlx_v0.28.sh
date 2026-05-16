# convert_gomlx_v0.28.sh helps converting the imports from gomlx v0.27 to v0.28
# 
# From an API perspective largely this was a fix on naming and reorganization of the packages
# into separate its separate repos (the backend is separate and with minimal dependencies).
#
# This script replaces these imports to their new locations.
# Most packages keep the same naming conventions, but some changed names (backends -> compute, context -> model)
# and will require manual fixing -- this script helps with that by also running `gofmt -r` to rename the 
# package references.

# Backends
gofmt -w -r '"github.com/gomlx/gomlx/backends" -> "github.com/gomlx/compute"' .
gofmt -w -r 'backends.X -> compute.X' .
gofmt -w -r '"github.com/gomlx/gomlx/backends/simplego" -> "github.com/gomlx/compute/gobackend"' .
gofmt -w -r '"github.com/gomlx/gomlx/backends/xla" -> "github.com/gomlx/go-xla/compute/xla"' .
gofmt -w -r '"github.com/gomlx/gomlx/pkg/core/dtypes" -> "github.com/gomlx/compute/dtypes"' .
gofmt -w -r '"github.com/gomlx/gomlx/pkg/core/shapes" -> "github.com/gomlx/compute/shapes"' .
gofmt -w -r '"github.com/gomlx/gomlx/pkg/core/distributed" -> "github.com/gomlx/compute/distributed"' .

# Removal of `pkg/` prefix from import paths of exported packages.

# - core/
gofmt -w -r '"github.com/gomlx/gomlx/pkg/core/tensors/images" -> "github.com/gomlx/gomlx/core/tensors/images"' .
gofmt -w -r '"github.com/gomlx/gomlx/pkg/core/graph/bucketing" -> "github.com/gomlx/gomlx/core/graph/bucketing"' .

# - support/

# - ml/
gofmt -w -r '"github.com/gomlx/gomlx/pkg/ml/nn" -> "github.com/gomlx/gomlx/ml/nn"' .
gofmt -w -r '"github.com/gomlx/gomlx/pkg/ml/context" -> "github.com/gomlx/gomlx/ml/model"' .
gofmt -w -r 'context.X -> model.X' .
gofmt -w -r '"github.com/gomlx/gomlx/pkg/ml/context/checkpoints" -> "github.com/gomlx/gomlx/ml/model/checkpoints"' .
gofmt -w -r '"github.com/gomlx/gomlx/pkg/ml/context/ctxtest" -> "github.com/gomlx/gomlx/ml/model/ctxtest"' .
gofmt -w -r '"github.com/gomlx/gomlx/pkg/ml/context/initializers" -> "github.com/gomlx/gomlx/ml/model/initializers"' .