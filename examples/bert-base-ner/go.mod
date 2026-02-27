module github.com/gomlx/gomlx/examples/bert-base-ner

go 1.24.3

require (
	github.com/gomlx/go-huggingface v0.3.1
	github.com/gomlx/gomlx v0.26.1-0.20260131161033-fde190ebb2c3
	github.com/gomlx/onnx-gomlx v0.3.5-0.20260130173634-2497f2c7652f
	github.com/janpfeifer/must v0.2.0
	k8s.io/klog/v2 v2.130.1
)

require (
	github.com/dustin/go-humanize v1.0.1 // indirect
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/gofrs/flock v0.13.0 // indirect
	github.com/gomlx/exceptions v0.0.3 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/pkg/errors v0.9.1 // indirect
	github.com/x448/float16 v0.8.4 // indirect
	golang.org/x/exp v0.0.0-20260112195511-716be5621a96 // indirect
	golang.org/x/sys v0.40.0 // indirect
	google.golang.org/protobuf v1.36.11 // indirect
)

replace github.com/gomlx/gomlx => ../..
