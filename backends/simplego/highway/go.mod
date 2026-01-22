module github.com/gomlx/gomlx/backends/simplego/highway

go 1.26rc2

require (
	github.com/ajroetker/go-highway v0.0.0-dev11
	github.com/gomlx/gomlx v0.0.0
	github.com/pkg/errors v0.9.1
	github.com/stretchr/testify v1.11.1
	github.com/x448/float16 v0.8.4
)

require (
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/kr/text v0.2.0 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	golang.org/x/exp v0.0.0-20251023183803-a4bb9ffd2546 // indirect
	golang.org/x/sys v0.40.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
	k8s.io/klog/v2 v2.130.1 // indirect
)

replace github.com/gomlx/gomlx => ../../..
