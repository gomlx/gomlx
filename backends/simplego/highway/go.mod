module github.com/gomlx/gomlx/backends/simplego/highway

go 1.26rc2

require (
	github.com/ajroetker/go-highway v0.0.0-dev12
	github.com/gomlx/gomlx v0.0.0
	github.com/pkg/errors v0.9.1
	github.com/stretchr/testify v1.11.1
	github.com/x448/float16 v0.8.4
)

require (
	github.com/davecgh/go-spew v1.1.2-0.20180830191138-d8f796af33cc // indirect
	github.com/dustin/go-humanize v1.0.1 // indirect
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/kr/text v0.2.0 // indirect
	github.com/pmezard/go-difflib v1.0.1-0.20181226105442-5d4384ee4fb2 // indirect
	golang.org/x/exp v0.0.0-20260112195511-716be5621a96 // indirect
	golang.org/x/sys v0.40.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
	k8s.io/klog/v2 v2.130.1 // indirect
)

replace github.com/gomlx/gomlx => ../../..

replace github.com/ajroetker/go-highway => /Users/ajroetker/go/src/github.com/ajroetker/go-highway
