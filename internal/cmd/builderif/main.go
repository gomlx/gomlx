package builderif

import (
	"os"

	"github.com/gomlx/gomlx/internal/must"
)

const outputFile = "gen_builder_interfaces.go"

func main() {
	f := must.M(os.Create(outputFile))
	defer f.Close()

}
