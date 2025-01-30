/*
 *	Copyright 2023 Rener Castro
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package mnist

import (
	"os"

	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/janpfeifer/must"
)

func CreateDefaultContext() *context.Context {
	ctx := context.New()
	return ctx
}

// TrainModel based on configuration and flags.
func TrainModel(ctx *context.Context, dataDir string, paramsSet []string) {
	dataDir = data.ReplaceTildeInDir(dataDir)
	if !data.FileExists(dataDir) {
		must.M(os.MkdirAll(dataDir, 0777))
	}

}
