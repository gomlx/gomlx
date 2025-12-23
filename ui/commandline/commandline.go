/*
 *	Copyright 2023 Jan Pfeifer
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

// Package commandline contains convenience UI training tools for the command line.
package commandline

import (
	"fmt"

	"github.com/gomlx/gomlx/pkg/ml/train"
)

// ReportEval reports on the command line the results of evaluating the datasets using trainer.Eval.
func ReportEval(trainer *train.Trainer, datasets ...train.Dataset) error {
	for _, ds := range datasets {
		fmt.Printf("Results on %s:\n", ds.Name())
		metricsValues, err := trainer.Eval(ds)
		if err != nil {
			return err
		}
		for metricIdx, metric := range trainer.EvalMetrics() {
			value := metricsValues[metricIdx]
			fmt.Printf("\t%s (%s): %s\n", metric.Name(), metric.ShortName(), metric.PrettyPrint(value))
		}
		ds.Reset()
	}
	return nil
}
