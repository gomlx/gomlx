// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

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
