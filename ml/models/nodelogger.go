package models

import "github.com/gomlx/gomlx/graph"

// SetNodeLogger with the function to be called after each execution for the nodes marked for logging during execution.
// If set this to nil and nothing will be logged (the values are just ignored).
//
// If setting a logger, it is common practice to check if the logger message is for your logger, and if not,
// call the previous logger, which can be read with GetNodeLogger.
func (e *Exec) SetNodeLogger(loggerFn graph.LoggerFn) {
	e.exec.SetNodeLogger(loggerFn)
}

// GetNodeLogger returns the currently registered LoggerFn. See SetNodeLogger.
func (e *Exec) GetNodeLogger() graph.LoggerFn {
	return e.exec.GetNodeLogger()
}
