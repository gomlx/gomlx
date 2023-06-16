// Package nanlogger collects `graph.Node` objects to monitor for
// NaN or infinity values.
//
// It does that by implementing `graph.LoggerFn` and hooking to the `graph.Exec` that
// executes the graph. If at the end of a graph.Exec call, if a NaN value is found, the
// first node where it appears is reported back, with information about the node -- including
// the stack trace if it was saved with `Graph.SetTrace(true)`.
package nanlogger
