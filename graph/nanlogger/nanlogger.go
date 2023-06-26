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

// Package nanlogger collects `graph.Node` objects to monitor for
// `NaN` ("not-a-number") or `Inf` (infinity) values.
//
// It does that by implementing `graph.LoggerFn` and hooking to the `graph.Exec` that
// executes the graph. If at the end of a graph.Exec call, if a `NaN` value is found, the
// first node where it appears (often `NaN` values spread through the graph) is reported back.
//
// The report includes a stack trace and an optional user set scoped context.
//
// Example:
//
//	func train() {
//		…
//		nanLogger := nanlogger.New()
//		trainer := train.NewTrainer(…)
//		trainer.OnExecCreation(func(exec *context.Exec, _ train.GraphType) {
//			nanLogger.Attach(exec)
//		})
//		…
//	}
//
//	func ModelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {) {
//		…
//		for ii := 0; ii < numBlocks; ii++ {
//			name := fmt.Sprintf("Residual-%d", ii+1)
//			nanLogger.PushScope(name)
//			x = ResidualBlock(ctx.In(name), x, lastNumChannels)
//			nanLogger.Trace(x)
//			nanLogger.PopScope()
//		}
//		…
//	}
package nanlogger

import (
	"fmt"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"math"
	"strings"
)

const UniqueMessageId = "#nanlogger"

// NanLogger uses the logger infrastructure to monitor for NaN (and Inf) values in your graph.
// You manually select the nodes you want to monitor, and it saves the stack where it was called
// along with user provided scope information.
// If during the execution any NaN appears, it panics with the stack trace where the monitor was
// set, along with the scope. Alternatively, instead of panicking, one can set a handler to be
// called if/when a NaN is observed.
//
// See example in package documentation.
type NanLogger struct {
	prevLoggerFn graph.LoggerFn
	handler      HandlerFn

	traces map[graph.GraphId]map[graph.NodeId]*Trace

	currentScope []string
}

// Trace information of a node that is set to monitor.
// This is what printed out when a `NaN` is found, or passed to a handler function, if one is set.
type Trace struct {
	// StackTrace of where the monitored node was created, stored as an error that can be printed.
	StackTrace error

	// Scope saved when monitor node was created.
	Scope []string
}

// ExecWithLogger represents any of the executors in GoMLX (or future): `graph.Exec` and `context.Exec`.
// What is required is that it supports setting the logger and reading the current logger.
type ExecWithLogger interface {
	SetNodeLogger(loggerFn graph.LoggerFn)
	GetNodeLogger() graph.LoggerFn
}

// New creates a NanLogger that can be used to debug where NaN happen in graphs.
// See NanLogger for details.
func New() *NanLogger {
	return &NanLogger{
		handler: DefaultHandler,
		traces:  make(map[graph.GraphId]map[graph.NodeId]*Trace),
	}
}

// Attach will set the NanLogger as the default logger in exec. NanLogger acts as a pass-through logger, anything
// that is not marked as nanlogger.UniqueMessageId is passed through to whatever was the previous logger configured
// in exec.
//
// A nil NanLogger is valid, and it will simply be a no-op.
func (l *NanLogger) Attach(exec ExecWithLogger) {
	if l == nil {
		return
	}
	l.prevLoggerFn = exec.GetNodeLogger()
	exec.SetNodeLogger(l.loggerFn)
}

// Trace the given node.
// This means the node is monitored and whenever a NaN is observed, the trace is printed and the program exit.
// Alternatively, a handler is called, see SetHandler.
//
// A user-provided scope can be given. If none is given, then it uses the current NanLogger scope.
// These are two different methods of providing scope, both optional -- it's also fine not to provide any.
//
// A nil NanLogger is valid, and it will simply be a no-op.
func (l *NanLogger) Trace(node *graph.Node, scope ...string) {
	if l == nil {
		return
	}
	if !node.Ok() {
		return
	}
	g := node.Graph()
	if !g.Ok() {
		// Graph is in error, nothing to do.
		return
	}

	// We actually only need to check the sum of all values: any NaN/Inf will trigger the sum to
	// be NaN/Inf.
	node = graph.ReduceAllSum(node)
	node.SetLogged(UniqueMessageId)

	// Create trace:
	trace := &Trace{
		StackTrace: errors.Errorf("Stack-trace"),
	}
	if len(scope) == 0 {
		trace.Scope = slices.Copy(l.currentScope)
	} else {
		trace.Scope = slices.Copy(scope)
	}

	gId := g.GraphId()
	graphMap, found := l.traces[gId]
	if !found {
		graphMap = make(map[graph.NodeId]*Trace)
		l.traces[gId] = graphMap
	}
	graphMap[node.Id()] = trace
}

// PushScope to current scope stack.
// These values are added by default to any new Trace.
//
// A nil NanLogger is valid, and it will simply be a no-op.
func (l *NanLogger) PushScope(scope string) {
	if l == nil {
		return
	}
	l.currentScope = append(l.currentScope, scope)
}

// PopScope removes the last entry in the current scope stack.
// These values are added by default to any new Trace.
//
// A nil NanLogger is valid, and it will simply be a no-op.
func (l *NanLogger) PopScope() {
	if l == nil {
		return
	}
	if len(l.currentScope) == 0 {
		klog.Warningf("NanLogger.PopScope() called on an already empty scope stack!?")
		return
	}
	_, l.currentScope = slices.Pop(l.currentScope)
}

// loggerFn implements graph.LoggerFn, it's the hook that listens to nodes for which we want to
// monitor for NaNs.
func (l *NanLogger) loggerFn(g *graph.Graph, messages []string, values []tensor.Tensor, nodes []graph.NodeId) {
	// Filtered logged values/messages: the ones not handled by NanLogger:
	filteredMessages := make([]string, 0, len(messages))
	filteredValues := make([]tensor.Tensor, 0, len(values))
	filteredNodes := make([]graph.NodeId, 0, len(nodes))

	firstNan := graph.InvalidNodeId
	var firstNanValue float64
	for ii, msg := range messages {
		if msg != UniqueMessageId {
			// Not managed by NanLogger.
			filteredMessages = append(filteredMessages, msg)
			filteredValues = append(filteredValues, values[ii])
			filteredNodes = append(filteredNodes, nodes[ii])
			continue
		}
		//fmt.Printf("Trace for %v\n", values[ii].Value())
		v := shapes.ConvertTo[float64](values[ii].Value())
		if math.IsNaN(v) || math.IsInf(v, 0) {
			nodeId := nodes[ii]
			if firstNan == graph.InvalidNodeId || nodeId < firstNan {
				firstNan = nodeId
				firstNanValue = v
			}
		}
	}

	if firstNan != graph.InvalidNodeId {
		// Report first NaN.
		gId := g.GraphId()
		graphMap, found := l.traces[gId]
		if found {
			var trace *Trace
			trace, found = graphMap[firstNan]
			if found {
				l.handler(firstNanValue, trace)
			}
		}
		if !found {
			klog.Warningf("NanLogger received trace for node that was not marked as traced: did you attach the wrong NanLogger to the executor?")
		}
	}

	if g.Ok() && l.prevLoggerFn != nil && len(filteredMessages) > 0 {
		// Call previous logger on remaining messages.
		l.prevLoggerFn(g, filteredMessages, filteredValues, filteredNodes)
	}
	return
}

// HandlerFn is the type of function to handle NaN traces.
type HandlerFn func(nanType float64, info *Trace)

// SetHandler sets the function called when a `NaN` is observed. The default is
// DefaultHandler which prints out all information on the node and exits.
func (l *NanLogger) SetHandler(handler HandlerFn) {
	if l == nil {
		return
	}
	l.handler = handler
}

// DefaultHandler when a `NaN` or `Inf` is observed: it prints all out all the information
// and exits.
func DefaultHandler(nanType float64, info *Trace) {
	var scopeTxt string
	if len(info.Scope) > 0 {
		scopeTxt = fmt.Sprintf("Scope:\n\t%s\n", strings.Join(info.Scope, "\n\t"))
	}
	klog.Exitf("NaNLogger observed a %f during execution of graph:\n%sStack-trace of node:\n%+v\n",
		nanType, scopeTxt, info.StackTrace)
}
