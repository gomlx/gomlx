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
// executes the graph -- it can also `AttachToTrainer`, so it automatically attaches to every
// graph the trainer creates.
//
// If at the end of a graph.Exec call, if a `NaN` value is found on the traced computation nodes,
// the first node where it appears (often `NaN` values spread through the graph) is reported back.
//
// The report includes a stack trace and an optional user set scoped context.
//
// Example: create a `NanLogger` and attaches it to the trainer, to it gets attached to every
// graph created (if more than one is created by the trainer).
//
//	func train() {
//		…
//		nanLogger := nanlogger.New()
//		ctx.SetParam(optimizers.ParamNanLogger, nanLogger)  // Gradients with NaNs get reported.
//		trainer := train.NewTrainer(…)
//		nanLogger.AttachToTrainer(trainer)
//		…
//	}
//
//	func ModelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
//		…
//		for ii := range numBlocks {
//			x = ResidualBlock(ctx.In(name), x, lastNumChannels)
//			nanLogger.TraceFirstNaN(x, fmt.Sprintf("Residual-%d", ii+1))
//		}
//		…
//	}
package nanlogger

import (
	"fmt"
	"slices"
	"strings"
	"sync/atomic"

	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

const (
	TracedMessageId = "#nanlogger(%d)"
)

// uniqueCounter is a thread-safe counter used to generate unique IDs for traced nodes
var uniqueCounter atomic.Int64

// NanLogger uses the logger infrastructure to monitor for NaN (and Inf) values in your graph.
// You manually select the nodes you want to monitor, and it saves the stack where it was called
// along with user-provided scope information.
//
// If during the execution any NaN appears, it calls your handler with the stack trace where the monitor was
// set, along with the scope.
//
// Alternatively, instead of panicking, one can set a custom handler to be called if/when a NaN is observed.
//
// See an example in the package documentation.
type NanLogger struct {
	uniqueMessageID        string
	prevLoggerFn           graph.LoggerFn
	handler, reportHandler HandlerFn

	traces map[graph.GraphId]map[graph.NodeId]*Trace

	currentScope []string
}

// stackTracer is implemented by the github.com/pkg/errors package.
type stackTracer interface {
	StackTrace() errors.StackTrace
}

// Trace information of a node that is set to monitor.
// This is what is printed out when a `NaN` is found or passed to a handler function if one is set.
type Trace struct {
	// StackTrace of where the monitored node was created, stored as an error that can be printed.
	StackTrace errors.StackTrace

	// Scope saved when the monitor node was created.
	Scope []string

	// FirstOnly traces only the first occurrence of a NaN: that means if more than one traced node
	// is or has a NaN/Inf value, only the handler is called only for the first one.
	FirstOnly bool
}

// ExecWithLogger represents any of the executors in GoMLX (or future): `graph.Exec` and `context.Exec`.
// What is required is that it supports setting the logger and reading the current logger.
type ExecWithLogger interface {
	SetNodeLogger(loggerFn graph.LoggerFn)
	GetNodeLogger() graph.LoggerFn
}

// New creates a NanLogger that can be used to debug where NaN happen in graphs.
//
// You manually select the nodes you want to monitor, and it saves the stack where it was called
// along with user-provided scope information.
//
// If during the execution any NaN appears, it calls your handler with the stack trace where the monitor was
// set, along with the scope.
//
// Alternatively, instead of panicking, one can set a custom handler to be called if/when a NaN is observed.
//
// See an example in the package documentation.
func New() *NanLogger {
	return &NanLogger{
		uniqueMessageID: fmt.Sprintf(TracedMessageId, uniqueCounter.Add(1)),
		handler:         DefaultBreakHandler,
		traces:          make(map[graph.GraphId]map[graph.NodeId]*Trace),
	}
}

// AttachToExec will set the NanLogger as the default logger in exec.
// NanLogger acts as a pass-through logger, anything that is not marked as nanlogger.TracedMessageId is passed
// through to whatever the previous logger configured in exec was.
//
// A nil NanLogger is valid, and it will simply be a no-op.
func (l *NanLogger) AttachToExec(exec ExecWithLogger) {
	if l == nil {
		return
	}
	l.prevLoggerFn = exec.GetNodeLogger()
	exec.SetNodeLogger(l.loggerFn)
}

// AttachToTrainer makes sure that the logger is attached to every graph created by the trainer.
//
// A nil NanLogger is valid, and it will simply be a no-op.
func (l *NanLogger) AttachToTrainer(trainer *train.Trainer) {
	if l == nil {
		return
	}
	trainer.OnExecCreation(func(exec *context.Exec, _ train.GraphType) {
		l.AttachToExec(exec)
	})
}

// TraceFirstNaN sets a trace on the given node, and if it ever becomes NaN, the trace of the first node
// the graph (closer to the input in graph ordering) is logged (if the DefaultHandler is used).
//
// Remember to attach the NanLogger once to the train.Trainer or to the graph.Exec/context.Exec you are running.
//
// A nil NanLogger is valid, and it will simply be a no-op.
func (l *NanLogger) TraceFirstNaN(node *graph.Node, scope ...string) {
	l.traceImpl(true, node, scope...)
}

// Trace sets a trace on the given node, and if it ever becomes NaN, the scope is printed -- but the execution
// continues normally.
//
// Args:
//
// Remember to attach the NanLogger once to the train.Trainer or to the graph.Exec/context.Exec you are running.
//
// A nil NanLogger is valid, and it will simply be a no-op.
func (l *NanLogger) Trace(node *graph.Node, scope ...string) {
	l.traceImpl(false, node, scope...)
}

func (l *NanLogger) traceImpl(firstOnly bool, node *graph.Node, scope ...string) {
	if l == nil {
		return
	}
	node.AssertValid()

	// Check whether any of the values are finite.
	var tracedNode *graph.Node
	if node.IsScalar() {
		tracedNode = graph.IsFinite(node)
	} else {
		tracedNode = graph.LogicalAll(graph.IsFinite(node))
	}
	tracedNode.SetLogged(l.uniqueMessageID)

	// Create trace, stripping this function from it:
	tracer := errors.Errorf("Stack-trace").(stackTracer)
	stackTrace := tracer.StackTrace()
	stackTrace = stackTrace[1:]
	trace := &Trace{
		StackTrace: stackTrace,
		FirstOnly:  firstOnly,
	}
	if len(scope) == 0 {
		trace.Scope = slices.Clone(l.currentScope)
	} else {
		trace.Scope = make([]string, 0, len(l.currentScope)+len(scope))
		trace.Scope = append(trace.Scope, l.currentScope...)
		trace.Scope = append(trace.Scope, scope...)
	}

	gId := tracedNode.Graph().GraphId()
	graphMap, found := l.traces[gId]
	if !found {
		graphMap = make(map[graph.NodeId]*Trace)
		l.traces[gId] = graphMap
	}
	graphMap[tracedNode.Id()] = trace
}

// PushScope to the current scope stack.
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
	_, l.currentScope = xslices.Pop(l.currentScope)
}

// loggerFn implements graph.LoggerFn, it's the hook that listens to nodes for which we want to
// monitor for NaNs.
func (l *NanLogger) loggerFn(g *graph.Graph, messages []string, values []*tensors.Tensor, nodes []graph.NodeId) {
	// Filtered logged values/messages: the ones not handled by NanLogger:
	firstNodeId := graph.InvalidNodeId
	var firstTrace *Trace
	var shiftIdx int

	tracesMap, found := l.traces[g.GraphId()]
	if !found {
		klog.Warningf("NanLogger (%q) received traced node for unknown Graph %d (%q)!?", l.uniqueMessageID, g.GraphId(), g.Name())
		if l.prevLoggerFn != nil {
			// Call previous logger on remaining messages.
			l.prevLoggerFn(g, messages, values, nodes)
		}
	}

	for idx, msg := range messages {
		if msg != l.uniqueMessageID {
			// Not managed by NanLogger.
			if shiftIdx == idx {
				// Nothing to do.
				continue
			}
			messages[shiftIdx] = messages[idx]
			values[shiftIdx] = values[idx]
			nodes[shiftIdx] = nodes[idx]
			shiftIdx++
			continue
		}
		isAllFinite := tensors.ToScalar[bool](values[idx])
		if !isAllFinite {
			nodeId := nodes[idx]
			trace, found := tracesMap[nodeId]
			if !found {
				klog.Warningf("NanLogger (%q) received traced node (id=%d, graph=%s) that it hasn't traced!?", l.uniqueMessageID, nodeId, g.Name())
				continue
			}
			if trace.FirstOnly {
				// Find the first traced node.
				if firstNodeId == graph.InvalidNodeId || nodeId < firstNodeId {
					firstNodeId = nodeId
					firstTrace = trace
				}
			} else {
				// Call handler on the traced node.
				l.handler(trace)
			}
		}
	}
	messages = messages[:shiftIdx]
	values = values[:shiftIdx]
	nodes = nodes[:shiftIdx]

	// Report the first NaN node, if only those that require only the first.
	if firstTrace != nil {
		l.handler(firstTrace)
	}

	// Report other values first, since they may help debug.
	if l.prevLoggerFn != nil && len(messages) > 0 {
		// Call previous logger on remaining messages.
		l.prevLoggerFn(g, messages, values, nodes)
	}
	return
}

// HandlerFn is the type of function to handle NaN traces.
type HandlerFn func(info *Trace)

// WithHandler sets the function called when a `NaN` is observed.
//
// The default is DefaultHandler, that is initialized with ReportScopeHandler.
// There are also ReportAllHandler and ReportAndPanicHandler handler predefined that can be used.
//
// If the handler is nil, set it to the default value.
func (l *NanLogger) WithHandler(handler HandlerFn) *NanLogger {
	if l == nil {
		handler = DefaultHandler
	}
	l.handler = handler
	return l
}

// DefaultBreakHandler is the default handler called when a `NaN` or `Inf` is observed in a break traced
// node.
//
// It prints all out all the information about the `NaN` trace.
func DefaultBreakHandler(info *Trace) {
	var scopeTxt string
	if len(info.Scope) > 0 {
		scopeTxt = fmt.Sprintf("Scope:\n\t%s\n", strings.Join(info.Scope, "\n\t"))
	}
	klog.Errorf("NaNLogger observed a NaN or Inf values during execution of graph:\n%sStack-trace of node:\n%+v\n",
		scopeTxt, info.StackTrace)
}

// DefaultHandler for a new NanLogger.
var DefaultHandler = ReportScopeHandler

// ReportScopeHandler reports the scope when a traced node with NaN/Inf values is found.
func ReportScopeHandler(info *Trace) {
	klog.Infof("NaN/Inf observed for scope %q", info.Scope)
}

// ReportAllHandler is a handler that reports the stack trace and scope of a traced node, and then panic.
func ReportAllHandler(info *Trace) {
	klog.Errorf("NaN/Inf observed for scope %q:\n%+v\n", info.Scope, info.StackTrace)
}

// ReportAndPanicHandler is a handler that reports the stack trace and scope of a traced node, and then panic.
func ReportAndPanicHandler(info *Trace) {
	klog.Errorf("NaN/Inf observed for scope %q:\n%+v\n", info.Scope, info.StackTrace)
	panic(errors.Errorf("NaN/Inf observer in scope %q", info.Scope))
}
