// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package model

import (
	"fmt"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	. "github.com/gomlx/gomlx/support/exceptions"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

const (
	// RNGStateVariablePrefix is the prefix for Store internal variables holding per-algorithm
	// Random Number Generator (RNG) states (e.g., "#rngState_philox", "#rngState_three_fry",
	// or "#rngState_default_<backend>").
	RNGStateVariablePrefix = "#rngState_"

	// RNGStateVariableName is the legacy single-RNG-state variable name.
	//
	// Deprecated: Store now maintains per-algorithm state variables under [RNGStateVariablePrefix];
	// use [Store.RNGStateVariable] to retrieve the active RNG state variable for a graph.
	RNGStateVariableName = "#rngState"
)

var (
	// ParamInitialSeed is the key for the hyperparameter to use for initial seed (int64). The default is 0,
	// which makes it non-deterministic. Set it to a value different from 0 for a deterministic (as long
	// as the model doesn't change) initialization.
	ParamInitialSeed = "initializer_seed"

	// ParamRNGAlgorithm is the hyperparameter key selecting the [compute.RNGAlgorithm] used by
	// [Store] / [Scope] random helpers (defaults to [compute.RNGDefault]).
	// Accepts either a [compute.RNGAlgorithm] value or its snake-case string name (e.g. "default",
	// "philox", "three_fry", "split_mix64", "pcg", "cha_cha8").
	ParamRNGAlgorithm = "rng_algorithm"
)

// RNGAlgorithm returns the [compute.RNGAlgorithm] configured on this Store via
// [SetRNGAlgorithm] or [ParamRNGAlgorithm] (defaulting to [compute.RNGDefault]).
func (s *Store) RNGAlgorithm() compute.RNGAlgorithm {
	val, found := s.params.Get("/", ParamRNGAlgorithm)
	if !found || val == nil {
		return compute.RNGDefault
	}
	switch v := val.(type) {
	case compute.RNGAlgorithm:
		return v
	case int:
		return compute.RNGAlgorithm(v)
	case string:
		algo, err := compute.RNGAlgorithmString(v)
		if err != nil {
			klog.Errorf("Invalid %s=%q, using RNGDefault: %+v", ParamRNGAlgorithm, v, err)
			return compute.RNGDefault
		}
		return algo
	default:
		klog.Errorf("Unexpected type %T for %s, using RNGDefault", val, ParamRNGAlgorithm)
		return compute.RNGDefault
	}
}

// SetRNGAlgorithm configures the [compute.RNGAlgorithm] used by this Store's random methods.
func (s *Store) SetRNGAlgorithm(algo compute.RNGAlgorithm) {
	s.SetParam(ParamRNGAlgorithm, algo)
}

// WithRNGAlgorithm configures the [compute.RNGAlgorithm] used by this Store's random methods
// and returns the Store for method chaining.
func (s *Store) WithRNGAlgorithm(algo compute.RNGAlgorithm) *Store {
	s.SetRNGAlgorithm(algo)
	return s
}

func resolveStoreRNGVarKey(backend compute.Backend, configured compute.RNGAlgorithm) (varName string, resolvedAlgo compute.RNGAlgorithm, stateShape shapes.Shape, err error) {
	caps := backend.Capabilities()
	resolvedAlgo = caps.ResolveRNGAlgorithm(configured)
	if resolvedAlgo != compute.RNGDefault {
		stateShape, err = caps.RNGStateShape(resolvedAlgo)
		if err != nil {
			if compute.IsNotImplemented(err) && resolvedAlgo.IsCounterBased() {
				stateShape = resolvedAlgo.StateShape()
				err = nil
			} else {
				return "", compute.RNGDefault, shapes.Invalid(), err
			}
		}
		return RNGStateVariablePrefix + resolvedAlgo.String(), resolvedAlgo, stateShape, nil
	}
	stateShape, err = caps.RNGStateShape(compute.RNGDefault)
	if err != nil {
		return "", compute.RNGDefault, shapes.Invalid(), err
	}
	return fmt.Sprintf("%sdefault_%s", RNGStateVariablePrefix, backend.Name()), compute.RNGDefault, stateShape, nil
}

func (s *Store) getRNGStateVarForSpec(varName string, stateShape shapes.Shape, spec rngVarSpec) *Variable {
	fullPath := JoinPath("/", varName)
	rngStateVar := s.GetVariable(fullPath)
	if rngStateVar != nil {
		if s.rngStateVars == nil {
			s.rngStateVars = make(map[string]rngVarSpec)
		}
		s.rngStateVars[fullPath] = spec
		return rngStateVar
	}
	rngStateVar = &Variable{
		name:         varName,
		fullPath:     fullPath,
		shape:        stateShape,
		Trainable:    false,
		shardingSpec: s.defaultShardingSpec,
	}
	if s.loader != nil && !s.hasRNGSeed && !s.rngRandomReset {
		if loaded, ok := s.loader.LoadVariable(s, fullPath); ok {
			if err := rngStateVar.SetValue(loaded); err != nil {
				klog.Errorf("Failed to load RNG state variable %q: %+v", fullPath, err)
			}
		}
	}
	s.setVariable(rngStateVar)
	if s.rngStateVars == nil {
		s.rngStateVars = make(map[string]rngVarSpec)
	}
	s.rngStateVars[fullPath] = spec
	return rngStateVar
}

func (s *Store) initRNGStateTensor(spec rngVarSpec) (*tensors.Tensor, error) {
	if s.hasRNGSeed {
		return graph.RNGStateFromSeedForBackend(spec.backend, spec.algo, s.rngSeed)
	}
	if !s.rngRandomReset {
		if seedAny, found := s.params.Get("/", ParamInitialSeed); found {
			seed, ok := seedAny.(int64)
			if !ok {
				klog.Errorf("Seed in %q not an int64, using 0 instead", ParamInitialSeed)
			}
			return graph.RNGStateFromSeedForBackend(spec.backend, spec.algo, seed)
		}
	}
	return graph.RNGStateForBackend(spec.backend, spec.algo)
}

// RNGStateVariable returns the Store's internal RNG state [Variable] corresponding to the
// graph's backend and configured [RNGAlgorithm], initializing its value if needed.
func (s *Store) RNGStateVariable(g *graph.Graph) *Variable {
	g.AssertValid()
	backend := g.Backend()
	configured := s.RNGAlgorithm()
	varName, resolvedAlgo, stateShape, err := resolveStoreRNGVarKey(backend, configured)
	if err != nil {
		panic(errors.Wrapf(err, "Store.RNGStateVariable: resolving RNG algorithm %s for backend %q", configured, backend.Name()))
	}
	spec := rngVarSpec{algo: resolvedAlgo, backend: backend}
	v := s.getRNGStateVarForSpec(varName, stateShape, spec)
	if !v.HasValue() {
		stateTensor, err := s.initRNGStateTensor(spec)
		if err != nil {
			panic(errors.Wrapf(err, "Store.RNGStateVariable: initializing %q", v.Path()))
		}
		if err := v.SetValue(stateTensor); err != nil {
			panic(errors.Wrapf(err, "Store.RNGStateVariable: setting value on %q", v.Path()))
		}
	} else if val := v.MustValue(); val.IsOnAnyDevice() {
		if valBackend, err := val.Backend(); err == nil && valBackend != backend {
			if err := val.ToLocal(); err != nil {
				panic(errors.Wrapf(err, "Store.RNGStateVariable: moving %q to local storage for backend %q", v.Path(), backend.Name()))
			}
		}
	}
	return v
}

func (s *Store) rngStateNode(g *graph.Graph) (*Variable, *Node, compute.RNGAlgorithm) {
	v := s.RNGStateVariable(g)
	spec := s.rngStateVars[v.Path()]
	node := v.NodeValue(g)
	return v, node, spec.algo
}

// ResetRNGState resets the Store's random number generator (RNG) state(s) to a cryptographically
// secure random seed (if the OS supports it), or to [ParamInitialSeed] if set.
//
// Any already-materialized per-algorithm RNG state variables are re-initialized immediately, and
// any future algorithms/backends accessed on this Store will initialize lazily from the reset.
func (s *Store) ResetRNGState() error {
	seedAny, found := s.params.Get("/", ParamInitialSeed)
	if found {
		seed, ok := seedAny.(int64)
		if !ok {
			klog.Errorf("Seed in %q not an int64, using 0 instead", ParamInitialSeed)
		}
		return s.SetRNGStateFromSeed(seed)
	}

	s.hasRNGSeed = false
	s.rngRandomReset = true
	for fullPath, spec := range s.rngStateVars {
		v := s.GetVariable(fullPath)
		if v == nil {
			continue
		}
		randomState, err := graph.RNGStateForBackend(spec.backend, spec.algo)
		if err != nil {
			return err
		}
		if err := v.SetValue(randomState); err != nil {
			return err
		}
	}
	return nil
}

// SetRNGStateFromSeed initializes the Store's random number generator (RNG) state(s) with
// a static seed.
//
// Safe to call before any backend or computation graph exists: any per-algorithm RNG state
// variables used later will initialize lazily from seed, and any already-materialized variables
// are re-seeded immediately.
func (s *Store) SetRNGStateFromSeed(seed int64) error {
	s.rngSeed = seed
	s.hasRNGSeed = true
	s.rngRandomReset = false
	for fullPath, spec := range s.rngStateVars {
		v := s.GetVariable(fullPath)
		if v == nil {
			continue
		}
		initialState, err := graph.RNGStateFromSeedForBackend(spec.backend, spec.algo, seed)
		if err != nil {
			return err
		}
		if err := v.SetValue(initialState); err != nil {
			return err
		}
	}
	return nil
}

// RandomNormal generates random numbers from a normal distribution, with mean 0.0
// and standard deviation 1.0.
func (s *Store) RandomNormal(g *graph.Graph, shape shapes.Shape) (values *Node) {
	rngStateVar, rngState, algo := s.rngStateNode(g)
	rngState, values = graph.RandomNormal(algo, rngState, shape)
	rngStateVar.SetNodeValue(rngState)
	return
}

// RandomUniform generates random uniform values from 0.0 to 1.0.
func (s *Store) RandomUniform(g *graph.Graph, shape shapes.Shape) (values *Node) {
	rngStateVar, rngState, algo := s.rngStateNode(g)
	rngState, values = graph.RandomUniform(algo, rngState, shape)
	rngStateVar.SetNodeValue(rngState)
	return
}

// RandomNormal proxy for Store.RandomNormal.
func (s *Scope) RandomNormal(g *graph.Graph, shape shapes.Shape) *Node {
	return s.store.RandomNormal(g, shape)
}

// RandomUniform proxy for Store.RandomUniform.
func (s *Scope) RandomUniform(g *graph.Graph, shape shapes.Shape) *Node {
	return s.store.RandomUniform(g, shape)
}

// RandomBernoulli generates 0s and 1s in the given shape (or True/False if shape dtype is Bool),
// with probability of 1s being prob.
func (s *Store) RandomBernoulli(prob *Node, shape shapes.Shape) *Node {
	g := prob.Graph()
	maskShape := shape.Clone()
	maskShape.DType = prob.DType()
	mask := s.RandomUniform(g, maskShape)
	mask = graph.LessThan(mask, prob)
	return graph.ConvertDType(mask, shape.DType)
}

// RandomBernoulli generates 0s and 1s in the given shape (or True/False if shape dtype is Bool),
// with probability of 1s being prob.
//
// It is a proxy to Store.RandomBernoulli.
func (s *Scope) RandomBernoulli(prob *Node, shape shapes.Shape) *Node {
	return s.store.RandomBernoulli(prob, shape)
}

// RandomIntN generates random numbers uniformly from 0 to N-1.
func (s *Store) RandomIntN(g *graph.Graph, N any, shape shapes.Shape) (values *Node) {
	rngStateVar, rngState, algo := s.rngStateNode(g)
	switch n := N.(type) {
	case *Node:
		rngState, values = graph.RandomIntN(algo, rngState, n, shape)
	case uint8:
		rngState, values = graph.RandomIntN(algo, rngState, n, shape)
	case uint16:
		rngState, values = graph.RandomIntN(algo, rngState, n, shape)
	case uint32:
		rngState, values = graph.RandomIntN(algo, rngState, n, shape)
	case uint64:
		rngState, values = graph.RandomIntN(algo, rngState, n, shape)
	case int8:
		rngState, values = graph.RandomIntN(algo, rngState, n, shape)
	case int16:
		rngState, values = graph.RandomIntN(algo, rngState, n, shape)
	case int32:
		rngState, values = graph.RandomIntN(algo, rngState, n, shape)
	case int64:
		rngState, values = graph.RandomIntN(algo, rngState, n, shape)
	default:
		Panicf("Store.RandomIntN: unsupported N type %T", N)
	}
	rngStateVar.SetNodeValue(rngState)
	return
}

// RandomIntN generates random numbers uniformly from 0 to N-1.
//
// It is a proxy to Store.RandomIntN.
func (s *Scope) RandomIntN(g *graph.Graph, N any, shape shapes.Shape) (values *Node) {
	return s.store.RandomIntN(g, N, shape)
}
