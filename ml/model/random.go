// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package model

import (
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"k8s.io/klog/v2"
)

const (
	// RNGStateVariableName is the name of a Store internal variable that holds the current
	// random number generator state.
	RNGStateVariableName = "#rngState"
)

var (
	// ParamInitialSeed is the key for the hyperparameter to use for initial seed (int64). The default is 0,
	// which makes it non-deterministic. Set it to a value different from 0 for a deterministic (as long
	// as the model doesn't change) initialization.
	ParamInitialSeed = "initializer_seed"
)

// getRNGStateVar panics if it fails to create the random state.
func (s *Store) getRNGStateVar() *Variable {
	fullPath := JoinPath("/", RNGStateVariableName)
	rngStateVar := s.GetVariable(fullPath)
	if rngStateVar != nil {
		return rngStateVar
	}
	rngStateVar = &Variable{
		name:         RNGStateVariableName,
		fullPath:     fullPath,
		shape:        graph.RNGStateShape,
		Trainable:    false,
		shardingSpec: s.defaultShardingSpec,
	}
	s.setVariable(rngStateVar)
	return rngStateVar
}

// mustGetRNGStateVarWithValue panics if it fails to create the random state.
func (s *Store) mustGetRNGStateVarWithValue() *Variable {
	v := s.getRNGStateVar()
	if v.HasValue() {
		return v
	}
	err := s.ResetRNGState()
	if err != nil {
		panic(err)
	}
	return v
}

// ResetRNGState resets the default store random number generator (RNG) to a cryptographically secure
// random seed, if the OS supports it.
//
// If ParamInitialSeed is set, it will be used instead of cryptographically secure random seed.
func (s *Store) ResetRNGState() error {
	v := s.getRNGStateVar()
	var randomState *tensors.Tensor
	seedAny, found := s.params.Get("/", ParamInitialSeed)
	if !found {
		var err error
		randomState, err = graph.RNGState()
		if err != nil {
			return err
		}
	} else {
		seed, ok := seedAny.(int64)
		if !ok {
			klog.Errorf("Seed in %q not an int64, using 0 instead", ParamInitialSeed)
		}
		var err error
		randomState, err = graph.RNGStateFromSeed(seed)
		if err != nil {
			return err
		}
	}
	err := v.SetValue(randomState)
	if err != nil {
		return err
	}
	return nil
}

// SetRNGStateFromSeed initializes the default store random number generator (RNG) state with a static seed.
func (s *Store) SetRNGStateFromSeed(seed int64) error {
	initialState, err := graph.RNGStateFromSeed(seed)
	if err != nil {
		return err
	}
	v := s.getRNGStateVar()
	return v.SetValue(initialState)
}

// RandomNormal generates random numbers from a normal distribution, with mean 0.0
// and standard deviation 1.0.
func (s *Store) RandomNormal(g *graph.Graph, shape shapes.Shape) (values *Node) {
	rngStateVar := s.mustGetRNGStateVarWithValue()
	rngState := rngStateVar.NodeValue(g)
	rngState, values = graph.RandomNormal(rngState, shape)
	rngStateVar.SetNodeValue(rngState)
	return
}

// RandomUniform generates random uniform values from 0.0 to 1.0.
func (s *Store) RandomUniform(g *graph.Graph, shape shapes.Shape) (values *Node) {
	rngStateVar := s.mustGetRNGStateVarWithValue()
	rngState := rngStateVar.NodeValue(g)
	rngState, values = graph.RandomUniform(rngState, shape)
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
	rngStateVar := s.mustGetRNGStateVarWithValue()
	rngState := rngStateVar.NodeValue(g)
	switch n := N.(type) {
	case *Node:
		rngState, values = graph.RandomIntN(rngState, n, shape)
	case uint8:
		rngState, values = graph.RandomIntN(rngState, n, shape)
	case uint16:
		rngState, values = graph.RandomIntN(rngState, n, shape)
	case uint32:
		rngState, values = graph.RandomIntN(rngState, n, shape)
	case uint64:
		rngState, values = graph.RandomIntN(rngState, n, shape)
	case int8:
		rngState, values = graph.RandomIntN(rngState, n, shape)
	case int16:
		rngState, values = graph.RandomIntN(rngState, n, shape)
	case int32:
		rngState, values = graph.RandomIntN(rngState, n, shape)
	case int64:
		rngState, values = graph.RandomIntN(rngState, n, shape)
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
