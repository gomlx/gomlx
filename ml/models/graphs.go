package models

import (
	"sync"
	"weak"

	"github.com/gomlx/gomlx/graph"
)

var (
	graphToExecMu sync.RWMutex
	graphToExec   = make(map[graph.GraphId]weak.Pointer[Exec])
)

func getExecByGraphId(id graph.GraphId) *Exec {
	graphToExecMu.RLock()
	defer graphToExecMu.RUnlock()
	if ptr, exists := graphToExec[id]; exists {
		// Notice ptr.Value() will return nil if the object has already been garbage collected.
		return ptr.Value()
	}
	return nil
}

func registerGraphIdToExec(id graph.GraphId, exec *Exec) {
	graphToExecMu.Lock()
	defer graphToExecMu.Unlock()
	graphToExec[id] = weak.Make(exec)
}

func removeGraphIds(ids ...graph.GraphId) {
	graphToExecMu.Lock()
	defer graphToExecMu.Unlock()
	for _, id := range ids {
		delete(graphToExec, id)
	}
}
