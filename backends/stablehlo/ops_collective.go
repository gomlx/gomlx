package stablehlo

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/stablehlo"
)

func (b Builder) AllReduce(inputs []backends.Op, reduceOp backends.ReduceOpType,
	replicaGroups [][]int, channelID int) ([]backends.Op, error) {
	nodes, err := b.verifyAndCastValues("BroadcastInDim", inputs...)
	if err != nil {
		return nil, err
	}
	value, err := stablehlo.AllReduce(nodes[0].value, ShapeToStableHLO(outputShape), broadcastAxes)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil

}
