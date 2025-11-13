package stablehlo

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/gomlx/stablehlo"
	shlotypes "github.com/gomlx/stablehlo/types"
)

func (b Builder) AllReduce(operands []backends.Op, reduceOp backends.ReduceOpType,
	replicaGroups [][]int, channelID int) ([]backends.Op, error) {
	nodes, err := b.verifyAndCastValues("BroadcastInDim", operands...)
	if err != nil {
		return nil, err
	}
	var reduceFn *stablehlo.Function
	collectiveConfig := shlotypes.CollectiveConfig{
		ChannelID:   &channelID,
		ChannelType: shlotypes.CrossReplica,
	}
	values, err := stablehlo.AllReduce(
		xslices.Map(nodes, func(node *Node) *stablehlo.Value { return node.value }),
		replicaGroups, reduceFn, &collectiveConfig)
	_ = values
	if err != nil {
		return nil, err
	}
	return nil, nil
}
