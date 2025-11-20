package backends

// CollectiveOps is an interface for collective operations, that is, operations executed across multiple devices.
//
// EXPERIMENTAL: currently the best supported distribution model is "GSPMD" (Global Single Program Multiple Data),
// which automatically distributes the computation across all devices, without having to use explicit
// collective operations.
type CollectiveOps interface {
	// AllReduce is a distributed (multi-device) operation that reduces over the build the AllReduce operation
	// across replica groups.
	//
	// - operands: list of operands to be replicated -- often this operation is called over all the parameters
	//   of a model, hence the option to pass a variable number of parameters to them.
	// - reductionType: how the operands should be reduced.
	// - replicaGroups: a collection of replica groups: each replica group ([]int) is a collection of devices that
	//   will participate in the distributed operation. The devices are given as indices (hence []int) into the
	//   device assignments (not absolute DeviceNum).
	// - channelIDGenerator: issues unique IDs to be used for the operation. The number of channel IDs required
	//   may vary according to the backend (XLA requires one per dtype of the operands).
	//   It may also be used when (if) the gradient of the operators is calculated.
	AllReduce(operands []Op, reductionType ReduceOpType, replicaGroups [][]int, channelIDGenerator func() int) (
		[]Op, error)

	// CollectiveBroadcast broadcasts the value from the first replica (in each group) to all others.
	// The returned shape is the same as the source.
	// Devices not included in any replica group will return zeros as their output (the same shape as the input).
	//
	// - replicaGroups: a collection of replica groups: each replica group ([]int) is a collection of devices that
	//   will participate in the distributed operation. The devices are given as indices (hence []int) into the
	//   device assignments (not absolute DeviceNum).
	// - channelID: and identification of the communication channel used for this operation -- important when,
	//   for instance, many distributed operations are potentially happening concurrently. It must be the
	//   same across all participating devices.
	//CollectiveBroadcast(operand Op, replicaGroups [][]int, channelID int) (Op, error)

	// AllGather builds the AllGather operation.
	//AllGather(operands []Op, gatherAxis int, replicaGroups [][]int, channelID int) ([]Op, error)

	//// ReplicaId builds the ReplicaId operation.
	//ReplicaId(b Builder) (Op, error)

	//// PartitionId builds the PartitionId operation.
	//PartitionId(b Builder) (Op, error)
}
