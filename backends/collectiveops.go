package backends

// CollectiveOps is an interface for collective operations, that is, operations executed across multiple devices.
type CollectiveOps interface {
	// AllReduce is a multi-device operation that reduces over the build the AllReduce operation across replica groups.
	//
	// - inputs: list of operands to be replicated -- often this operation is called over all the parameters
	//   of a model, hence the option to pass a variable number of parameters to them.
	// - reduceOp: how the operands should be reduced.
	// - replicaGroups: a collection of replica groups: each replica group ([]int) is a collection of devices that
	//   will participate in the distributed operation. The devices are given as indices (hence []int) into the
	//   device assignments (not absolute DeviceNum).
	// - channelID: and identification of the communication channel used for this operation -- important when,
	//   for instance, many distributed operations are potentially happening concurrently. It must be the
	//   same across all participating devices.
	AllReduce(inputs []Op, reduceOp ReduceOpType, replicaGroups [][]int, channelID int) ([]Op, error)

	//// AllGather builds the AllGather operation.
	//AllGather(b Builder, gatherAxis int, input Op) (Op, error)
	//
	//// CollectiveBroadcast builds the CollectiveBroadcast operation.
	//CollectiveBroadcast(b Builder, input Op) (Op, error)
	//
	//// ReplicaId builds the ReplicaId operation.
	//ReplicaId(b Builder) (Op, error)
	//
	//// PartitionId builds the PartitionId operation.
	//PartitionId(b Builder) (Op, error)
}
