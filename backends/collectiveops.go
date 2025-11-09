package backends

// CollectiveOps is an interface for collective operations, that is, operations executed across multiple devices.
type CollectiveOps interface {
	//// AllReduce is a multi-device operation that reduces over the build the AllReduce operation across the configured devices.
	//AllReduce(b Builder, reduceOp ReduceOpType, input Op, axis int) (Op, error)
	//
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
