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

package xla

// ErrorCode defined on a separate file, so it will work with stringer -- it doesn't work with files using cgo.

// ErrorCode is used by the underlying TensorFlow/XLA libraries, in Status objects.
type ErrorCode int

//go:generate stringer -type=ErrorCode errorcode.go

// Values copied from tensorflow/core/protobuf/error_codes.proto.
// TODO: convert the protos definitions to Go and use that instead.
const (
	OK                  ErrorCode = 0
	CANCELLED           ErrorCode = 1
	UNKNOWN             ErrorCode = 2
	INVALID_ARGUMENT    ErrorCode = 3
	DEADLINE_EXCEEDED   ErrorCode = 4
	NOT_FOUND           ErrorCode = 5
	ALREADY_EXISTS      ErrorCode = 6
	PERMISSION_DENIED   ErrorCode = 7
	UNAUTHENTICATED     ErrorCode = 16
	RESOURCE_EXHAUSTED  ErrorCode = 8
	FAILED_PRECONDITION ErrorCode = 9
	ABORTED             ErrorCode = 10
	OUT_OF_RANGE        ErrorCode = 11
)
