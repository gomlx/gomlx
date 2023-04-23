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

// Package xla wraps the XLA functionality, plus extra dependencies.
//
// To set a default platform to use, set the environment variable GOMLX_PLATFORM to the value you want (eg. "Host", "CUDA", "TPU").
//
// One can configure XLA C++ logging with the environment variable TF_CPP_MIN_LOG_LEVEL.
//
// To compile this library requires (see README.md for details on how to install these):
//
//   - gomlx_xla library installed
//   - tcmalloc installed. See README.md.
package xla

// #cgo LDFLAGS: -lgomlx_xla -ltcmalloc
import "C"
