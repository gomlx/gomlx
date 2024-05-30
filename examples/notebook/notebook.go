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

// Package notebook allows one to check if running within a notebook.
package notebook

import "os"

// IsPresent returns whether running inside a Jupyter notebook.
func IsPresent() bool {
	return IsBashKernel() || IsGoNB()
}

const bashKernelEnv = "NOTEBOOK_BASH_KERNEL_CAPABILITIES"

// IsBashKernel returns whether running in a Jupyter notebook with a bash_kernel.
func IsBashKernel() bool {
	_, found := os.LookupEnv(bashKernelEnv)
	return found
}

const goNBKernelEnv = "GONB_PIPE"

// IsGoNB returns whether running in a Jupyter notebook with a GoNB kernel.
func IsGoNB() bool {
	_, found := os.LookupEnv(goNBKernelEnv)
	return found
}
