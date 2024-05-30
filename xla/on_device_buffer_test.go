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

package xla_test

import (
	"testing"

	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/xla"

	_ "github.com/gomlx/gomlx/xla/cpu"
)

func TestOnDeviceBuffer(t *testing.T) {
	client, err := xla.NewClient("Host", 1, -1)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	want := shapes.Make(shapes.Float64, 2, 3)
	literal := xla.NewLiteralFromShape(want)
	buffer, err := literal.ToOnDeviceBuffer(client, 0)
	if err != nil {
		t.Fatalf("failed to move Literal to device:%v", err)
	}
	got := buffer.Shape()
	if !got.Eq(want) {
		t.Fatalf("Created literal with shape %s, got ScopedBuffer of shape %s.", want, got)
	}

	got2, err := xla.FromOnDeviceBuffer(buffer)
	if err != nil {
		t.Fatalf("failed to move Literal from device:%v", err)
	}
	if !got2.Shape().Eq(want) {
		t.Fatalf("Created literal with shape %s, converted to and back from ScopedBuffer and got shape %s.", want, got2.Shape())
	}
}
