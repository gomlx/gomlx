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

import (
	"fmt"
	"testing"
)

// LeakThreshold for MemoryLeakTest.
const LeakThreshold int64 = 50000

// MemoryLeakTest records memory before and after running `fn`, and reports a fatal error
// if its larger than a minimum threshold, see LeakThreshold.
func MemoryLeakTest(t *testing.T, fn func(), msg string) {
	used := MemoryUsedByFn(fn, msg)
	if used > LeakThreshold {
		t.Fatalf("MemoryLeakTest failed: %d more bytes used in %q", used, msg)
	}
}

func TestSizeOf(t *testing.T) {
	got, want := int(SizeOf[int]()), 8
	if int(got) != want {
		t.Errorf("sizeof(int)=%d, wanted %d", got, want)
	}
	got, want = int(SizeOf[uint8]()), 1
	if int(got) != want {
		t.Errorf("sizeof(uint8)=%d, wanted %d", got, want)
	}
	got, want = int(SizeOf[float32]()), 4
	if int(got) != want {
		t.Errorf("sizeof(float32)=%d, wanted %d", got, want)
	}
}

func TestStrFree(t *testing.T) {
	//fmt.Printf("MemoryStats before test:\n%s\n", MemoryStats())
	MemoryLeakTest(t,
		func() {
			for n := 0; n < 100000; n++ {
				got := NumberToString(n)
				want := fmt.Sprintf("%d", n)
				if got != want {
					t.Fatalf("numberToString(%d): Wanted %q, got %q", n, want, got)
				}
			}
		}, "Handling C strings")
}
