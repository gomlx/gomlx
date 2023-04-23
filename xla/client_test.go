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

func TestBuildClient(t *testing.T) {
	platforms, err := GetPlatforms()
	if err != nil {
		t.Fatalf("Failed to enumerate platforms: %v", err)
	}
	fmt.Printf("Plaforms available: %v\n", platforms)
	for _, platform := range platforms {
		fmt.Printf("\nPlatform: %q\n", platform)
		_, err := NewClient(platform, 1, -1)
		if err != nil {
			t.Errorf("\tFailed to create Client for %q: %v", platform, err)
			continue
		}
		/*
			if !o.RunError() {
				t.Errorf("\tOrchestrator %q failed.", name)
				continue
			}
			shape, data := o.Result()
			wantShape := []int64{3}
			wantData := []float32{10.5, 21, 32}
			fmt.Printf("\tGot: Shape=%v, Data=%v\n", shape, data)
			if !reflect.DeepEqual(shape, wantShape) || !reflect.DeepEqual(data, wantData) {
				fmt.Printf("Want: Shape=%v, Data=%v\n", wantShape, wantData)
			}
		*/
	}
}
