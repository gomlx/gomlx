// Code generated by "stringer -type=Type"; DO NOT EDIT.

package nest

import "strconv"

func _() {
	// An "invalid array index" compiler error signifies that the constant values have changed.
	// Re-run the stringer command to generate them again.
	var x [1]struct{}
	_ = x[InvalidNest-0]
	_ = x[ValueNest-1]
	_ = x[SliceNest-2]
	_ = x[MapNest-3]
}

const _Type_name = "InvalidNestValueNestSliceNestMapNest"

var _Type_index = [...]uint8{0, 11, 20, 29, 36}

func (i Type) String() string {
	if i >= Type(len(_Type_index)-1) {
		return "Type(" + strconv.FormatInt(int64(i), 10) + ")"
	}
	return _Type_name[_Type_index[i]:_Type_index[i+1]]
}
