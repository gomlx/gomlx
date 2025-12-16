package dtypes

// File copied from github.com/gomlx/go-xla/pkg/types/dtypes/gen_dtype_enum.go.
// As much as possible, to keep data types aligned, we should upate it accordingly,
// but they may diverge if new dtypes are introduced by other backends.

// DType is an enum represents the data type of a buffer or a scalar.
//
// The names were populated from the XLA C/C++ constants, so they are not Go idiomatic.
// The package provides some aliases.
type DType int32

const (
	// InvalidDType is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_INVALID).
	// Invalid primitive type to serve as default.
	InvalidDType DType = 0

	// Bool is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_PRED).
	// Predicates are two-state booleans.
	Bool DType = 1

	// Int8 is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_S8).
	// Signed integral values of fixed width.
	Int8 DType = 2

	// Int16 is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_S16).
	Int16 DType = 3

	// Int32 is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_S32).
	Int32 DType = 4

	// Int64 is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_S64).
	Int64 DType = 5

	// Uint8 is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_U8).
	// Unsigned integral values of fixed width.
	Uint8 DType = 6

	// Uint16 is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_U16).
	Uint16 DType = 7

	// Uint32 is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_U32).
	Uint32 DType = 8

	// Uint64 is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_U64).
	Uint64 DType = 9

	// Float16 is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_F16).
	// Floating-point values of fixed width.
	Float16 DType = 10

	// Float32 is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_F32).
	Float32 DType = 11

	// Float64 is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_F64).
	Float64 DType = 12

	// BFloat16 is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_BF16).
	// Truncated 16 bit floating-point format. This is similar to IEEE's 16 bit
	// floating-point format, but uses 1 bit for the sign, 8 bits for the exponent
	// and 7 bits for the mantissa.
	BFloat16 DType = 13

	// Complex64 is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_C64).
	// Complex values of fixed width.
	//
	// Paired F32 (real, imag), as in std::complex<float>.
	Complex64 DType = 14

	// Complex128 is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_C128).
	// Paired F64 (real, imag), as in std::complex<double>.
	Complex128 DType = 15

	// F8E5M2 is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_F8E5M2).
	// Truncated 8 bit floating-point formats.
	F8E5M2 DType = 16

	// F8E4M3FN is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_F8E4M3FN).
	F8E4M3FN DType = 17

	// F8E4M3B11FNUZ is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_F8E4M3B11FNUZ).
	F8E4M3B11FNUZ DType = 18

	// F8E5M2FNUZ is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_F8E5M2FNUZ).
	F8E5M2FNUZ DType = 19

	// F8E4M3FNUZ is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_F8E4M3FNUZ).
	F8E4M3FNUZ DType = 20

	// S4 is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_S4).
	// 4-bit integer types
	S4 DType = 21

	// U4 is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_U4).
	U4 DType = 22

	// TOKEN is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_TOKEN).
	TOKEN DType = 23

	// S2 is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_S2).
	// 2-bit integer types
	S2 DType = 24

	// U2 is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_U2).
	U2 DType = 25

	// F8E4M3 is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_F8E4M3).
	// More truncated 8 bit floating-point formats.
	F8E4M3 DType = 26

	// F8E3M4 is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_F8E3M4).
	F8E3M4 DType = 27

	// F8E8M0FNU is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_F8E8M0FNU).
	F8E8M0FNU DType = 28

	// F4E2M1FN is a 1:1 mapping of the corresponding C enum value defined in pjrt_c_api.h (as PJRT_Buffer_Type_F4E2M1FN).
	// 4-bit MX floating-point format.
	F4E2M1FN DType = 29
)

// Aliases from PJRT C API.
const (
	// INVALID (or PJRT_Buffer_Type_INVALID) is the C enum name for InvalidDType.
	INVALID = InvalidDType

	// PRED (or PJRT_Buffer_Type_PRED) is the C enum name for Bool.
	PRED = Bool

	// S8 (or PJRT_Buffer_Type_S8) is the C enum name for Int8.
	S8 = Int8

	// S16 (or PJRT_Buffer_Type_S16) is the C enum name for Int16.
	S16 = Int16

	// S32 (or PJRT_Buffer_Type_S32) is the C enum name for Int32.
	S32 = Int32

	// S64 (or PJRT_Buffer_Type_S64) is the C enum name for Int64.
	S64 = Int64

	// U8 (or PJRT_Buffer_Type_U8) is the C enum name for Uint8.
	U8 = Uint8

	// U16 (or PJRT_Buffer_Type_U16) is the C enum name for Uint16.
	U16 = Uint16

	// U32 (or PJRT_Buffer_Type_U32) is the C enum name for Uint32.
	U32 = Uint32

	// U64 (or PJRT_Buffer_Type_U64) is the C enum name for Uint64.
	U64 = Uint64

	// F16 (or PJRT_Buffer_Type_F16) is the C enum name for Float16.
	F16 = Float16

	// F32 (or PJRT_Buffer_Type_F32) is the C enum name for Float32.
	F32 = Float32

	// F64 (or PJRT_Buffer_Type_F64) is the C enum name for Float64.
	F64 = Float64

	// BF16 (or PJRT_Buffer_Type_BF16) is the C enum name for BFloat16.
	BF16 = BFloat16

	// C64 (or PJRT_Buffer_Type_C64) is the C enum name for Complex64.
	C64 = Complex64

	// C128 (or PJRT_Buffer_Type_C128) is the C enum name for Complex128.
	C128 = Complex128
)

// MapOfNames to their dtypes. It includes also aliases to the various dtypes.
// It is also later initialized to include the lower-case version of the names.
var MapOfNames = map[string]DType{
	"InvalidDType":  InvalidDType,
	"INVALID":       InvalidDType,
	"Bool":          Bool,
	"PRED":          Bool,
	"Int8":          Int8,
	"S8":            Int8,
	"Int16":         Int16,
	"S16":           Int16,
	"Int32":         Int32,
	"S32":           Int32,
	"Int64":         Int64,
	"S64":           Int64,
	"Uint8":         Uint8,
	"U8":            Uint8,
	"Uint16":        Uint16,
	"U16":           Uint16,
	"Uint32":        Uint32,
	"U32":           Uint32,
	"Uint64":        Uint64,
	"U64":           Uint64,
	"Float16":       Float16,
	"F16":           Float16,
	"Float32":       Float32,
	"F32":           Float32,
	"Float64":       Float64,
	"F64":           Float64,
	"BFloat16":      BFloat16,
	"BF16":          BFloat16,
	"Complex64":     Complex64,
	"C64":           Complex64,
	"Complex128":    Complex128,
	"C128":          Complex128,
	"F8E5M2":        F8E5M2,
	"F8E4M3FN":      F8E4M3FN,
	"F8E4M3B11FNUZ": F8E4M3B11FNUZ,
	"F8E5M2FNUZ":    F8E5M2FNUZ,
	"F8E4M3FNUZ":    F8E4M3FNUZ,
	"S4":            S4,
	"U4":            U4,
	"TOKEN":         TOKEN,
	"S2":            S2,
	"U2":            U2,
	"F8E4M3":        F8E4M3,
	"F8E3M4":        F8E3M4,
	"F8E8M0FNU":     F8E8M0FNU,
	"F4E2M1FN":      F4E2M1FN,
}
