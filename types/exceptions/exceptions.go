// Package exceptions provides helper functions to leverage Go's `panic`, `recover` and `defer`
// as an "exceptions" system.
//
// The `panic`, `recover` and the added runtime type checking is slow when compared to
// simply returning errors.
// So it should be used where a little latency in case of errors is not an issue.
//
// It defines `Try` and `TryCatch[E any]`.
package exceptions

// Try calls `fn` and return any exception (`panic`) that may occur.
//
// Example:
//
//	var x ResultType
//	e := Try(func() { x = DoSomething() })
//	if e != nil {
//		if eInt, ok := e.(int); ok {
//			errorInt = e
//		} else if err, ok := e.(err); ok {
//			klog.Errorf("%v", e)
//		} else {
//			panic(e)
//		}
//	}
func Try(fn func()) (exception any) {
	defer func() {
		exception = recover()
	}()
	fn()
	return
}

// TryCatch executes `fn` and in case of `panic`, it recovers if of the type `E`.
// For a `panic` of any other type, it simply re-throws the `panic`.
//
// Example:
//
//	var x ResultType
//	err := TryCatch[error](func() { x = DoSomething() })
//	if err != nil {
//		// Handle error ...
//	}
func TryCatch[E any](fn func()) (exception E) {
	defer func() {
		e := recover()
		if e == nil {
			// No exceptions.
			return
		}
		var ok bool
		exception, ok = e.(E)
		if !ok {
			// Re-throw an exception of a different type.
			panic(e)
		}
		// Return the recovered exception.
		return
	}()
	fn()
	return
}
