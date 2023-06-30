// Package exceptions provides helper functions to leverage Go's `panic`, `recover` and `defer`
// as an "exceptions" system.
//
// It is relatively slow (when compared to simply returning an error), but more ergonomic
// in some cases, and can be used where a little latency in case of errors is not an issue.
package exceptions

// Catch calls `handler` if an exception occurs of the given type.
//
// This should be called on a deferred statement. And multiple deferred Catch statements
// are allowed, for different types of exceptions.
//
// Example:
//
//	func SomeFunc() {
//		defer exceptions.Catch(func(e MyError) { fixIt(e) })
//		defer exceptions.Catch(func(e OtherInfo) { log(e) })
//		… // code that may throw (panic)...
//	}
func Catch[E any](handler func(exception E)) {
	exception := recover()
	if exception == nil {
		// No exception.
		return
	}
	exceptionE, ok := exception.(E)
	if !ok {
		// "Re-throw" the exception.
		panic(exception)
	}
	handler(exceptionE) // Assign panic.
}

// Try calls fn and returns any exception (`panic`) that may have occurred.
// If not panic happened, it returns nil.
func Try(fn func()) (exception any) {
	defer func() {
		exception = recover()
	}()
	fn()
	return
}

// TryFor calls `fn` and recover from any exceptions (panic) of type `E`. If no exception
// happened it returns the zero value for E.
//
// If a panic happened of a type different then `E`, it is not caught.
func TryFor[E any](fn func()) (exception E) {
	defer Catch(func(e E) { exception = e })
	fn()
	return
}

// Throw is an alias to `panic`, for those who prefer the usual exceptions' jargon.
func Throw(exception any) {
	panic(exception)
}
