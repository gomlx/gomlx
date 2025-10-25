package checkpoints

// BinFormat defines the type for representing binary file compression formats.
type BinFormat int

const (

	// BinGZIP represents the GZIP compressed binary file format.
	BinGZIP BinFormat = iota
	// BinUncompressed represents the uncompressed binary file format.  This is the format used up until version 0.24.1
	BinUncompressed
)

// String implements the Stringer interface.
func (bf BinFormat) String() string {
	switch bf {
	case BinGZIP:
		return "gzip"
	case BinUncompressed:
		return "uncompressed"
	default:
		return "unknown"
	}
}

type handlerOptions struct {
	binFormat BinFormat
}

// HandlerOptions allows parameterizing Handler's function
type HandlerOptions func(opts *handlerOptions)

func collectHandlerOptions(options ...HandlerOptions) *handlerOptions {
	opts := &handlerOptions{}
	for _, option := range options {
		option(opts)
	}
	return opts
}

// WithCompression defines the compression format of the binary files.  The default mode is BinGZIP.
func WithCompression(bf BinFormat) HandlerOptions {
	return func(op *handlerOptions) {
		op.binFormat = bf
		if bf != BinGZIP && bf != BinUncompressed {
			op.binFormat = BinGZIP
		}
	}
}
