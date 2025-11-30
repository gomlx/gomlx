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

// Package checkpoints implements checkpoint management: saving and loading of checkpoints to file,
// or loading a checkpoint from an embedded checkpoint.
//
// The main object is the Handler, that should be created by calling Build, followed by the
// various options setting and finally calling Config.Done.
// Once create, if a previous saved checkpoint exists, it will automatically load variables and parameters
// for your model into Context.
// And as the model trains, one can call Handler.Save() at any time to save a new checkpoint --
// typically one will do that inside train.EveryNSteps().
//
// Example: After creating the Context, it checks if a checkpoint directory was set (`*flagCheckpoint`)
// and if yes, creates a checkpoints.Handler to save checkpoints every 100 steps, keeping the last
// `*flagCheckpointKeep` steps.
//
//	…
//	ctx := context.New()
//	ctx.SetParam(optimizers.ParamLearningRate, *flagLearningRate)
//
//	var checkpoint *checkpoints.Handler
//	if *flagCheckpoint != "" {
//		var err error
//		checkpoint, err = checkpoints.Build(ctx).Dir(*flagCheckpoint).Keep(*flagCheckpointKeep).Done()
//		Must(err)  // Panics if err != nil.
//	}
//	…
//	// Build training loop.
//	loop := train.NewLoop(trainer)
//	commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.
//	if checkpoint != nil {
//		const priority = 100  // Large number here, means it runs last.
//		train.EveryNSteps(loop, 100, "checkpointing", priority, checkpoint.OnStepFn)
//	}
//	…
//
// Example 2: To load a checkpoint from an embedded checkpoint, something usually used to distribute a model for
// inference:
//
//	//go:embed "my_model/checkpoint.json"
//	var myModelJson string
//
//	//go:embed "my_model/checkpoint.bin"
//	var myModelBin []byte
//
//	...
//	_ = checkpoints.Build(ctx).
//		FromEmbed(myModelJson, myModelBin).
//		Done()
//
// TODO:
//  1. Allow to specify parts of the model to load / scope where they should be loaded to, for
//     transfer learning.
package checkpoints

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"

	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/gomlx/gomlx/pkg/support/fsutil"
	"github.com/gomlx/gomlx/pkg/support/sets"
	"github.com/gomlx/gomlx/pkg/support/xslices"
)

var (
	// DirPermMode is the default directory creation permission (before umask) used.
	DirPermMode = os.FileMode(0770)

	// ErrUnsupportedCompression signifies an error when a compression type is not supported.
	ErrUnsupportedCompression = errors.New("unsupported compression")
)

// BinFormat defines the type for representing binary file compression formats.
type BinFormat int

const (

	// BinGZIP represents the GZIP compressed binary file format.
	BinGZIP BinFormat = iota
	// BinUncompressed represents the uncompressed binary file format.  This is the format used up until version 0.24.1
	BinUncompressed
)

// Config for the checkpoints' Handler to be created. This is created with Build() and
// configured with the various methods. Once finished, call Done() and it will output
// a checkpoints.Handler that loads (if there are any previously saved checkpoints) and
// saves checkpoints.
type Config struct {
	ctx *context.Context

	err error

	// One of the two are set: dir or jsonReader+binReader.
	dir                   string
	jsonReader, binReader io.Reader

	immediate bool
	keep      int
	mustLoad  bool

	includeParams   bool             // whether to includeParams in loading/saving.
	paramsToExclude sets.Set[string] // specific parameter names to exclude from loading.

	backend  backends.Backend // used when taking the mean.
	takeMean int

	varsToExclude sets.Set[*context.Variable]

	binFormat BinFormat // the compression format
}

// Build a configuration for building a checkpoints.Handler. After configuring the
// Config object returned, call `Done` to get the configured checkpoints.Handler.
//
// The new checkpoints.Handler will load ("lazy" by default) a checkpoint to the context
// (see Config.Dir, Config.DirFromBase or Config.FromEmbed to specify where to load/save)
// if it exists, otherwise it creates a new directory and can simply be used to save checkpoints.
//
// When a checkpoint is "lazily loaded", its variables are not listed by default (if one uses Context.EnumerateVariables
// or Context.IterVariables). But if they are directly accessed, they are on-the-fly loaded. This is convenient
// when loading only part of the variables for inference (if one doesn't care about the training/optimizer variables),
// or for transfer learning part of a model. It also works to continue training a model loaded from a checkpoint.
// But if you need to variables to be loaded immediately, use Config.Immediate() -- an inspecting tool, like
// gomlx_checkpoints, will want to do that.
//
// See Config.Dir, Config.DirFromBase or Config.FromEmbed to specify where to load/save.
func Build(ctx *context.Context) *Config {
	c := &Config{
		ctx:             ctx,
		includeParams:   true,
		keep:            1,
		takeMean:        1,
		paramsToExclude: sets.Make[string](),
		varsToExclude:   sets.Make[*context.Variable](),
	}
	return c
}

// Load creates the configuration to load a checkpoint.
// It's identical to Build, except it will fail if the checkpoint does not already exist.
//
// Use Dir or DirWithBase to configure the location of the checkpoint.
// Once configured, call Config.Done to actually load it.
func Load(ctx *context.Context) *Config {
	c := Build(ctx)
	c.mustLoad = true
	return c
}

func (c *Config) setError(err error) {
	if c.err != nil {
		c.err = err
	}
}

// Dir sets the directory where to save / load the checkpoints.
//
// One must be set either Dir, DirFromBase, or TempDir before building the checkpoints.Handler.
func (c *Config) Dir(dir string) *Config {
	c.dir = fsutil.MustReplaceTildeInDir(dir)
	fi, err := os.Stat(dir)
	if err != nil && !os.IsNotExist(err) {
		c.setError(errors.Wrapf(err, "failed to os.Stat(%q)", dir))
		return c
	}
	if err == nil && !fi.IsDir() {
		c.setError(errors.Errorf("checkpiont directory %q exists but it's a normal file, not a directory", dir))
		return c
	}
	if err == nil {
		// Directory exists, all fine.
		return c
	}
	if c.mustLoad {
		c.setError(errors.Wrapf(err, "checkpoint directory %q does not exist or cannot be accessed", dir))
		return c
	}

	// Create the directory.
	err = os.MkdirAll(dir, DirPermMode)
	if err != nil {
		c.setError(errors.Wrapf(err, "trying to create dir %q", dir))
	}
	return c
}

// DirFromBase sets the directory where to save / load the checkpoints.
// If `dir` is not an absolute path, assumes it is a subdirectory of baseDir.
//
// One must be set either Dir, DirFromBase, or TempDir before building the checkpoints.Handler.
func (c *Config) DirFromBase(dir, baseDir string) *Config {
	dir = fsutil.MustReplaceTildeInDir(dir)
	if !path.IsAbs(dir) {
		baseDir = fsutil.MustReplaceTildeInDir(baseDir)
		dir = path.Join(baseDir, dir)
	}
	return c.Dir(dir)
}

// FromEmbed allows one to load a checkpoint from an embedded checkpoint (using the go:embed tag).
//
// You must set only one of Dir(or DirFromBase) or FromEmbed, but not both.
//
// Notice that after Done() is called, it releases the references to the passed JSON and binary blobs,
// potentially freeing the resources.
//
// Example:
//
//	//go:embed "my_model/checkpoint.json"
//	var myModelJson []byte
//
//	//go:embed "my_model/checkpoint.bin"
//	var myModelBin []byte
//
//	...
//	_ = checkpoints.Build(ctx).FromEmbed(myModelJson, myModelBin).Done()
func (c *Config) FromEmbed(json string, binary []byte) *Config {
	c.jsonReader = bytes.NewBufferString(json)
	c.binReader, _ = getLoadVarFilesFromReader(bytes.NewReader(binary))
	return c
}

// Immediate forces immediate load of all variables, as opposed to dynamically load
// variables from checkpoint as they are being used when building the model.
//
// Not normally needed, but may be handy for testing. See also [context.Context.InspectVariableIfLoaded].
//
// It may trigger use more memory if not all variables are not used by the model -- not all training data (e.g.: optimizer
// variables) is used for inference.
func (c *Config) Immediate() *Config {
	c.immediate = true
	return c
}

// TempDir creates a temporary directory under dir, with the pattern name, and uses this
// directory to load / save checkpoints. It's a convenience wrapper to os.MkdirTemp.
//
// If dir is the empty string, MkdirTemp uses the default directory for temporary files, as returned
// by os.TempDir.
//
// The new directory's name is generated by adding a random string to the end of the pattern.
// If `pattern` includes a "*", the random string replaces the last "*" instead (see os.MkdirTemp).
//
// Any errors are reported on the return to the call to the method Done.
//
// One must be set either Dir, DirFromBase, or TempDir before building the checkpoints.Handler.
func (c *Config) TempDir(dir, pattern string) *Config {
	newDir, err := os.MkdirTemp(dir, pattern)
	if err != nil {
		c.setError(errors.Wrapf(err, "failed to create os.MkdirTemp(%q, %q)", dir, pattern))
		return c
	}
	c.dir = newDir
	err = os.Chmod(c.dir, DirPermMode)
	if err != nil {
		c.setError(errors.Wrapf(err, "failed to os.Chmod(%q, %s)", dir, DirPermMode))
		return c
	}
	return c
}

// ExcludeAllParams configures Handler to exclude Context parameters (values usually
// read/written by Context.GetParam and context.SetParam) from being read.
//
// By default, Params are loaded and set into Context the moment Handler is created
// (when Done() is called), overriding values already present in the Context.
//
// See also ExcludeParams to exclude specific params from being read.
func (c *Config) ExcludeAllParams() *Config {
	c.includeParams = false
	return c
}

// ExcludeParams configures Handler to exclude certain Context parameters (values usually
// read/written by Context.GetParam and context.SetParam) from being read.
// It can be called multiple times; each call adds new parameters to be excluded.
//
// For values in paramsToExclude that don't include a preceding scope (separated by "/"), the exclusion applies to all scopes.
// Otherwise, it applies only to the specific scope. See context.JoinScope to merge scope and name.
//
// By default, no parameters are excluded.
//
// See also ExcludeAllParams to exclude all params from being read.
func (c *Config) ExcludeParams(paramsToExclude ...string) *Config {
	for _, name := range paramsToExclude {
		c.paramsToExclude.Insert(name)
	}
	return c
}

// ExcludeVars enumerate variables to be excluded from saving.
// The function can be called multiple times, adding variables to be excluded from saving.
//
// It can also be called after the [Handler] object is built as new variables are created.
func (c *Config) ExcludeVars(vars ...*context.Variable) *Config {
	for _, v := range vars {
		c.varsToExclude.Insert(v)
	}
	return c
}

// Keep configures the number of checkpoint files to keep. If set to -1, it will never erase older checkpoints.
// The default is 1.
func (c *Config) Keep(n int) *Config {
	c.keep = n
	return c
}

// TakeMean loads the mean of the last `n` checkpoints.
// If `n <= 0`, take the mean of all available checkpoints.
// Notice that only trainable variables are averaged.
// Variables that have integer values or are not marked as trainable (e.g., the global step)
// are taken from the most recent checkpoint instead.
//
// The default is 1, so only load the most recent checkpoint.
//
// If n != 1, it requires a backend that will be used to calculate the mean.
// If n == 1, the backend argument is ignored and can be nil.
//
// Notice the mean is taken one tensor at a time, so at any time there is only one copy
// of the model weights in memory, plus the tensor being merged.
//
// If a mean is calculated, the values of the variables will be stored on-device. This can be good -- if their values
// are going to be used on-device anyway -- or bad -- if they are not needed on-device, and it's using the limited
// on-device space. Consider *tensors.Tensor.MaterializeLocal and *tensors.Tensor.InvalidateOnDevice to have them
// moved locally if so desired.
func (c *Config) TakeMean(n int, backend backends.Backend) *Config {
	if c.err != nil {
		return c
	}
	c.takeMean = n
	c.backend = backend
	if n != 1 && backend == nil {
		c.err = errors.Errorf(
			"TakeMean(n=%d, backend=nil) requires a valid backend for n != 1, it is the backend used to calculate the mean",
			n,
		)
	}
	return c
}

// WithCompression sets the binary format to the provided value.  The default configuration is BinGZIP.
func (c *Config) WithCompression(bf BinFormat) *Config {
	c.binFormat = bf
	if bf != BinGZIP && bf != BinUncompressed {
		c.binFormat = BinGZIP
	}
	return c
}

// Done creates a Handler with the current configuration. It returns an error if
// the configuration is invalid or if it's missing information.
func (c *Config) Done() (*Handler, error) {
	if c.err != nil {
		return nil, c.err
	}
	if c.dir == "" && c.jsonReader == nil {
		return nil, errors.Errorf("directory for checkpoints not configured or empty, and no embedded model configured")
	}
	if c.dir != "" && c.jsonReader != nil {
		return nil, errors.Errorf("cannot use both Dir/DirFromBase and FromEmbed at the same time, choose one.")
	}
	handler := &Handler{config: c, serialized: &serializedData{
		Params:    nil,
		Variables: nil,
	}}

	if c.dir != "" {
		// Load (if checkpoints exist) from a directory.
		checkpoints, err := handler.ListCheckpoints()
		if err != nil {
			return nil, err
		}
		if len(checkpoints) == 0 && c.mustLoad {
			return nil, errors.Errorf("no checkpoints found in %q", c.dir)
		}
		handler.checkpointsCount = maxCheckPointCountFromCheckpoints(checkpoints) + 1
		if len(checkpoints) > 0 {
			takeMean := c.takeMean
			if takeMean < 0 || takeMean > len(checkpoints) {
				takeMean = len(checkpoints)
			}
			if c.takeMean == 1 {
				// Just load most recent checkpoint.
				err = handler.loadCheckpointFromFile(checkpoints[len(checkpoints)-1], false, 0)
			} else {
				err = handler.takeMean(checkpoints[len(checkpoints)-takeMean:])
			}
			if err != nil {
				return nil, err
			}
		}
	} else {
		// Load from an embedded checkpoint.
		err := handler.loadCheckpoint(c.jsonReader, c.binReader, false, 0)
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to load checkpoint from embedded checkpoint (json+bin blobs) given")
		}
		// Don't keep links to the data, since it's no longer used -- and potentially releasing the memory.
		handler.config.jsonReader = nil
		handler.config.binReader = nil
	}

	if c.immediate {
		ctxToSet := c.ctx.Checked(false)
		for paramName, value := range handler.variableValues {
			scope, name := context.VariableScopeAndNameFromParameterName(paramName)
			v := ctxToSet.GetVariableByScopeAndName(scope, name)
			if v != nil {
				v.MustSetValue(value)
			} else {
				ctxToSet.InAbsPath(scope).VariableWithValue(name, value)
			}
		}
		// Empty remaining variableValues.
		handler.variableValues = make(map[string]*tensors.Tensor)
	} else {
		// Force overwriting variables already present in the context: e.g., global_step.
		ctxToSet := c.ctx.Checked(false)
		for v := range ctxToSet.IterVariables() {
			value, found := handler.LoadedVariables()[v.ParameterName()]
			if !found {
				continue
			}
			v.MustSetValue(value)
			delete(handler.variableValues, v.ParameterName())
		}
	}
	handler.attachTo(c.ctx)
	return handler, nil
}

// MustDone constructs the checkpoints.Handler. It panics if there was an error.
func (c *Config) MustDone() *Handler {
	h, err := c.Done()
	if err != nil {
		panic(errors.Wrap(err, "Failed to create checkpoints.Handler"))
	}
	return h
}

// Handler handles saving and loading of checkpoints for a context.Context. See an example in the
// package documentation.
//
// It is created and configured using Build(), followed by options setting and then calling
// Config.Done().
//
// Loading data into Handler happens at its creation time: it loads from the latest checkpoint.
// (Hyper-)Parameters are immediately loaded into the context then (if not Config.ExcludeAllParams)
// but the loaded variable values are only "consumed" (used) one at a time, as the variables are
// created during the graph building (e.g., when building the model).
//
// Saving of checkpoints is explicit, by calling Handler.Save(). Usually this is
// done by configuring train.Loop to call it using train.EveryNSteps or train.NTimesDuringLoop.
// When saving all variables in Context are saved, along with any previous variables loaded
// by the Handler that were not used by Context and with the `Params` for all scopes (including
// changed values).
//
// There can be more than one Handler attached to a Context -- they are used for loading in order
// they are created (so the first one created takes priority). Multiple Handler set up can
// be used, for instance, for transfer learning, where parts of the model are loaded from somewhere
// else.
//
// A Handler can only be "attached" to one context.Context. If one wants to load the same
// checkpoint to two different contexts, another Handler object needs to be created.
// This is because once a variable is loaded, it is transferred to Context, and handler does
// not keep it.
type Handler struct {
	config            *Config
	ctx               *context.Context
	prevContextLoader context.Loader

	serialized     *serializedData
	variableValues map[string]*tensors.Tensor
	mergeExec      *graph.Exec

	checkpointsCount int
}

// serializedData is how the information is read and written from storage.
type serializedData struct {
	Params []serializedParam

	// Variables maps context.Variable.GetParameterName() to its position in storage.
	Variables []serializedVar
	// BinFormat describes the format used by the binary file.  It is informative.
	// The current valid values are "gzip" and "uncompressed"
	BinFormat string
}

// serializedVar contains information about the variable that was serialized.
type serializedVar struct {
	// ParameterName is a Variable unique id.
	ParameterName string

	// Dimensions of the shape.
	Dimensions []int

	// DType of the shape.
	DType dtypes.DType

	// Pos, Length in bytes in the file.
	Pos, Length int
}

// serializedParam represents a serialized context parameter.
// It includes the original ValueType, because Json decoder may
// not be capable of recovering the original type in anonymous (any) Value.
type serializedParam struct {
	Scope, Key string
	Value      any
	ValueType  string
}

// jsonDecodedTypeConvert attempts to convert the Value decoded by Json into
// the original ValueType.
//
// E.g.: Json decoder will decode all numbers to float64. So we cast it to the
// given ValueType.
func (p *serializedParam) jsonDecodeTypeConvert() {
	// Switch on current Json type:
	switch value := p.Value.(type) {
	case float64:
		// All numbers when converted to `any` by the json decoders become float64,
		// here we convert them back.
		switch p.ValueType {
		case "int":
			p.Value = int(value)
		case "int8":
			p.Value = int8(value)
		case "int32":
			p.Value = int32(value)
		case "int64":
			p.Value = int64(value)
		case "uint8":
			p.Value = uint8(value)
		case "uint32":
			p.Value = uint32(value)
		case "float32":
			p.Value = float32(value)
		}

	case []any:
		switch p.ValueType {
		case "[]int":
			p.Value = xslices.Map(value, func(fAny any) int {
				f, _ := fAny.(float64) // Json decoder converts any numbers to float64.
				return int(f)
			})
		case "[]float64":
			p.Value = xslices.Map(value, func(fAny any) float64 {
				f, _ := fAny.(float64) // Json decoder converts any numbers to float64.
				return f
			})
		case "[]string":
			p.Value = xslices.Map(value, func(sAny any) string {
				s, _ := sAny.(string) // Json decoder converts any numbers to float64.
				return s
			})
		}
	default:
		// No other types converted for now.
		return
	}
}

// String implements Stringer.
func (h *Handler) String() string {
	return fmt.Sprintf("checkpoints.Handler(%q)", h.config.dir)
}

// newCheckpointBaseName returns the base name for the checkpoint files.
func (h *Handler) newCheckpointBaseName(globalStep int64) string {
	now := time.Now().Format("20060102-150405")
	baseName := fmt.Sprintf("%sn%07d-%s", baseNamePrefix, h.checkpointsCount, now)
	if globalStep > 0 {
		return fmt.Sprintf("%s-step-%08d", baseName, globalStep)
	} else {
		return fmt.Sprintf("%s-initial", baseName)
	}
}

const (
	baseNamePrefix = "checkpoint-"

	// JsonNameSuffix for the JSON files returned by Handler.ListCheckpoints.
	JsonNameSuffix = ".json"

	// BinDataSuffix for the data files (holding the tensor values) returned by Handler.ListCheckpoints.
	BinDataSuffix = ".bin"

	// BackupDir is the name of the (sub-)directory under the model checkpoints directory that holds
	// the backups. See Handler.Backup.
	BackupDir = "backup"
)

// ListCheckpoints returns the base file paths of the checkpoints in the directory in time order (older first).
//
// The actual paths are these base file paths suffixed with JsonNameSuffix and BinDataSuffix.
func (h *Handler) ListCheckpoints() (checkpoints []string, err error) {
	entries, err := os.ReadDir(h.config.dir)
	if err != nil {
		return nil, errors.Wrapf(err, "%s listing checkpoints", h)
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		fileName := entry.Name()
		if !strings.HasPrefix(fileName, baseNamePrefix) || !strings.HasSuffix(fileName, JsonNameSuffix) {
			continue
		}
		baseName := fileName[:len(fileName)-len(JsonNameSuffix)]
		checkpoints = append(checkpoints, baseName)
	}
	sort.Strings(checkpoints)
	return checkpoints, nil
}

// HasCheckpoints returns whether there are any checkpoints saved.
func (h *Handler) HasCheckpoints() (bool, error) {
	list, err := h.ListCheckpoints()
	return len(list) > 0, err
}

var checkpointCountRegex = regexp.MustCompile(`^checkpoint-n(\d+)-`)

// maxCheckPointCountFromCheckpoints returns the largest `checkpointCount` in the saved
// checkpoints -- so the next checkpoint saved uses this count+1.
//
// The input should be the output of Handler.ListCheckpoints.
func maxCheckPointCountFromCheckpoints(checkpoints []string) int {
	maxId := -1
	for _, name := range checkpoints {
		matches := checkpointCountRegex.FindAllStringSubmatch(name, 1)
		if len(matches) != 1 || len(matches[0]) != 2 {
			continue
		}
		id, err := strconv.Atoi(matches[0][1])
		if err != nil {
			continue
		}
		if id > maxId {
			maxId = id
		}
	}
	return maxId
}

// loadCheckpointFromFile loads a specific checkpoint file. This needs to happen before attachTo,
// since otherwise it may not have any effect.
//
// Usually this does not need to be called: when the Handler is created (when calling `Build()....Done()`)
// it will automatically load the latest checkpoint. But it can be used to load some specific
// checkpoint.
//
// If `merge` is set to false, loading a different checkpoint discards the previous checkpoint read.
// If `merge` is set to true, only trainable weights are merged into the current values, using
// `mergeWeight` for the current weight. For merging one must set up `h.mergeExec` as well.
func (h *Handler) loadCheckpointFromFile(baseName string, merge bool, mergeWeight float64) error {
	if klog.V(1).Enabled() {
		klog.Infof("loading: %q\n", baseName)
	}
	if h.ctx != nil {
		return errors.Errorf(
			"%s tried to loadCheckpointFromFile(%q) after being attached to a Context, this is not allowed",
			h, baseName)
	}

	// Open files for reading.
	binFileName := filepath.Join(h.config.dir, baseName+BinDataSuffix)
	f, err := os.Open(binFileName)
	if err != nil {
		return errors.Wrapf(err, "%s: failed to open checkpoint data file %s", h, binFileName)
	}
	defer func() { _ = f.Close() }()
	binFile, err := getLoadVarFilesFromReader(f)
	if err != nil {
		return errors.Wrapf(err, "%s: failed to read checkpoint data file %s", h, binFileName)
	}

	jsonFileName := filepath.Join(h.config.dir, baseName+JsonNameSuffix)
	jsonFile, err := os.Open(jsonFileName)
	if err != nil {
		return errors.Wrapf(err, "%s: failed to open checkpoint metadata file %s", h, jsonFileName)
	}
	defer func() { _ = jsonFile.Close() }()
	if err = h.loadCheckpoint(jsonFile, binFile, merge, mergeWeight); err != nil {
		err = errors.WithMessagef(err,
			"failed loading checkpoint from %s{%s,%s}", baseName, JsonNameSuffix, BinDataSuffix)
		return err
	}
	return nil
}

// loadCheckpoint from a jsonReader (io.Reader) for configuration, and a binReader with the actual data for the variables.
//
// If `merge` is set to false, loading a different checkpoint discards the previous checkpoint read.
// If `merge` is set to true, only trainable weights are merged into the current values, using
// `mergeWeight` for the current weight. For merging one must set up `h.mergeExec` as well.
func (h *Handler) loadCheckpoint(jsonReader, binReader io.Reader, merge bool, mergeWeight float64) error {
	// Read metadata.
	dec := json.NewDecoder(jsonReader)
	var serialized *serializedData
	if err := dec.Decode(&serialized); err != nil {
		return errors.Wrapf(err, "%s: failed to decode contents of checkpoint", h)
	}
	if h.config.includeParams {
		for ii := range serialized.Params {
			// Recover original type where possible.
			serialized.Params[ii].jsonDecodeTypeConvert()
		}
	} else {
		// Discard loaded Params, if they were not included.
		serialized.Params = nil
	}
	if !merge {
		// We are loading all the variables, as opposed to merging them.
		h.serialized = serialized
		h.variableValues = make(map[string]*tensors.Tensor, len(h.serialized.Variables))
	}

	// Load variable values: we assume they are stored in order.
	var memoryPos int
	for _, varInfo := range serialized.Variables {
		tensor := tensors.FromShape(shapes.Make(varInfo.DType, varInfo.Dimensions...))
		if varInfo.Pos != memoryPos {
			return errors.Errorf("variable %s (%s) position at %d is out-of-order, expected it to be in %d -- "+
				"if you need to handle checkpoints with variables out-of-order, please open a feature request, it is doable",
				varInfo.ParameterName, tensor.Shape(), varInfo.Pos+1, memoryPos)
		}
		memoryPos += varInfo.Length
		var n, memoryLen int
		var readErr error
		err := tensor.MutableBytes(func(data []byte) {
			memoryLen = len(data)
			n, readErr = binReader.Read(data)
		})
		if err != nil {
			return errors.WithMessagef(err, "failed to access tensor data for variable %q",
				varInfo.ParameterName)
		}
		if readErr != nil {
			return errors.Wrapf(
				readErr,
				"%s: failed to read variable contents of checkpoint binary file at position %d",
				h,
				varInfo.Pos,
			)
		}
		if n != memoryLen {
			return errors.Errorf("%s: failed to read variable contents of checkpoint binary file "+
				"at position %d -- read %d bytes, wanted %d bytes", h, varInfo.Pos, n, memoryLen)
		}

		if !merge {
			// Load the value.
			h.variableValues[varInfo.ParameterName] = tensor
		} else {
			// Merge value:
			current, found := h.variableValues[varInfo.ParameterName]
			if !found || !varInfo.DType.IsFloat() {
				// Variable was not found in the last checkpoint or not merge-able, just ignore it.
				continue
			}
			var results []*tensors.Tensor
			err := TryCatch[error](func() {
				results = h.mergeExec.MustExec(current, tensor, shapes.CastAsDType(mergeWeight, varInfo.DType))
			})
			if err != nil {
				panic(errors.WithMessagef(err, "when taking the mean of variable %q", varInfo.ParameterName))
			}
			current.MustFinalizeAll()
			h.variableValues[varInfo.ParameterName] = results[0]
			tensor.MustFinalizeAll()
		}
	}
	return nil
}

// takeMean will load the checkpoints pointed by baseNames and take the mean of those.
// It takes the mean only for trainable float variables, everything else it just takes
// the value from the last checkpoint.
//
// Notice the mean is taken one tensor at a time, so at any time there is only one copy
// of the model weights in memory, plus the tensor being merged.
func (h *Handler) takeMean(baseNames []string) error {
	// First load the last checkpoint.
	err := h.loadCheckpointFromFile(xslices.Last(baseNames), false, 0)
	if err != nil {
		return err
	}

	// Create merger graph executor.
	h.mergeExec = graph.MustNewExec(h.config.backend, func(a, b, bWeight *graph.Node) *graph.Node {
		return graph.Add(
			graph.Mul(a, graph.OneMinus(bWeight)),
			graph.Mul(b, bWeight))
	})
	h.mergeExec.SetMaxCache(-1) // No max number of graphs, since each variable may be of different shape.

	// Then merge all other weights -- the order doesn't matter.
	for ii, baseName := range baseNames[:len(baseNames)-1] {
		mergeWeight := 1.0 / (float64(ii) + 2.0)
		err = h.loadCheckpointFromFile(baseName, true, mergeWeight)
		if err != nil {
			return err
		}
	}

	// Free merge executor.
	h.mergeExec.Finalize()
	h.mergeExec = nil
	return nil
}

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

// Save creates a new checkpoint and save the context variables and (optionally) Params.
//
// All variables in the context are saved, as well as those previously loaded -- this allows one
// to load the variables only for a part of the model, update that part, and save again with everything.
//
// Params is (de-) serialized with package json.
//
// If the handler is nil, this is a no-op: so it's safe to simply be called, even if the user hasn't configured a
// checkpoint.
//
// By default, the binary file is compressed.  The option WithCompression overrides the default behavior.  This
// information is reported in the JSON file.
func (h *Handler) Save() error {
	if h == nil {
		return nil
	}
	if h.ctx == nil {
		return errors.Errorf("%s not attached to a context.Context yet.", h)
	}

	// Read globalStep if one is set.
	globalStep := optimizers.GetGlobalStep(h.ctx)
	// Update the binary format in JSON.
	h.serialized.BinFormat = h.config.binFormat.String()

	// Copy over Params.
	if h.config.includeParams {
		h.serialized.Params = nil
		h.ctx.EnumerateParams(func(scope, name string, value any) {
			h.serialized.Params = append(h.serialized.Params,
				serializedParam{
					Scope: scope, Key: name, Value: value, ValueType: fmt.Sprintf("%T", value)})
		})
	}

	// Create files.
	baseName := h.newCheckpointBaseName(globalStep)
	h.checkpointsCount += 1 // Bump unique number.
	varFileName := filepath.Join(h.config.dir, baseName+BinDataSuffix)
	varFile, err := getSaveVarFiles(varFileName, h.config.binFormat)
	if err != nil {
		return errors.Wrapf(err, "%s: failed to create checkpoint data file %s", h, varFileName)
	}
	jsonFileName := filepath.Join(h.config.dir, baseName+JsonNameSuffix)
	var jsonFile *os.File
	jsonFile, err = os.Create(jsonFileName)
	if err != nil {
		return errors.Wrapf(err, "%s: failed to create checkpoint metadata file %s", h, jsonFileName)
	}

	// Copy over and set variables: both from Context and previously loaded ones, that haven't yet
	// been loaded into context.
	h.serialized.Variables = make([]serializedVar, 0, h.ctx.NumVariables()+len(h.variableValues))
	pos := 0
	// * Closure to save the contents of a variable.
	saveVar := func(name string, tensor *tensors.Tensor) error {
		shape := tensor.Shape()
		var writeErr error
		var n, memoryLen int
		err := tensor.ConstBytes(func(rawData []byte) {
			memoryLen = len(rawData)
			n, writeErr = varFile.Write(rawData)
		})
		if err != nil {
			return errors.Wrapf(err, "%s: failed to access tensor data for variable %s", h, name)
		}
		if writeErr != nil {
			return errors.Wrapf(err, "%s: failed to write variable %s", h, name)
		}
		if n != memoryLen {
			return errors.Errorf(
				"%s: failed to write variable %s -- %d bytes requested, %d bytes written",
				h,
				name,
				memoryLen,
				n,
			)
		}
		h.serialized.Variables = append(h.serialized.Variables, serializedVar{
			ParameterName: name,
			Dimensions:    shape.Dimensions,
			DType:         shape.DType,
			Pos:           pos,
			Length:        memoryLen,
		})
		pos += memoryLen
		return nil
	}

	// * Loop over variables in context.
	h.ctx.EnumerateVariables(func(v *context.Variable) {
		if err != nil {
			return
		}
		if h.config.varsToExclude.Has(v) {
			return
		}
		err = saveVar(v.ParameterName(), v.MustValue())
	})

	// * Loop over current loaded variables.
	for name, tensor := range h.variableValues {
		if err != nil {
			break
		}
		err = saveVar(name, tensor)
	}
	if err != nil {
		return err
	}
	if err := varFile.Flush(); err != nil {
		return errors.Wrapf(err, "%s: failed to flush checkpoint data file %s", h, varFileName)
	}
	if err := varFile.Close(); err != nil {
		return errors.Wrapf(err, "%s: failed to close checkpoint data file %s", h, varFileName)
	}
	// Write all the metadata, including Params.
	enc := json.NewEncoder(jsonFile)
	enc.SetIndent("", "\t")
	err = enc.Encode(&h.serialized)
	if err != nil {
		return errors.Wrapf(err, "%s: failed to write checkpoint metadata file %s", h, jsonFileName)
	}
	err = jsonFile.Close()
	if err != nil {
		return errors.Wrapf(err, "%s: failed to close checkpoint metadata file %s", h, jsonFileName)
	}
	// Remove excess checkpoints.
	return h.keepNCheckpoints()
}

// Backup links (or copies) the latest checkpoint to a separate sub-directory under the model directory called
// "backup" (constant in checkpoints.BackupDir).
//
// This way the backed up checkpoint doesn't get automatically deleted as the model training progresses.
//
// Useful, for instance, to back up the checkpoints used to collect the plot points. But can be used for any other
// reason.
func (h *Handler) Backup() error {
	baseNames, err := h.ListCheckpoints()
	if err != nil {
		return errors.WithMessagef(err, "failed Backup() finding currenct checkpoints")
	}
	if len(baseNames) == 0 {
		// If there are n
		return errors.Errorf("there are no saved checkpoints in %q: maybe call Save() before Backup() ?", h.Dir())
	}
	baseName := xslices.Last(baseNames)
	varFilePath := filepath.Join(h.config.dir, baseName+BinDataSuffix)
	jsonFilePath := filepath.Join(h.config.dir, baseName+JsonNameSuffix)
	backupDir := path.Join(h.Dir(), BackupDir)
	err = os.MkdirAll(backupDir, DirPermMode)
	if err != nil {
		return errors.Wrapf(err, "trying to create dir %q", backupDir)
	}
	for _, srcFilePath := range []string{varFilePath, jsonFilePath} {
		newPath := path.Join(backupDir, path.Base(srcFilePath))
		err := os.Link(srcFilePath, newPath)
		if err != nil {
			return errors.Wrapf(err, "failed to link %q to %q", srcFilePath, newPath)
		}
	}
	return nil
}

// OnStepFn implements `train.OnStepFn`, and make it convenient to attach to a training loop.
// It simply calls save.
func (h *Handler) OnStepFn(_ *train.Loop, _ []*tensors.Tensor) error {
	return h.Save()
}

// keepNCheckpoints checks if there are more than the configured number of checkpoints, and remove
// the excess.
func (h *Handler) keepNCheckpoints() error {
	if h.config.keep < 0 {
		return nil
	}
	list, err := h.ListCheckpoints()
	if err != nil {
		return errors.Wrapf(err, "%s failed ot list saved checkpoints", h)
	}
	if len(list) <= h.config.keep {
		return nil
	}

	// Remove the excess checkpoints, starting from the earlier ones.
	list = list[:len(list)-h.config.keep]
	for _, baseName := range list {
		varFileName := filepath.Join(h.config.dir, baseName+BinDataSuffix)
		jsonFileName := filepath.Join(h.config.dir, baseName+JsonNameSuffix)
		for _, fileName := range []string{varFileName, jsonFileName} {
			err = os.Remove(fileName)
			if err != nil && !os.IsNotExist(err) {
				return errors.Wrapf(err, "%s failed to remove excess checkpoint file %q", h, fileName)
			}
		}
	}
	return nil
}

// attachTo attaches Handler to a context.Context. The first thing it does if there is a checkpoint
// loaded is to set the Context's Params from the loaded values (except if the Handler was configured
// with ExcludeAllParams).
//
// attachTo can only be called once. It will fail, and set the given context to an error state if
// requested to attach more than once.
func (h *Handler) attachTo(ctx *context.Context) {
	if h.ctx != nil {
		Panicf("%s already attached to a Context, can not attach to another one", h.config.dir)
	}
	h.ctx = ctx
	h.prevContextLoader = ctx.Loader()
	ctx.SetLoader(h)

	// Sets ctx.Params with values read, if any.
	if h.config.includeParams {
		for _, p := range h.serialized.Params {
			// Check for un-scoped and scoped exclusions:
			if h.config.paramsToExclude.Has(p.Key) || h.config.paramsToExclude.Has(context.JoinScope(p.Scope, p.Key)) {
				continue
			}
			tmpCtx := ctx.InAbsPath(p.Scope)
			tmpCtx.SetParam(p.Key, p.Value)
		}
	}
}

// Dir returns the directory the Handler is configured to.
// It cannot be changed once the Handler was created.
//
// It returns "" (empty) if the Handler is `nil`.
func (h *Handler) Dir() string {
	if h == nil {
		return ""
	}
	return h.config.dir
}

// LoadVariable implements context.Loader.
// This is called by context.Context when the variable is used for the first time.
// The user may want to use this function to inspect loaded values for testing.
func (h *Handler) LoadVariable(ctx *context.Context, scope, name string) (value *tensors.Tensor, found bool) {
	// Priority is based on the installation order. That means we attempt first the previously configured loaders.
	if h.prevContextLoader != nil {
		value, found = h.prevContextLoader.LoadVariable(ctx, scope, name)
		if found {
			// Previous manager found value (or issued an error), return that.
			return
		}
	}

	// Try to find variable in our currently loaded checkpoint.
	varParamName := context.VariableParameterNameFromScopeAndName(scope, name)
	value, found = h.variableValues[varParamName]
	if !found {
		return
	}

	// "Consume" value, meaning remove it from Handler.
	delete(h.variableValues, varParamName)
	return
}

// DeleteVariable implements context.Loader.
// It is called whenever Context.DeleteVariable is called. The deletion should cascade to the
// loader, otherwise the variable will reappear after deletion.
func (h *Handler) DeleteVariable(ctx *context.Context, scope, name string) error {
	if h.prevContextLoader != nil {
		h.prevContextLoader.DeleteVariable(ctx, scope, name)
	}
	varParamName := context.VariableParameterNameFromScopeAndName(scope, name)
	delete(h.variableValues, varParamName)
	return nil
}

// LoadedVariables for inspection. These are the values loaded -- but not necessarily immediately available in
// context, since they are actually used only when a model asks for the variable.
//
// The Handler owns the returned map, don't change it -- the behavior is undefined if you do.
func (h *Handler) LoadedVariables() map[string]*tensors.Tensor {
	return h.variableValues
}

// ExcludeVarsFromSaving enumerates variables to be excluded from saving.
// The function can be called multiple times, adding variables to be excluded from saving.
func (h *Handler) ExcludeVarsFromSaving(vars ...*context.Variable) {
	for _, v := range vars {
		h.config.varsToExclude.Insert(v)
	}
}

const (
	binHeader     = "gomlx_checkpoints"
	lenBinHeader  = len(binHeader)
	gzipHeader    = "gzip"
	lenGzipHeader = uint8(len(gzipHeader))
)

// Format header
//
// ----------------------------------------------
// | 0                 16 | 17  | 18    17 +len |
// ----------------------------------------------
// |  "gomlx_checkpoints" | len |  "gzip"       |

// getLoadVarFilesFromReader returns a reader to the decompressed binary variables.  It is compliant with legacy format,
// i.e., non-compressed.
func getLoadVarFilesFromReader(f io.ReadSeeker) (io.Reader, error) {
	buf := make([]byte, lenBinHeader)
	_, err := f.Read(buf)
	if err != nil {
		return nil, errors.Wrap(err, "read header")
	}
	if string(buf) != binHeader {
		_, err = f.Seek(0, io.SeekStart)
		if err != nil {
			return nil, errors.Wrap(err, "seek header")
		}
		return f, nil
	}
	var headerZipLen uint8
	if err := binary.Read(f, binary.BigEndian, &headerZipLen); err != nil {
		return nil, errors.Wrap(err, "read header")
	}
	buf1 := make([]byte, headerZipLen)
	_, err = f.Read(buf1)
	if err != nil {
		return nil, errors.Wrap(err, "read header")
	}
	if string(buf1) != gzipHeader {
		return nil, ErrUnsupportedCompression
	}
	rd, err := gzip.NewReader(f)
	if err != nil {
		return nil, errors.Wrap(err, "read gzip header")
	}
	defer func() { _ = rd.Close() }()
	var rd1 bytes.Buffer
	_, err = rd1.ReadFrom(rd)
	if err != nil {
		return nil, errors.Wrap(err, "read zip")
	}
	return &rd1, nil
}

type flushWriter interface {
	Write([]byte) (int, error)
	Close() error
	Flush() error
}

type flushNullWriter struct {
	io.WriteCloser
}

func (fw flushNullWriter) Flush() error {
	return nil
}

// getSaveVarFiles creates a new file at the specified path, writes a predefined header "gomlx_00", and returns a gzip
// writer for the file.  It is the responsibility of the caller to call the writer's Flush function before closing.
// Returns an error if the file creation or header writing process fails.
func getSaveVarFiles(path string, bf BinFormat) (flushWriter, error) {
	f, err := os.Create(path)
	if err != nil {
		return nil, errors.Wrap(err, "create file")
	}
	if bf == BinUncompressed {
		return &flushNullWriter{f}, nil
	}
	var h []byte
	h = append(h, []byte(binHeader)...)
	h = append(h, []byte{byte(lenGzipHeader)}...)
	h = append(h, []byte(gzipHeader)...)
	_, err = f.Write(h)
	if err != nil {
		return nil, errors.Wrap(err, "write header")
	}
	return gzip.NewWriter(f), nil
}
