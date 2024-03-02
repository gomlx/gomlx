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

// Package checkpoints implements checkpoint management: saving and loading of checkpoints.
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
// ```
//
//	…
//	ctx := context.NewContext(manager)
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
// ```
//
// TODO:
//  1. Compress checkpoints.
//  2. Allow to specify parts of the model to load / scope where they should be loaded to, for
//     transfer learning.
package checkpoints

import (
	"encoding/json"
	"fmt"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types"
	. "github.com/gomlx/gomlx/types/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/pkg/errors"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

var (
	// DirPermMode is the default directory creation permission (before umask) used.
	DirPermMode = os.FileMode(0770)
)

// Config for the checkpoints Handler to be created. This is created with Build() and
// configured with the various methods. Once finished, call Done() and it will output
// a checkpoints.Handler that loads (if there are any previously saved checkpoints) and
// saves checkpoints.
type Config struct {
	ctx *context.Context

	err error

	dir             string
	includeParams   bool
	keep            int
	takeMean        int
	excludeFromSave types.Set[*context.Variable]
}

// Build a configuration for building a checkpoints.Handler. After configuring the
// Config object returned, call `Done` to get the configured checkpoints.Handler.
func Build(ctx *context.Context) *Config {
	c := &Config{
		ctx:             ctx,
		includeParams:   true,
		keep:            1,
		takeMean:        1,
		excludeFromSave: types.MakeSet[*context.Variable](),
	}
	return c
}

func (c *Config) setError(err error) {
	if c.err != nil {
		c.err = err
	}
}

// Dir sets the directory where to save / load the checkpoints.
//
// One must be set either Dir, DirFromBase or TempDir before building the checkpoints.Handler.
func (c *Config) Dir(dir string) *Config {
	c.dir = data.ReplaceTildeInDir(dir)
	fi, err := os.Stat(dir)
	if err != nil && !os.IsNotExist(err) {
		c.setError(errors.Wrapf(err, "failed to os.Stat(%q)", dir))
		return c
	}
	if err == nil && !fi.IsDir() {
		c.setError(errors.Errorf("directory name %q exists but it's a normal file, not a directory", dir))
		return c
	}
	if err == nil {
		// Directory exists, all fine.
		return c
	}

	// Create directory.
	err = os.MkdirAll(dir, DirPermMode)
	if err != nil {
		c.setError(errors.Wrapf(err, "trying to create dir %q", dir))
	}
	return c
}

// DirFromBase sets the directory where to save / load the checkpoints.
// If `dir` is not an absolute path, assumes it is a subdirectory of baseDir.
//
// One must be set either Dir, DirFromBase or TempDir before building the checkpoints.Handler.
func (c *Config) DirFromBase(dir, baseDir string) *Config {
	dir = data.ReplaceTildeInDir(dir)
	if !path.IsAbs(dir) {
		baseDir = data.ReplaceTildeInDir(baseDir)
		dir = path.Join(baseDir, dir)
	}
	return c.Dir(dir)
}

// TempDir creates a temporary directory under dir, with the pattern name, and uses this
// directory to load / save checkpoints. It's a convenience wrapper to os.MkdirTemp.
//
// If dir is the empty string, MkdirTemp uses the default directory for temporary files, as returned
// by os.TempDir.
//
// The new directory's name is generated by adding a random string to the end of pattern.
// If `pattern` includes a "*", the random string replaces the last "*" instead (see os.MkdirTemp).
//
// Any errors are reported on the return to the call to the method Done.
//
// One must be set either Dir, DirFromBase or TempDir before building the checkpoints.Handler.
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

// ExcludeParams configures Handler to exclude the Context parameters (values usually
// read/written by Context.GetParam and context.SetParam).
//
// By default, Params are loaded and set into Context the moment Handler is created
// (when Done() is called), overriding values already present in the Context.
func (c *Config) ExcludeParams() *Config {
	c.includeParams = false
	return c
}

// ExcludeVarsFromSaving enumerate variables to be excluded from saving.
// The function can be called multiple times, adding variables to be excluded from saving.
func (c *Config) ExcludeVarsFromSaving(vars ...*context.Variable) *Config {
	for _, v := range vars {
		c.excludeFromSave.Insert(v)
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
// Notice that only trainable variables are averaged. Variables that have
// integer values or are not marked as trainable (e.g. the global step),
// are taken from the most recent checkpoint instead.
//
// The default is 1, so only load the most recent checkpoint.
//
// Notice the mean is taken one tensor at a time, so at any time there is only one copy
// of the model weights in memory, plus the tensor being merged.
func (c *Config) TakeMean(n int) *Config {
	c.takeMean = n
	return c
}

// Done creates a Handler with the current configuration. It returns an error if
// the configuration is invalid, or if it's missing information.
func (c *Config) Done() (*Handler, error) {
	if c.err != nil {
		return nil, c.err
	}
	if c.dir == "" {
		return nil, errors.Errorf("directory for checkpoints not configured or empty")
	}
	handler := &Handler{config: c, serialized: &serializedData{
		Params:    nil,
		Variables: nil,
	}}
	checkpoints, err := handler.ListCheckpoints()
	handler.checkpointsCount = maxCheckPointCountFromCheckpoints(checkpoints) + 1
	if err != nil {
		return nil, err
	}
	if len(checkpoints) > 0 {
		takeMean := c.takeMean
		if takeMean < 0 || takeMean > len(checkpoints) {
			takeMean = len(checkpoints)
		}
		if c.takeMean == 1 {
			// Just load most recent checkpoint.
			err = handler.loadCheckpoint(checkpoints[len(checkpoints)-1], false, 0)
		} else {
			err = handler.takeMean(checkpoints[len(checkpoints)-takeMean:])
		}
		if err != nil {
			return nil, err
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

// Handler handles saving and loading of checkpoints for a context.Context. See example in
// package documentation.
//
// It is created and configured using Build(), followed by options setting and then calling
// Config.Done().
//
// Loading data into Handler happens at its creation time: it loads from the latest checkpoint.
// (Hyper-)Parameters are immediately loaded into the context then (if not Config.ExcludeParams)
// but the loaded variable values are only "consumed" (used) one at a time, as the variables are
// created during the graph building (e.g: when building the model).
//
// Saving of checkpoints is explicit, by calling Handler.Save(). Usually this is
// done by configuring train.Loop to call it using train.EveryNSteps or train.NTimesDuringLoop.
// When saving all variables in Context are saved, along with any previous variables loaded
// by the Handler that were not used by Context and with the `Params` for all scopes (including
// changed values).
//
// There can be more than one Handler attached to a Context -- they are used for loading in order
// they are created (so the first one created takes priority). Multiple Handler set up can
// be used for instance for transfer learning, where parts of the model are loaded from somewhere
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
	variableValues map[string]tensor.Tensor
	mergeExec      *graph.Exec

	checkpointsCount int
}

// serializedData is how the information is read and written from storage.
type serializedData struct {
	Params []serializedParam

	// Variables maps context.Variable.ParameterName() to its position in storage.
	Variables []serializedVar
}

// serializedVar contains information about the variable that was serialized.
type serializedVar struct {
	// ParameterName is a Variable unique id.
	ParameterName string

	// Dimensions of the shape.
	Dimensions []int

	// DType of the shape.
	DType shapes.DType

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
			p.Value = slices.Map(value, func(fAny any) int {
				f, _ := fAny.(float64) // Json decoder converts any numbers to float64.
				return int(f)
			})
		case "[]float64":
			p.Value = slices.Map(value, func(fAny any) float64 {
				f, _ := fAny.(float64) // Json decoder converts any numbers to float64.
				return f
			})
		case "[]string":
			p.Value = slices.Map(value, func(sAny any) string {
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
	jsonNameSuffix = ".json"
	varDataSuffix  = ".bin"
)

// ListCheckpoints returns the base file name of the checkpoints in the directory in time order (older first).
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
		if !strings.HasPrefix(fileName, baseNamePrefix) || !strings.HasSuffix(fileName, jsonNameSuffix) {
			continue
		}
		baseName := fileName[:len(fileName)-len(jsonNameSuffix)]
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

var checkpointCountRexex = regexp.MustCompile(`^checkpoint-n(\d+)-`)

// maxCheckPointCountFromCheckpoints returns the largest `checkpointCount` in the saved
// checkpoints -- so the next checkpoint saved uses this count+1.
//
// The input should be the output of Handler.ListCheckpoints.
func maxCheckPointCountFromCheckpoints(checkpoints []string) int {
	maxId := -1
	for _, name := range checkpoints {
		matches := checkpointCountRexex.FindAllStringSubmatch(name, 1)
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

// loadCheckpoint loads a specific checkpoint file. This needs to happen before attachTo,
// since otherwise it may not have any effect.
//
// Usually this does not need to be called: when the Handler is created (when calling `Build()....Done()`)
// it will automatically load the latest checkpoint. But it can be used to load some specific
// checkpoint.
//
// If `merge` is set to false, loading a different checkpoint discards the previous checkpoint read.
// If `merge` is set to true, only trainable weights are merged into the current values, using
// `mergeWeight` for the current weight. For merging one must set up `h.mergeExec` as well.
func (h *Handler) loadCheckpoint(baseName string, merge bool, mergeWeight float64) error {
	fmt.Printf("loading: %q\n", baseName)
	if h.ctx != nil {
		return errors.Errorf("%s tried to loadCheckpoint(%q) after being attached to a Context, this is not allowed", h, baseName)
	}

	// Open files for reading.
	varFileName := filepath.Join(h.config.dir, baseName+varDataSuffix)
	varFile, err := os.Open(varFileName)
	if err != nil {
		return errors.Wrapf(err, "%s: failed to open checkpoint data file %s", h, varFileName)
	}
	jsonFileName := filepath.Join(h.config.dir, baseName+jsonNameSuffix)
	jsonFile, err := os.Open(jsonFileName)
	if err != nil {
		return errors.Wrapf(err, "%s: failed to open checkpoint metadata file %s", h, jsonFileName)
	}

	// Read metadata.
	dec := json.NewDecoder(jsonFile)
	var serialized *serializedData
	if err = dec.Decode(&serialized); err != nil {
		return errors.Wrapf(err, "%s: failed to decode contents of checkpoint metadata file %s", h, jsonFileName)
	}
	if err = jsonFile.Close(); err != nil {
		return errors.Wrapf(err, "%s: failed to close checkpoint metadata file %s", h, jsonFileName)
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
		h.variableValues = make(map[string]tensor.Tensor, len(h.serialized.Variables))
	}

	// Load variable values.
	for _, varInfo := range serialized.Variables {
		localT := tensor.FromShape(shapes.Make(varInfo.DType, varInfo.Dimensions...))
		dataRef := localT.AcquireData()
		rawBytes := dataRef.Bytes()
		var n int
		n, err = varFile.Read(rawBytes)
		dataRef.Release()

		if err != nil {
			return errors.Wrapf(err, "%s: failed to read variable contents of checkpoint data file %s at position %d", h, varFileName, varInfo.Pos)
		}
		if n != len(rawBytes) {
			return errors.Errorf("%s: failed to read variable contents of checkpoint data file %s "+
				"at position %d -- read %d bytes, wanted %d bytes", h, varFileName, varInfo.Pos, n, len(rawBytes))
		}

		if !merge {
			// Load the value.
			h.variableValues[varInfo.ParameterName] = localT
		} else {
			// Make sure we have enough variations of mergeExec for each variable, if they
			// are of different shape.
			h.mergeExec.SetMaxCache(len(h.variableValues))

			// Merge value:
			current, found := h.variableValues[varInfo.ParameterName]
			if !found || !varInfo.DType.IsFloat() {
				// Variable not found in last checkpoint or not merge-able, just ignore it.
				continue
			}
			var results []tensor.Tensor
			err := TryCatch[error](func() {
				results = h.mergeExec.Call(current, localT, shapes.CastAsDType(mergeWeight, varInfo.DType))
			})
			if err != nil {
				panic(errors.WithMessagef(err, "when taking the mean of variable %q", varInfo.ParameterName))
			}
			current.FinalizeAll()
			h.variableValues[varInfo.ParameterName] = results[0]
			localT.FinalizeAll()
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
	err := h.loadCheckpoint(slices.Last(baseNames), false, 0)
	if err != nil {
		return err
	}

	// Create merger graph executor.
	manager := graph.BuildManager().Platform("CPU").Done()
	h.mergeExec = graph.NewExec(manager, func(a, b, bWeight *graph.Node) *graph.Node {
		return graph.Add(
			graph.Mul(a, graph.OneMinus(bWeight)),
			graph.Mul(b, bWeight))
	})

	// Then merge all other weights -- the order doesn't matter.
	for ii, baseName := range baseNames[:len(baseNames)-1] {
		mergeWeight := 1.0 / (float64(ii) + 2.0)
		err = h.loadCheckpoint(baseName, true, mergeWeight)
		if err != nil {
			return err
		}
	}

	// Free merge executor.
	h.mergeExec.Finalize()
	h.mergeExec = nil

	// Move all variables to a local tensor.
	for key, t := range h.variableValues {
		deviceT := t.CurrentDevice()
		if deviceT != nil {
			// Tensor is on device: convert to local, and free the device version.
			h.variableValues[key] = deviceT.Local()
			deviceT.Finalize()
		}
	}
	return nil
}

// Save creates a new checkpoint and save the context variables and (optionally) Params.
//
// All variables in the context are saved, as well as those previously loaded -- this allows one
// to load the variables only for a part of the model, update that part and save again with everything.
//
// Params is (de-) serialized with package json.
func (h *Handler) Save() error {
	if h.ctx == nil {
		return errors.Errorf("%s not attached to a context.Context yet.", h)
	}

	// Read globalStep if one is set.
	globalStep := optimizers.GetGlobalStep(h.ctx)

	// Copy over Params.
	if h.config.includeParams {
		h.serialized.Params = nil
		h.ctx.EnumerateParams(func(scope, key string, value any) {
			h.serialized.Params = append(h.serialized.Params,
				serializedParam{
					Scope: scope, Key: key, Value: value, ValueType: fmt.Sprintf("%T", value)})
		})
	}

	// Create files.
	baseName := h.newCheckpointBaseName(globalStep)
	h.checkpointsCount += 1 // Bump unique number.
	varFileName := filepath.Join(h.config.dir, baseName+varDataSuffix)
	varFile, err := os.Create(varFileName)
	if err != nil {
		return errors.Wrapf(err, "%s: failed to create checkpoint data file %s", h, varFileName)
	}
	jsonFileName := filepath.Join(h.config.dir, baseName+jsonNameSuffix)
	var jsonFile *os.File
	jsonFile, err = os.Create(jsonFileName)
	if err != nil {
		return errors.Wrapf(err, "%s: failed to create checkpoint metadata file %s", h, jsonFileName)
	}

	// Copy over and set variables: both from Context and previously loaded ones.
	h.serialized.Variables = make([]serializedVar, 0, h.ctx.NumVariables()+len(h.variableValues))
	pos := 0
	// * Closure to save the contents of a variable.
	saveVar := func(name string, value *tensor.Local) error {
		shape := value.Shape()
		dataRef := value.AcquireData()
		defer dataRef.Release()
		rawData := dataRef.Bytes()
		n, err := varFile.Write(rawData)
		if err != nil {
			return errors.Wrapf(err, "%s: failed to write variable %s", h, name)
		}
		if n != len(rawData) {
			return errors.Errorf("%s: failed to write variable %s -- %d bytes requested, %d bytes written", h, name, len(rawData), n)
		}
		h.serialized.Variables = append(h.serialized.Variables, serializedVar{
			ParameterName: name,
			Dimensions:    shape.Dimensions,
			DType:         shape.DType,
			Pos:           pos,
			Length:        len(rawData),
		})
		pos += len(rawData)
		return nil
	}
	// * Loop over variables in context.
	h.ctx.EnumerateVariables(func(v *context.Variable) {
		if err != nil {
			return
		}
		if h.config.excludeFromSave.Has(v) {
			return
		}
		err = saveVar(v.ParameterName(), v.Value().Local())
	})
	// * Loop over current loaded variables.
	for name, value := range h.variableValues {
		if err != nil {
			break
		}
		err = saveVar(name, value.Local())
	}
	if err != nil {
		return err
	}
	err = varFile.Close()
	if err != nil {
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

// OnStepFn implements `train.OnStepFn`, and make it convenient to attach to a training loop.
// It simply calls save.
func (h *Handler) OnStepFn(_ *train.Loop, _ []tensor.Tensor) error {
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
		varFileName := filepath.Join(h.config.dir, baseName+varDataSuffix)
		jsonFileName := filepath.Join(h.config.dir, baseName+jsonNameSuffix)
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
// with ExcludeParams).
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
// This will is called by context.Context when the variable is used for the first time.
// The user may want to use this function to inspect loaded values for testing.
func (h *Handler) LoadVariable(ctx *context.Context, v *context.Variable) (value tensor.Tensor, found bool) {
	// Priority is based on the installation order. That means we attempt first the previously configured loaders.
	if h.prevContextLoader != nil {
		value, found = h.prevContextLoader.LoadVariable(ctx, v)
		if found {
			// Previous manager found value (or issued an error), return that.
			return
		}
	}

	// Try to find variable in our currently loaded checkpoint.
	value, found = h.variableValues[v.ParameterName()]
	if !found {
		return
	}
	if !value.Shape().Eq(v.Shape()) {
		Panicf("shape requested for variable %s is different from value shape %s loaded from %s",
			v.Shape(), value.Shape(), h)
	}
	// "Consume" value, meaning remove it from Handler.
	delete(h.variableValues, v.ParameterName())
	return
}

// LoadedVariables for inspection. These are the values loaded -- but not necessarily immediately available in
// context, since they are actually used only when a model asks for the variable.
//
// The Handler owns the returned map, don't change it -- the behavior is undefined if you do.
func (h *Handler) LoadedVariables() map[string]tensor.Tensor {
	return h.variableValues
}
