// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package commandline

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/gomlx/compute/support/sets"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/support/fsutil"
	"github.com/gomlx/gomlx/support/xslices"
	"github.com/pkg/errors"
)

// ParseContextSettings from settings -- typically the contents of a flag set by the user.
// The settings are a list separated by ";": e.g.: "param1=value1;param2=value2;...".
//
// All the parameters "param1", "param2", etc. must be already set with default values
// in the context `scope`. The default values are also used to set the type to which the
// string values will be parsed to.
//
// It updates `scope` parameters accordingly and returns an error in case a parameter
// is unknown or the parsing failed.
//
// Note, one can also provide a scope for the parameters: "layer_1/l2_regularization=0.1"
// will work, as long as a default "l2_regularization" is defined in `scope`.
//
// For integer types, "_" is removed: it allows one to enter large numbers using it as a separator, like
// in Go. E.g.: 1_000_000 = 1000000.
//
// See the example in CreateContextSettingsFlag, which will create a flag for the settings.
//
// Example usage:
//
//	func main() {
//		store := createDefaultModelStore()
//		settings := commandline.CreateContextSettingsFlag(store.RootScope(), "")
//		flag.Parse()
//		err := commandline.ParseContextSettings(store.RootScope(), *settings)
//		if err != nil { panic(err) }
//		fmt.Println(commandline.SprintContextSettings(scope))
//		...
//	}
func ParseSettings(store *model.Store, settings string) (modifiedParams []string, err error) {
	settingsList := strings.SplitSeq(settings, ";")
	for setting := range settingsList {
		modifiedParams, err = parseSetting(store, setting, modifiedParams)
		if err != nil {
			return
		}
	}
	return
}

func parseSetting(store *model.Store, setting string, modifiedParams []string) (newModifiedParams []string, err error) {
	newModifiedParams = modifiedParams
	if setting == "" {
		return
	}
	if after, ok := strings.CutPrefix(setting, "file:"); ok {
		// Read parameters from a file.
		filePath := after
		filePath, err = fsutil.ReplaceTildeInDir(filePath)
		if err != nil {
			err = errors.Wrapf(err, "failed to replace tilde in file path %q", filePath)
			return
		}
		var contents []byte
		contents, err = os.ReadFile(filePath)
		if err != nil {
			err = errors.Wrapf(err, "failed to read settings from file %q", filePath)
			return
		}
		lines := strings.SplitSeq(string(contents), "\n")
		for line := range lines {
			line = strings.TrimSpace(line)
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			settings := strings.SplitSeq(line, ";")
			for setting := range settings {
				newModifiedParams, err = parseSetting(store, setting, newModifiedParams)
				if err != nil {
					return
				}
			}
		}
		return
	}

	parts := strings.Split(setting, "=")
	if len(parts) != 2 {
		err = errors.Errorf("can't parse settings %q: each setting requires the format \"<param>=<value>\", got %q",
			setting, setting)
		return nil, err
	}
	paramPath, valueStr := parts[0], parts[1]
	paramName := model.BasePath(paramPath)
	if paramName == "" {
		return nil, errors.Errorf("can't set parameter named %q (full path=%q) because it's name is empty",
			paramName, paramPath)
	}
	value, found := store.GetParam(paramName)
	if !found {
		err = errors.Errorf(
			"can't set parameter %q (full path=%q) because the base parameter %q is not known in the root scope",
			paramPath, paramPath, paramName)
		return nil, err
	}

	// Parse value accordingly.
	// Is there a better way of doing this using reflection?
	switch v := value.(type) {
	case int:
		valueStr = strings.ReplaceAll(valueStr, "_", "")
		err = json.Unmarshal([]byte(valueStr), &v)
		value = v
	case int32:
		valueStr = strings.ReplaceAll(valueStr, "_", "")
		err = json.Unmarshal([]byte(valueStr), &v)
		value = v
	case int64:
		valueStr = strings.ReplaceAll(valueStr, "_", "")
		err = json.Unmarshal([]byte(valueStr), &v)
		value = v
	case uint:
		valueStr = strings.ReplaceAll(valueStr, "_", "")
		err = json.Unmarshal([]byte(valueStr), &v)
		value = v
	case uint32:
		valueStr = strings.ReplaceAll(valueStr, "_", "")
		err = json.Unmarshal([]byte(valueStr), &v)
		value = v
	case uint64:
		valueStr = strings.ReplaceAll(valueStr, "_", "")
		err = json.Unmarshal([]byte(valueStr), &v)
		value = v
	case float64:
		err = json.Unmarshal([]byte(valueStr), &v)
		value = v
	case float32:
		err = json.Unmarshal([]byte(valueStr), &v)
		value = v
	case bool:
		err = json.Unmarshal([]byte(valueStr), &v)
		value = v
	case string:
		value = valueStr
	case []string:
		value = strings.Split(valueStr, ",")
	case []int:
		parts := strings.Split(valueStr, ",")
		value = xslices.Map(parts, func(str string) int {
			var asInt int
			str = strings.ReplaceAll(str, "_", "")
			newErr := json.Unmarshal([]byte(str), &asInt)
			if newErr != nil {
				err = newErr
			}
			return asInt
		})
	case []float64:
		parts := strings.Split(valueStr, ",")
		value = xslices.Map(parts, func(str string) float64 {
			var asNum float64
			newErr := json.Unmarshal([]byte(str), &asNum)
			if newErr != nil {
				err = newErr
			}
			return asNum
		})
	default:
		err = fmt.Errorf("don't know how to parse type %T for setting parameter %q -- it's easy to write a parser to a new type, ask in github if you need something standard",
			value, setting)
	}
	if err != nil {
		err = errors.Wrapf(err, "failed to parse value %q for parameter %q (default value is %#v)", valueStr, paramPath, value)
		return nil, err
	}
	store.SetParam(paramPath, value)
	newModifiedParams = append(newModifiedParams, paramPath)
	return
}

// CreateSettingsFlag create a string flag with the given flagName (if empty it will be named
// "set") and with a description of the current defined parameters in the context `ctx`.
//
// The flag should be created before the call to `flags.Parse()`.
//
// Example usage:
//
//	func main() {
//		store := createModelStore()
//		settings := commandline.CreateSettingsFlag(store, "")
//		flag.Parse()
//		err := commandline.ParseSettings(store, *settings)
//		if err != nil { panic(err) }
//		fmt.Println(commandline.SprintSettings(ctx))
//		...
//	}
func CreateSettingsFlag(store *model.Store, flagName string) *string {
	if flagName == "" {
		flagName = "set"
	}
	var parts []string
	parts = append(parts, fmt.Sprintf(
		`Set hyperparameters defining the model. `+
			`It should be a list of elements "param=value" separated by ";". `+
			`Scoped settings are allowed, by using %q to separated scopes. `+
			`It can also be given an entry like: "file:settings_file.txt", in `+
			`which case the file will be read and the settings will be parsed, `+
			`with new-lines working as ";" to separate settings and lines starting with "#" are considered comments. `+
			`Current available parameters that can be set:`,
		model.ScopeSeparator))
	for paramPath, value := range store.IterParams() {
		scope, name := model.SplitPath(paramPath)
		if scope != model.RootScopePath && scope != "" {
			continue
		}
		parts = append(parts, fmt.Sprintf("%q: default value is %v", name, value))
	}
	usage := strings.Join(parts, "\n")
	var settings string
	flag.StringVar(&settings, flagName, "", usage)
	return &settings
}

// SprintSettings pretty-print values for the current hyperparameters settings into a string.
func SprintSettings(storeOrScope model.StoreProvider) string {
	store := storeOrScope.Store()
	var parts []string
	for paramPath, value := range store.IterParams() {
		parts = append(parts, fmt.Sprintf("\t%q: (%T) %v", paramPath, value, value))
	}
	return strings.Join(parts, "\n")
}

// SprintModifiedSettings pretty-print values of the modified settings into a string.
func SprintModifiedSettings(storeOrScope model.StoreProvider, modifiedParams []string) string {
	store := storeOrScope.Store()
	var parts []string
	paramsSet := sets.MakeWith(modifiedParams...)
	for _, paramPath := range xslices.SortedKeys(paramsSet) {
		value, found := store.GetParam(paramPath)
		if !found {
			continue
		}
		parts = append(parts, fmt.Sprintf("\t%q: (%T) %v", paramPath, value, value))
	}
	return strings.Join(parts, "\n")
}
