package commandline

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"slices"
	"strings"

	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/pkg/support/fsutil"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/pkg/errors"
)

// ParseContextSettings from settings -- typically the contents of a flag set by the user.
// The settings are a list separated by ";": e.g.: "param1=value1;param2=value2;...".
//
// All the parameters "param1", "param2", etc. must be already set with default values
// in the context `ctx`. The default values are also used to set the type to which the
// string values will be parsed to.
//
// It updates `ctx` parameters accordingly and returns an error in case a parameter
// is unknown or the parsing failed.
//
// Note, one can also provide a scope for the parameters: "layer_1/l2_regularization=0.1"
// will work, as long as a default "l2_regularization" is defined in `ctx`.
//
// For integer types, "_" is removed: it allows one to enter large numbers using it as a separator, like
// in Go. E.g.: 1_000_000 = 1000000.
//
// See the example in CreateContextSettingsFlag, which will create a flag for the settings.
//
// Example usage:
//
//	func main() {
//		ctx := createDefaultContext()
//		settings := commandline.CreateContextSettingsFlag(ctx, "")
//		flag.Parse()
//		err := commandline.ParseContextSettings(ctx, *settings)
//		if err != nil { panic(err) }
//		fmt.Println(commandline.SprintContextSettings(ctx))
//		...
//	}
func ParseContextSettings(ctx *context.Context, settings string) (paramsSet []string, err error) {
	settingsList := strings.Split(settings, ";")
	for _, setting := range settingsList {
		paramsSet, err = parseContextSetting(ctx, setting, paramsSet)
		if err != nil {
			return
		}
	}
	return
}

func parseContextSetting(ctx *context.Context, setting string, paramsSet []string) (newParamsSet []string, err error) {
	newParamsSet = paramsSet
	if setting == "" {
		return
	}
	if strings.HasPrefix(setting, "file:") {
		// Read parameters from a file.
		filePath := strings.TrimPrefix(setting, "file:")
		filePath = fsutil.MustReplaceTildeInDir(filePath)
		var contents []byte
		contents, err = os.ReadFile(filePath)
		if err != nil {
			err = errors.Wrapf(err, "failed to read settings from file %q", filePath)
			return
		}
		lines := strings.Split(string(contents), "\n")
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			settings := strings.Split(line, ";")
			for _, setting := range settings {
				newParamsSet, err = parseContextSetting(ctx, setting, newParamsSet)
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
		return
	}
	paramPath, valueStr := parts[0], parts[1]
	paramScope, paramName := context.SplitScope(paramPath)
	if strings.Index(paramName, context.ScopeSeparator) != -1 {
		err = errors.Errorf("can't set parameter %q  because some scope is set, but it is not absolue (it does not start with %q)",
			paramPath, context.ScopeSeparator)
		return
	}
	value, found := ctx.GetParam(paramName)
	if !found {
		err = errors.Errorf("can't set parameter %q (scope=%q)  because the param %q is not known in the root context",
			paramPath, paramScope, paramName)
		return
	}

	// Set the new parameter in the selected scope.
	ctxInScope := ctx
	if paramScope != "" {
		ctxInScope = ctxInScope.InAbsPath(paramScope)
	}

	// Parse value accordingly.
	// Is there a better way of doing this using reflection?
	switch v := value.(type) {
	case int:
		valueStr = strings.Replace(valueStr, "_", "", -1)
		err = json.Unmarshal([]byte(valueStr), &v)
		value = v
	case int32:
		valueStr = strings.Replace(valueStr, "_", "", -1)
		err = json.Unmarshal([]byte(valueStr), &v)
		value = v
	case int64:
		valueStr = strings.Replace(valueStr, "_", "", -1)
		err = json.Unmarshal([]byte(valueStr), &v)
		value = v
	case uint:
		valueStr = strings.Replace(valueStr, "_", "", -1)
		err = json.Unmarshal([]byte(valueStr), &v)
		value = v
	case uint32:
		valueStr = strings.Replace(valueStr, "_", "", -1)
		err = json.Unmarshal([]byte(valueStr), &v)
		value = v
	case uint64:
		valueStr = strings.Replace(valueStr, "_", "", -1)
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
			str = strings.Replace(str, "_", "", -1)
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
		return
	}
	ctxInScope.SetParam(paramName, value)
	newParamsSet = append(newParamsSet, paramPath)
	return
}

// CreateContextSettingsFlag create a string flag with the given flagName (if empty it will be named
// "set") and with a description of the current defined parameters in the context `ctx`.
//
// The flag should be created before the call to `flags.Parse()`.
//
// Example usage:
//
//	func main() {
//		ctx := createDefaultContext()
//		settings := commandline.CreateContextSettingsFlag(ctx, "")
//		flag.Parse()
//		err := commandline.ParseContextSettings(ctx, *settings)
//		if err != nil { panic(err) }
//		fmt.Println(commandline.SprintContextSettings(ctx))
//		...
//	}
func CreateContextSettingsFlag(ctx *context.Context, flagName string) *string {
	if flagName == "" {
		flagName = "set"
	}
	var parts []string
	parts = append(parts, fmt.Sprintf(
		`Set context parameters defining the model. `+
			`It should be a list of elements "param=value" separated by ";". `+
			`Scoped settings are allowed, by using %q to separated scopes. `+
			`It can also be given an entry like: "file:settings_file.txt", in `+
			`which case the file will be read and the settings will be parsed, `+
			`with new-lines working as ";" to separate settings and lines starting with "#" are considered comments. `+
			`Current available parameters that can be set:`,
		context.ScopeSeparator))
	ctx.EnumerateParams(func(scope, key string, value any) {
		if scope != context.RootScope {
			return
		}
		parts = append(parts, fmt.Sprintf("%q: default value is %v", key, value))
	})
	usage := strings.Join(parts, "\n")
	var settings string
	flag.StringVar(&settings, flagName, "", usage)
	return &settings
}

// SprintContextSettings pretty-print values for the current hyperparameters settings into a string.
func SprintContextSettings(ctx *context.Context) string {
	var parts []string
	ctx.EnumerateParams(func(scope, key string, value any) {
		if scope == context.RootScope {
			scope = ""
		}
		parts = append(parts, fmt.Sprintf("\t\"%s/%s\": (%T) %v", scope, key, value, value))
	})
	return strings.Join(parts, "\n")
}

func SprintModifiedContextSettings(ctx *context.Context, paramsSet []string) string {
	var parts []string
	paramsSet = slices.Clone(paramsSet)
	slices.Sort(paramsSet)
	var lastParamPath string
	for _, paramPath := range paramsSet {
		// Remove duplicates.
		if paramPath == lastParamPath {
			continue
		}
		lastParamPath = paramPath
		paramScope, paramName := context.SplitScope(paramPath)
		if paramScope == "" {
			paramScope = context.RootScope
		}
		value, found := ctx.InAbsPath(paramScope).GetParam(paramName)
		if !found {
			continue
		}
		parts = append(parts, fmt.Sprintf("\t%q: (%T) %v", paramPath, value, value))
	}
	return strings.Join(parts, "\n")
}
