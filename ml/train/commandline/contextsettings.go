package commandline

import (
	"encoding/json"
	"flag"
	"fmt"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/pkg/errors"
	"strings"
)

// ParseContextSettings from settings -- typically the contents of a flag set by the user.
// The settings are a list separated by ";": e.g.: "param1=value1;param2=value2;...".
//
// All the parameters "param1", "param2", etc. must be already set with default values
// in the context `ctx`. The default values are also used to set the type to which the
// string values will be parsed to.
//
// It updates `ctx` parameters accordingly, and returns an error in case a parameter
// is unknown or the parsing failed.
//
// Note, one can also provide a scope for the parameters: "layer_1/l2_regularization=0.1"
// will work, as long as a default "l2_regularization" is defined in `ctx`.
//
// For integer types, "_" is removed: it allows one to enter large numbers using it as a separator, like
// in Go. E.g.: 1_000_000 = 1000000.
//
// See example in CreateContextSettingsFlag, which will create a flag for the settings.
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
func ParseContextSettings(ctx *context.Context, settings string) error {
	settingsList := strings.Split(settings, ";")
	for _, setting := range settingsList {
		if setting == "" {
			continue
		}
		parts := strings.Split(setting, "=")
		if len(parts) != 2 {
			return errors.Errorf("can't parse settings %q: each setting requires the format \"<param>=<value>\", got %q",
				settings, setting)
		}
		paramPath, valueStr := parts[0], parts[1]
		paramPathParts := strings.Split(paramPath, context.ScopeSeparator)
		key := paramPathParts[len(paramPathParts)-1]
		value, found := ctx.GetParam(key)
		if !found {
			return errors.Errorf("can't set parameter %q because the param %q is not known in the root context",
				paramPath, key)
		}

		// Set the new parameter in the selected scope.
		ctxInScope := ctx
		if len(paramPathParts) > 1 {
			for _, part := range paramPathParts[:len(paramPathParts)-1] {
				if part == "" {
					continue
				}
				ctxInScope = ctxInScope.In(part)
			}
		}

		// Parse value accordingly.
		// Is there a better way of doing this using reflection ?
		var err error
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
		default:
			err = fmt.Errorf("don't know how to parse type %T for setting parameter %q",
				value, setting)
		}
		if err != nil {
			return errors.Wrapf(err, "failed to parse value %q for parameter %q (default value is %#v)", valueStr, paramPath, value)
		}
		ctxInScope.SetParam(key, value)
	}
	return nil
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
	parts = append(parts, "Context hyperparameters:")
	ctx.EnumerateParams(func(scope, key string, value any) {
		if scope == context.RootScope {
			parts = append(parts, fmt.Sprintf("%q: (%T) %v", key, value, value))
		} else {
			parts = append(parts, fmt.Sprintf("%q / %q: (%T) %q", scope, key, value, value))
		}
	})
	return strings.Join(parts, "\n\t")
}
