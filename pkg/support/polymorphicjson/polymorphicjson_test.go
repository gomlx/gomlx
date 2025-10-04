package polymorphicjson_test

import (
	"encoding/json"
	"fmt"
	"testing"

	. "github.com/gomlx/gomlx/models/polymorphicjson" // Import the library being tested
)

// --- Test Interfaces (Core Business Logic) ---

// OptimizerIface is the raw interface definition used as the generic constraint (the contract).
// It embeds JSONIdentifiable to satisfy the polymorphic constraint.
type OptimizerIface interface {
	JSONIdentifiable
	Tune() string
}

// Optimizer is the final, clean, exported type the user puts in their structs.
// It wraps the interface with code to properly encode/decode to JSON.
type Optimizer struct {
	Wrapper[OptimizerIface]
}

// NewOptimizer convert an OptimizerIface to the JSON handling proxy.
func NewOptimizer(opt OptimizerIface) Optimizer {
	// This uses positional initialization for the anonymous embedded field.
	return Optimizer{Wrapper[OptimizerIface]{Value: opt}}
}

// Tune proxies the call to the underlying interface value.
func (o Optimizer) Tune() string {
	if any(o.Value) == nil {
		return "Optimizer is nil"
	}
	return o.Value.Tune()
}

// SchedulerIface is the raw interface definition used as the generic constraint (the contract).
// It embeds JSONIdentifiable to satisfy the polymorphic constraint.
type SchedulerIface interface {
	JSONIdentifiable
	Plan() string
}

// Scheduler is the final, clean, exported type for schedulers.
type Scheduler struct {
	Wrapper[SchedulerIface]
}

// NewScheduler from a SchedulerIface
func NewScheduler(opt SchedulerIface) Scheduler {
	// This uses positional initialization for the anonymous embedded field.
	return Scheduler{Wrapper[SchedulerIface]{Value: opt}}
}

// Plan proxies the call to the underlying interface value.
func (s Scheduler) Plan() string {
	if any(s.Value) == nil {
		return "Scheduler is nil"
	}
	return s.Value.Plan()
}

// --- Concrete Type 1: MyOptimizer (json_type="my") ---

// MyOptimizer implements OptimizerIface.
type MyOptimizer struct {
	LearningRate float64 `json:"learning_rate"`
}

func (o *MyOptimizer) Tune() string {
	return fmt.Sprintf("Optimizer tuned with rate %.4f", o.LearningRate)
}

func (o *MyOptimizer) JSONTags() (interfaceName, concreteType string) {
	return "OptimizerIface", "my"
}

// Constructor helper
func NewMyOptimizer(rate float64) *MyOptimizer {
	return &MyOptimizer{
		LearningRate: rate,
	}
}

func init() {
	// Register the Optimizer implementation here, near its definition.
	Register(func() OptimizerIface { return NewMyOptimizer(0) })
}

// --- Concrete Type 2: MyScheduler (json_type="my") ---

// MyScheduler implements SchedulerIface.
type MyScheduler struct {
	WarmupSteps int `json:"warmup_steps"`
}

func (s *MyScheduler) Plan() string {
	return fmt.Sprintf("Scheduler planned with %d steps", s.WarmupSteps)
}

func (s *MyScheduler) JSONTags() (interfaceName, concreteType string) {
	return "SchedulerIface", "my"
}

// Constructor helper
func NewMyScheduler(steps int) *MyScheduler {
	return &MyScheduler{
		WarmupSteps: steps,
	}
}

func init() {
	// Register the Scheduler implementation here, near its definition.
	Register(func() SchedulerIface { return NewMyScheduler(0) })
}

// ----------------------------------------------------------------------
// --- USER-FACING API DEFINITION (The Clean API Proxy) ---
// ----------------------------------------------------------------------

// --- Test Model ---

// TestModel is the user's model using the clean type aliases.
type TestModel struct {
	OptCfg   Optimizer `json:"optimizer_config"` // User sees the clean 'Optimizer'
	SchedCfg Scheduler `json:"scheduler_config"` // User sees the clean 'Scheduler'
}

// --- Actual Test Function ---

func TestPolymorphicSameJSONTypeResolution(t *testing.T) {
	originalModel := TestModel{
		OptCfg:   NewOptimizer(NewMyOptimizer(0.005)),
		SchedCfg: NewScheduler(NewMyScheduler(500)),
	}

	// 1. Marshal (Serialization)
	jsonData, err := json.MarshalIndent(originalModel, "", "  ")
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	expectedJSON := `{
  "optimizer_config": {
    "interface_name": "my",
    "concrete_type": "OptimizerIface",
    "value": {
      "learning_rate": 0.005
    }
  },
  "scheduler_config": {
    "interface_name": "my",
    "concrete_type": "SchedulerIface",
    "value": {
      "warmup_steps": 500
    }
  }
}`
	if string(jsonData) != expectedJSON {
		fmt.Printf("Expected:\n%s\nGot:\n%s\n", expectedJSON, string(jsonData))
		t.Fatal("Marshaled JSON mismatch.")
	}

	// 2. Unmarshal (Deserialization)
	var loadedModel TestModel
	err = json.Unmarshal(jsonData, &loadedModel)
	if err != nil {
		t.Fatalf("Unmarshal failed: %+v", err)
	}

	// 3. Validation: Test the clean API access using the proxy methods
	if loadedModel.OptCfg.Tune() != "Optimizer tuned with rate 0.0050" {
		t.Errorf("Optimizer method call failed. Expected 'Optimizer tuned with rate 0.0050', Got '%s'", loadedModel.OptCfg.Tune())
	}
	if loadedModel.SchedCfg.Plan() != "Scheduler planned with 500 steps" {
		t.Errorf("Scheduler method call failed. Expected 'Scheduler planned with 500 steps', Got '%s'", loadedModel.SchedCfg.Plan())
	}

	// Final check on the underlying concrete type
	if _, ok := loadedModel.OptCfg.Value.(*MyOptimizer); !ok {
		t.Errorf("Optimizer did not unmarshal to *MyOptimizer, got %T", loadedModel.OptCfg.Value)
	}
	if _, ok := loadedModel.SchedCfg.Value.(*MyScheduler); !ok {
		t.Errorf("Scheduler did not unmarshal to *MyScheduler, got %T", loadedModel.SchedCfg.Value)
	}
}
