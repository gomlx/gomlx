// Package models provide helpers to train, execute (inference), save and load models with their weights and hyperparameters.
//
// Note: This package aims to replace the `ml/context` package and all the layers libraries.
// **It's still in beta**. Any feedback is very welcome.
//
// And this package defines:
//
//   - Variable: holds the model weights. It has a "concrete" view, with the actual value as a Tensor, and a graph node view
//     that can be used and updated during the building of the graph.
//   - Exec: it mimics the graph.Exec interface to execute computation graphs, but it also handles Variable objects: they are
//     automatically passed as side inputs to the graph when used, and side outputs to automatically update their values, if
//     they are updated in the graph.
//
// It hinges on the following abstraction of "model", which can be any user-defined struct (`any` type in Go), which can contain:
//
//   - Fields with static "hyperparameters" (e.g.: learning rate, batch size, number of layers, etc.)
//   - Fields of the type *Variable with model's weights (trainable or not).
//   - Slices, arrays, sub-structs, maps (with string or number keys) of "model" (a recursive definition).
//
// Example: A model that has a counter, and an increment function.
//
//	myModel := &struct{
//		counter *models.Variable
//	} {
//		counter: must.M1(models.VariableWithValue("counter", int32(0))),
//	}
//	incFn := func(g *graph.Graph) *graph.Node {
//		currentValue := myModel.counter.ValueGraph(g)
//		nextValue := graph.AddScalar(currentValue, 1)
//		myModel.counter.SetValueGraph(nextValue)  // Updates the counter.
//		return currentValue
//	}
//	incExec := must.M1(models.NewExec(backend, incFn))  // Executor that increments the counter.
//	incExec.Call1() // -> 0
//	incExec.Call1() // -> 1
//	fmt.Printf("current myModel state: %s\n", myModel.counter.Value()) // -> 2
//	models.Checkpoint(myModel, "~/work/my_counter")
package exec

// Generate ExecFnSet interface, and a conversion tool to a canonical form.
//go:generate go run ../../../internal/cmd/builderiface/
