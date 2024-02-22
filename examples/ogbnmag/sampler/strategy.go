package sampler

import (
	"fmt"
	"strings"
)

// Strategy is created by [Sampler]. A [Sampler] can create multiple [Strategy]s, a typical
// example is creating one for training, one for validation and one for testing.
//
// After creation (see [Sampler.NewStrategy]), one defines what and how to sample a subgraph,
// by creating "rules" ([Rule]) that will translate to sampled nodes.
//
// Once the strategy is defined, it can be used to create one or more datasets -- and after datasets are created,
// the strategy can no longer be changed.
type Strategy struct {
	sampler *Sampler
	frozen  bool // If set to true, it can no longer be modified.

	rules map[string]*Rule
}

// String returns a multi-line informative description of the strategy.
func (st *Strategy) String() string {
	parts := make([]string, 0, 1+len(st.rules))
	var frozenDesc string
	if st.frozen {
		frozenDesc = ", Frozen"
	}
	parts = append(parts, fmt.Sprintf("Sampling strategy: (%d rules%s)", len(st.rules), frozenDesc))
	for _, rule := range st.rules {
		if !rule.IsNode() {
			// These will be included under their root nodes.
			continue
		}
		parts = appendRulesRecursively(parts, rule, 1)
	}
	return strings.Join(parts, "\n")
}

// appendRulesRecursively enumerates rule descriptions (strings) recursively.
func appendRulesRecursively(parts []string, rule *Rule, indent int) []string {
	parts = append(parts, fmt.Sprintf("%s%s", strings.Repeat("\t", indent), rule))
	indent++
	for _, subRule := range rule.dependents {
		parts = appendRulesRecursively(parts, subRule, indent)
	}
	return parts
}
