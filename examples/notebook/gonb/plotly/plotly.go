// Package plotly uses GoNB plotly support (`github.com/janpfeifer/gonb/gonbui/plotly`) to plot both
// on dynamic plots while training (see [DynamicPlot]) and to quickly plot the results of a previously
// saved plot results in a checkpoints directory with [StaticPlot].
//
// In either case it allows adding baseline plots of previous checkpoints.
//
// The advantage of `plotly` over `margaid` plots is that it uses Javascript to make the plot interactive (it displays
// information on mouse hover).
//
// The disadvantage is that saving doesn't work, because of the javascript nature.
package plotly
