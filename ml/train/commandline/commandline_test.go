package commandline

import (
	"github.com/gomlx/gomlx/ml/context"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

func createTestContext() *context.Context {
	ctx := context.New()
	ctx.SetParam("x", 11.0)
	ctx.SetParam("y", 7)
	ctx.SetParam("z", false)
	ctx.SetParam("s", "foo")
	ctx.SetParam("list_int", []int{})
	ctx.SetParam("list_float", []float64{})
	ctx.SetParam("list_str", []string{})
	return ctx
}

func TestParseContextSettings(t *testing.T) {
	ctx := createTestContext()

	require.NoError(t, ParseContextSettings(ctx,
		"x=13;a/z=true;/a/b/y=3;s=bar;list_int=1,3,7;list_float=0.1,1.2,3e3;list_str=a,b;"))
	x, found := ctx.GetParam("x")
	assert.True(t, found)
	assert.Equal(t, 13.0, x.(float64))

	y, found := ctx.GetParam("y")
	assert.True(t, found)
	assert.Equal(t, 7, y)
	y, _ = ctx.In("a").GetParam("y")
	assert.Equal(t, 7, y)
	y, _ = ctx.In("a").In("b").GetParam("y")
	assert.Equal(t, 3, y)

	z, found := ctx.GetParam("z")
	assert.True(t, found)
	assert.False(t, z.(bool))
	z, _ = ctx.In("a").GetParam("z")
	assert.True(t, z.(bool))

	s, found := ctx.GetParam("s")
	assert.True(t, found)
	assert.Equal(t, "bar", s.(string))

	assert.Equal(t, []int{1, 3, 7}, context.GetParamOr(ctx, "list_int", []int{}))
	assert.Equal(t, []float64{0.1, 1.2, 3e3}, context.GetParamOr(ctx, "list_float", []float64{}))
	assert.Equal(t, []string{"a", "b"}, context.GetParamOr(ctx, "list_str", []string{}))

	// Parameter "q" is unknown.
	require.Error(t, ParseContextSettings(ctx, "q=3"))

	// Parameter "q" is still unknown in root.
	ctx.In("c").SetParam("q", 13)
	require.Error(t, ParseContextSettings(ctx, "q=3"))

	// Cannot set the wrong type of value.
	require.Error(t, ParseContextSettings(ctx, "y=3.14"))
}
