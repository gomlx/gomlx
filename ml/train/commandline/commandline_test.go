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
	return ctx
}

func TestParseContextSettings(t *testing.T) {
	ctx := createTestContext()

	require.NoError(t, ParseContextSettings(ctx,
		"x=13;a/z=true;/a/b/y=3;s=bar"))
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

	// Parameter "q" is unknown.
	require.Error(t, ParseContextSettings(ctx, "q=3"))

	// Parameter "q" is still unknown in root.
	ctx.In("c").SetParam("q", 13)
	require.Error(t, ParseContextSettings(ctx, "q=3"))

	// Cannot set the wrong type of value.
	require.Error(t, ParseContextSettings(ctx, "y=3.14"))
}
