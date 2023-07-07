package diffusion

import (
	"bytes"
	"fmt"
	flowers "github.com/gomlx/gomlx/examples/oxfordflowers102"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/metrics"
	"github.com/gomlx/gomlx/models/inceptionv3"
	"github.com/gomlx/gomlx/types/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/types/tensor"
	timage "github.com/gomlx/gomlx/types/tensor/image"
	"github.com/janpfeifer/gonb/gonbui"
	"image"
	"io"
	"math"
	"math/rand"
	"os"
	"path"
	"strings"
	"text/template"
)

// PlotImagesTensor plots images in tensor format, all in one row.
// It assumes image's MaxValue of 255.
//
// This only works in a Jupyter (GoNB kernel) notebook.
func PlotImagesTensor(imagesT tensor.Tensor) {
	if gonbui.IsNotebook {
		images := MustNoError(timage.ToImage().MaxValue(255.0).Batch(imagesT))
		PlotImages(images)
	}
}

// PlotImages all in one row. The image size in the HTML is set to the value given.
//
// This only works in a Jupyter (GoNB kernel) notebook.
func PlotImages(images []image.Image) {
	if gonbui.IsNotebook {
		var parts []string
		for _, img := range images {
			imgSrc, err := gonbui.EmbedImageAsPNGSrc(img)
			AssertNoError(err)
			parts = append(parts, fmt.Sprintf(`<img src="%s">`, imgSrc))
		}
		//gonbui.DisplayHTML(fmt.Sprintf(
		//	"<table style=\"overflow-x: auto\"><tr><td>%s</td></tr></table>", strings.Join(parts, "\n</td><td>\n")))
		gonbui.DisplayHTML(fmt.Sprintf(
			"<div style=\"overflow-x: auto\">\n\t%s</div>\n", strings.Join(parts, "\n\t")))
	}
}

// PlotModelEvolution plots the saved sampled generated images of a model in the current configured checkpoint.
//
// It outputs at most imagesPerSample per checkpoint sampled.
func PlotModelEvolution(imagesPerSample int, animate bool) {
	if !gonbui.IsNotebook {
		return
	}
	Init()
	modelDir := data.ReplaceTildeInDir(*flagCheckpoint)
	if !path.IsAbs(modelDir) {
		modelDir = path.Join(DataDir, modelDir)
	}
	entries := MustNoError(os.ReadDir(modelDir))
	var generatedFiles []string
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		fileName := entry.Name()
		if !strings.HasPrefix(fileName, GeneratedSamplesPrefix) || !strings.HasSuffix(fileName, ".tensor") {
			continue
		}
		generatedFiles = append(generatedFiles, fileName)
	}

	if len(generatedFiles) == 0 {
		gonbui.DisplayHTML(fmt.Sprintf("<b>No generated samples in <pre>%s</pre>.</b>", modelDir))
		return
	}

	gonbui.DisplayHTML(fmt.Sprintf("<b>Generated samples in <pre>%s</pre>.</b>", modelDir))
	if !animate {
		// Simply display all images:
		for _, generatedFile := range generatedFiles {
			imagesT := MustNoError(tensor.Load(path.Join(modelDir, generatedFile)))
			images := MustNoError(timage.ToImage().MaxValue(255.0).Batch(imagesT))
			images = images[:imagesPerSample]
			PlotImages(images)
		}
		return
	}

	params := struct {
		Id              string
		Images          [][]string
		FrameRateMs     int
		Size, Width     int
		ImagesPerSample int
	}{
		Id:              fmt.Sprintf("%X", rand.Int63()),
		Images:          make([][]string, len(generatedFiles)),
		FrameRateMs:     200,
		Size:            ImageSize,
		Width:           imagesPerSample * ImageSize,
		ImagesPerSample: imagesPerSample,
	}
	for ii, generatedFile := range generatedFiles {
		imagesT := MustNoError(tensor.Load(path.Join(modelDir, generatedFile)))
		images := MustNoError(timage.ToImage().MaxValue(255.0).Batch(imagesT))
		images = images[:imagesPerSample]
		params.Images[ii] = slices.Map(images[:imagesPerSample], func(img image.Image) string {
			//return fmt.Sprintf("timestep_%d", ii)
			return MustNoError(gonbui.EmbedImageAsPNGSrc(img))
		})
	}

	var jsTemplate = MustNoError(template.New("PlotModelEvolution").Parse(`
	<canvas id="canvas_{{.Id}}" height="{{.Size}}px" width="{{.Width}}px"></canvas>
	<script>
	var canvas_{{.Id}} = document.getElementById("canvas_{{.Id}}"); 
	var ctx_{{.Id}} = canvas_{{.Id}}.getContext("2d"); 
	var currentFrame_{{.Id}} = 0;
	var frameRate_{{.Id}} = {{.FrameRateMs}};
	var imagePaths_{{.Id}} = [{{range .Images}}
	[
		{{range .}}"{{.}}",{{end}}
	],
{{end}}
];
	// Load images
	var images_{{.Id}} = [];
	for (var ii = 0; ii < imagePaths_{{.Id}}.length; ii++) {
		var images = [];
		for (var jj = 0; jj < {{.ImagesPerSample}}; jj++) {
			var image = new Image();
			image.src = imagePaths_{{.Id}}[ii][jj];
			images.push(image);
		}
		images_{{.Id}}.push(images);
	}
	
	function animate_{{.Id}}() {
		var ctx = ctx_{{.Id}};
		var canvas = canvas_{{.Id}};
		ctx.clearRect(0, 0, canvas.width, canvas.height);
		var images = images_{{.Id}}[currentFrame_{{.Id}}];
		for (var jj = 0; jj < {{.ImagesPerSample}}; jj++) {
			ctx.drawImage(images[jj], jj*{{.Size}}, 0);
		}
		currentFrame_{{.Id}} = (currentFrame_{{.Id}} + 1) % images_{{.Id}}.length;
		var timeout = frameRate_{{.Id}};
		if (currentFrame_{{.Id}} == 0) {
			timeout = 3000;
		}
		setTimeout(animate_{{.Id}}, timeout);
	}
	animate_{{.Id}}();
	</script>
`))
	var buf bytes.Buffer
	AssertNoError(jsTemplate.Execute(&buf, params))
	gonbui.DisplayHTML(buf.String())
}

// DenoiseStepGraph executes one step of separating the noise and images from noisy images.
func DenoiseStepGraph(ctx *context.Context, noisyImages, diffusionTime, nextDiffusionTime, flowerIds *Node) (
	predictedImages, nextNoisyImages *Node) {
	numImages := noisyImages.Shape().Dimensions[0]
	diffusionTimes := BroadcastToDims(ConvertType(diffusionTime, DType), numImages, 1, 1, 1)
	signalRatios, noiseRatios := DiffusionSchedule(diffusionTimes, false)
	var predictedNoises *Node
	predictedImages, predictedNoises = Denoise(ctx, noisyImages, signalRatios, noiseRatios, flowerIds)

	nextDiffusionTimes := BroadcastToDims(ConvertType(nextDiffusionTime, DType), numImages, 1, 1, 1)
	nextSignalRatios, nextNoiseRatios := DiffusionSchedule(nextDiffusionTimes, false)
	nextNoisyImages = Add(
		Mul(predictedImages, nextSignalRatios),
		Mul(predictedNoises, nextNoiseRatios))
	return
}

// GenerateImages using reverse diffusion. If displayEveryNSteps is not 0, it will display
// intermediary results every n results -- it also displays the initial noise and final image.
//
// Plotting results only work if in a Jupyter (with GoNB kernel) notebook.
func GenerateImages(numImages int, numDiffusionSteps int, displayEveryNSteps int) (predictedImages tensor.Tensor) {
	ctx := context.NewContext(manager).Checked(false)
	_, _, _ = LoadCheckpointToContext(ctx)
	ctx.RngStateReset()
	noise := GenerateNoise(numImages)
	flowerIds := GenerateFlowerIds(numImages)
	generator := NewImagesGenerator(ctx, noise, flowerIds, numDiffusionSteps, displayEveryNSteps)
	return generator.Generate()
}

// GenerateImagesOfFlowerType is similar to GenerateImages, but it limits itself to generating images of only one
// flower type.
func GenerateImagesOfFlowerType(numImages int, flowerType int32, numDiffusionSteps int) (predictedImages tensor.Tensor) {
	ctx := context.NewContext(manager).Checked(false)
	_, _, _ = LoadCheckpointToContext(ctx)
	ctx.RngStateReset()
	noise := GenerateNoise(numImages)
	flowerIds := tensor.FromValue(slices.SliceWithValue(numImages, flowerType))
	generator := NewImagesGenerator(ctx, noise, flowerIds, numDiffusionSteps, 0)
	return generator.Generate()
}

// GenerateImagesOfAllFlowerTypes takes one random noise, and generate the flower for each of the 102 types.
func GenerateImagesOfAllFlowerTypes(numDiffusionSteps int) (predictedImages tensor.Tensor) {
	numImages := flowers.NumLabels
	ctx := context.NewContext(manager).Checked(false)
	_, _, _ = LoadCheckpointToContext(ctx)
	ctx.RngStateReset()
	noise := NewExec(manager, func(g *Graph) *Node {
		state := Const(g, RngState())
		_, noise := RandomNormal(state, shapes.Make(DType, 1, ImageSize, ImageSize, 3))
		noise = BroadcastToDims(noise, numImages, ImageSize, ImageSize, 3)
		return noise
	}).Call()[0]
	flowerIds := tensor.FromValue(slices.Iota(int32(0), numImages))
	generator := NewImagesGenerator(ctx, noise, flowerIds, numDiffusionSteps, 0)
	return generator.Generate()
}

type ImagesGenerator struct {
	ctx                                   *context.Context
	noise, flowerIds                      tensor.Tensor
	numImages                             int
	numDiffusionSteps, displayEveryNSteps int
	denormalizerExec                      *Exec
	diffusionStepExec                     *context.Exec
}

func NewImagesGenerator(ctx *context.Context, noise, flowerIds tensor.Tensor, numDiffusionSteps, displayEveryNSteps int) *ImagesGenerator {
	ctx = ctx.Reuse()
	return &ImagesGenerator{
		ctx:                ctx,
		noise:              noise,
		flowerIds:          flowerIds,
		numImages:          noise.Shape().Dimensions[0],
		numDiffusionSteps:  numDiffusionSteps,
		displayEveryNSteps: displayEveryNSteps,
		denormalizerExec:   NewExec(manager, DenormalizeImages),
		diffusionStepExec:  context.NewExec(manager, ctx, DenoiseStepGraph),
	}
}

// Generate images from the original noise.
//
// It can be called multiple times if the context changed, if the model was further trained.
// Otherwise, it will always return the same images.
func (g *ImagesGenerator) Generate() (predictedImages tensor.Tensor) {
	if g.displayEveryNSteps > 0 {
		fmt.Printf("GenerateImages(%d images, %d steps): noise.shape=%s\n", g.numImages, g.numDiffusionSteps, g.noise.Shape())
		fmt.Printf("\tModel #params:\t%d\n", g.ctx.NumParameters())
		fmt.Printf("\t Model memory:\t%s\n", data.ByteCountIEC(g.ctx.Memory()))
	}

	noisyImages := g.noise
	if g.displayEveryNSteps > 0 {
		gonbui.DisplayHTML("<p><b>Noise</b></p>")
		PlotImagesTensor(noisyImages)
	}

	stepSize := 1.0 / float64(g.numDiffusionSteps)
	for step := 0; step < g.numDiffusionSteps; step++ {
		diffusionTime := 1.0 - float64(step)*stepSize
		nextDiffusionTime := math.Max(diffusionTime-stepSize, 0)
		parts := g.diffusionStepExec.Call(noisyImages, diffusionTime, nextDiffusionTime, g.flowerIds)
		if predictedImages != nil {
			predictedImages.FinalizeAll() // Immediate release of (GPU) memory for intermediary results.
		}
		if noisyImages != nil && step > 0 {
			noisyImages.FinalizeAll() // Immediate release of (GPU) memory for intermediary results.
		}
		predictedImages, noisyImages = parts[0], parts[1]
		if g.displayEveryNSteps > 0 && step%g.displayEveryNSteps == 0 {
			displayImages := g.denormalizerExec.Call(predictedImages)[0]
			gonbui.DisplayHTML(fmt.Sprintf("<p><b>Images @ step=%d, diffusion_time=%.3f</b></p>", step, diffusionTime))
			PlotImagesTensor(displayImages)
		}
	}
	predictedImages = g.denormalizerExec.Call(predictedImages)[0]
	if g.displayEveryNSteps > 0 {
		gonbui.DisplayHTML("<p><b>Final Images</b></p>")
		PlotImagesTensor(predictedImages)
	}
	return
}

// GenerateNoise generates random noise that can be used to generate images.
func GenerateNoise(numImages int) tensor.Tensor {
	return NewExec(manager, func(g *Graph) *Node {
		state := Const(g, RngState())
		_, noise := RandomNormal(state, shapes.Make(DType, numImages, ImageSize, ImageSize, 3))
		return noise
	}).Call()[0]
}

// GenerateFlowerIds generates random flower ids: this is the type of flowers, one of the 102.
func GenerateFlowerIds(numImages int) tensor.Tensor {
	flowerIds := make([]int32, numImages)
	for ii := range flowerIds {
		flowerIds[ii] = int32(rand.Intn(flowers.NumLabels))
	}
	return tensor.FromValue(flowerIds)
}

// KidGenerator generates the [Kernel Inception Distance (KID)](https://arxiv.org/abs/1801.01401) metric.
type KidGenerator struct {
	ctxGenerator, ctxInceptionV3 *context.Context
	ds                           train.Dataset
	generator                    *ImagesGenerator
	kid                          metrics.Interface
	evalExec                     *context.Exec
}

// NewKidGenerator allows to generate the Kid metric.
// The ctx passed is the context for the diffusion model.
// It uses a different context for the InceptionV3 KID metric, so that it's weights are not included
// in the generator model.
func NewKidGenerator(ctx *context.Context, evalDS train.Dataset, numDiffusionStep int) *KidGenerator {
	Init()
	ctx = ctx.Checked(false)
	noise := GenerateNoise(EvalBatchSize)
	flowerIds := GenerateFlowerIds(EvalBatchSize)

	i3Path := path.Join(DataDir, "inceptionV3")
	AssertNoError(inceptionv3.DownloadAndUnpackWeights(i3Path))
	kg := &KidGenerator{
		ctxGenerator:   ctx,
		ctxInceptionV3: context.NewContext(manager).Checked(false),
		ds:             evalDS,
		generator:      NewImagesGenerator(ctx, noise, flowerIds, numDiffusionStep, 0),
		kid:            inceptionv3.KidMetric(i3Path, inceptionv3.MinimumImageSize, 255.0, timage.ChannelsLast),
	}
	kg.evalExec = context.NewExec(manager, kg.ctxInceptionV3, kg.EvalStepGraph)
	return kg
}

func (kg *KidGenerator) EvalStepGraph(ctx *context.Context, allImages []*Node) (metric *Node) {
	g := allImages[0].Graph()
	ctx.SetTraining(g, false) // Some layers behave differently in train/eval.

	// Get metrics and updates: the generated images are the inputs, and the
	generatedImages := allImages[0]
	datasetImages := PreprocessImages(allImages[1], false)
	metric = kg.kid.UpdateGraph(ctx, []*Node{datasetImages}, []*Node{generatedImages})
	return
}

func (kg *KidGenerator) Eval() (metric tensor.Tensor) {
	kg.ds.Reset()
	kg.kid.Reset(kg.ctxInceptionV3)
	generatedImages := kg.generator.Generate()
	count := 0
	for {
		_, inputs, _, err := kg.ds.Yield()
		if err == io.EOF {
			break
		}
		count++

		datasetImages := inputs[0]
		if metric != nil {
			metric.FinalizeAll()
		}
		metric = kg.evalExec.Call(generatedImages, datasetImages)[0]
	}
	if count == 0 {
		exceptions.Panicf("evaluation dataset %s yielded no batches, no data to evaluate KID", kg.ds)
	}
	return
}
