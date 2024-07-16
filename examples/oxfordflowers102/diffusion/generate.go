package diffusion

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"github.com/gomlx/exceptions"
	flowers "github.com/gomlx/gomlx/examples/oxfordflowers102"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/metrics"
	"github.com/gomlx/gomlx/models/inceptionv3"
	"github.com/gomlx/gomlx/types/shapes"
	timage "github.com/gomlx/gomlx/types/tensor/image"
	"github.com/janpfeifer/gonb/cache"
	"github.com/janpfeifer/gonb/common"
	"github.com/janpfeifer/gonb/gonbui"
	"github.com/janpfeifer/gonb/gonbui/dom"
	"github.com/janpfeifer/gonb/gonbui/widgets"
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
func PlotImagesTensor(imagesT tensors.Tensor) {
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
		gonbui.DisplayHTML(ImagesToHtml(images))
	}
}

// ImagesToHtml converts slice of images to a list of images side-by-side in HTML format,
// that can be easily displayed.
func ImagesToHtml(images []image.Image) string {
	var parts []string
	for _, img := range images {
		imgSrc, err := gonbui.EmbedImageAsPNGSrc(img)
		AssertNoError(err)
		parts = append(parts, fmt.Sprintf(`<img src="%s">`, imgSrc))
	}
	return fmt.Sprintf(
		"<div style=\"overflow-x: auto\">\n\t%s</div>\n", strings.Join(parts, "\n\t"))
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
			imagesT := MustNoError(tensors.Load(path.Join(modelDir, generatedFile)))
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
		imagesT := MustNoError(tensors.Load(path.Join(modelDir, generatedFile)))
		images := MustNoError(timage.ToImage().MaxValue(255.0).Batch(imagesT))
		images = images[:imagesPerSample]
		params.Images[ii] = xslices.Map(images[:imagesPerSample], func(img image.Image) string {
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
	diffusionTimes := BroadcastToDims(ConvertDType(diffusionTime, DType), numImages, 1, 1, 1)
	signalRatios, noiseRatios := DiffusionSchedule(diffusionTimes, false)
	var predictedNoises *Node
	predictedImages, predictedNoises = Denoise(ctx, noisyImages, signalRatios, noiseRatios, flowerIds)

	nextDiffusionTimes := BroadcastToDims(ConvertDType(nextDiffusionTime, DType), numImages, 1, 1, 1)
	nextSignalRatios, nextNoiseRatios := DiffusionSchedule(nextDiffusionTimes, false)
	nextNoisyImages = Add(
		Mul(predictedImages, nextSignalRatios),
		Mul(predictedNoises, nextNoiseRatios))
	return
}

// DisplayImagesAcrossDiffusionSteps using reverse diffusion. If displayEveryNSteps is not 0, it will display
// intermediary results every n results -- it also displays the initial noise and final image.
//
// Plotting results only work if in a Jupyter (with GoNB kernel) notebook.
func DisplayImagesAcrossDiffusionSteps(numImages int, numDiffusionSteps int, displayEveryNSteps int) {
	ctx := context.NewContext(manager).Checked(false)
	_, _, _ = LoadCheckpointToContext(ctx)
	ctx.RngStateReset()
	noise := GenerateNoise(numImages)
	flowerIds := GenerateFlowerIds(numImages)

	generator := NewImagesGenerator(ctx, noise, flowerIds, numDiffusionSteps)
	denoisedImages, diffusionSteps, diffusionTimes := generator.GenerateEveryN(displayEveryNSteps)

	fmt.Printf("DisplayImagesAcrossDiffusionSteps(%d images, %d steps): noise.shape=%s\n", numImages, numDiffusionSteps, noise.Shape())
	fmt.Printf("\tModel #params:\t%d\n", ctx.NumParameters())
	fmt.Printf("\t Model memory:\t%s\n", data.ByteCountIEC(ctx.Memory()))
	gonbui.DisplayHtml("<p><b>Noise</b></p>")
	PlotImagesTensor(noise)

	for ii, denoisedImage := range denoisedImages {
		gonbui.DisplayHtml(fmt.Sprintf("<p>%.1f%% Denoised -- Step %d/%d",
			(1.0-diffusionTimes[ii])*100.0, diffusionSteps[ii]+1, numDiffusionSteps))
		PlotImagesTensor(denoisedImage)
	}
}

// SliderDiffusionSteps creates and animates a slider that shows images at different diffusion steps.
// It handles the slider on a separate goroutine.
// Trigger the returned latch to stop it.
//
// If `cacheKey` empty, cache is by-passed. Otherwise, try to load images from cache first if available,
// or save generated images in cache for future use.
func SliderDiffusionSteps(cacheKey string, ctx *context.Context, numImages int, numDiffusionSteps int, htmlId string) *common.Latch {
	// Generate images.
	type ImagesAndDiffusions struct {
		Images    []string
		Diffusion []float64
	}
	generateFn := func() *ImagesAndDiffusions {
		noise := GenerateNoise(numImages)
		noisesHtml := ImagesToHtml(MustNoError(timage.ToImage().MaxValue(255.0).Batch(noise)))
		flowerIds := GenerateFlowerIds(numImages)
		generator := NewImagesGenerator(ctx, noise, flowerIds, numDiffusionSteps)
		denoisedImagesT, _, diffusionTimes := generator.GenerateEveryN(1)
		denoisedImages := make([]string, len(denoisedImagesT))
		for ii, imgT := range denoisedImagesT {
			denoisedImages[ii] = ImagesToHtml(
				MustNoError(
					timage.ToImage().MaxValue(255.0).Batch(imgT),
				))
		}
		return &ImagesAndDiffusions{
			Images:    append([]string{noisesHtml}, denoisedImages...),
			Diffusion: append([]float64{1.0}, diffusionTimes...),
		}
	}

	// Use cache if available.
	var imagesAndDiffusions *ImagesAndDiffusions
	gob.Register(imagesAndDiffusions)
	imagesAndDiffusions = cache.Cache[*ImagesAndDiffusions](cacheKey, generateFn)

	// Create HTML content and containers.
	denoiseHtmlId := "denoise_" + gonbui.UniqueId()
	dom.Append(
		htmlId, fmt.Sprintf(`Denoising to flowers: &nbsp;<span id="%s" style="font-family: monospace; font-style: italic; font-size: small; border: 1px solid; border-style: inset; padding-right:5px;"> </span><br/>`, denoiseHtmlId))
	slider := widgets.Slider(0, numDiffusionSteps, 0).AppendTo(htmlId).Done()
	plotId := "plot_" + gonbui.UniqueId()
	dom.Append(htmlId, fmt.Sprintf(`<div id="%s"></div>`, plotId))

	// Create listeners, and inject first value.
	sliderChan := slider.Listen().LatestOnly()
	sliderChan.C <- 0

	done := common.NewLatch()
	go func() {
		for {
			select {
			case value := <-sliderChan.C:
				dom.SetInnerHtml(denoiseHtmlId, fmt.Sprintf(
					"%8.1f%%", 100.0*(1.0-imagesAndDiffusions.Diffusion[value])))
				dom.SetInnerHtml(plotId, imagesAndDiffusions.Images[value])
			case <-done.WaitChan():
				sliderChan.Close()
				return
			}
		}
	}()
	return done
}

// GenerateImagesOfFlowerType is similar to DisplayImagesAcrossDiffusionSteps, but it limits itself to generating images of only one
// flower type.
func GenerateImagesOfFlowerType(numImages int, flowerType int32, numDiffusionSteps int) (predictedImages tensors.Tensor) {
	ctx := context.NewContext(manager).Checked(false)
	_, _, _ = LoadCheckpointToContext(ctx)
	ctx.RngStateReset()
	noise := GenerateNoise(numImages)
	flowerIds := tensors.FromValue(xslices.SliceWithValue(numImages, flowerType))
	generator := NewImagesGenerator(ctx, noise, flowerIds, numDiffusionSteps)
	return generator.Generate()
}

// DropdownFlowerTypes creates a drop-down that shows images at different diffusion steps.
//
// If `cacheKey` empty, cache is by-passed. Otherwise, try to load images from cache first if available,
// or save generated images in cache for future use.
func DropdownFlowerTypes(cacheKey string, ctx *context.Context, numImages, numDiffusionSteps int, htmlId string) *common.Latch {
	numFlowerTypes := flowers.NumLabels
	generateFn := func() []string {
		htmlImages := make([]string, numFlowerTypes)
		noise := GenerateNoise(numImages)
		statusId := "flower_types_status_" + gonbui.UniqueId()
		gonbui.UpdateHtml(statusId, "Generating flowers ...")
		for flowerType := 0; flowerType < numFlowerTypes; flowerType++ {
			flowerIds := tensors.FromValue(xslices.SliceWithValue(numImages, flowerType))
			generator := NewImagesGenerator(ctx, noise, flowerIds, numDiffusionSteps)
			denoisedImages := generator.Generate()
			htmlImages[flowerType] = ImagesToHtml(
				MustNoError(
					timage.ToImage().MaxValue(255.0).Batch(denoisedImages),
				))
			gonbui.UpdateHtml(statusId, fmt.Sprintf(
				"Generating flowers: %q<br/>%s", flowers.Names[flowerType],
				htmlImages[flowerType]))
		}
		gonbui.UpdateHtml(statusId, "")
		return htmlImages
	}
	htmlImages := cache.Cache(cacheKey, generateFn)

	dom.Append(htmlId, "<b>Denoise Conditioned On Flower Type:</b><br/>")
	dom.Append(htmlId, "Flower Type: ")
	dropDown := widgets.Select(flowers.Names).AppendTo(htmlId).Done()
	plotId := "plot_" + gonbui.UniqueId()
	dom.Append(htmlId, fmt.Sprintf(`<div id="%s"></div>`, plotId))

	// Create listeners, and inject first value.
	selChan := dropDown.Listen().LatestOnly()
	selChan.C <- 0

	done := common.NewLatch()
	go func() {
		for {
			select {
			case value := <-selChan.C:
				dom.SetInnerHtml(plotId, htmlImages[value])
			case <-done.WaitChan():
				selChan.Close()
				return
			}
		}
	}()
	return done
}

// GenerateImagesOfAllFlowerTypes takes one random noise, and generate the flower for each of the 102 types.
func GenerateImagesOfAllFlowerTypes(numDiffusionSteps int) (predictedImages tensors.Tensor) {
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
	flowerIds := tensors.FromValue(xslices.Iota(int32(0), numImages))
	generator := NewImagesGenerator(ctx, noise, flowerIds, numDiffusionSteps)
	return generator.Generate()
}

// ImagesGenerator given noise and the flowerIds.
// Use it with NewImagesGenerator.
type ImagesGenerator struct {
	ctx               *context.Context
	noise, flowerIds  tensors.Tensor
	numImages         int
	numDiffusionSteps int
	denormalizerExec  *Exec
	diffusionStepExec *context.Exec
}

// NewImagesGenerator generates flowers given initial `noise` and `flowerIds`, in `numDiffusionSteps`.
// Typically, 20 diffusion steps will suffice.
func NewImagesGenerator(ctx *context.Context, noise, flowerIds tensors.Tensor, numDiffusionSteps int) *ImagesGenerator {
	ctx = ctx.Reuse()
	if numDiffusionSteps <= 0 {
		exceptions.Panicf("Expected numDiffusionSteps > 0, got %d", numDiffusionSteps)
	}
	numImages := noise.Shape().Dimensions[0]
	if flowerIds.Shape().Dimensions[0] != numImages || noise.Rank() != 4 || flowerIds.Rank() != 1 {
		exceptions.Panicf("Shapes of noise (%s) and flowerIds (%s) are incompatible: "+
			"they must have the same number of images, noise must be rank 4 and flowerIds must "+
			"be rank 1", noise.Shape(), flowerIds.Shape())
	}
	return &ImagesGenerator{
		ctx:               ctx,
		noise:             noise,
		flowerIds:         flowerIds,
		numImages:         numImages,
		numDiffusionSteps: numDiffusionSteps,
		diffusionStepExec: context.NewExec(manager, ctx, DenoiseStepGraph),
		denormalizerExec:  NewExec(manager, DenormalizeImages),
	}
}

// GenerateEveryN images from the original noise.
// While iteratively undoing diffusion, it will keep every `n` intermediary images.
// It will always return the last image generated.
//
// It can be called multiple times if the context changed, if the model was further trained.
// Otherwise, it will always return the same images.
//
// It returns a slice of batches of images, one batch per intermediary diffusion step,
// a slice with the step used for each batch, and another slice with the "diffusionTime"
// of the intermediary images (it will be 1.0 for the last)
func (g *ImagesGenerator) GenerateEveryN(n int) (predictedImages []tensors.Tensor,
	diffusionSteps []int, diffusionTimes []float64) {
	noisyImages := g.noise

	var imagesBatch tensors.Tensor
	stepSize := 1.0 / float64(g.numDiffusionSteps)
	for step := 0; step < g.numDiffusionSteps; step++ {
		diffusionTime := 1.0 - float64(step)*stepSize
		nextDiffusionTime := math.Max(diffusionTime-stepSize, 0)
		parts := g.diffusionStepExec.Call(noisyImages, diffusionTime, nextDiffusionTime, g.flowerIds)
		if imagesBatch != nil {
			imagesBatch.FinalizeAll() // Immediate release of (GPU) memory for intermediary results.
		}
		if noisyImages != nil && step > 0 {
			noisyImages.FinalizeAll() // Immediate release of (GPU) memory for intermediary results.
		}
		imagesBatch, noisyImages = parts[0], parts[1]
		if (n > 0 && step%n == 0) || step == g.numDiffusionSteps-1 {
			diffusionSteps = append(diffusionSteps, step)
			diffusionTimes = append(diffusionTimes, nextDiffusionTime)
			predictedImages = append(predictedImages, g.denormalizerExec.Call(imagesBatch)[0])
		}
	}
	return
}

// Generate images from the original noise.
//
// It can be called multiple times if the context changed, if the model was further trained.
// Otherwise, it will always return the same images.
func (g *ImagesGenerator) Generate() (batchedImages tensors.Tensor) {
	allBatches, _, _ := g.GenerateEveryN(0)
	return allBatches[0]
}

// GenerateNoise generates random noise that can be used to generate images.
func GenerateNoise(numImages int) tensors.Tensor {
	return NewExec(manager, func(g *Graph) *Node {
		state := Const(g, RngState())
		_, noise := RandomNormal(state, shapes.Make(DType, numImages, ImageSize, ImageSize, 3))
		return noise
	}).Call()[0]
}

// GenerateFlowerIds generates random flower ids: this is the type of flowers, one of the 102.
func GenerateFlowerIds(numImages int) tensors.Tensor {
	flowerIds := make([]int32, numImages)
	for ii := range flowerIds {
		flowerIds[ii] = int32(rand.Intn(flowers.NumLabels))
	}
	return tensors.FromValue(flowerIds)
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
		generator:      NewImagesGenerator(ctx, noise, flowerIds, numDiffusionStep),
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

func (kg *KidGenerator) Eval() (metric tensors.Tensor) {
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
