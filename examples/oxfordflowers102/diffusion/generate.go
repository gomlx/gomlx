// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package diffusion

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"image"
	"io"
	"math"
	"math/rand"
	"os"
	"path"
	"regexp"
	"strconv"
	"strings"
	"text/template"

	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/humanize"
	"github.com/gomlx/compute/support/xslices"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	timage "github.com/gomlx/gomlx/core/tensors/images"
	"github.com/gomlx/gomlx/examples/inceptionv3"
	flowers "github.com/gomlx/gomlx/examples/oxfordflowers102"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/metric"
	"github.com/gomlx/gomlx/support/exceptions"
	"github.com/gomlx/gomlx/support/xsync"
	"github.com/janpfeifer/gonb/cache"
	"github.com/janpfeifer/gonb/gonbui"
	"github.com/janpfeifer/gonb/gonbui/dom"
	"github.com/janpfeifer/gonb/gonbui/widgets"
)

// PlotImagesTensor plots images in tensor format, all in one row.
// It assumes image's MaxValue of 255.
//
// This only works in a Jupyter (GoNB kernel) notebook.
func PlotImagesTensor(imagesT *tensors.Tensor) {
	if gonbui.IsNotebook {
		images := timage.ToImage().MaxValue(255.0).Batch(imagesT)
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
	parts := make([]string, 0, len(images))
	for _, img := range images {
		imgSrc := must1(gonbui.EmbedImageAsPNGSrc(img))
		parts = append(parts, fmt.Sprintf(`<img src="%s">`, imgSrc))
	}
	return fmt.Sprintf(
		"<div style=\"overflow-x: auto\">\n\t%s</div>\n", strings.Join(parts, "\n\t"))
}

var generateSamplesRegex = regexp.MustCompile(`generated_samples_(\d+).tensor`)

// PlotModelEvolution plots the saved sampled generated images of a model in the current configured checkpointHandler.
//
// It outputs at most imagesPerSample per checkpoint sampled.
func (c *Config) PlotModelEvolution(imagesPerSample int, animate bool) {
	if c.Checkpoint == nil {
		exceptions.Panicf("PlotModelEvolution requires a model loaded from a checkpoint, see Config.AttachCheckpoint.")
	}
	if !gonbui.IsNotebook {
		return
	}
	modelDir := c.Checkpoint.Dir()
	entries := must1(os.ReadDir(modelDir))
	var generatedFiles []string
	var generateGlobalSteps []int
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		fileName := entry.Name()
		nameMatches := generateSamplesRegex.FindStringSubmatch(fileName)
		if len(nameMatches) != 2 || nameMatches[0] != fileName {
			continue
		}
		generatedFiles = append(generatedFiles, fileName)
		generateGlobalSteps = append(generateGlobalSteps, must1(strconv.Atoi(nameMatches[1])))
	}

	if len(generatedFiles) == 0 {
		gonbui.DisplayHTML(fmt.Sprintf("<b>No generated samples in <pre>%s</pre>.</b>", modelDir))
		return
	}

	gonbui.DisplayMarkdown(fmt.Sprintf("**Generated samples in `%s`:**", modelDir))
	if !animate {
		// Simply display all images:
		for ii, generatedFile := range generatedFiles {
			imagesT := must1(tensors.Load(path.Join(modelDir, generatedFile)))
			images := timage.ToImage().MaxValue(255.0).Batch(imagesT)
			images = images[:imagesPerSample]
			gonbui.DisplayMarkdown(fmt.Sprintf("- global_step %d:\n", generateGlobalSteps[ii]))
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
		Size:            c.ImageSize,
		Width:           imagesPerSample * c.ImageSize,
		ImagesPerSample: imagesPerSample,
	}
	for ii, generatedFile := range generatedFiles {
		imagesT := must1(tensors.Load(path.Join(modelDir, generatedFile)))
		images := timage.ToImage().MaxValue(255.0).Batch(imagesT)
		images = images[:imagesPerSample]
		params.Images[ii] = xslices.Map(images[:imagesPerSample], func(img image.Image) string {
			//return fmt.Sprintf("timestep_%d", ii)
			return must1(gonbui.EmbedImageAsPNGSrc(img))
		})
	}

	var jsTemplate = must1(template.New("PlotModelEvolution").Parse(`
	<canvas id="canvas_{{.Id}}" height="{{.Size}}px" width="{{.Width}}px"></canvas>
	<script>
	var canvas_{{.Id}} = document.getElementById("canvas_{{.Id}}");
	var ctx_{{.Id}} = canvas_{{.Id}}.getScope("2d");
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
		var Scope = ctx_{{.Id}};
		var canvas = canvas_{{.Id}};
		Scope.clearRect(0, 0, canvas.width, canvas.height);
		var images = images_{{.Id}}[currentFrame_{{.Id}}];
		for (var jj = 0; jj < {{.ImagesPerSample}}; jj++) {
			Scope.drawImage(images[jj], jj*{{.Size}}, 0);
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
	check(jsTemplate.Execute(&buf, params))
	gonbui.DisplayHTML(buf.String())
}

// DenoiseStepGraph executes one step of separating the noise and images from noisy images.
func DenoiseStepGraph(scope *model.Scope, noisyImages, diffusionTime, nextDiffusionTime, flowerIds *Node) (
	predictedImages, nextNoisyImages *Node) {
	dtype := noisyImages.DType()
	numImages := noisyImages.Shape().Dimensions[0]
	diffusionTimes := BroadcastToDims(ConvertDType(diffusionTime, dtype), numImages, 1, 1, 1)
	signalRatios, noiseRatios := DiffusionSchedule(scope, diffusionTimes, false)
	var predictedNoises *Node
	predictedImages, predictedNoises = Denoise(scope, noisyImages, signalRatios, noiseRatios, flowerIds)

	nextDiffusionTimes := BroadcastToDims(ConvertDType(nextDiffusionTime, dtype), numImages, 1, 1, 1)
	nextSignalRatios, nextNoiseRatios := DiffusionSchedule(scope, nextDiffusionTimes, false)
	nextNoisyImages = Add(
		Mul(predictedImages, nextSignalRatios),
		Mul(predictedNoises, nextNoiseRatios))
	return
}

// DisplayImagesAcrossDiffusionSteps using reverse diffusion. If displayEveryNSteps is not 0, it will display
// intermediary results every n results -- it also displays the initial noise and final image.
//
// Plotting results only work if in a Jupyter (with GoNB kernel) notebook.
func (c *Config) DisplayImagesAcrossDiffusionSteps(numImages int, numDiffusionSteps int, displayEveryNSteps int) {
	if c.Checkpoint == nil {
		exceptions.Panicf(
			"DisplayImagesAcrossDiffusionSteps requires a model loaded from a checkpoint, see Config.AttachCheckpoint.",
		)
	}
	store := c.Store
	scope := c.Store.RootScope()
	_ = store.ResetRNGState()
	noise := c.GenerateNoise(numImages)
	flowerIds := c.GenerateFlowerIds(numImages)

	generator := c.NewImagesGenerator(noise, flowerIds, numDiffusionSteps)
	denoisedImages, diffusionSteps, diffusionTimes := generator.GenerateEveryN(displayEveryNSteps)

	fmt.Printf(
		"DisplayImagesAcrossDiffusionSteps(%d images, %d steps): noise.shape=%s\n",
		numImages,
		numDiffusionSteps,
		noise.Shape(),
	)
	fmt.Printf("\tModel #params:\t%d\n", scope.NumParameters())
	fmt.Printf("\t Model memory:\t%s\n", humanize.Bytes(scope.ByteSize()))
	gonbui.DisplayHtml("<p><b>Noise</b></p>")
	PlotImagesTensor(noise)

	for ii, denoisedImage := range denoisedImages {
		gonbui.DisplayHtml(fmt.Sprintf("<p>%.1f%% Denoised -- Step %d/%d",
			(1.0-diffusionTimes[ii])*100.0, diffusionSteps[ii]+1, numDiffusionSteps))
		PlotImagesTensor(denoisedImage)
	}
}

// SliderDiffusionSteps creates and animates a slider that shows images at different diffusion steps.
func (c *Config) SliderDiffusionSteps(
	cacheKey string,
	scope *model.Scope,
	numImages int,
	numDiffusionSteps int,
	htmlId string,
) *xsync.Latch {
	// Generate images.
	type ImagesAndDiffusions struct {
		Images    []string
		Diffusion []float64
	}
	generateFn := func() *ImagesAndDiffusions {
		noise := c.GenerateNoise(numImages)
		noisesHtml := ImagesToHtml(timage.ToImage().MaxValue(255.0).Batch(noise))
		flowerIds := c.GenerateFlowerIds(numImages)
		generator := c.NewImagesGenerator(noise, flowerIds, numDiffusionSteps)
		denoisedImagesT, _, diffusionTimes := generator.GenerateEveryN(1)
		denoisedImages := make([]string, len(denoisedImagesT))
		for ii, imgT := range denoisedImagesT {
			denoisedImages[ii] = ImagesToHtml(timage.ToImage().MaxValue(255.0).Batch(imgT))
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
		htmlId,
		fmt.Sprintf(
			`Denoising to flowers: &nbsp;<span id="%s" style="font-family: monospace; font-style: italic; font-size: small; border: 1px solid; border-style: inset; padding-right:5px;"> </span><br/>`,
			denoiseHtmlId,
		),
	)
	slider := widgets.Slider(0, numDiffusionSteps, 0).AppendTo(htmlId).Done()
	plotId := "plot_" + gonbui.UniqueId()
	dom.Append(htmlId, fmt.Sprintf(`<div id="%s"></div>`, plotId))

	// Create listeners, and inject first value.
	sliderChan := slider.Listen().LatestOnly()
	sliderChan.C <- 0

	done := xsync.NewLatch()
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
func (c *Config) GenerateImagesOfFlowerType(
	numImages int,
	flowerType int32,
	numDiffusionSteps int,
) (predictedImages *tensors.Tensor) {
	c.Store.ResetRNGState()
	noise := c.GenerateNoise(numImages)
	flowerIds := tensors.FromValue(xslices.SliceWithValue(numImages, flowerType))
	generator := c.NewImagesGenerator(noise, flowerIds, numDiffusionSteps)
	return generator.Generate()
}

// DropdownFlowerTypes creates a drop-down that shows images at different diffusion steps.
func (c *Config) DropdownFlowerTypes(cacheKey string, numImages, numDiffusionSteps int, htmlId string) *xsync.Latch {
	numFlowerTypes := flowers.NumLabels
	generateFn := func() []string {
		htmlImages := make([]string, numFlowerTypes)
		noise := c.GenerateNoise(numImages)
		statusId := "flower_types_status_" + gonbui.UniqueId()
		gonbui.UpdateHtml(statusId, "Generating flowers ...")
		for flowerType := range numFlowerTypes {
			flowerIds := tensors.FromValue(xslices.SliceWithValue(numImages, flowerType))
			generator := c.NewImagesGenerator(noise, flowerIds, numDiffusionSteps)
			denoisedImages := generator.Generate()
			htmlImages[flowerType] = ImagesToHtml(timage.ToImage().MaxValue(255.0).Batch(denoisedImages))
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

	done := xsync.NewLatch()
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
func (c *Config) GenerateImagesOfAllFlowerTypes(numDiffusionSteps int) (predictedImages *tensors.Tensor) {
	numImages := flowers.NumLabels
	_ = c.Store.ResetRNGState()
	imageSize := c.ImageSize
	noise := MustNewExec(c.Backend, func(g *Graph) *Node {
		state := RNGStateForGraph(g)
		_, noise := RandomNormal(state, shapes.Make(c.DType, 1, imageSize, imageSize, 3))
		noise = BroadcastToDims(noise, numImages, imageSize, imageSize, 3)
		return noise
	}).MustCall1()
	flowerIds := tensors.FromValue(xslices.Iota(int32(0), numImages))
	generator := c.NewImagesGenerator(noise, flowerIds, numDiffusionSteps)
	return generator.Generate()
}

// ImagesGenerator given noise and the flowerIds.
// Use it with NewImagesGenerator.
type ImagesGenerator struct {
	config            *Config
	store             *model.Store
	noise, flowerIds  *tensors.Tensor
	numImages         int
	numDiffusionSteps int
	denormalizerExec  *Exec
	diffusionStepExec *model.Exec
}

// NewImagesGenerator generates flowers given initial `noise` and `flowerIds`, in `numDiffusionSteps`.
// Typically, 20 diffusion steps will suffice.
func (c *Config) NewImagesGenerator(noise, flowerIds *tensors.Tensor, numDiffusionSteps int) *ImagesGenerator {
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
		config:            c,
		store:             c.Store,
		noise:             noise,
		flowerIds:         flowerIds,
		numImages:         numImages,
		numDiffusionSteps: numDiffusionSteps,
		diffusionStepExec: model.MustNewExec(c.Backend, c.Store, DenoiseStepGraph),
		denormalizerExec: MustNewExec(c.Backend, func(image *Node) *Node {
			return c.DenormalizeImages(image)
		}),
	}
}

// GenerateEveryN images from the original noise.
func (g *ImagesGenerator) GenerateEveryN(n int) (predictedImages []*tensors.Tensor,
	diffusionSteps []int, diffusionTimes []float64) {
	noisyImages := g.noise

	var imagesBatch *tensors.Tensor
	stepSize := 1.0 / float64(g.numDiffusionSteps)
	for step := 0; step < g.numDiffusionSteps; step++ {
		diffusionTime := 1.0 - float64(step)*stepSize
		nextDiffusionTime := math.Max(diffusionTime-stepSize, 0)
		parts := g.diffusionStepExec.MustCall(noisyImages, diffusionTime, nextDiffusionTime, g.flowerIds)
		if imagesBatch != nil {
			imagesBatch.MustFinalizeAll() // Immediate release of (GPU) memory for intermediary results.
		}
		if noisyImages != nil && step > 0 {
			noisyImages.MustFinalizeAll() // Immediate release of (GPU) memory for intermediary results.
		}
		imagesBatch, noisyImages = parts[0], parts[1]
		if (n > 0 && step%n == 0) || step == g.numDiffusionSteps-1 {
			diffusionSteps = append(diffusionSteps, step)
			diffusionTimes = append(diffusionTimes, nextDiffusionTime)
			predictedImages = append(predictedImages, g.denormalizerExec.MustExec(imagesBatch)[0])
		}
	}
	return
}

// Generate images from the original noise.
func (g *ImagesGenerator) Generate() (batchedImages *tensors.Tensor) {
	allBatches, _, _ := g.GenerateEveryN(0)
	return allBatches[0]
}

// GenerateNoise generates random noise that can be used to generate images.
func (c *Config) GenerateNoise(numImages int) *tensors.Tensor {
	return MustNewExec(c.Backend, func(g *Graph) *Node {
		state := RNGStateForGraph(g)
		_, noise := RandomNormal(state, shapes.Make(c.DType, numImages, c.ImageSize, c.ImageSize, 3))
		return noise
	}).MustExec1()
}

// GenerateFlowerIds generates random flower ids: this is the type of flowers, one of the 102.
func (c *Config) GenerateFlowerIds(numImages int) *tensors.Tensor {
	flowerIds := make([]int32, numImages)
	for ii := range flowerIds {
		flowerIds[ii] = int32(rand.Intn(flowers.NumLabels))
	}
	return tensors.FromValue(flowerIds)
}

// KidGenerator generates the [Kernel Inception Distance (KID)](https://arxiv.org/abs/1801.01401) metric.
type KidGenerator struct {
	config           *Config
	scopeInceptionV3 *model.Scope
	ds               train.Dataset
	generator        *ImagesGenerator
	kid              metric.Interface
	evalExec         *model.Exec
}

// NewKidGenerator allows to generate the Kid metric.
func (c *Config) NewKidGenerator(evalDS train.Dataset, numDiffusionStep int) *KidGenerator {
	noise := c.GenerateNoise(c.EvalBatchSize)
	flowerIds := c.GenerateFlowerIds(c.EvalBatchSize)
	i3Path := path.Join(c.DataDir, "inceptionV3")
	check(inceptionv3.DownloadAndUnpackWeights(i3Path))
	kg := &KidGenerator{
		config:           c,
		scopeInceptionV3: model.NewStore().RootScope(),
		ds:               evalDS,
		generator:        c.NewImagesGenerator(noise, flowerIds, numDiffusionStep),
		kid:              inceptionv3.KidMetric(i3Path, inceptionv3.MinimumImageSize, 255.0, timage.ChannelsLast),
	}
	kg.evalExec = model.MustNewExec(c.Backend, kg.scopeInceptionV3.Store(), kg.EvalStepGraph)
	return kg
}

func (kg *KidGenerator) EvalStepGraph(scope *model.Scope, allImages []*Node) (metric *Node) {
	g := allImages[0].Graph()
	scope.Store().SetTraining(g, false) // Some layers behave differently in train/eval.

	// Get metrics and updates: the generated images are the inputs, and the
	generatedImages := allImages[0]
	datasetImages := kg.config.PreprocessImages(allImages[1], false)
	metric = kg.kid.UpdateGraph(scope, []*Node{datasetImages}, []*Node{generatedImages})
	return
}

func (kg *KidGenerator) Eval() (metric *tensors.Tensor) {
	kg.ds.Reset()
	kg.kid.Reset(kg.scopeInceptionV3)
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
			metric.MustFinalizeAll()
		}
		metric = kg.evalExec.MustCall(generatedImages, datasetImages)[0]
	}
	if count == 0 {
		exceptions.Panicf("evaluation dataset %s yielded no batches, no data to evaluate KID", kg.ds)
	}
	return
}
