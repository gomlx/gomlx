package fm

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"image"
	"io"
	"math/rand"
	"os"
	"path"
	"regexp"
	"strconv"
	"strings"
	"text/template"

	"github.com/gomlx/gomlx/examples/inceptionv3"
	flowers "github.com/gomlx/gomlx/examples/oxfordflowers102"
	"github.com/gomlx/gomlx/examples/oxfordflowers102/diffusion"
	"github.com/gomlx/gomlx/internal/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	timage "github.com/gomlx/gomlx/pkg/core/tensors/images"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/ml/train/metrics"
	"github.com/gomlx/gomlx/pkg/support/fsutil"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/gomlx/gomlx/pkg/support/xsync"
	"github.com/janpfeifer/gonb/cache"
	"github.com/janpfeifer/gonb/gonbui"
	"github.com/janpfeifer/gonb/gonbui/dom"
	"github.com/janpfeifer/gonb/gonbui/widgets"
	"github.com/janpfeifer/must"
)

// GenerateNoise generates random noise that can be used to generate images.
func GenerateNoise(cfg *diffusion.Config, numImages int) *tensors.Tensor {
	return MustExecOnce(cfg.Backend, func(g *Graph) *Node {
		state := RNGStateForGraph(g)
		_, noise := RandomNormal(state, shapes.Make(cfg.DType, numImages, cfg.ImageSize, cfg.ImageSize, 3))
		return noise
	})
}

// GenerateFlowerIds generates random flower ids: this is the type of flowers, one of the 102.
func GenerateFlowerIds(cfg *diffusion.Config, numImages int) *tensors.Tensor {
	flowerIds := make([]int32, numImages)
	for ii := range flowerIds {
		flowerIds[ii] = int32(rand.Intn(flowers.NumLabels))
	}
	return tensors.FromValue(flowerIds)
}

// MidPointODEStep using "Midpoint Method" (https://en.wikipedia.org/wiki/Midpoint_method)
//
// Parameters:
//   - noisyImages: the X_t being integrated from t=0 to t=1. Shaped [numImages, width, height, channels].
//   - flowerIds: shaped [numImages, 1].
//   - startTime and endTime can either be scalars or shaped [numImages, 1]. They must be contrained to
//     0 <= startTime < 1 and startTime < endTime <= 1.
//
// Returns the sample images moved ΔT (ΔT=endTime-startTime) towards the target distribution.
func MidPointODEStep(ctx *context.Context, noisyImages, flowerIds, startTime, endTime *Node) *Node {
	numImages := noisyImages.Shape().Dimensions[0]
	normalizeTimeFn := func(x *Node) *Node {
		x = ConvertDType(x, noisyImages.DType())
		if !x.IsScalar() {
			x = Reshape(x, numImages, 1, 1, 1)
		} else {
			x = BroadcastToDims(x, numImages, 1, 1, 1)
		}
		return x
	}
	startTime = normalizeTimeFn(startTime)
	endTime = normalizeTimeFn(endTime)

	velocity0 := diffusion.UNetModelGraph(ctx, nil, noisyImages, startTime, flowerIds)
	// slope0 := u(ctx, xyT, tStart)
	ΔT := Sub(endTime, startTime)
	halfΔT := DivScalar(ΔT, 2)
	midPoint := Add(noisyImages, Mul(velocity0, halfΔT))
	velocity1 := diffusion.UNetModelGraph(ctx, nil, midPoint, Add(startTime, halfΔT), flowerIds)
	return Add(noisyImages, Mul(velocity1, ΔT))
}

// ImagesGenerator given noise and the flowerIds.
// Use it with NewImagesGenerator.
type ImagesGenerator struct {
	config           *diffusion.Config
	ctx              *context.Context
	noise, flowerIds *tensors.Tensor
	numImages        int
	numSteps         int
	denormalizerExec *Exec
	stepExec         *context.Exec
}

// NewImagesGenerator generates flowers given initial `noise` and `flowerIds`, in `numSteps`.
func NewImagesGenerator(cfg *diffusion.Config, noise, flowerIds *tensors.Tensor, numSteps int) *ImagesGenerator {
	ctx := cfg.Context.Reuse()
	if numSteps <= 0 {
		exceptions.Panicf("Expected numSteps > 0, got %d", numSteps)
	}
	numImages := noise.Shape().Dimensions[0]
	if flowerIds.Shape().Dimensions[0] != numImages || noise.Rank() != 4 || flowerIds.Rank() != 1 {
		exceptions.Panicf("Shapes of noise (%s) and flowerIds (%s) are incompatible: "+
			"they must have the same number of images, noise must be rank 4 and flowerIds must "+
			"be rank 1", noise.Shape(), flowerIds.Shape())
	}
	return &ImagesGenerator{
		config:    cfg,
		ctx:       ctx,
		noise:     noise,
		flowerIds: flowerIds,
		numImages: numImages,
		numSteps:  numSteps,
		stepExec:  context.MustNewExec(cfg.Backend, ctx, MidPointODEStep),
		denormalizerExec: MustNewExec(cfg.Backend, func(image *Node) *Node {
			return cfg.DenormalizeImages(image)
		}),
	}
}

// GenerateEveryN images from the original noise.
// They are generating by transposing the random noise to the distribution of the flowers images in numSteps steps.
// It always returns the last generated image, plus every n intermediary image generated.
//
// It can be called more than once if the context changed, if the model was further trained.
// Otherwise, it will always return the same images.
//
// It returns a slice of batches of images, one batch per intermediary diffusion step
// and another slice with the "time" of each step, 0 <= time <= 1, time = 1 being the last.
func (g *ImagesGenerator) GenerateEveryN(n int) (predictedImages []*tensors.Tensor, times []float64) {
	// Copy tensor: this tensor will be overwritten at each interation, and we want
	// to preserve the original g.noise.
	imagesBatch := must.M1(g.noise.LocalClone())

	backend := g.config.Backend
	stepSize := 1.0 / float64(g.numSteps-1)
	for step := 0; step < g.numSteps; step++ {
		var startTime, endTime float64
		// We skip step 0.
		if step > 0 {
			startTime = float64(step-1) * stepSize
			endTime = float64(step) * stepSize
			if step == g.numSteps-1 {
				endTime = 1.0 // Avoiding numeric issues.
			}
			buf := must.M1(DonateTensorBuffer(imagesBatch, backend, 0))
			imagesBatch = must.M1(g.stepExec.Exec1(buf, g.flowerIds, startTime, endTime))
		}
		if (n > 0 && step%n == 0) || step == g.numSteps-1 {
			times = append(times, endTime)
			predictedImages = append(predictedImages, g.denormalizerExec.MustExec(imagesBatch)[0])
		}
	}
	return
}

// Generate images from the original noise.
//
// It can be called multiple times if the context changed, if the model was further trained.
// Otherwise, it will always return the same images.
func (g *ImagesGenerator) Generate() (batchedImages *tensors.Tensor) {
	allBatches, _ := g.GenerateEveryN(0)
	return allBatches[0]
}

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
	var parts []string
	for _, img := range images {
		imgSrc := must.M1(gonbui.EmbedImageAsPNGSrc(img))
		parts = append(parts, fmt.Sprintf(`<img src="%s">`, imgSrc))
	}
	return fmt.Sprintf(
		"<div style=\"overflow-x: auto\">\n\t%s</div>\n", strings.Join(parts, "\n\t"))
}

var generateSamplesRegex = regexp.MustCompile(`generated_samples_(\d+).tensor`)

// PlotModelEvolution plots the saved sampled generated images of a model in the current configured checkpoint.
//
// If animate is true it will do an animation from first to last image, staying a few seconds on the last image.
//
// If one globaStepLimits is given, it will take the latest image whose global step <= than the one given.
//
// If two globalStepLimits are given, they are considered a range (start, end) of global step limits.
//
// It outputs at most imagesPerSample per checkpoint sampled.
func PlotModelEvolution(cfg *diffusion.Config, imagesPerSample int, animate bool, globalStepLimits ...int) {
	if cfg.Checkpoint == nil {
		exceptions.Panicf("PlotModelEvolution requires a model loaded from a checkpoint, see Config.AttachCheckpoint.")
	}
	if !gonbui.IsNotebook {
		return
	}
	modelDir := cfg.Checkpoint.Dir()
	entries := must.M1(os.ReadDir(modelDir))
	startGlobalStep, endGlobalStep := -1, -1
	if len(globalStepLimits) == 1 {
		endGlobalStep = globalStepLimits[0]
	} else if len(globalStepLimits) == 2 {
		startGlobalStep = globalStepLimits[0]
		endGlobalStep = globalStepLimits[1]
	} else if len(globalStepLimits) > 2 {
		exceptions.Panicf("PlotModelEvolution: expected 0, 1 or 2 global step limits, got %d", len(globalStepLimits))
	}
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
		globalStep := must.M1(strconv.Atoi(nameMatches[1]))
		if startGlobalStep > 0 && globalStep < startGlobalStep {
			continue
		}
		if endGlobalStep > 0 {
			if globalStep > endGlobalStep {
				continue
			}
			if startGlobalStep < 0 {
				// We just want to keep the latest file that is earlier than endGlobalStep
				if len(generatedFiles) == 1 {
					if globalStep > generateGlobalSteps[0] {
						generatedFiles[0] = fileName
						generateGlobalSteps[0] = globalStep
					}
					continue
				}
			}

		}
		generatedFiles = append(generatedFiles, fileName)
		generateGlobalSteps = append(generateGlobalSteps, globalStep)
	}

	if len(generatedFiles) == 0 {
		gonbui.DisplayHTML(fmt.Sprintf("<b>No generated samples in <pre>%s</pre>.</b>", modelDir))
		return
	}

	gonbui.DisplayMarkdown(fmt.Sprintf("**Generated samples in `%s`:**", modelDir))
	if !animate {
		// Simply display all images:
		for ii, generatedFile := range generatedFiles {
			imagesT := must.M1(tensors.Load(path.Join(modelDir, generatedFile)))
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
		Size:            cfg.ImageSize,
		Width:           imagesPerSample * cfg.ImageSize,
		ImagesPerSample: imagesPerSample,
	}
	for ii, generatedFile := range generatedFiles {
		imagesT := must.M1(tensors.Load(path.Join(modelDir, generatedFile)))
		images := timage.ToImage().MaxValue(255.0).Batch(imagesT)
		images = images[:imagesPerSample]
		params.Images[ii] = xslices.Map(images[:imagesPerSample], func(img image.Image) string {
			//return fmt.Sprintf("timestep_%d", ii)
			return must.M1(gonbui.EmbedImageAsPNGSrc(img))
		})
	}

	var jsTemplate = must.M1(template.New("PlotModelEvolution").Parse(`
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
		var Context = ctx_{{.Id}};
		var canvas = canvas_{{.Id}};
		Context.clearRect(0, 0, canvas.width, canvas.height);
		var images = images_{{.Id}}[currentFrame_{{.Id}}];
		for (var jj = 0; jj < {{.ImagesPerSample}}; jj++) {
			Context.drawImage(images[jj], jj*{{.Size}}, 0);
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
	must.M(jsTemplate.Execute(&buf, params))
	gonbui.DisplayHTML(buf.String())
}

// DisplayImagesAcrossTime creates numImages series of images of the images starting from gaussian
// noise being transposed to generate images.
//
// It transports the random noise to generated images in numSteps, of which displayEveryNSteps are
// actually displayed.
//
// Plotting results only work if in a Jupyter (with GoNB kernel) notebook.
func DisplayImagesAcrossTime(cfg *diffusion.Config, numImages int, numSteps int, displayEveryNSteps int) {
	if !gonbui.IsNotebook {
		exceptions.Panicf("DisplayImagesAcrossTime requires a Jupyter notebook.")
	}
	if cfg.Checkpoint == nil {
		exceptions.Panicf("DisplayImagesAcrossDiffusionSteps requires a model loaded from a checkpoint, see " +
			"Config.AttachCheckpoint.")
	}
	ctx := cfg.Context.Checked(false)
	ctx.ResetRNGState()
	noise := cfg.GenerateNoise(numImages)
	flowerIds := cfg.GenerateFlowerIds(numImages)

	generator := NewImagesGenerator(cfg, noise, flowerIds, numSteps)
	generatedImages, generationTimes := generator.GenerateEveryN(displayEveryNSteps)

	fmt.Printf("DisplayImagesAcrossDiffusionSteps(%d images, %d steps): noise.shape=%s\n",
		numImages, numSteps, noise.Shape())
	fmt.Printf("\tModel #params:\t%d\n", ctx.NumParameters())
	fmt.Printf("\t Model memory:\t%s\n", fsutil.ByteCountIEC(ctx.Memory()))
	for ii, generatedImage := range generatedImages {
		gonbui.DisplayHTMLF("<p>%.2f%% Transformed</p>", generationTimes[ii]*100.0)
		PlotImagesTensor(generatedImage)
	}
}

// SliderDiffusionSteps creates and animates a slider that shows images at different diffusion steps.
// It handles the slider on a separate goroutine.
// Trigger the returned latch to stop it.
//
// If `cacheKey` empty, cache is by-passed. Otherwise, try to load images from cache first if available,
// or save generated images in cache for future use.
func SliderDiffusionSteps(
	cfg *diffusion.Config,
	cacheKey string,
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
		noise := cfg.GenerateNoise(numImages)
		noisesHtml := ImagesToHtml(timage.ToImage().MaxValue(255.0).Batch(noise))
		flowerIds := cfg.GenerateFlowerIds(numImages)
		generator := NewImagesGenerator(cfg, noise, flowerIds, numDiffusionSteps)
		generatedImagesT, generationTimes := generator.GenerateEveryN(1)
		generatedImages := make([]string, len(generatedImagesT))
		for ii, imgT := range generatedImagesT {
			generatedImages[ii] = ImagesToHtml(timage.ToImage().MaxValue(255.0).Batch(imgT))
		}
		return &ImagesAndDiffusions{
			Images:    append([]string{noisesHtml}, generatedImages...),
			Diffusion: append([]float64{1.0}, generationTimes...),
		}
	}

	// Use cache if available.
	var imagesAndDiffusions *ImagesAndDiffusions
	gob.Register(imagesAndDiffusions)
	imagesAndDiffusions = cache.Cache[*ImagesAndDiffusions](cacheKey, generateFn)

	// Create HTML content and containers.
	denoiseHtmlId := "fm_transform_" + gonbui.UniqueId()
	dom.Append(
		htmlId,
		fmt.Sprintf(
			`Tranforming Noise to Flowers: &nbsp;<span id="%s" style="font-family: monospace; font-style: italic; font-size: small; border: 1px solid; border-style: inset; padding-right:5px;"> </span><br/>`,
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
					"%8.1f%%", 100.0*(imagesAndDiffusions.Diffusion[value])))
				dom.SetInnerHtml(plotId, imagesAndDiffusions.Images[value])
			case <-done.WaitChan():
				sliderChan.Close()
				return
			}
		}
	}()
	return done
}

// GenerateImagesOfFlowerType is similar to DisplayImagesAcrossTime, but it limits itself to generating images of only one
// flower type.
//
// paramsSet are hyperparameters overridden, that it should not load from the checkpoint (see commandline.ParseContextSettings).
func GenerateImagesOfFlowerType(
	cfg *diffusion.Config,
	numImages int,
	flowerType int32,
	numDiffusionSteps int,
) (predictedImages *tensors.Tensor) {
	ctx := cfg.Context
	ctx.ResetRNGState()
	noise := cfg.GenerateNoise(numImages)
	flowerIds := tensors.FromValue(xslices.SliceWithValue(numImages, flowerType))
	generator := NewImagesGenerator(cfg, noise, flowerIds, numDiffusionSteps)
	return generator.Generate()
}

// DropdownFlowerTypes creates a drop-down that shows images at different diffusion steps.
//
// If `cacheKey` empty, cache is by-passed. Otherwise, try to load images from cache first if available,
// or save generated images in cache for future use.
func DropdownFlowerTypes(
	cfg *diffusion.Config,
	cacheKey string,
	numImages, numDiffusionSteps int,
	htmlId string,
) *xsync.Latch {
	numFlowerTypes := flowers.NumLabels
	generateFn := func() []string {
		htmlImages := make([]string, numFlowerTypes)
		noise := cfg.GenerateNoise(numImages)
		statusId := "flower_types_status_" + gonbui.UniqueId()
		gonbui.UpdateHTML(statusId, "Generating flowers ...")
		for flowerType := 0; flowerType < numFlowerTypes; flowerType++ {
			flowerIds := tensors.FromValue(xslices.SliceWithValue(numImages, flowerType))
			generator := NewImagesGenerator(cfg, noise, flowerIds, numDiffusionSteps)
			denoisedImages := generator.Generate()
			htmlImages[flowerType] = ImagesToHtml(timage.ToImage().MaxValue(255.0).Batch(denoisedImages))
			gonbui.UpdateHTML(statusId, fmt.Sprintf(
				"Generating flowers: %q<br/>%s", flowers.Names[flowerType],
				htmlImages[flowerType]))
		}
		gonbui.UpdateHTML(statusId, "")
		return htmlImages
	}
	htmlImages := cache.Cache(cacheKey, generateFn)

	dom.Append(htmlId, "<b>Generation Conditioned On Flower Type:</b><br/>")
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
//
// paramsSet are hyperparameters overridden, that it should not load from the checkpoint (see commandline.ParseContextSettings).
func GenerateImagesOfAllFlowerTypes(cfg *diffusion.Config, numDiffusionSteps int) (predictedImages *tensors.Tensor) {
	ctx := cfg.Context
	numImages := flowers.NumLabels
	ctx.ResetRNGState()
	imageSize := cfg.ImageSize
	noise := MustNewExec(cfg.Backend, func(g *Graph) *Node {
		state := RNGStateForGraph(g)
		_, noise := RandomNormal(state, shapes.Make(cfg.DType, 1, imageSize, imageSize, 3))
		noise = BroadcastToDims(noise, numImages, imageSize, imageSize, 3)
		return noise
	}).MustExec()[0]
	flowerIds := tensors.FromValue(xslices.Iota(int32(0), numImages))
	generator := NewImagesGenerator(cfg, noise, flowerIds, numDiffusionSteps)
	return generator.Generate()
}

// KidGenerator generates the [Kernel Inception Distance (KID)](https://arxiv.org/abs/1801.01401) metric.
type KidGenerator struct {
	config         *diffusion.Config
	ctxInceptionV3 *context.Context
	ds             train.Dataset
	generator      *ImagesGenerator
	kid            metrics.Interface
	evalExec       *context.Exec
}

// NewKidGenerator allows to generate the Kid metric.
// The Context passed is the context for the diffusion model.
// It uses a different context for the InceptionV3 KID metric, so that it's weights are not included
// in the generator model.
func NewKidGenerator(cfg *diffusion.Config, evalDS train.Dataset, numDiffusionStep int) *KidGenerator {
	noise := cfg.GenerateNoise(cfg.EvalBatchSize)
	flowerIds := cfg.GenerateFlowerIds(cfg.EvalBatchSize)
	i3Path := path.Join(cfg.DataDir, "inceptionV3")
	must.M(inceptionv3.DownloadAndUnpackWeights(i3Path))
	kg := &KidGenerator{
		config:         cfg,
		ctxInceptionV3: context.New().Checked(false),
		ds:             evalDS,
		generator:      NewImagesGenerator(cfg, noise, flowerIds, numDiffusionStep),
		kid:            inceptionv3.KidMetric(i3Path, inceptionv3.MinimumImageSize, 255.0, timage.ChannelsLast),
	}
	kg.evalExec = context.MustNewExec(cfg.Backend, kg.ctxInceptionV3, kg.EvalStepGraph)
	return kg
}

func (kg *KidGenerator) EvalStepGraph(ctx *context.Context, allImages []*Node) (metric *Node) {
	g := allImages[0].Graph()
	ctx.SetTraining(g, false) // Some layers behave differently in train/eval.

	// Get metrics and updates: the generated images are the inputs, and the
	generatedImages := allImages[0]
	datasetImages := kg.config.PreprocessImages(allImages[1], false)
	metric = kg.kid.UpdateGraph(ctx, []*Node{datasetImages}, []*Node{generatedImages})
	return
}

func (kg *KidGenerator) Eval() (metric *tensors.Tensor) {
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
			metric.MustFinalizeAll()
		}
		metric = kg.evalExec.MustExec(generatedImages, datasetImages)[0]
	}
	if count == 0 {
		exceptions.Panicf("evaluation dataset %s yielded no batches, no data to evaluate KID", kg.ds)
	}
	return
}
