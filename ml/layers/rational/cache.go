package rational

import (
	"github.com/gomlx/gomlx/types/tensors"
	"sync"
)

// approximationToVarianceGain maps an initial approximation function to its variance gain
// (gain(F(x)) = Var[x]/E[F(x)^2], where E[] is the expectation operator), that we use
// to initialize w, to preserve the variance across layers. See KAT paper [1], table 3.
//
// This is used only if the cache doesn't provide a value.
var approximationToVarianceGain = map[string]float64{
	"identity": 1,
	"relu":     2,
	"gelu":     2.3568,
	"swish":    2.8178,
	"silu":     2.8178, // alias to swish
	//"geglu": 0.7112,
	//"swishglu": 0.8434,
}

// initCacheKey used to initialize the coefficients of the rational functions in some preset
// configurations. See Config.Approximate.
type initCacheKey struct {
	// Approximation name set with Config.Approximate.
	Approximation string

	// Version of the function set with Config.Version.
	Version string

	NumeratorDegree, DenominatorDegree int
}

type initCacheValue struct {
	// Numerator and Denominator values.
	Num, Den     []float64
	GainEstimate float64 // Not always set.

	// Numerator and Denominator tensors, only created once used.
	NumTensor, DenTensor *tensors.Tensor
}

// GetTensors converts the float64 slices to a tensor (of type float64) and returns that.
func (cv *initCacheValue) GetTensors() (numerator, denominator *tensors.Tensor) {
	if cv.NumTensor == nil {
		cv.NumTensor = tensors.FromValue(cv.Num)
	}
	if cv.DenTensor == nil {
		cv.DenTensor = tensors.FromValue(cv.Den)
	}
	return cv.NumTensor, cv.DenTensor
}

// getInitTensors for given configuration.
func (c *Config) getInitTensorsAndGain() (numerator, denominator *tensors.Tensor, gain float64) {
	muInitializationCoefficients.Lock()
	defer muInitializationCoefficients.Unlock()
	key := initCacheKey{Approximation: c.initApproximation, Version: c.version, NumeratorDegree: c.numeratorDegree, DenominatorDegree: c.denominatorDegree}
	value, found := initializationCoefficients[key]
	if !found {
		if c.initApproximation != "identity" && c.initApproximation != "" {
			return
		}
		value = &initCacheValue{
			Num:          make([]float64, c.numeratorDegree+1),
			Den:          make([]float64, c.denominatorDegree),
			GainEstimate: 1.0,
		}
		value.Num[1] = 1
		initializationCoefficients[key] = value
	}
	numerator, denominator = value.GetTensors()
	gain = value.GainEstimate
	if gain == 0 {
		gain = approximationToVarianceGain[c.initApproximation]
	}
	return
}

var (
	muInitializationCoefficients sync.Mutex

	// initializationCoefficients for rational functions approximations to various other functions,
	// adapted from:
	// https://github.com/ml-research/rational_activations/blob/master/rational/rationals_config.json
	//
	// To create new ones, either follow the steps prescribed in
	// https://rational-activations.readthedocs.io/en/latest/tutorials/tutorials.1_find_weights_for_initialization.html
	//
	// Or simply create a new gradient descent function with a squared loss to match the desired function.
	initializationCoefficients = map[initCacheKey]*initCacheValue{
		{Approximation: "identity", Version: "N", NumeratorDegree: 2, DenominatorDegree: 2}: {
			Num: []float64{2.0768813552409573e-20, 0.9999999997812081, 0.0006867760240578934}, Den: []float64{0.0006867760240050522, -1.4587398861117518e-11}},
		{Approximation: "leaky_relu", Version: "B", NumeratorDegree: 7, DenominatorDegree: 6}: {
			Num: []float64{0.01964840427008628, 0.8663826315551684, 6.171795038916364, 17.115491923105473, 22.193921388440426, 14.340632785553574, 4.445676524738527, 0.5371403305595667}, Den: []float64{7.010335862486178, 13.809752918048835, 23.82775258043521, 16.51163509824627, 2.5658857005547717, 0.9068983776485914}},
		{Approximation: "leaky_relu", Version: "A", NumeratorDegree: 7, DenominatorDegree: 6}: {
			Num: []float64{0.01007645697195976, 0.6317967788773284, 7.084951122518462, 28.513835912920452, 47.1330586589471, 37.7353261538062, 13.273737403073005, 1.7899654547972275}, Den: []float64{3.825862317531991, 39.44222535222467, -30.642504274599432, 49.69365309008858, 9.223191325027036, 2.302627109817824}},
		{Approximation: "relu", Version: "B", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{0.033897129202224346, 0.4999985439606278, 1.6701363611130988, 1.9901021632350815, 0.9413089613384323, 0.1509133373584318}, Den: []float64{-2.1040152094202414e-05, 3.980247851167207, -3.166344237241501e-05, 0.30183382300945066}},
		{Approximation: "gelu", Version: "B", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{-0.0004221071456647063, 0.49999955516606254, 0.40270535451115536, 0.07366976895222031, -0.012954788054484537, -0.0037414002583076983}, Den: []float64{4.9585381197087913e-05, 0.1472977407631199, 1.1645825701440633e-05, -0.007483871514842074}},
		{Approximation: "swish", Version: "B", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{3.2597006782059475e-07, 0.500000007624051, 0.2500169540800127, 0.05327519598927768, 0.005804580590860159, 0.00027529600705133663}, Den: []float64{3.917294261258692e-05, 0.1065308384054887, 3.7991514117509955e-06, 0.0005503052126186067}},
		{Approximation: "leaky_relu_0.1", Version: "B", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{0.03050741536904778, 0.5499984745977262, 1.5031192607489545, 2.1891150893190687, 0.8471738041675438, 0.1660052303771313}, Den: []float64{-2.354586180306346e-05, 3.980247923802633, -3.368710799869651e-05, 0.30183383024180266}},
		{Approximation: "identity", Version: "B", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{0, 1, 0, 0, 0, 0}, Den: []float64{0, 0, 0, 0}},
		{Approximation: "leaky_relu", Version: "B", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{0.03355815733452378, 0.5050059186781659, 1.6534697310952233, 2.0100501552600014, 0.9319195586484518, 0.15242646398275836}, Den: []float64{4.812831860534153e-05, 3.9802479055059194, 1.5837587665720168e-05, 0.30183382839661377}},
		{Approximation: "tanh", Version: "B", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{2.1172949817857366e-09, 0.9999942495075363, 6.276332768876106e-07, 0.10770864506559906, 2.946556898117109e-08, 0.000871124373591946}, Den: []float64{6.376908337817277e-07, 0.44101418051922986, 2.2747661404467182e-07, 0.014581039909092108}},
		{Approximation: "sigmoid", Version: "B", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{0.5000000002774382, 0.2500039727332485, 0.05544474230118124, 0.006888449237990345, 0.00048491391666921244, 1.5646015289718136e-05}, Den: []float64{7.956371345839366e-06, 0.11088550952772189, 7.76547864226066e-07, 0.0009697684428133153}},
		{Approximation: "tanh", Version: "D", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{2.1172949817857366e-09, 0.9999942495075363, 6.276332768876106e-07, 0.10770864506559906, 2.946556898117109e-08, 0.000871124373591946}, Den: []float64{6.376908337817277e-07, 0.44101418051922986, 2.2747661404467182e-07, 0.014581039909092108}},
		{Approximation: "gelu", Version: "D", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{-0.0004221071456647063, 0.49999955516606254, 0.40270535451115536, 0.07366976895222031, -0.012954788054484537, -0.0037414002583076983}, Den: []float64{4.9585381197087913e-05, 0.1472977407631199, 1.1645825701440633e-05, -0.007483871514842074}},
		{Approximation: "sigmoid", Version: "D", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{0.5000000002774382, 0.2500039727332485, 0.05544474230118124, 0.006888449237990345, 0.00048491391666921244, 1.5646015289718136e-05}, Den: []float64{7.956371345839366e-06, 0.11088550952772189, 7.76547864226066e-07, 0.0009697684428133153}},
		{Approximation: "swish", Version: "D", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{3.2597006782059475e-07, 0.500000007624051, 0.2500169540800127, 0.05327519598927768, 0.005804580590860159, 0.00027529600705133663}, Den: []float64{3.917294261258692e-05, 0.1065308384054887, 3.7991514117509955e-06, 0.0005503052126186067}},
		{Approximation: "identity", Version: "D", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{0, 1, 0, 0, 0, 0}, Den: []float64{0, 0, 0, 0}},
		{Approximation: "relu", Version: "D", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{0.033897129202224346, 0.4999985439606278, 1.6701363611130988, 1.9901021632350815, 0.9413089613384323, 0.1509133373584318}, Den: []float64{-2.1040152094202414e-05, 3.980247851167207, -3.166344237241501e-05, 0.30183382300945066}},
		{Approximation: "leaky_relu", Version: "D", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{0.03355815733452378, 0.5050059186781659, 1.6534697310952233, 2.0100501552600014, 0.9319195586484518, 0.15242646398275836}, Den: []float64{4.812831860534153e-05, 3.9802479055059194, 1.5837587665720168e-05, 0.30183382839661377}},
		{Approximation: "tanh", Version: "N", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{-3.145423530423389e-09, 0.9999942494014291, -1.9721597767154503e-06, 0.10770862804964325, -9.136256462122441e-08, 0.0008711236928997527}, Den: []float64{-2.0011607852015447e-06, 0.44101416266969534, -7.123442514694839e-07, 0.014581034132660431}},
		{Approximation: "sigmoid", Version: "N", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{0.4999999997037559, 0.2499957412799507, 0.05544062622341843, 0.006887644058850528, 0.00048485212592602815, 1.5645989044598313e-05}, Den: []float64{-8.506532784269549e-06, 0.11088550238903279, -8.302465236326423e-07, 0.0009697677495734157}},
		{Approximation: "leaky_relu", Version: "N", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{0.0335582132067107, 0.505003740582471, 1.653458603975765, 2.010035526870634, 0.9319123754639458, 0.1524253263766341}, Den: []float64{2.9474567624015346e-05, 3.9802415438832024, 5.162142903703752e-06, 0.3018331889980995}},
		{Approximation: "silu", Version: "N", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{3.266881610395981e-07, 0.49999999081326113, 0.24997332715914988, 0.05325340025674068, 0.0058003403315322, 0.00027497533639232177}, Den: []float64{-4.80777614312505e-05, 0.10653079960025053, -4.663835554309269e-06, 0.0005503029408021047}},
		{Approximation: "swish", Version: "N", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{3.266881610395981e-07, 0.49999999081326113, 0.24997332715914988, 0.05325340025674068, 0.0058003403315322, 0.00027497533639232177}, Den: []float64{-4.80777614312505e-05, 0.10653079960025053, -4.663835554309269e-06, 0.0005503029408021047}},
		{Approximation: "gelu", Version: "N", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{-0.001080882882487456, 0.5025161652007099, 0.6283336330558847, 0.2973720903550247, 0.06322520452943567, 0.005097017287773147}, Den: []float64{0.44045431304569604, 0.24638482058938668, 0.08444750846536668, 0.0047540424624096115}},
		{Approximation: "leaky_relu", Version: "A", NumeratorDegree: 3, DenominatorDegree: 2}: {
			Num: []float64{0.11440311001189646, 0.5891610756805713, 0.5423986871505561, 0.13286948390430828}, Den: []float64{0.2074994452826854, 0.20726656124953777}},
		{Approximation: "identity", Version: "A", NumeratorDegree: 3, DenominatorDegree: 2}: {
			Num: []float64{1.2892809281246332e-20, 1.000000000614517, 7.883537508306838e-12, 0.6183612729447092}, Den: []float64{8.511708691315752e-10, 0.6183612726994397}},
		{Approximation: "leaky_relu_0.1", Version: "A", NumeratorDegree: 3, DenominatorDegree: 2}: {
			Num: []float64{0.10612873905150731, 0.6133592016772789, 0.4602314910152753, 0.1334141245745798}, Den: []float64{0.14236896839932014, 0.20440997200950106}},
		{Approximation: "leaky_relu_0.1", Version: "B", NumeratorDegree: 3, DenominatorDegree: 2}: {
			Num: []float64{0.11143814042285753, 0.5500043254094745, 0.38748030194742117, 0.10859116398267224}, Den: []float64{7.370237749086212e-06, 0.19743788349488567}},
		{Approximation: "leaky_relu_0.1", Version: "C", NumeratorDegree: 3, DenominatorDegree: 2}: {
			Num: []float64{1299.252221054247, 6412.385498792421, 4517.481746168571, 1266.0305574666183}, Den: []float64{11658.855469503093, -0.1578883069439065, 2301.9197200860253}},
		{Approximation: "tanh", Version: "A", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{-1.0804622559204184e-08, 1.0003008043819048, -2.5878199375289335e-08, 0.09632129918392647, 3.4775841628196104e-09, 0.0004255709234726337}, Den: []float64{-0.0013027181209176277, 0.428349017422072, 1.4524304083061898e-09, 0.010796648111337176}},
		{Approximation: "sigmoid", Version: "A", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{0.4999992534599381, 0.25002157564685185, 0.14061924500301096, 0.049420492431596394, 0.00876714851885483, 0.0006442412789159799}, Den: []float64{2.1694506382753683e-09, 0.28122766100417684, 1.0123620714203357e-05, 0.017531988049946}},
		{Approximation: "relu", Version: "A", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{0.029963801610813613, 0.6168978366891341, 2.37534759733888, 3.0659900472408443, 1.5246831881677423, 0.2528070864040542}, Den: []float64{-1.191550121923625, 4.4080487697236626, 0.9110357113686055, 0.34884977946384615}},
		{Approximation: "gelu", Version: "A", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{-0.0012423594497499122, 0.5080497063245629, 0.41586363182937475, 0.13022718688035761, 0.024355900098993424, 0.00290283948155535}, Den: []float64{-0.06675015696494944, 0.17927646217001553, 0.03746682605496631, 1.6561610853276082e-10}},
		{Approximation: "swish", Version: "A", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{3.054879741161051e-07, 0.5000007853744493, 0.24999783422824703, 0.05326628273219478, 0.005803034571292244, 0.0002751961022402342}, Den: []float64{-4.111554955950634e-06, 0.10652899335007572, -1.2690007399796238e-06, 0.0005502331264140556}},
		{Approximation: "leaky_relu_0.1", Version: "A", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{0.028001623256966722, 0.6356606448768645, 1.9230178392192485, 2.96752700768763, 1.1911336843997844, 0.23922607056505849}, Den: []float64{-0.7802079437303349, 4.278190623059944, 0.5865922075979962, 0.33447021316466974}},
		{Approximation: "identity", Version: "A", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{0, 1, 0, 0, 0, 0}, Den: []float64{0, 0, 0, 0}},
		{Approximation: "leaky_relu", Version: "A", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{0.029792778657264946, 0.6183735264987601, 2.323309062531321, 3.051936237265109, 1.4854203263828845, 0.2510244961111299}, Den: []float64{-1.1419548357285474, 4.393159974992486, 0.8714712309957245, 0.34719662339598834}},
		{Approximation: "gelu", Version: "C", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{-0.06332215990776038, 75.00371336515258, 60.40299979439254, 11.046215295208759, -1.944610878554029, -0.5613502889957269}, Den: []float64{149.90738995374193, -0.00414880098336288, 22.09585877904437, -0.0009128752792880584, -1.122625040450191}},
		{Approximation: "sigmoid", Version: "C", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{9.292539203868202, 4.646322961463492, 1.0304346137104188, 0.12802035070214404, 0.009012006389028321, 0.000290782121558725}, Den: []float64{18.485078400296143, 0.00010692547716301122, 2.0608158064816653, 1.0436027843534009e-05, 0.01802321462478323}},
		{Approximation: "swish", Version: "C", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{9.623223463794738e-06, 14.75722171199872, 7.379106424221765, 1.5723856435823387, 0.17131862639152923, 0.008125189146807921}, Den: []float64{29.41444303593795, 0.0011467081629387187, 3.144198880793362, 0.00011121774973276025, 0.016241982787967915}},
		{Approximation: "leaky_relu_0.1", Version: "C", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{93.05767974217531, 1677.672893036559, 4584.985076631346, 6677.44618254537, 2584.1222750745555, 506.36164108885333}, Den: []float64{3050.2117646277247, -0.028096613493514083, 12140.898972387418, -0.09299740884806114, 920.6758597596518}},
		{Approximation: "identity", Version: "C", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{0, 1, 0, 0, 0, 0}, Den: []float64{0.9, 0, 0, 0, 0}},
		{Approximation: "relu", Version: "C", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{0.7865035431863266, 11.601593145754968, 38.7535199511595, 46.17884158100411, 21.842701976543186, 3.501924367528041}, Den: []float64{23.103140929589287, 0.0004299501319296557, 92.356987965797, 0.000308676388304253, 7.003817416523842}},
		{Approximation: "leaky_relu", Version: "C", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{0.863943144302673, 13.001131096231564, 42.56763919941753, 51.74757697869782, 23.99165145314087, 3.924120833060723}, Den: []float64{25.644974747512343, -0.0013495866374991824, 102.47308203485366, -0.0017931970367811253, 7.77090005615082}},
		{Approximation: "tanh", Version: "C", NumeratorDegree: 5, DenominatorDegree: 4}: {
			Num: []float64{9.143817420311582e-08, 46.87601105103956, 9.606864506795431e-05, 5.048980033620479, 4.469361939672058e-06, 0.04083504407681448}, Den: []float64{46.77628057640709, 9.74704061164436e-05, 20.67310382546532, 3.473303618435723e-05, 0.6835046974230089}},

		// Add new cache entries here:
		// (see rational.ipynb)
		{Approximation: "swish", Version: "B", NumeratorDegree: 6, DenominatorDegree: 5}: {
			Num:          []float64{-0.0012891008526019756, 0.5621181587194732, 0.2948299778301195, 0.13422493978570096, 0.0382134224174028, 0.004988841589258075, 0.00022819454081435546},
			Den:          []float64{-0.331710580046256, -0.013566191937184724, -0.07888001168090944, 0.0003079036213871363, -0.00047259821266641564},
			GainEstimate: 2.8154571846095324}}
)
