// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package oxfordflowers102 provides tools to download and cache the dataset
// and a `train.Dataset` implementation that can be used to train models
// using GoMLX (http://github.com/gomlx/gomlx/).
//
// Details in the README.md file. The dataset's home page is in
// https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
//
// Usage example:
package oxfordflowers102

// This file contains constants of the dataset, including the flowers names.

// NumLabels is 102, hence the name of the dataset being Oxford Flowers 102.
const NumLabels = 102

var (
	// AllLabels of the dataset. Converted to 0-based (0 to 101).
	// Only available after DownloadAndParse is successfully called.
	AllLabels []int32

	// AllImages of the dataset, the path to the images that is.
	// Only available after DownloadAndParse is successfully called.
	AllImages []string

	// NumExamples is the number of examples (images and labels) in the dataset.
	// Only available after DownloadAndParse is successfully called.
	NumExamples int

	// ImagesDir where images are stored. Only available after DownloadAndParse is
	// successfully called.
	ImagesDir string

	// Names of all the 102 flowers in the dataset.
	Names = []string{
		"pink primrose",
		"hard-leaved pocket orchid",
		"canterbury bells",
		"sweet pea",
		"english marigold",
		"tiger lily",
		"moon orchid",
		"bird of paradise",
		"monkshood",
		"globe thistle",
		"snapdragon",
		"colt's foot",
		"king protea",
		"spear thistle",
		"yellow iris",
		"globe-flower",
		"purple coneflower",
		"peruvian lily",
		"balloon flower",
		"giant white arum lily",
		"fire lily",
		"pincushion flower",
		"fritillary",
		"red ginger",
		"grape hyacinth",
		"corn poppy",
		"prince of wales feathers",
		"stemless gentian",
		"artichoke",
		"sweet william",
		"carnation",
		"garden phlox",
		"love in the mist",
		"mexican aster",
		"alpine sea holly",
		"ruby-lipped cattleya",
		"cape flower",
		"great masterwort",
		"siam tulip",
		"lenten rose",
		"barbeton daisy",
		"daffodil",
		"sword lily",
		"poinsettia",
		"bolero deep blue",
		"wallflower",
		"marigold",
		"buttercup",
		"oxeye daisy",
		"common dandelion",
		"petunia",
		"wild pansy",
		"primula",
		"sunflower",
		"pelargonium",
		"bishop of llandaff",
		"gaura",
		"geranium",
		"orange dahlia",
		"pink-yellow dahlia?",
		"cautleya spicata",
		"japanese anemone",
		"black-eyed susan",
		"silverbush",
		"californian poppy",
		"osteospermum",
		"spring crocus",
		"bearded iris",
		"windflower",
		"tree poppy",
		"gazania",
		"azalea",
		"water lily",
		"rose",
		"thorn apple",
		"morning glory",
		"passion flower",
		"lotus",
		"toad lily",
		"anthurium",
		"frangipani",
		"clematis",
		"hibiscus",
		"columbine",
		"desert-rose",
		"tree mallow",
		"magnolia",
		"cyclamen",
		"watercress",
		"canna lily",
		"hippeastrum",
		"bee balm",
		"ball moss",
		"foxglove",
		"bougainvillea",
		"camellia",
		"mallow",
		"mexican petunia",
		"bromelia",
		"blanket flower",
		"trumpet creeper",
		"blackberry lily",
	}
)
