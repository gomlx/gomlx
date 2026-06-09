// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"k8s.io/klog/v2"
)

// notebooks maps the relative path of the Jupyter notebook to the relative path of the exported HTML.
var notebooks = map[string]string{
	"examples/tutorial/tutorial.ipynb":     "docs/notebooks/tutorial.html",
	"examples/adult/uci-adult.ipynb":       "docs/notebooks/uci-adult.html",
	"examples/cifar/cifar.ipynb":           "docs/notebooks/cifar.html",
	"examples/mnist/mnist.ipynb":           "docs/notebooks/mnist.html",
	"examples/dogsvscats/dogsvscats.ipynb": "docs/notebooks/dogsvscats.html",
	"examples/spiral/Spiral.ipynb":         "docs/notebooks/Spiral.html",
	"examples/imdb/imdb.ipynb":             "docs/notebooks/imdb.html",
	// "examples/oxfordflowers102/OxfordFlowers102_Diffusion.ipynb": "docs/notebooks/OxfordFlowers102_Diffusion.html",
	// "examples/ogbnmag/ogbn-mag.ipynb":                            "docs/notebooks/ogbn-mag.html",
	// "examples/fft/fft.ipynb":                                     "docs/notebooks/fft.html",
	// "examples/discretekan/kans_shapes.ipynb":                     "docs/notebooks/kans_shapes.html",
	// "examples/discretekan/discrete-kan.ipynb":                   "docs/notebooks/discrete-kan.html",
	// "examples/FlowMatching/flow_matching.ipynb":                  "docs/notebooks/flow_matching.html",
	// "ml/layers/rational/rational.ipynb":                          "docs/notebooks/rational.html",
}

func main() {
	flagNotebook := flag.String("n", "", "Specify only one exact notebook to run (checks for exact match or suffix match)")
	flagForce := flag.Bool("force", false, "Force re-running of notebooks even if they are up-to-date")
	flag.Parse()

	_, err := os.Getwd()
	if err != nil {
		klog.Errorf("Failed to get current working directory: %v\n", err)
		os.Exit(1)
	}

	hasRunAny := false
	for ipynbPath, htmlPath := range notebooks {
		if *flagNotebook != "" {
			target := strings.TrimPrefix(*flagNotebook, "./")
			if ipynbPath != target && !strings.HasSuffix(ipynbPath, target) {
				continue
			}
		}

		hasRunAny = true
		klog.V(1).Infof("Checking notebook: %s -> %s\n", ipynbPath, htmlPath)

		// Check if execution is needed based on modification times
		runNeeded := false
		if *flagForce {
			runNeeded = true
		} else {
			ipynbInfo, err := os.Stat(ipynbPath)
			if err != nil {
				klog.Errorf("Error stating notebook %s: %v\n", ipynbPath, err)
				os.Exit(1)
			}
			htmlInfo, err := os.Stat(htmlPath)
			if err != nil {
				if os.IsNotExist(err) {
					runNeeded = true
				} else {
					klog.Errorf("Error stating html %s: %v\n", htmlPath, err)
					os.Exit(1)
				}
			} else {
				if ipynbInfo.ModTime().After(htmlInfo.ModTime()) {
					runNeeded = true
				}
			}
		}

		if !runNeeded {
			fmt.Printf("✅ %s is up-to-date, skipping (use -force to run anyway)\n", ipynbPath)
			continue
		}

		fmt.Printf("  Running notebook %s...", ipynbPath)
		err = runNotebook(ipynbPath, htmlPath)
		if err != nil {
			klog.Errorf("Failed to run notebook %s: %v\n", ipynbPath, err)
			os.Exit(1)
		}
		fmt.Printf("\r✅ Successfully executed and exported %s\n", ipynbPath)
	}

	if *flagNotebook != "" && !hasRunAny {
		klog.Errorf("No notebooks matched filter: %q\n", *flagNotebook)
		os.Exit(1)
	}
}

// runNotebook executes a single notebook using nbexec and handles special validation for tutorial.ipynb.
func runNotebook(ipynbPath, htmlPath string) error {
	args := []string{
		"run",
		"github.com/janpfeifer/gonb/cmd/nbexec",
		"-n", ipynbPath,
		"-export_html", htmlPath,
		"-vmodule=nbexec=1",
		"-check_cells",
	}

	cmd := exec.Command("go", args...)

	// Prepend virtualenv path if jupyter is not globally available in the current environment
	cmd.Env = os.Environ()
	if _, err := exec.LookPath("jupyter"); err != nil {
		venvBin := "/home/janpf/.venv/jupyter/bin"
		if _, err := os.Stat(filepath.Join(venvBin, "jupyter")); err == nil {
			for i, e := range cmd.Env {
				if strings.HasPrefix(e, "PATH=") {
					cmd.Env[i] = "PATH=" + venvBin + string(os.PathListSeparator) + e[5:]
					break
				}
			}
		}
	}

	var outBuf bytes.Buffer
	cmd.Stdout = &outBuf
	cmd.Stderr = &outBuf

	err := cmd.Run()
	outputStr := outBuf.String()

	isTutorial := strings.HasSuffix(ipynbPath, "tutorial.ipynb")
	if isTutorial {
		// Tutorial notebook contains a deliberate error to demonstrate error reporting.
		// nbexec with -check_cells will exit with exit code 1 when cell execution fails.
		exitErr, ok := err.(*exec.ExitError)
		if !ok || exitErr.ExitCode() != 1 {
			klog.Errorf("nbexec output:\n%s\n", outputStr)
			return fmt.Errorf("tutorial notebook expected exit code 1, got error: %v", err)
		}

		// Verify exactly one cell failed execution.
		errCount := strings.Count(outputStr, "failed execution:")
		if errCount != 1 {
			klog.Errorf("nbexec output:\n%s\n", outputStr)
			return fmt.Errorf("tutorial notebook expected exactly 1 failing cell, found %d", errCount)
		}

		// Verify the specific expected error message is in the traceback output.
		expectedMsg := `cannot broadcast Float64 and Int64 for "Add": they have different dtypes`
		if !strings.Contains(outputStr, expectedMsg) {
			klog.Errorf("nbexec output:\n%s\n", outputStr)
			return fmt.Errorf("tutorial notebook did not contain the expected error message: %q", expectedMsg)
		}

		// Return nil as this failure matches our target test condition
		return nil
	}

	// For normal notebooks, no errors should occur and cmd.Run() should complete with code 0.
	if err != nil {
		klog.Errorf("nbexec output:\n%s\n", outputStr)
		return err
	}

	return nil
}
