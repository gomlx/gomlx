package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"

	"github.com/gomlx/gomlx/internal/must"
	"k8s.io/klog/v2"
)

const AltPrefix = "//alt:"

// processLine transforms a single line based on the target tag.
// If the line has a comment like `//alt:tag1|tag2|...` at the start, and any of the tags match
// `targetTag`, the comment is moved to the end, effectively "uncommenting" the code.
// Otherwise, the line is returned as is.
func processLine(line string, targetTag string) string {
	// Check for the presence of alt tag
	idx := strings.Index(line, AltPrefix)
	if idx == -1 {
		return line
	}

	// Find where the tag ends (at the first space or end of the line)
	tagStart := idx + len(AltPrefix)
	spaceIdx := strings.IndexAny(line[tagStart:], " \t")
	var lineTag string
	var rest string

	if spaceIdx == -1 {
		// No space found - the entire rest is the tag
		lineTag = line[tagStart:]
		rest = ""
	} else {
		lineTag = line[tagStart : tagStart+spaceIdx]
		rest = line[tagStart+spaceIdx+1:]
	}

	// Remove the alt tag part from the line
	beforeTag := line[:idx]
	beforeTag = strings.TrimRight(beforeTag, " \t")

	// Combine all parts
	// Split lineTag into multiple tags if pipe separator exists
	tags := strings.Split(lineTag, "|")
	for _, tag := range tags {
		if tag == targetTag {
			// Move tag to the end
			if rest == "" && beforeTag == "" {
				return AltPrefix + lineTag
			}
			return strings.TrimSpace(beforeTag+" "+rest) + " " + AltPrefix + lineTag
		}
	}

	// Move tag to the beginning
	combined := strings.TrimSpace(beforeTag + " " + rest)
	if combined == "" {
		return AltPrefix + lineTag
	}
	return AltPrefix + lineTag + " " + combined
}

func main() {
	// 1. Define and parse command-line flags.
	baseFile := flag.String("base", "", "The base Go file to process (e.g., app.go).")
	tagsStr := flag.String("tags", "", "A comma-separated list of tags (e.g., free,pro).")
	flag.Parse()

	if *baseFile == "" || *tagsStr == "" {
		fmt.Println("❌ Both -base and -tags flags are required.")
		flag.Usage()
		os.Exit(1)
	}

	// 2. Read the entire base file into memory.
	content, err := os.ReadFile(*baseFile)
	if err != nil {
		klog.Fatalf("🚨 Failed to read base file %s: %v", *baseFile, err)
	}
	lines := strings.Split(string(content), "\n")

	// 3. Get information for naming the output files.
	tags := strings.Split(*tagsStr, ",")
	baseFileName := filepath.Base(*baseFile)
	baseName := strings.TrimSuffix(baseFileName, filepath.Ext(baseFileName))

	// 4. Process the file for each specified tag.
	for _, tag := range tags {
		trimmedTag := strings.TrimSpace(tag)
		if trimmedTag == "" {
			continue
		}
		processFileForTag(trimmedTag, baseName, lines)
	}
}

func processFileForTag(tag string, baseName string, lines []string) {
	// Create the output file.
	outputFileName := fmt.Sprintf("gen_%s_%s.go", baseName, tag)
	outputFileName = path.Join(must.M1(os.Getwd()), outputFileName)
	outFile, err := os.Create(outputFileName)
	if err != nil {
		klog.Fatalf("🚨 Failed to create output file %s: %v", outputFileName, err)
		return
	}
	defer outFile.Close()
	writer := bufio.NewWriter(outFile)

	// Process each line and write to the new file.
	for i, line := range lines {
		// Avoid writing an extra newline if the original file ends with one.
		if i == len(lines)-1 && line == "" {
			continue
		}
		processedLine := processLine(line, tag)
		fmt.Fprintln(writer, processedLine)
	}

	if err := writer.Flush(); err != nil {
		klog.Fatalf("🚨 Failed to write to %s: %v", outputFileName, err)
	}

	// Run go fmt on the generated file
	cmd := exec.Command("go", "fmt", outputFileName)
	if err := cmd.Run(); err != nil {
		klog.Warningf("Failed to run go fmt on %s: %v", outputFileName, err)
	}
	fmt.Printf("✅ alternates_generator:\tsuccessfully generated %s\n", outputFileName)
}
