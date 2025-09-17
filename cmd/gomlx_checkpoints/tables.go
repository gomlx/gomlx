package main

import (
	"github.com/charmbracelet/lipgloss"
	lgtable "github.com/charmbracelet/lipgloss/table"
)

var (
	headerRowStyle = lipgloss.NewStyle().Reverse(true).
			Padding(0, 2, 0, 2).Align(lipgloss.Center)
	oddRowStyle = lipgloss.NewStyle().Faint(false).
			PaddingLeft(1).PaddingRight(1)
	evenRowStyle = lipgloss.NewStyle().Faint(true).
			PaddingLeft(1).PaddingRight(1)
	redRowStyle = lipgloss.NewStyle().
			Foreground(lipgloss.AdaptiveColor{Light: "9", Dark: "9"}).
			Bold(true).
			PaddingLeft(1).PaddingRight(1)
)

func newPlainTable(withHeader bool, alignments ...lipgloss.Position) *lgtable.Table {
	t := newPlainTableWithReds(withHeader, alignments...)
	return t.Table
}

type TableWithReds struct {
	Table *lgtable.Table
	Count int
	Reds  map[int]bool
}

func (t *TableWithReds) Row(isRed bool, row ...string) {
	if isRed {
		t.Reds[t.Count] = true
	}
	t.Table.Row(row...)
	t.Count++
}

func newPlainTableWithReds(withHeader bool, alignments ...lipgloss.Position) *TableWithReds {
	t := &TableWithReds{
		Reds: make(map[int]bool),
	}
	t.Table = lgtable.New().
		Border(lipgloss.NormalBorder()).
		BorderStyle(lipgloss.NewStyle().Foreground(lipgloss.Color("99"))).
		StyleFunc(func(row, col int) (s lipgloss.Style) {
			if row < 0 {
				s = headerRowStyle
				return
			}
			if t.Reds[row] {
				s = redRowStyle
			} else {
				switch {
				case row%2 == 0:
					// Even row style.
					s = oddRowStyle
				default:
					// Odd row style
					s = evenRowStyle
				}
			}
			alignment := lipgloss.Left
			if col < len(alignments) {
				alignment = alignments[col]
			} else if len(alignments) > 0 {
				alignment = alignments[len(alignments)-1]
			}
			s = s.Align(alignment)
			return
		})
	return t
}

func isAllEqual[E comparable](s []E) bool {
	if len(s) == 0 {
		return true
	}
	v := s[0]
	for ii := range len(s) - 1 {
		if v != s[ii+1] {
			return false
		}
	}
	return true
}
