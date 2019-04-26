/*
MIT License

Copyright (c) 2019 문동선 (NaniteFactory)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/gosuri/uitable"
	"github.com/nanitefactory/naneat"
)

func main() {
	flag.Usage = func() {
		println("Usage: naneatviewer [flags] [target1] [target2]\n")

		println("Flags:\n")
		flag.PrintDefaults()
		println()

		println("Targets:\n")
		for _, m := range []struct{ name, desc string }{
			{"target1", "(mandatory) filepath or filename to a json you want to open and view"},
			{"target2", "(optional) filepath or filename to a json you want to open and view"},
		} {
			fmt.Printf("  %v%v%v\n", m.name, strings.Repeat(" ", 12-len(m.name)), m.desc)
		}
		println()

		os.Exit(0)
	}
	pbShort := flag.Bool("short", false, "Omit details if given.")
	flag.Parse()
	filepath1, filepath2 := flag.Arg(0), flag.Arg(1)
	if filepath1 == "" {
		log.Fatalln("Please provide filepath to your JSON.")
	}
	experimenter1, experimenter2 := func() (e1, e2 naneat.Experimenter) {
		var err error
		ark1, err := naneat.NewArkFromFile(filepath1)
		if err != nil {
			log.Fatalln("Failed to load from JSON. Reason:", err)
		}
		if e1, err = ark1.New(); err != nil {
			log.Fatalln("Failed to load Experimenter:", err)
		}
		if filepath2 != "" {
			ark2, err := naneat.NewArkFromFile(filepath2)
			if err != nil {
				log.Fatalln("Failed to load from JSON. Reason:", err)
			}
			if e2, err = ark2.New(); err != nil {
				log.Fatalln("Failed to load Experimenter:", err)
			}
		}
		return e1, e2
	}()

	{
		// tableSpecies
		tableSpecies := uitable.New()
		tableSpecies.MaxColWidth = 40
		tableSpecies.Wrap = false
		schemeSpecies := []interface{}{"Species", "Niche", "Population", "Stagnancy", "TopFitness", "TopFitness.Adj", "AvgFitness.Adj", "Dominance.Top", "Dominance.Avg"}
		if experimenter2 != nil {
			schemeSpecies = append(schemeSpecies, schemeSpecies...)
		}
		tableSpecies.AddRow(schemeSpecies...)
		// tableOrgans
		tableOrgans := uitable.New()
		tableOrgans.MaxColWidth = 40
		tableOrgans.Wrap = false
		schemeOrgans := []interface{}{"Species", "Organism", "Fitness", "IsMeasured"}
		if experimenter2 != nil {
			schemeOrgans = append(schemeOrgans, schemeOrgans...)
		}
		tableOrgans.AddRow(schemeOrgans...)
		// def
		updateSpecies := func(e naneat.Experimenter) {
			e.Self().MutexClasses.Lock()
			for _, group := range e.Self().Classes { // update species
				_, _, err := group.EvaluateGeneration() // update stagnancy and topFitness
				if err != nil {
					log.Fatalln("Cannot evaluate species:", err)
				}
			}
			e.Self().MutexClasses.Unlock() // although mutex isn't really necessary
		}
		getRowsOfTableSpecies := func(e naneat.Experimenter) [][]interface{} {
			sumFitTopAdj, fitTopsAdj := e.Self().AdjTopFitnessesOfSpecies()
			sumFitAvgAdj, fitAvgsAdj := e.Self().AdjAvgFitnessesOfSpecies()
			rows := make([][]interface{}, len(e.Status().Breeds()))
			for iSpecies, species := range e.Status().Breeds() {
				species.Sort()
				topFitAdj := fitTopsAdj[iSpecies]
				avgFitAdj := fitAvgsAdj[iSpecies]
				dominanceByTopFitAdj := topFitAdj / sumFitTopAdj
				dominanceByAvgFitAdj := avgFitAdj / sumFitAvgAdj
				rows[iSpecies] = []interface{}{
					iSpecies,
					species.Niche(),
					species.Size(),
					species.Stagnancy,
					species.TopFitness,
					fmt.Sprint(topFitAdj, " ("+fmt.Sprint(math.Floor(dominanceByTopFitAdj*100))+"%)"),
					fmt.Sprint(avgFitAdj, " ("+fmt.Sprint(math.Floor(dominanceByAvgFitAdj*100))+"%)"),
					fmt.Sprint(math.Floor(dominanceByTopFitAdj*float64(e.Self().Config.SizePopulation)), " ("+fmt.Sprint(math.Floor(dominanceByTopFitAdj*100))+"%)"),
					fmt.Sprint(math.Floor(dominanceByAvgFitAdj*float64(e.Self().Config.SizePopulation)), " ("+fmt.Sprint(math.Floor(dominanceByAvgFitAdj*100))+"%)"),
				}
			}
			return rows
		}
		getRowsOfTableOrgans := func(e naneat.Experimenter) [][]interface{} {
			if *pbShort { // perform tableOrgans
				rows := [][]interface{}{
					[]interface{}{"-", "-", "-", "-"},
				}
				return rows
			}
			rows := make([][]interface{}, len(e.Status().Breeds())+len(e.Status().Organisms()))
			e.Self().MutexClasses.Lock() // although mutex isn't really necessary
			sortByProminence, err := e.Self().GetSpeciesOrderedByProminence()
			e.Self().MutexClasses.Unlock() // although mutex isn't really necessary
			if err != nil {
				log.Fatalln("Failed to get species sort by prominence:", err)
			}
			i := 0
			for _, species := range sortByProminence {
				rows[i] = []interface{}{species.Niche(), "", "", ""}
				i++
				for iOrgan, organ := range species.Livings {
					rows[i] = []interface{}{
						"",
						iOrgan,
						organ.Fitness,
						organ.IsMeasured,
					}
					i++
				}
			}
			return rows
		}
		// perform
		if experimenter2 == nil {
			updateSpecies(experimenter1)
			rowsSpecies := getRowsOfTableSpecies(experimenter1)
			rowsOrgans := getRowsOfTableOrgans(experimenter1)
			for _, row := range rowsSpecies {
				tableSpecies.AddRow(row...)
			}
			for _, row := range rowsOrgans {
				tableOrgans.AddRow(row...)
			}
		} else {
			updateSpecies(experimenter1)
			rowsSpecies1 := getRowsOfTableSpecies(experimenter1)
			rowsOrgans1 := getRowsOfTableOrgans(experimenter1)
			updateSpecies(experimenter2)
			rowsSpecies2 := getRowsOfTableSpecies(experimenter2)
			rowsOrgans2 := getRowsOfTableOrgans(experimenter2)
			for len(rowsSpecies1) < len(rowsSpecies2) {
				rowsSpecies1 = append(rowsSpecies1, []interface{}{"-", "-", "-", "-", "-", "-", "-", "-", "-"})
			}
			for len(rowsSpecies1) > len(rowsSpecies2) {
				rowsSpecies2 = append(rowsSpecies2, []interface{}{"-", "-", "-", "-", "-", "-", "-", "-", "-"})
			}
			for len(rowsOrgans1) < len(rowsOrgans2) {
				rowsOrgans1 = append(rowsOrgans1, []interface{}{"-", "-", "-", "-"})
			}
			for len(rowsOrgans1) > len(rowsOrgans2) {
				rowsOrgans2 = append(rowsOrgans2, []interface{}{"-", "-", "-", "-"})
			}
			for iSpecies, row1 := range rowsSpecies1 {
				tableSpecies.AddRow(append(row1, rowsSpecies2[iSpecies]...)...)
			}
			for iOrgan, row1 := range rowsOrgans1 {
				tableOrgans.AddRow(append(row1, rowsOrgans2[iOrgan]...)...)
			}
		}
		// print
		fmt.Println()
		fmt.Println(" ~ Species Rank ~ ")
		fmt.Println()
		fmt.Println(filepath.Base(filepath1), "vs.", filepath.Base(filepath2))
		fmt.Println()
		fmt.Println(tableSpecies)
		fmt.Println()

		fmt.Println()
		fmt.Println(" ~ Individual Rank ~ ")
		fmt.Println()
		fmt.Println(filepath.Base(filepath1), "vs.", filepath.Base(filepath2))
		fmt.Println()
		fmt.Println(tableOrgans)
		fmt.Println()
	}
}
