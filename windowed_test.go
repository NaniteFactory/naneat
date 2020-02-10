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

package naneat_test // black box

import (
	"flag"
	"fmt"
	"math"
	"os"
	"testing"
	"time"

	"github.com/faiface/pixel"
	"github.com/nanitefactory/naneat"
	"github.com/nanitefactory/visual"
	"github.com/sqweek/dialog"
	"golang.org/x/image/colornames"
)

// ----------------------------------------------------------------------------
// TestMain

// TestMain is important.
func TestMain(*testing.M) {
	{
		var bFlagHeadless = false
		flag.BoolVar(&bFlagHeadless, "headless", false, "If set to true, the test runs in non-windowed mode.")
		flag.Parse()
		if bFlagHeadless {
			return
		}
	}

	loadExperimenterFromJSON := func(defaultConfig *naneat.Configuration) (experimenter naneat.Experimenter) {
		askToLoadJSON := func() *naneat.Ark {
			for { // Load JSON
				cwd, _ := os.Getwd()
				filepath, err := dialog.File().Title("Load JSON").
					Filter("JSON Format (*.json)", "json").
					Filter("All Files (*.*)", "*").
					SetStartDir(cwd).Load()
				if err != nil {
					if err.Error() == "Cancelled" {
						return nil
					}
					dialog.Message("%s", "Invalid file path."+"\r\n"+"\r\n"+fmt.Sprint(err)).Title("Failed to load JSON").Error()
					continue
				}
				ark, err := naneat.NewArkFromFile(filepath)
				if err != nil {
					dialog.Message("%s", "Reason:"+"\r\n"+"\r\n"+fmt.Sprint(err)).Title("Failed to load from JSON").Error()
					continue
				}
				return ark
			}
		}
		if ark := askToLoadJSON(); ark != nil {
			var err error
			experimenter, err = ark.New()
			if err != nil {
				experimenter = naneat.New(defaultConfig)
			}
		} else {
			experimenter = naneat.New(defaultConfig)
		}
		return experimenter
	}

	testRandomFitness := func() {
		// mainthread
		cfg := visual.Config{
			Bg:                  pixel.ToRGBA(colornames.Mediumseagreen),
			OnPaused:            nil,
			OnResumed:           nil,
			OnDrawn:             nil,
			OnUpdated:           nil,
			OnResized:           nil,
			OnClose:             nil,
			OnHandlingEvents:    nil,
			OnLogging:           func(v ...interface{}) { panic(fmt.Sprint(v...)) },
			WinCentered:         false,
			Undecorated:         false,
			Title:               "Random Fitness",
			Version:             "undefined version",
			Width:               60000.0,
			Height:              20000.0,
			WinWidth:            800.0,
			WinHeight:           800.0,
			InitialZoomLevel:    -3.0,
			InitialRotateDegree: -360.0,
		}
		visualizer := visual.NewVisualizer(cfg, nil)
		const L, R = 0, 1
		const nNets, nInputs, nOutputs = 2, 38 * 28, 7 // 5, 5
		const width, height = 600, 400
		const maxTrial1, maxTrial2, maxTrial3 = 1000000, 1000000, 1000000
		rectCentered := pixel.R(0, 0, width, height).Moved(cfg.PosCenterGame())

		// supplier's
		sup := func(agent *naneat.Agent, cntTrial *int, maxTrial int, boundL, boundR pixel.Rect) {
			// defer func() {
			// 	if r := recover(); r != nil { //
			// 		log.Println("sup(): panic recovered")
			// 		log.Println("reason for panic:", r)
			// 		log.Println("stacktrace from panic: \r\n" + string(debug.Stack()))
			// 	}
			// }()
			var brainActorL, brainActorR naneat.NeuralNetworkActor
			// run loopy run
			for *cntTrial = 0; *cntTrial < maxTrial; (*cntTrial)++ {
				brains := func() []*naneat.NeuralNetwork { // awake
					brains := agent.Receive()
					if !visualizer.RemoveActor(brainActorL) && brainActorL != nil {
						panic(fmt.Sprint("failed to remove `brainActorL` from `visualizer`:", brainActorL))
					}
					if !visualizer.RemoveActor(brainActorR) && brainActorR != nil {
						panic(fmt.Sprint("failed to remove `brainActorR` from `visualizer`:", brainActorR))
					}
					brainActorL, brainActorR = func(l, r *naneat.NeuralNetwork, boundL, boundR pixel.Rect) (actorL, actorR naneat.NeuralNetworkActor) {
						var err error
						actorL, err = naneat.NewNeuralNetworkActor(
							l, 1,
							boundL, 10, 10, 10, 10,
							16, 4, true, true, true,
							false, true, false, false,
							nil, nil,
						)
						if err != nil {
							panic(err)
						}
						actorR, err = naneat.NewNeuralNetworkActor(
							r, 1,
							boundR, 10, 10, 10, 10,
							16, 4, true, true, true,
							false, true, false, false,
							nil, nil,
						)
						if err != nil {
							panic(err)
						}
						return actorL, actorR
					}(brains[L], brains[R], boundL, boundR) // actorize
					visualizer.PushActors(brainActorL, brainActorR)
					return brains
				}()
				{ // eval l
					brainTopoSortLeft, err := brains[L].Sort()
					if err != nil {
						panic(err)
					}
					input := make([]naneat.NeuralValue, nInputs)
					for i := 0; i < len(input); i++ {
						input[i] = naneat.NeuralValueMax
					}
					brains[L].Evaluate(input)
					// visualize
					if len(brainActorL.Channel()) < cap(brainActorL.Channel()) {
						toSend := make([]naneat.NeuralNode, len(brainTopoSortLeft))
						for i, ptr := range brainTopoSortLeft {
							toSend[i] = *ptr
						}
						select {
						case brainActorL.Channel() <- toSend:
							// None.
						default:
							// None.
						}
					}
				}
				{ // eval r
					brainTopoSortRight, err := brains[R].Sort()
					if err != nil {
						panic(err)
					}
					input := make([]naneat.NeuralValue, nInputs)
					for i := 0; i < len(input); i++ {
						input[i] = naneat.NeuralValueMax
					}
					brains[R].Evaluate(input)
					// visualize
					if len(brainActorR.Channel()) < cap(brainActorR.Channel()) {
						toSend := make([]naneat.NeuralNode, len(brainTopoSortRight))
						for i, ptr := range brainTopoSortRight {
							toSend[i] = *ptr
						}
						select {
						case brainActorR.Channel() <- toSend:
							// None.
						default:
							// None.
						}
					}
				}
				agent.Send(naneat.Gaussian(0, 10000)) // feedback
			}
		}

		// consumer thread(goroutine)
		experimenter := loadExperimenterFromJSON(naneat.NewConfigurationSimple(nNets, nInputs, nOutputs))
		go experimenter.Run()
		defer experimenter.Shutdown()

		// supplier thread(goroutine) 1
		agent1 := naneat.NewAgent() // port
		experimenter.RegisterMeasurer(agent1)
		// brain context
		cntTrial1 := 0
		{ // run
			boundL := rectCentered.Moved(pixel.V(-width, height/2))
			boundR := rectCentered.Moved(pixel.V(0, height/2))
			go sup(agent1, &cntTrial1, maxTrial1, boundL, boundR)
		}

		// supplier thread(goroutine) 2
		agent2 := naneat.NewAgent() // port
		experimenter.RegisterMeasurer(agent2)
		// brain context
		cntTrial2 := 0
		{ // run
			boundL := rectCentered.Moved(pixel.V(-width, -height*3/2))
			boundR := rectCentered.Moved(pixel.V(0, -height*3/2))
			go sup(agent2, &cntTrial2, maxTrial2, boundL, boundR)
		}

		// supplier thread(goroutine) 3
		agent3 := naneat.NewAgent() // port
		experimenter.RegisterMeasurer(agent3)
		// brain context
		cntTrial3 := 0
		{ // run
			boundL := rectCentered.Moved(pixel.V(-width, -height/2))
			boundR := rectCentered.Moved(pixel.V(0, -height/2))
			go sup(agent3, &cntTrial3, maxTrial3, boundL, boundR)
		}

		// title bar
		go func() {
			for {
				_, title, _ := visualizer.Title()
				visualizer.SetTitle(title, fmt.Sprint(
					cntTrial1,
					cntTrial2,
					cntTrial3,
				))
				time.Sleep(time.Second)
			}
		}()

		{ // HUD from foreign package
			actorHUD := naneat.NewBanner(true, false)
			visualizer.PushHUDs(actorHUD)
			go func() {
				for {
					_, topFitness, population, numSpecies, generation, innovation, nicheCount := experimenter.Status().Info()
					actorHUD.Self().UpdateDesc(fmt.Sprint(
						cntTrial1,
						cntTrial2,
						cntTrial3,
						" ", "Gen", " ", generation,
						" ", "Pop", " ", population,
						" ", "TopFitness", " ", int(topFitness),
						" ", "Innovation", " ", innovation,
						" ", "NicheCount", " ", nicheCount,
						" ", "Num.Species", " ", numSpecies,
					))
					time.Sleep(time.Second)
				}
			}()
		}

		// mainthread runs
		visualizer.Run()
	}

	testXOR := func() {
		// mainthread
		cfg := visual.Config{
			Bg:                  pixel.ToRGBA(colornames.Mediumseagreen),
			OnPaused:            nil,
			OnResumed:           nil,
			OnDrawn:             nil,
			OnUpdated:           nil,
			OnResized:           nil,
			OnClose:             nil,
			OnHandlingEvents:    nil,
			OnLogging:           func(v ...interface{}) { panic(fmt.Sprint(v...)) },
			WinCentered:         false,
			Undecorated:         false,
			Title:               "XOR Test",
			Version:             "undefined version",
			Width:               60000.0,
			Height:              20000.0,
			WinWidth:            800.0,
			WinHeight:           800.0,
			InitialZoomLevel:    -3.0,
			InitialRotateDegree: -360.0,
		}
		visualizer := visual.NewVisualizer(cfg, nil)
		const nNets, nInputs, nOutputs = 1, 2, 1
		const width, height = 1200, 400
		const maxTrial1, maxTrial2, maxTrial3 = 1000000, 1000000, 1000000
		rectCentered := pixel.R(0, 0, width, height).Moved(cfg.PosCenterGame())

		// supplier's
		sup := func(agent *naneat.Agent, cntTrial *int, maxTrial int, bound pixel.Rect) {
			// defer func() {
			// 	if r := recover(); r != nil { //
			// 		log.Println("sup(): panic recovered")
			// 		log.Println("reason for panic:", r)
			// 		log.Println("stacktrace from panic: \r\n" + string(debug.Stack()))
			// 	}
			// }()
			var brainActor naneat.NeuralNetworkActor
			// run loopy run
			for *cntTrial = 0; *cntTrial < maxTrial; (*cntTrial)++ {
				brain := func() (retBrain *naneat.NeuralNetwork) { // awake
					retBrain = agent.Receive()[0]
					if !visualizer.RemoveActor(brainActor) && brainActor != nil {
						panic(fmt.Sprint("failed to remove `brainActor` from `visualizer`:", brainActor))
					}
					brainActor = func(nn *naneat.NeuralNetwork, bound pixel.Rect) (actor naneat.NeuralNetworkActor) {
						var err error
						actor, err = naneat.NewNeuralNetworkActor(
							nn, 1,
							bound, 10, 10, 10, 10,
							16, 4, true, true, true,
							true, true, true, true,
							nil, nil,
						)
						if err != nil {
							panic(err)
						}
						return actor
					}(retBrain, bound) // actorize
					visualizer.PushActors(brainActor)
					return retBrain
				}()
				nCorrect, bonus, _ := func() (nCorrect int, bonus, diff float64) { // measure fitness
					brainTopoSort, err := brain.Sort()
					if err != nil {
						panic(err)
					}
					eval := func(input []naneat.NeuralValue) (outputs []naneat.NeuralNode, err error) {
						outputs, err = brain.Evaluate(input)
						// visualize
						if len(brainActor.Channel()) < cap(brainActor.Channel()) {
							toSend := make([]naneat.NeuralNode, len(brainTopoSort))
							for i, ptr := range brainTopoSort {
								toSend[i] = *ptr
							}
							select {
							case brainActor.Channel() <- toSend:
								// None.
							default:
								// None.
							}
						}
						return outputs, err
					}
					bonus1, bonus2, bonus3, bonus4 := 0, 0, 0, 0
					// true (positive) // XOR Gate
					for x1 := naneat.NeuralValueMid + 1; x1 < naneat.NeuralValueMax+1; x1++ { // true (positive)
						for x2 := naneat.NeuralValueMid - 1; x2 > naneat.NeuralValueMin-1; x2-- { // false (negative)
							outputs, err := eval([]naneat.NeuralValue{x1, x2})
							if err != nil {
								panic(err)
							}
							output, err := outputs[0].Output()
							if err != nil {
								panic(err)
							}
							diff += float64(naneat.NeuralValueMax - output)
							switch output.Kind() {
							case naneat.NVKindPositive:
								nCorrect++
								bonus1 = 1
							case naneat.NVKindNegative:
								// Noop. Do nothing.
							case naneat.NVKindNeutral:
								// Noop. Do nothing.
							case naneat.NVKindExceptional:
								fallthrough
							default:
								panic("NVKind error")
							}
						}
					}
					// true (positive) // XOR Gate
					for x1 := naneat.NeuralValueMid - 1; x1 > naneat.NeuralValueMin-1; x1-- { // false (negative)
						for x2 := naneat.NeuralValueMid + 1; x2 < naneat.NeuralValueMax+1; x2++ { // true (positive)
							outputs, err := eval([]naneat.NeuralValue{x1, x2})
							if err != nil {
								panic(err)
							}
							output, err := outputs[0].Output()
							if err != nil {
								panic(err)
							}
							diff += float64(naneat.NeuralValueMax - output)
							switch output.Kind() {
							case naneat.NVKindPositive:
								nCorrect++
								bonus2 = 1
							case naneat.NVKindNegative:
								// Noop. Do nothing.
							case naneat.NVKindNeutral:
								// Noop. Do nothing.
							case naneat.NVKindExceptional:
								fallthrough
							default:
								panic("NVKind error")
							}
						}
					}
					// false (negative) // XOR Gate
					for x1 := naneat.NeuralValueMid - 1; x1 > naneat.NeuralValueMin-1; x1-- { // false (negative)
						for x2 := naneat.NeuralValueMid - 1; x2 > naneat.NeuralValueMin-1; x2-- { // false (negative)
							outputs, err := eval([]naneat.NeuralValue{x1, x2})
							if err != nil {
								panic(err)
							}
							output, err := outputs[0].Output()
							if err != nil {
								panic(err)
							}
							diff -= float64(naneat.NeuralValueMin - output)
							switch output.Kind() {
							case naneat.NVKindPositive:
								// Noop. Do nothing.
							case naneat.NVKindNegative:
								nCorrect++
								bonus3 = 1
							case naneat.NVKindNeutral:
								// Noop. Do nothing.
							case naneat.NVKindExceptional:
								fallthrough
							default:
								panic("NVKind error")
							}
						}
					}
					// false (negative) // XOR Gate
					for x1 := naneat.NeuralValueMid + 1; x1 < naneat.NeuralValueMax+1; x1++ { // true (positive)
						for x2 := naneat.NeuralValueMid + 1; x2 < naneat.NeuralValueMax+1; x2++ { // true (positive)
							outputs, err := eval([]naneat.NeuralValue{x1, x2})
							if err != nil {
								panic(err)
							}
							output, err := outputs[0].Output()
							if err != nil {
								panic(err)
							}
							diff -= float64(naneat.NeuralValueMin - output)
							switch output.Kind() {
							case naneat.NVKindPositive:
								// Noop. Do nothing.
							case naneat.NVKindNegative:
								nCorrect++
								bonus4 = 1
							case naneat.NVKindNeutral:
								// Noop. Do nothing.
							case naneat.NVKindExceptional:
								fallthrough
							default:
								panic("NVKind error")
							}
						}
					}
					return nCorrect, float64(bonus1 + bonus2 + bonus3 + bonus4), diff
				}()
				agent.Send((bonus * 100000 * 10) + float64(nCorrect*10) /*+ ((16777216.0 - diff) / 16777216.0)*/) // feedback
			}
		}

		// consumer thread(goroutine)
		experimenter := loadExperimenterFromJSON(func() (conf *naneat.Configuration) {
			conf = naneat.NewConfiguration()
			conf.ExperimentName = "XOR-Test"
			conf.SizePopulation = 500
			conf.PercentageCulling = 0.8
			conf.MaxCntArmageddon = 2
			conf.MaxCntApocalypse = 4
			conf.IsProtectiveElitism = true
			conf.IsFitnessRemeasurable = false
			//
			conf.ScaleCrossoverMultipointRnd = 1
			conf.ScaleCrossoverMultipointAvg = 0
			conf.ScaleCrossoverSinglepointRnd = 0
			conf.ScaleCrossoverSinglepointAvg = 0
			conf.ScaleFission = 0
			//
			conf.ChanceIsMutational = true
			conf.ChanceAddNode = 0.1
			conf.ChanceAddLink = 0.1
			conf.ChanceAddBias = 0.1
			conf.ChancePerturbWeight = 1.0
			conf.ChanceNullifyWeight = 0.0
			conf.ChanceTurnOn = 0.0
			conf.ChanceTurnOff = 0.0
			conf.ChanceBump = 0.0
			conf.TendencyPerturbWeight = 0.0
			conf.TendencyPerturbDice = 0.0
			conf.StrengthPerturbWeight = 0.1
			conf.StrengthPerturbDice = 0.08
			//
			conf.CompatThreshold = 5.0
			conf.CompatIsNormalizedForSize = true
			conf.CompatCoeffDisjoint = 5.0
			conf.CompatCoeffExcess = 5.0
			conf.CompatCoeffWeight = 3.0
			conf.CompatCoeffChance = 2.0
			//
			conf.NumChromosomes = nNets
			conf.NumBiases = 1
			conf.NumNonBiasInputs = nInputs
			conf.NumOutputs = nOutputs
			return conf
		}())
		go experimenter.Run()
		defer experimenter.Shutdown()

		// supplier thread(goroutine) 1
		agent1 := naneat.NewAgent() // port
		experimenter.RegisterMeasurer(agent1)
		// brain context
		cntTrial1 := 0
		// run
		go sup(agent1, &cntTrial1, maxTrial1, rectCentered.Moved(pixel.V(-width/2, height/2)))

		// supplier thread(goroutine) 2
		agent2 := naneat.NewAgent() // port
		experimenter.RegisterMeasurer(agent2)
		// brain context
		cntTrial2 := 0
		// run
		go sup(agent2, &cntTrial2, maxTrial2, rectCentered.Moved(pixel.V(-width/2, -height*3/2)))

		// supplier thread(goroutine) 3
		agent3 := naneat.NewAgent() // port
		experimenter.RegisterMeasurer(agent3)
		// brain context
		cntTrial3 := 0
		// run
		go sup(agent3, &cntTrial3, maxTrial3, rectCentered.Moved(pixel.V(-width/2, -height/2)))

		// title bar
		go func() {
			for {
				_, title, _ := visualizer.Title()
				visualizer.SetTitle(title, fmt.Sprint(
					cntTrial1,
					cntTrial2,
					cntTrial3,
				))
				time.Sleep(time.Second)
			}
		}()

		{ // HUD from foreign package
			actorHUD := naneat.NewBanner(true, false)
			visualizer.PushHUDs(actorHUD)
			go func() {
				for {
					_, topFitness, population, numSpecies, generation, innovation, nicheCount := experimenter.Status().Info()
					actorHUD.Self().UpdateDesc(fmt.Sprint(
						cntTrial1,
						cntTrial2,
						cntTrial3,
						" ", "Gen", " ", generation,
						" ", "Pop", " ", population,
						" ", "TopFitness", " ", fmt.Sprintf("%.2f", topFitness),
						" ", "Goal", " ", int(4655360),
						" ", "Innovation", " ", innovation,
						" ", "NicheCount", " ", nicheCount,
						" ", "Num.Species", " ", numSpecies,
					))
					time.Sleep(time.Second)
				}
			}()
		}

		// mainthread runs
		visualizer.Run()
	}

	testAND := func() {
		// mainthread
		cfg := visual.Config{
			Bg:                  pixel.ToRGBA(colornames.Mediumseagreen),
			OnPaused:            nil,
			OnResumed:           nil,
			OnDrawn:             nil,
			OnUpdated:           nil,
			OnResized:           nil,
			OnClose:             nil,
			OnHandlingEvents:    nil,
			OnLogging:           func(v ...interface{}) { panic(fmt.Sprint(v...)) },
			WinCentered:         false,
			Undecorated:         false,
			Title:               "AND Test",
			Version:             "undefined version",
			Width:               60000.0,
			Height:              20000.0,
			WinWidth:            800.0,
			WinHeight:           800.0,
			InitialZoomLevel:    -3.0,
			InitialRotateDegree: -360.0,
		}
		visualizer := visual.NewVisualizer(cfg, nil)
		const nNets, nInputs, nOutputs = 1, 2, 1
		const width, height = 1200, 400
		const maxTrial1, maxTrial2, maxTrial3 = 1000000, 1000000, 1000000
		rectCentered := pixel.R(0, 0, width, height).Moved(cfg.PosCenterGame())

		// supplier's
		sup := func(agent *naneat.Agent, cntTrial *int, maxTrial int, bound pixel.Rect) {
			// defer func() {
			// 	if r := recover(); r != nil { //
			// 		log.Println("sup(): panic recovered")
			// 		log.Println("reason for panic:", r)
			// 		log.Println("stacktrace from panic: \r\n" + string(debug.Stack()))
			// 	}
			// }()
			var brainActor naneat.NeuralNetworkActor
			// run loopy run
			for *cntTrial = 0; *cntTrial < maxTrial; (*cntTrial)++ {
				brain := func() (retBrain *naneat.NeuralNetwork) { // awake
					retBrain = agent.Receive()[0]
					if !visualizer.RemoveActor(brainActor) && brainActor != nil {
						panic(fmt.Sprint("failed to remove `brainActor` from `visualizer`:", brainActor))
					}
					brainActor = func(nn *naneat.NeuralNetwork, bound pixel.Rect) (actor naneat.NeuralNetworkActor) {
						var err error
						actor, err = naneat.NewNeuralNetworkActor(
							nn, 1,
							bound, 10, 10, 10, 10,
							16, 4, true, true, true,
							true, true, true, true,
							nil, nil,
						)
						if err != nil {
							panic(err)
						}
						return actor
					}(retBrain, bound) // actorize
					visualizer.PushActors(brainActor)
					return retBrain
				}()
				nCorrect, bonus := func() (nCorrect int, bonus float64) { // measure fitness
					brainTopoSort, err := brain.Sort()
					if err != nil {
						panic(err)
					}
					eval := func(input []naneat.NeuralValue) (outputs []naneat.NeuralNode, err error) {
						outputs, err = brain.Evaluate(input)
						// visualize
						if len(brainActor.Channel()) < cap(brainActor.Channel()) {
							toSend := make([]naneat.NeuralNode, len(brainTopoSort))
							for i, ptr := range brainTopoSort {
								toSend[i] = *ptr
							}
							select {
							case brainActor.Channel() <- toSend:
								// None.
							default:
								// None.
							}
						}
						return outputs, err
					}
					// false (negative) // AND Gate
					for x1 := naneat.NeuralValueMid + 1; x1 < naneat.NeuralValueMax+1; x1++ { // true (positive)
						for x2 := naneat.NeuralValueMid - 1; x2 > naneat.NeuralValueMin-1; x2-- { // false (negative)
							outputs, err := eval([]naneat.NeuralValue{x1, x2})
							if err != nil {
								panic(err)
							}
							output, err := outputs[0].Output()
							if err != nil {
								panic(err)
							}
							switch output.Kind() {
							case naneat.NVKindPositive:
								// Noop. Do nothing.
							case naneat.NVKindNegative:
								nCorrect++
								bonus += math.Abs(float64(output))
							case naneat.NVKindNeutral:
								// Noop. Do nothing.
							case naneat.NVKindExceptional:
								fallthrough
							default:
								panic("NVKind error")
							}
						}
					}
					// false (negative) // AND Gate
					for x1 := naneat.NeuralValueMid - 1; x1 > naneat.NeuralValueMin-1; x1-- { // false (negative)
						for x2 := naneat.NeuralValueMid + 1; x2 < naneat.NeuralValueMax+1; x2++ { // true (positive)
							outputs, err := eval([]naneat.NeuralValue{x1, x2})
							if err != nil {
								panic(err)
							}
							output, err := outputs[0].Output()
							if err != nil {
								panic(err)
							}
							switch output.Kind() {
							case naneat.NVKindPositive:
								// Noop. Do nothing.
							case naneat.NVKindNegative:
								nCorrect++
								bonus += math.Abs(float64(output))
							case naneat.NVKindNeutral:
								// Noop. Do nothing.
							case naneat.NVKindExceptional:
								fallthrough
							default:
								panic("NVKind error")
							}
						}
					}
					// false (negative) // AND Gate
					for x1 := naneat.NeuralValueMid - 1; x1 > naneat.NeuralValueMin-1; x1-- { // false (negative)
						for x2 := naneat.NeuralValueMid - 1; x2 > naneat.NeuralValueMin-1; x2-- { // false (negative)
							outputs, err := eval([]naneat.NeuralValue{x1, x2})
							if err != nil {
								panic(err)
							}
							output, err := outputs[0].Output()
							if err != nil {
								panic(err)
							}
							switch output.Kind() {
							case naneat.NVKindPositive:
								// Noop. Do nothing.
							case naneat.NVKindNegative:
								nCorrect++
								bonus += math.Abs(float64(output))
							case naneat.NVKindNeutral:
								// Noop. Do nothing.
							case naneat.NVKindExceptional:
								fallthrough
							default:
								panic("NVKind error")
							}
						}
					}
					// true (positive) // AND Gate
					for x1 := naneat.NeuralValueMid + 1; x1 < naneat.NeuralValueMax+1; x1++ { // true (positive)
						for x2 := naneat.NeuralValueMid + 1; x2 < naneat.NeuralValueMax+1; x2++ { // true (positive)
							outputs, err := eval([]naneat.NeuralValue{x1, x2})
							if err != nil {
								panic(err)
							}
							output, err := outputs[0].Output()
							if err != nil {
								panic(err)
							}
							switch output.Kind() {
							case naneat.NVKindPositive:
								nCorrect++
								bonus += math.Abs(float64(output))
							case naneat.NVKindNegative:
								// Noop. Do nothing.
							case naneat.NVKindNeutral:
								// Noop. Do nothing.
							case naneat.NVKindExceptional:
								fallthrough
							default:
								panic("NVKind error")
							}
						}
					}
					return nCorrect, bonus
				}()
				agent.Send(float64(nCorrect)*naneat.NeuralValueMax.Float64()*9 + bonus) // feedback
			}
		}

		// consumer thread(goroutine)
		experimenter := loadExperimenterFromJSON(func() (conf *naneat.Configuration) {
			conf = naneat.NewConfiguration()
			conf.ExperimentName = "AND-Test"
			conf.SizePopulation = 300
			conf.PercentageCulling = 0.8
			conf.MaxCntArmageddon = 2
			conf.MaxCntApocalypse = 4
			conf.IsProtectiveElitism = true
			conf.IsFitnessRemeasurable = false
			//
			conf.ScaleCrossoverMultipointRnd = 1
			conf.ScaleCrossoverMultipointAvg = 1
			conf.ScaleCrossoverSinglepointRnd = 0
			conf.ScaleCrossoverSinglepointAvg = 0
			conf.ScaleFission = 1
			//
			conf.ChanceIsMutational = true
			conf.ChanceAddNode = 0.1
			conf.ChanceAddLink = 0.1
			conf.ChanceAddBias = 0.1
			conf.ChancePerturbWeight = 2.0
			conf.ChanceNullifyWeight = 0.1
			conf.ChanceTurnOn = 0.0
			conf.ChanceTurnOff = 0.0
			conf.ChanceBump = 0.0
			conf.TendencyPerturbWeight = 0.0
			conf.TendencyPerturbDice = 0.0
			conf.StrengthPerturbWeight = 0.03
			conf.StrengthPerturbDice = 0.08
			//
			conf.CompatThreshold = 5.0
			conf.CompatIsNormalizedForSize = true
			conf.CompatCoeffDisjoint = 5.0
			conf.CompatCoeffExcess = 5.0
			conf.CompatCoeffWeight = 4.0
			conf.CompatCoeffChance = 2.0
			//
			conf.NumChromosomes = nNets
			conf.NumBiases = 1
			conf.NumNonBiasInputs = nInputs
			conf.NumOutputs = nOutputs
			return conf
		}())
		go experimenter.Run()
		defer experimenter.Shutdown()

		// supplier thread(goroutine) 1
		agent1 := naneat.NewAgent() // port
		experimenter.RegisterMeasurer(agent1)
		// brain context
		cntTrial1 := 0
		// run
		go sup(agent1, &cntTrial1, maxTrial1, rectCentered.Moved(pixel.V(-width/2, height/2)))

		// supplier thread(goroutine) 2
		agent2 := naneat.NewAgent() // port
		experimenter.RegisterMeasurer(agent2)
		// brain context
		cntTrial2 := 0
		// run
		go sup(agent2, &cntTrial2, maxTrial2, rectCentered.Moved(pixel.V(-width/2, -height*3/2)))

		// supplier thread(goroutine) 3
		agent3 := naneat.NewAgent() // port
		experimenter.RegisterMeasurer(agent3)
		// brain context
		cntTrial3 := 0
		// run
		go sup(agent3, &cntTrial3, maxTrial3, rectCentered.Moved(pixel.V(-width/2, -height/2)))

		// title bar
		go func() {
			for {
				_, title, _ := visualizer.Title()
				visualizer.SetTitle(title, fmt.Sprint(
					cntTrial1,
					cntTrial2,
					cntTrial3,
				))
				time.Sleep(time.Second)
			}
		}()

		{ // HUD from foreign package
			actorHUD := naneat.NewBanner(true, false)
			visualizer.PushHUDs(actorHUD)
			go func() {
				for {
					_, topFitness, population, numSpecies, generation, innovation, nicheCount := experimenter.Status().Info()
					actorHUD.Self().UpdateDesc(fmt.Sprint(
						cntTrial1,
						cntTrial2,
						cntTrial3,
						" ", "Gen", " ", generation,
						" ", "Pop", " ", population,
						" ", "TopFitness", " ", fmt.Sprintf("%.2f", topFitness),
						" ", "Num.Count", " ", int(experimenter.Self().TopFitness/1280),
						" ", "Innovation", " ", innovation,
						" ", "NicheCount", " ", nicheCount,
						" ", "Num.Species", " ", numSpecies,
					))
					time.Sleep(time.Second)
				}
			}()
		}

		// mainthread runs
		visualizer.Run()
	}

	testMaxPositive := func() {
		// mainthread
		cfg := visual.Config{
			Bg:                  pixel.ToRGBA(colornames.Mediumseagreen),
			OnPaused:            nil,
			OnResumed:           nil,
			OnDrawn:             nil,
			OnUpdated:           nil,
			OnResized:           nil,
			OnClose:             nil,
			OnHandlingEvents:    nil,
			OnLogging:           func(v ...interface{}) { panic(fmt.Sprint(v...)) },
			WinCentered:         false,
			Undecorated:         false,
			Title:               "Max Positivity Test",
			Version:             "undefined version",
			Width:               60000.0,
			Height:              20000.0,
			WinWidth:            800.0,
			WinHeight:           800.0,
			InitialZoomLevel:    -3.0,
			InitialRotateDegree: -360.0,
		}
		visualizer := visual.NewVisualizer(cfg, nil)
		const nNets, nInputs, nOutputs = 1, 2, 1
		const width, height = 1200, 400
		const maxTrial1, maxTrial2, maxTrial3 = 1000000, 1000000, 1000000
		rectCentered := pixel.R(0, 0, width, height).Moved(cfg.PosCenterGame())

		// supplier's
		sup := func(agent *naneat.Agent, cntTrial *int, maxTrial int, bound pixel.Rect) {
			// defer func() {
			// 	if r := recover(); r != nil { //
			// 		log.Println("sup(): panic recovered")
			// 		log.Println("reason for panic:", r)
			// 		log.Println("stacktrace from panic: \r\n" + string(debug.Stack()))
			// 	}
			// }()
			var brainActor naneat.NeuralNetworkActor
			// run loopy run
			for *cntTrial = 0; *cntTrial < maxTrial; (*cntTrial)++ {
				brain := func() (retBrain *naneat.NeuralNetwork) { // awake
					retBrain = agent.Receive()[0]
					if !visualizer.RemoveActor(brainActor) && brainActor != nil {
						panic(fmt.Sprint("failed to remove `brainActor` from `visualizer`:", brainActor))
					}
					brainActor = func(nn *naneat.NeuralNetwork, bound pixel.Rect) (actor naneat.NeuralNetworkActor) {
						var err error
						actor, err = naneat.NewNeuralNetworkActor(
							nn, 1,
							bound, 10, 10, 10, 10,
							16, 4, true, true, true,
							true, true, true, true,
							nil, nil,
						)
						if err != nil {
							panic(err)
						}
						return actor
					}(retBrain, bound) // actorize
					visualizer.PushActors(brainActor)
					return retBrain
				}()
				fitness := func() (positivity float64) { // measure fitness
					brainTopoSort, err := brain.Sort()
					if err != nil {
						panic(err)
					}
					eval := func(input []naneat.NeuralValue) (outputs []naneat.NeuralNode, err error) {
						outputs, err = brain.Evaluate(input)
						// visualize
						if len(brainActor.Channel()) < cap(brainActor.Channel()) {
							toSend := make([]naneat.NeuralNode, len(brainTopoSort))
							for i, ptr := range brainTopoSort {
								toSend[i] = *ptr
							}
							select {
							case brainActor.Channel() <- toSend:
								// None.
							default:
								// None.
							}
						}
						return outputs, err
					}
					for i := 0; i < 1000; i++ {
						outputs, err := eval([]naneat.NeuralValue{
							naneat.NV(0.1),
							naneat.NV(-0.1),
						})
						if err != nil {
							panic(err)
						}
						nv, err := outputs[0].Output()
						if err != nil {
							panic(err)
						}
						positivity += nv.Float64()
					}
					return positivity
				}()
				agent.Send(float64(fitness)) // feedback
			}
		}

		// consumer thread(goroutine)
		experimenter := loadExperimenterFromJSON(func() (conf *naneat.Configuration) {
			conf = naneat.NewConfiguration()
			conf.ExperimentName = "Max-Positivity-Test"
			conf.SizePopulation = 300
			conf.PercentageCulling = 0.8
			conf.MaxCntArmageddon = 2
			conf.MaxCntApocalypse = 4
			conf.IsProtectiveElitism = false
			conf.IsFitnessRemeasurable = false
			//
			conf.ScaleCrossoverMultipointRnd = 1
			conf.ScaleCrossoverMultipointAvg = 1
			conf.ScaleCrossoverSinglepointRnd = 0
			conf.ScaleCrossoverSinglepointAvg = 0
			conf.ScaleFission = 1
			//
			conf.ChanceIsMutational = true
			conf.ChanceAddNode = 0.1
			conf.ChanceAddLink = 0.1
			conf.ChanceAddBias = 0.1
			conf.ChancePerturbWeight = 1.0
			conf.ChanceNullifyWeight = 0.1
			conf.ChanceTurnOn = 0.0
			conf.ChanceTurnOff = 0.0
			conf.ChanceBump = 0.0
			conf.TendencyPerturbWeight = 0.0
			conf.TendencyPerturbDice = 0.0
			conf.StrengthPerturbWeight = 0.05
			conf.StrengthPerturbDice = 0.08
			//
			conf.CompatThreshold = 5.0
			conf.CompatIsNormalizedForSize = true
			conf.CompatCoeffDisjoint = 5.0
			conf.CompatCoeffExcess = 5.0
			conf.CompatCoeffWeight = 4.0
			conf.CompatCoeffChance = 2.0
			//
			conf.NumChromosomes = nNets
			conf.NumBiases = 1
			conf.NumNonBiasInputs = nInputs
			conf.NumOutputs = nOutputs
			return conf
		}())
		go experimenter.Run()
		defer experimenter.Shutdown()

		// supplier thread(goroutine) 1
		agent1 := naneat.NewAgent() // port
		experimenter.RegisterMeasurer(agent1)
		// brain context
		cntTrial1 := 0
		// run
		go sup(agent1, &cntTrial1, maxTrial1, rectCentered.Moved(pixel.V(-width/2, height/2)))

		// supplier thread(goroutine) 2
		agent2 := naneat.NewAgent() // port
		experimenter.RegisterMeasurer(agent2)
		// brain context
		cntTrial2 := 0
		// run
		go sup(agent2, &cntTrial2, maxTrial2, rectCentered.Moved(pixel.V(-width/2, -height*3/2)))

		// supplier thread(goroutine) 3
		agent3 := naneat.NewAgent() // port
		experimenter.RegisterMeasurer(agent3)
		// brain context
		cntTrial3 := 0
		// run
		go sup(agent3, &cntTrial3, maxTrial3, rectCentered.Moved(pixel.V(-width/2, -height/2)))

		// title bar
		go func() {
			for {
				_, title, _ := visualizer.Title()
				visualizer.SetTitle(title, fmt.Sprint(
					cntTrial1,
					cntTrial2,
					cntTrial3,
				))
				time.Sleep(time.Second)
			}
		}()

		{ // HUD from foreign package
			actorHUD := naneat.NewBanner(true, false)
			visualizer.PushHUDs(actorHUD)
			go func() {
				for {
					_, topFitness, population, numSpecies, generation, innovation, nicheCount := experimenter.Status().Info()
					actorHUD.Self().UpdateDesc(fmt.Sprint(
						cntTrial1,
						cntTrial2,
						cntTrial3,
						" ", "Gen", " ", generation,
						" ", "Pop", " ", population,
						" ", "TopFitness", " ", int(topFitness),
						" ", "Innovation", " ", innovation,
						" ", "NicheCount", " ", nicheCount,
						" ", "Num.Species", " ", numSpecies,
					))
					time.Sleep(time.Second)
				}
			}()
		}

		// mainthread runs
		visualizer.Run()
	}

	// this is when __name__ == "__main__"

	// defer func() {
	// 	if r := recover(); r != nil { //
	// 		log.Println("main(): panic recovered")
	// 		log.Println("reason for panic:", r)
	// 		log.Println("stacktrace from panic: \r\n" + string(debug.Stack()))
	// 	}
	// }()

	if dialog.Message("%s", "Proceed? (No if unsure)").Title("Random Fitness Test").YesNo() {
		testRandomFitness()
	} else if dialog.Message("%s", "Proceed? (No if unsure)").Title("XOR Test").YesNo() {
		testXOR()
	} else if dialog.Message("%s", "Proceed? (No if unsure)").Title("AND Test").YesNo() {
		testAND()
	} else if dialog.Message("%s", "Proceed? (No if unsure)").Title("Max Positivity Test").YesNo() {
		testMaxPositive()
	}
}
