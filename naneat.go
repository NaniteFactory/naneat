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

package naneat

import (
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
	"reflect"
	"runtime/debug"
	"sort"
	"strconv"
	"sync"
	"syscall"
	"time"

	"github.com/campoy/unique"
	"github.com/gofrs/uuid"

	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/simple"
	"gonum.org/v1/gonum/graph/topo"
)

// ----------------------------------------------------------------------------
// Package

// Initialization of this package.
func init() {
	syscall.Write(syscall.Handle(os.Stdout.Fd()), nil) // The standard out is flushed.
	rand.Seed(time.Now().UTC().UnixNano())             // Seed for random mutations etc.
}

// ----------------------------------------------------------------------------
// Experimenter

// Experimenter exposes a set of user-friendly methods.
// Facade of this library.
type Experimenter interface {
	Self() *Universe
	Status() UniverseStatusGetter
	RegisterMeasurer(agent Measurer)
	UnregisterMeasurer(agent Measurer)
	Run() error
	Shutdown()
	IsPumping() bool
}

// UniverseStatusGetter exposes a set of thread-safe universe state getters.
type UniverseStatusGetter interface {
	Info() (
		conf Configuration, topFitness float64,
		population, numSpecies, generation, innovation, nicheCount int,
	)
	Measurers() (ret map[Measurer]*Subscribed)
	Breeds() []*Species
	Organisms() []*Organism
	BiasGenes() []*NodeGene
	NonBiasInputGenes() []*NodeGene
	GetInputGenes() []*NodeGene
	GetOutputGenes() []*NodeGene
}

// New is where everything starts.
//
// These two are the same:
//  - func New(config *Configuration) Experimenter
//  - func (config *Configuration) New() Experimenter
//
func New(config *Configuration) Experimenter {
	return config.New()
}

// ----------------------------------------------------------------------------
// Measurer

// Measurer is a consumer interface for what measures an organism's fitness.
// It could be an emulator thread or a network server or any test case.
// Implement it however you'd like.
type Measurer interface {
	// Notice the side effect of this method as an impure function that
	// with fitness it returns it might also update the state of our NN.
	MeasureAsync(organism []*NeuralNetwork) (postman Swallow, err error)
	String() string
}

// Swallow is a bird that will deliver our fortune.
// Or it's a channel our organism's fitness is transferred over.
type Swallow <-chan float64

// Agent is a Measurer: Agent implements Measurer interface.
type Agent struct {
	name      string
	chReceive chan []*NeuralNetwork
	chSend    chan float64
	isWorking bool
}

// NewAgent is a constructor.
func NewAgent() *Agent {
	return &Agent{
		name:      "Agent",
		chReceive: make(chan []*NeuralNetwork, 1),
		chSend:    make(chan float64),
		isWorking: false,
	}
}

// String callback of Agent implements Measurer.
func (a *Agent) String() string {
	return a.name
}

// Close all channels of this agent.
//
// Effects of closing channels:
//  - Channels are closed.
//  - The other goroutine is notified of the fact that the data flow was shut quite tough.
//  - Possible memory leaks are prevented in case our GC is working so lazy.
//
func (a *Agent) Close() {
	close(a.chReceive)
	close(a.chSend)
}

// Send blocks the thread.
// The supplier uses this method.
func (a *Agent) Send(fitness float64) {
	a.chSend <- fitness
	a.isWorking = false
}

// Receive blocks the thread.
// The supplier uses this method.
func (a *Agent) Receive() (brains []*NeuralNetwork) {
	a.isWorking = true
	brains = <-a.chReceive
	return brains
}

// IsWorking when this agent as a supplier thinks it is working,
// not when the consumer thinks it is.
// The consumer might use this method.
func (a *Agent) IsWorking() bool {
	return a.isWorking
}

// Measure blocks the thread.
// The consumer might use this method.
func (a *Agent) Measure(brains []*NeuralNetwork) (fitness float64) {
	a.chReceive <- brains
	return <-a.chSend
}

// MeasureAsync does not block the thread.
// The consumer uses this method.
func (a *Agent) MeasureAsync(brains []*NeuralNetwork) (messenger Swallow, err error) {
	select {
	case a.chReceive <- brains:
		return a.chSend, nil
	default:
		return nil, errors.New("failed to make an order: this agent has already been ordered")
	}
}

// ----------------------------------------------------------------------------
// Configuration

// Configuration stores the universe's constants. A set of hyper parameters.
type Configuration struct {
	// Label for the universe
	ExperimentName string
	// Max number of creatures
	SizePopulation int
	// The percentage bottom which percentage of organisms in a species is culled off every epoch.
	// The range of this value is [0, 1].
	PercentageCulling float64
	// The Nth generation the stagnant species is going to face its extermination. xD
	// The species not improving over this much of generations will get penalized.
	// Negative value for this means infinity.
	MaxCntArmageddon int // Age threshold to dropoff. Rotten species are not allowed to reproduce.
	// If the top fitness of the entire universe does not improve for more than this much of generations,
	// only the top two species are allowed to reproduce, refocusing the search into the most promising spaces.
	// Negative value for this means infinity.
	MaxCntApocalypse int // Threshold to the universe's stagnancy.
	// If false, entire population of a species besides the champion is replaced by their offsprings at each reproduce procedure.
	// If true, there are un-culled organisms in species: reproducible organisms are saved and lives on to next generation.
	IsProtectiveElitism   bool
	IsFitnessRemeasurable bool // If true, the fitness of every organism that's already measured gets remeasured in the measuring stage of each generation.
	// Ratio of different reproduction methods performed upon breeding species
	ScaleCrossoverMultipointRnd  int // Sexual reproduction 1
	ScaleCrossoverMultipointAvg  int // Sexual reproduction 2
	ScaleCrossoverSinglepointRnd int // Sexual reproduction 3
	ScaleCrossoverSinglepointAvg int // Sexual reproduction 4
	ScaleFission                 int // Asexual breeding (Binary fission)
	// Base of M-Rate (ChanceGene)
	ChanceIsMutational  bool    // Genetic(dynamic) or constant.
	ChanceAddNode       float64 // Add-Node mutation chance.
	ChanceAddLink       float64 // Add-Link mutation chance only for non-bias nodes.
	ChanceAddBias       float64 // Add-Link mutation chance only for bias.
	ChancePerturbWeight float64 // Synaptic weight mutation chance.
	ChanceNullifyWeight float64 // Mutation chance of synaptic weight becoming zero.
	ChanceTurnOn        float64 // Enable mutation chance.
	ChanceTurnOff       float64 // Disable mutation chance.
	ChanceBump          float64 // Bump mutation chance.
	// What defines the synaptic weight mutation.
	// This is a size factor determining the deviation of a Gaussian distribution.
	TendencyPerturbWeight float64 // The mean(offset). Set to 0.0 by default.
	TendencyPerturbDice   float64 // The mean(offset). Set by default to a very small positive value.
	StrengthPerturbWeight float64 // Adjusted deviation. Synaptic weight is perturbed by this percent of it.
	StrengthPerturbDice   float64 // Adjusted deviation. Mutational rate is perturbed by this percent of it.
	// Measuring the compatibility distance
	CompatThreshold           float64 // The line separating species.
	CompatIsNormalizedForSize bool    // The characterizing variables of disjointedness and excess get expressed in normalized percent, or, rather in the absolute which ignores the size of two genetic encoders.
	CompatCoeffDisjoint       float64 // The maximum disjointedness when expressed in normalized percent or simply a multiplier to the number of disjoint genes a chromosome has.
	CompatCoeffExcess         float64 // The maximum excessiveness when expressed in normalized percent or simply a multiplier to the number of excessive genes a chromosome has.
	CompatCoeffWeight         float64 // The maximum or simply a multiplier regarding the mutational difference of weights.
	CompatCoeffChance         float64 // The maximum or simply a multiplier regarding the mutational difference of chance gene's.
	// Seeds of genetics (Genome/Chromosome)
	NumChromosomes   int // The number of Chromosome-s for a single Genome. The number of NNs a single Organism consists of.
	NumBiases        int // The number of biases.
	NumNonBiasInputs int // The number of input nodes. This doesn't count for biases.
	NumOutputs       int // The number of output nodes.
}

// NewConfiguration is a constructor.
// This simply creates an empty object.
// Feel free to edit what's returned however you'd like.
func NewConfiguration() *Configuration {
	return &Configuration{}
}

// NewConfigurationSimple is a constructor.
// It returns what's filled up with a default setting.
//
// A set of params I used to test with:
//  - nNet = 2
//  - nIn = 38 * 28
//  - nOut = 7
//
func NewConfigurationSimple(nNet, nIn, nOut int) *Configuration {
	return &Configuration{
		ExperimentName:        "UU",
		SizePopulation:        400,
		PercentageCulling:     0.7,
		MaxCntArmageddon:      8,
		MaxCntApocalypse:      20,
		IsProtectiveElitism:   true,
		IsFitnessRemeasurable: false,
		//
		ScaleCrossoverMultipointRnd:  8, // 80%
		ScaleCrossoverMultipointAvg:  1, // 10%
		ScaleCrossoverSinglepointRnd: 0, // 0%
		ScaleCrossoverSinglepointAvg: 0, // 0%
		ScaleFission:                 1, // 10%
		//
		ChanceIsMutational:    true,
		ChanceAddNode:         0.5,   // 50%
		ChanceAddLink:         4.0,   // 400%
		ChanceAddBias:         0.9,   // 90%
		ChancePerturbWeight:   1.0,   // 100%
		ChanceNullifyWeight:   0.0,   // 0%
		ChanceTurnOn:          0.1,   // 10%
		ChanceTurnOff:         0.4,   // 40%
		ChanceBump:            0.1,   // 10%
		TendencyPerturbWeight: 0.0,   // 0%p (val offset +)
		TendencyPerturbDice:   0.001, // 0.1%p (val offset +)
		StrengthPerturbWeight: 0.15,  // 15% of 50%
		StrengthPerturbDice:   0.15,  // 15%
		//
		CompatThreshold:           10.0,
		CompatIsNormalizedForSize: true,
		CompatCoeffDisjoint:       20.0,
		CompatCoeffExcess:         20.0,
		CompatCoeffWeight:         10.0,
		CompatCoeffChance:         10.0,
		//
		NumChromosomes:   nNet,
		NumBiases:        1,
		NumNonBiasInputs: nIn,
		NumOutputs:       nOut,
	}
}

// BirthRatioSimplified returns the breeding methods ratio in smallest integers.
func (config *Configuration) BirthRatioSimplified() (
	weightCrossoverMultipointRnd,
	weightCrossoverMultipointAvg,
	weightCrossoverSinglepointRnd,
	weightCrossoverSinglepointAvg,
	weightFission,
	weightSterile float64,
) {
	r1 := config.ScaleCrossoverMultipointRnd
	r2 := config.ScaleCrossoverMultipointAvg
	r3 := config.ScaleCrossoverSinglepointRnd
	r4 := config.ScaleCrossoverSinglepointAvg
	r5 := config.ScaleFission
	gcd := GCD(r1, r2, r3, r4, r5)
	weightCrossoverMultipointRnd = float64(r1 / gcd)
	weightCrossoverMultipointAvg = float64(r2 / gcd)
	weightCrossoverSinglepointRnd = float64(r3 / gcd)
	weightCrossoverSinglepointAvg = float64(r4 / gcd)
	weightFission = float64(r5 / gcd)
	return
}

// NewUniverse is a constructor.
func (config *Configuration) NewUniverse() *Universe {
	retUniv := &Universe{
		Config: *config,
		//
		ChStopSign: make(chan chan struct{}),
		IsRunning:  false,
		MutexRun:   sync.Mutex{},
		//
		Agents:      map[Measurer]*Subscribed{},
		MutexAgents: sync.Mutex{},
		//
		Livings:      map[*Organism]struct{}{},
		Classes:      []*Species{},
		MutexLivings: sync.Mutex{},
		MutexClasses: sync.Mutex{},
		TopFitness:   0,
		Generation:   0,
		Innovation:   0,
		NicheCount:   0,
		//
		InputGenes:  make([]*NodeGene, config.NumBiases+config.NumNonBiasInputs),
		OutputGenes: make([]*NodeGene, config.NumOutputs),
	}
	// sow
	{ // inputs
		nBias := config.NumBiases
		for i := 0; i < nBias; i++ {
			retUniv.InputGenes[i] = NewNodeGene("bias_"+strconv.Itoa(i), InputNodeBias, i)
		}
		for i := nBias; i < len(retUniv.InputGenes); i++ {
			retUniv.InputGenes[i] = NewNodeGene("primal_in_"+strconv.Itoa(i-nBias), InputNodeNotBias, i)
		}
	}
	for i := 0; i < len(retUniv.OutputGenes); i++ { // outputs
		retUniv.OutputGenes[i] = NewNodeGene("primal_out_"+strconv.Itoa(i), OutputNode, i)
	}
	// creation + speciation
	for len(retUniv.Livings) < retUniv.Config.SizePopulation {
		newLife, err := retUniv.NewOrganismBasic()
		if err != nil {
			// log.Println("fatal:", err) // debug //
			panic(err)
		}
		if err := retUniv.AddOrganism(newLife); err != nil {
			// log.Println("fatal:", err) // debug //
			panic(err)
		}
		if err := retUniv.Speciate(newLife); err != nil {
			// log.Println("fatal:", err) // debug //
			panic(err)
		}
	}
	// The initial generation doesn't get more than a chance.
	for _, s := range retUniv.Classes {
		s.Stagnancy = retUniv.Config.MaxCntArmageddon
	}
	return retUniv
}

// New is where everything starts.
//
// These two are the same:
//  - func New(config *Configuration) Experimenter
//  - func (config *Configuration) New() Experimenter
//
func (config *Configuration) New() Experimenter {
	return config.NewUniverse()
}

// ----------------------------------------------------------------------------
// Ark (Import)

// Ark JSON export of Universe. Backup state data.
type Ark struct {
	ReferenceByUUID map[uuid.UUID]*NodeGene
	Classes         []*Species
	TopFitness      float64
	Generation      int
	Innovation      int
	NicheCount      int
	Config          Configuration
	InputGenes      []*NodeGene
	OutputGenes     []*NodeGene
}

// NewArkFromFile loads one universe state from a JSON file.
// Data in.
func NewArkFromFile(filepath string) (*Ark, error) {
	jsonRaw, err := ioutil.ReadFile(filepath)
	if err != nil {
		return nil, err
	}
	var glove Ark
	err = json.Unmarshal(jsonRaw, &glove)
	if err != nil {
		return nil, err
	}
	// log.Println(glove) // debug //
	return &glove, nil
}

// New is the coolest constructor where everything starts.
func (pack *Ark) New() (ret Experimenter, err error) {
	return pack.NewUniverse()
}

// NewUniverse is a constructor.
// Data out.
func (pack *Ark) NewUniverse() (retUniv *Universe, err error) {
	retUniv = &Universe{
		Config: pack.Config,
		//
		ChStopSign: make(chan chan struct{}),
		IsRunning:  false,
		MutexRun:   sync.Mutex{},
		//
		Agents:      map[Measurer]*Subscribed{},
		MutexAgents: sync.Mutex{},
		//
		Livings:    map[*Organism]struct{}{},
		Classes:    pack.Classes,
		TopFitness: pack.TopFitness,
		Generation: pack.Generation,
		Innovation: pack.Innovation,
		NicheCount: pack.NicheCount,
		//
		InputGenes:  pack.InputGenes,
		OutputGenes: pack.OutputGenes,
	}
	// IO nodes
	for i, nodeGene := range retUniv.InputGenes { // NodeGenesFromUUID
		retUniv.InputGenes[i] = pack.ReferenceByUUID[nodeGene.UUID]
	}
	for i, nodeGene := range retUniv.OutputGenes { // NodeGenesFromUUID
		retUniv.OutputGenes[i] = pack.ReferenceByUUID[nodeGene.UUID]
	}
	ioNodeGenes := func() []*NodeGene {
		ret := append([]*NodeGene{}, retUniv.InputGenes...)
		ret = append(ret, retUniv.OutputGenes...)
		return ret
	}()
	// Links & Hidden nodes
	for _, species := range retUniv.Classes {
		for _, organism := range species.Livings {
			// Genome (chrome)
			for _, chrome := range organism.GenoType().Chromosomes() {
				chrome.IONodeGenes = append([]*NodeGene{}, ioNodeGenes...)
				for i, linkGene := range chrome.LinkGenes {
					chrome.LinkGenes[i].Topo.From = pack.ReferenceByUUID[linkGene.Topo.From.UUID]
					chrome.LinkGenes[i].Topo.To = pack.ReferenceByUUID[linkGene.Topo.To.UUID]
				}
				chrome.Sort()
			}
			// Phenome
			phenotype := make([]*NeuralNetwork, organism.GenoType().Length())
			for iChrome, chromosome := range organism.GenoType().Chromosomes() {
				phenotype[iChrome], err = chromosome.NewNeuralNetwork()
				if err != nil {
					return nil, err
				}
			}
			organism.Phenotype = phenotype
			// After that...
			organism.Breed = species               // Speciate
			retUniv.Livings[organism] = struct{}{} // Populate
		}
	}
	return retUniv, nil
}

// ----------------------------------------------------------------------------
// Universe

// Universe is NEAT context in the highest level,
// which gets you an access to all available training data of a training session.
// It is strongly recommended not to access any of these members directly outside this package,
// unless you understand what those really are.
// Otherwise use this struct's methods and its constructor instead.
type Universe struct {
	// Static info relates to all creations and destructions.
	Config Configuration // Constants.
	// Trivial runtime settings.
	ChStopSign chan chan struct{} // Private used only for the Shutdown().
	IsRunning  bool               // What tells whether this universe is actually running or not.
	MutexRun   sync.Mutex         // Sync tool used only for Experimenter implementations. For IsRunning bool.
	// Interface to talk to other possible modules.
	Agents      map[Measurer]*Subscribed // A set of agents measuring fitnesses of our creatures.
	MutexAgents sync.Mutex               // Sync tool used only for Experimenter implementations.
	// Regarding space we have in this universe.
	Livings      map[*Organism]struct{} // A set of all creatures available in this universe.
	Classes      []*Species             // Biological category of creatures.
	TopFitness   float64                // Of an organism in this universe.
	MutexLivings sync.Mutex             // Sync tool used only for Experimenter implementations.
	MutexClasses sync.Mutex             // Sync tool used only for Experimenter implementations.
	// Regarding time how far we've been through.
	// These can only go forward and there is no way for them to be winded back besides resetting the whole universe.
	// So what's not necessary at all: Any method that would decrement any of these global (innovation/historical) number.
	Generation int // The Nth generation. We could say the age of this universe is N generation old.
	Innovation int // The innovation number, which should be/is global throughout this universe.
	NicheCount int // This is a counter that tells exactly how many of niches have appeared since the creation of this universe. Niche is an identifier historical and unique for each of species in this universe.
	// The ancestor of all creatures born and raised in this universe.
	InputGenes  []*NodeGene
	OutputGenes []*NodeGene
}

// ----------------------------------------------------------------------------
// Universe - Subscribed (Component)

// Subscribed of a Measurer.
type Subscribed struct {
	// Measurer receives an NN to be evaluated.
	S Swallow   // Channel assigned to a Measurer.
	O *Organism // The creature being measured by a Measurer.
}

// NewSubscribed is a constructor.
func NewSubscribed(mailFrom Swallow, mailTo *Organism) *Subscribed {
	return &Subscribed{
		S: mailFrom,
		O: mailTo,
	}
}

// String callback of this.
func (subbed *Subscribed) String() string {
	return fmt.Sprint(*subbed)
}

// ----------------------------------------------------------------------------
// Universe - Export

// NewArk of this universe.
func (univ *Universe) NewArk() *Ark {
	return &Ark{
		ReferenceByUUID: univ.NodeGenesByUUID(nil),
		Classes:         univ.Classes,
		TopFitness:      univ.TopFitness,
		Generation:      univ.Generation,
		Innovation:      univ.Innovation,
		NicheCount:      univ.NicheCount,
		Config:          univ.Config,
		InputGenes:      univ.InputGenes,
		OutputGenes:     univ.OutputGenes,
	}
}

// NodeGenesByUUID returns a map of all node genes available in this Universe.
// The parameter can be nil.
func (univ *Universe) NodeGenesByUUID(fillMe map[uuid.UUID]*NodeGene) (filledOut map[uuid.UUID]*NodeGene) {
	if fillMe == nil {
		filledOut = map[uuid.UUID]*NodeGene{}
	} else {
		filledOut = fillMe
	}
	for _, species := range univ.Classes {
		for _, organism := range species.Livings {
			filledOut = organism.GenoType().NodeGenesByUUID(filledOut)
		}
	}
	for _, nodeGene := range univ.InputGenes {
		filledOut[nodeGene.UUID] = nodeGene
	}
	for _, nodeGene := range univ.OutputGenes {
		filledOut[nodeGene.UUID] = nodeGene
	}
	return filledOut
}

// Save this universe to a JSON file.
func (univ *Universe) Save(filename string) (err error) {
	// copy
	jsonRawBackup, err := json.Marshal(univ.NewArk())
	if err != nil {
		return err
	}
	// dump
	err = ioutil.WriteFile(filename, jsonRawBackup, 0644)
	if err != nil {
		return err
	}
	return nil
}

// ----------------------------------------------------------------------------
// Universe - Run (EP)

// Self implements the Experimenter interface.
func (univ *Universe) Self() *Universe {
	return univ
}

// Status implements the Experimenter interface.
func (univ *Universe) Status() UniverseStatusGetter {
	return univ
}

// RegisterMeasurer to this universe.
func (univ *Universe) RegisterMeasurer(agent Measurer) {
	univ.MutexAgents.Lock()
	defer univ.MutexAgents.Unlock()

	univ.Agents[agent] = nil
}

// UnregisterMeasurer to this universe.
func (univ *Universe) UnregisterMeasurer(agent Measurer) {
	univ.MutexAgents.Lock()
	defer univ.MutexAgents.Unlock()

	delete(univ.Agents, agent)
}

// Run this universe. Entry point to our GA.
// *This* procedure of *this* universe can be run only one at a time.
// Otherwise it meets the only condition this can error.
// Give this blocking synchronous function a thread or a goroutine.
func (univ *Universe) Run() error {
	// init
	univ.MutexRun.Lock()
	if univ.IsRunning {
		univ.MutexRun.Unlock()
		return errors.New("already running")
	}
	univ.IsRunning = true
	univ.MutexRun.Unlock()

	// clean up
	defer func() {
		univ.MutexRun.Lock()
		defer univ.MutexRun.Unlock()
		univ.IsRunning = false
	}()

	// handled error
	type shutdownError struct {
		error
		chReply chan struct{}
	}
	newShutdownError := func(chReply chan struct{}) *shutdownError {
		return &shutdownError{errors.New("stop signed"), chReply}
	}

	// Def.
	measure := func(univ *Universe) error {
		// Note: univ.Agents map[Measurer]Subscribed // 1 // We know by this, the number of Measurers and their work info.
		mapSetMeasuringBirds := map[Swallow]Measurer{}       // 2 // Messengers in a set. So we get an idea about the number of students being evaluated(mailed).
		setOrgFroshesYetMeasured := map[*Organism]struct{}{} // 3 // A set temporarily storing newbs to be evaluated. (So we can count them.)
		// Set newbs.
		univ.MutexLivings.Lock()
		for organism := range univ.Livings {
			if univ.Config.IsFitnessRemeasurable || !organism.IsFitnessMeasured() {
				setOrgFroshesYetMeasured[organism] = struct{}{}
			}
		}
		univ.MutexLivings.Unlock()
		for len(mapSetMeasuringBirds) > 0 || len(setOrgFroshesYetMeasured) > 0 {
			select {
			case chReply := <-univ.ChStopSign: // don't reply and throw that to other routine
				return newShutdownError(chReply) // Stop running this measure().
			default:
				// Noop. Do nothing.
			}
			{ // 1. Send enough
				univ.MutexAgents.Lock()
				for measurer, postman := range univ.Agents {
					if postman != nil {
						continue
					}
					for newbie := range setOrgFroshesYetMeasured {
						bird, err := measurer.MeasureAsync(newbie.Phenotype[:])
						if err != nil { // 'This agent is already working' error.
							// log.Println("fatal:", err) // debug //
							panic(err)
						}
						log.Println("started measuring:", newbie) //
						// 123
						univ.Agents[measurer] = NewSubscribed(bird, newbie) // 1
						mapSetMeasuringBirds[bird] = measurer               // 2
						delete(setOrgFroshesYetMeasured, newbie)            // 3
						//
						break // Only a single item is retrieved from this map.
					}
				}
				univ.MutexAgents.Unlock()
			}
			{ // 2. Recv one
				if !(len(mapSetMeasuringBirds) > 0) {
					continue
				}
				birdCases := make([]reflect.SelectCase, len(mapSetMeasuringBirds)+1)
				{ // dynamic N cases
					birdCases[0] = reflect.SelectCase{
						Dir:  reflect.SelectRecv,
						Chan: reflect.ValueOf(univ.ChStopSign),
					}
					i := 1
					for messenger := range mapSetMeasuringBirds {
						birdCases[i] = reflect.SelectCase{
							Dir:  reflect.SelectRecv,
							Chan: reflect.ValueOf(messenger),
						}
						i++
					}
				}
				// select case of dynamic N cases
				if iMessenger, itfMessageFitness, notClosed := reflect.Select(birdCases); notClosed { // This blocks the thread.
					if iMessenger == 0 { // if univ.ChStopSign
						chReply := itfMessageFitness.Interface().(chan struct{}) // don't reply and throw that to other routine
						return newShutdownError(chReply)                         // Stop running this measure().
					}
					univ.MutexAgents.Lock()
					// set vars
					messenger := birdCases[iMessenger].Chan.Interface().(Swallow)
					messageFitness := itfMessageFitness.Float()
					measurer := mapSetMeasuringBirds[messenger]
					organism := univ.Agents[measurer].O
					// update
					organism.UpdateFitness(messageFitness)
					univ.TopFitness = math.Max(univ.TopFitness, messageFitness)
					log.Println("measured:", organism) //
					// 123
					univ.Agents[measurer] = nil             // 1
					delete(mapSetMeasuringBirds, messenger) // 2
					// 3 // 3 is already done.
					univ.MutexAgents.Unlock()
				} else { // This must be errornous case - a channel close.
					if iMessenger == 0 { // if univ.ChStopSign
						panic(fmt.Sprint(birdCases[iMessenger], " must not be closed"))
					}
					univ.MutexAgents.Lock()
					log.Println("warning: a closed channel is selected") //
					messenger := birdCases[iMessenger].Chan.Interface().(Swallow)
					measurer := mapSetMeasuringBirds[messenger]
					delete(univ.Agents, measurer)           // 1 // unregister measurer
					delete(mapSetMeasuringBirds, messenger) // 2
					// 3 // 3 is already done.
					// End of abnormal event handler.
					univ.MutexAgents.Unlock()
				}
			}
			// spin loop
			// for messenger, organism := range mapSetMeasuringBirds {
			// 	select {
			// 	case fitness := <-messenger:
			// 		organism.UpdateFitness(fitness)
			// 		delete(mapSetMeasuringBirds, messenger)
			// 	default:
			// 		// Noop. Do nothing.
			// 	}
			// }
		} // for
		univ.Save("akashic." + univ.Config.ExperimentName + "." + strconv.Itoa(univ.Generation) + ".measured.json")
		return nil
	}
	extinctify := func(univ *Universe) {
		// Update species.
		univ.MutexClasses.Lock()
		for _, group := range univ.Classes {
			// Check stagnancy.
			_ /*stagnancy*/, _ /*topFitnessOfSpecies*/, err := group.EvaluateGeneration() // update stagnancy and topFitness
			if err != nil {
				// log.Println("fatal:", err) // debug //
				panic(err)
			}
		}
		univ.MutexClasses.Unlock()
		// Presidential election.
		boss, inPlaceOfBoss := func(univ *Universe) (immuneToExtinction *Species, vicePresident *Species) { // get one leading species
			univ.MutexClasses.Lock()
			sortByProminence, err := univ.GetSpeciesOrderedByProminence()
			univ.MutexClasses.Unlock()
			if err != nil {
				// log.Println("fatal:", err) // debug //
				panic(err)
			}
			if len(sortByProminence) >= 2 {
				return sortByProminence[0], sortByProminence[1]
			}
			return sortByProminence[0], nil // assuming at least one species is there in univ.
		}(univ)
		log.Println("top leading species:", boss) //
		if inPlaceOfBoss != nil {
			log.Println("second leading species:", inPlaceOfBoss) //
		}
		// The black spot distribution.
		const (
			_ = iota
			extinctionReasonOverStagnantGlobal
			extinctionReasonOverStagnantLocal
			extinctionReasonTooWeakAvgFitness
			extinctionReasonTooWeakTopFitness
			extinctionReasonTooWeakHomeAlone
		)
		sumFitAvgAdj, fitAvgsAdj := univ.AdjAvgFitnessesOfSpecies() // reads
		sumFitTopAdj, fitTopsAdj := univ.AdjTopFitnessesOfSpecies() // reads
		indicesSpeciesExtinct := []int{}
		reasonsSpeciesExtinct := map[*Species]int{}
		univ.MutexClasses.Lock()
		for iGroup, group := range univ.Classes { // univ.Classes because of univ.AverageFitnessesOfSpecies()
			// Blackspot because RemoveSpecies() modifies `univ.Classes` thus can't be called while iterating over it.
			if group == boss { // the boss is immune to any of these death conditions
				continue
			}
			if univ.Config.MaxCntApocalypse >= 0 && boss.Stagnancy > univ.Config.MaxCntApocalypse { // Apocalypse not Armageddon
				if group != inPlaceOfBoss {
					indicesSpeciesExtinct = append(indicesSpeciesExtinct, iGroup)
					reasonsSpeciesExtinct[group] = extinctionReasonOverStagnantGlobal
				}
				continue
			}
			if univ.Config.MaxCntArmageddon >= 0 && group.Stagnancy > univ.Config.MaxCntArmageddon { // general over-stagnant
				indicesSpeciesExtinct = append(indicesSpeciesExtinct, iGroup)
				reasonsSpeciesExtinct[group] = extinctionReasonOverStagnantLocal
				continue
			}
			if 1 > int(math.Floor((fitAvgsAdj[iGroup]/sumFitAvgAdj)*float64(univ.Config.SizePopulation))) { // too weak in avg-fitness to even become a species with at least 1 population
				indicesSpeciesExtinct = append(indicesSpeciesExtinct, iGroup)
				reasonsSpeciesExtinct[group] = extinctionReasonTooWeakAvgFitness
				continue
			}
			if 1 > int(math.Floor((fitTopsAdj[iGroup]/sumFitTopAdj)*float64(univ.Config.SizePopulation))) { // too weak in top-fitness to even become a species with at least 1 population
				indicesSpeciesExtinct = append(indicesSpeciesExtinct, iGroup)
				reasonsSpeciesExtinct[group] = extinctionReasonTooWeakTopFitness
				continue
			}
			if group.Size() < 2 && group.Stagnancy > 0 { // species with only a single population doesn't need to get more than one chance
				indicesSpeciesExtinct = append(indicesSpeciesExtinct, iGroup)
				reasonsSpeciesExtinct[group] = extinctionReasonTooWeakHomeAlone
				continue
			}
		}
		univ.MutexClasses.Unlock()
		// Extinction.
		classesRemovedAndEmpty, expelledOrgsByClasses, err := univ.RemoveClasses(indicesSpeciesExtinct...) // should be called outside the univ.Classes iteration.
		if err != nil {
			// log.Println("fatal:", err) // debug //
			panic(err)
		}
		for iClass, expelledOrgs := range expelledOrgsByClasses {
			classEmptied := classesRemovedAndEmpty[iClass]
			for _, organismExpelled := range expelledOrgs {
				if err := univ.RemoveOrganism(organismExpelled); err != nil {
					// log.Println("fatal:", err) // debug //
					panic(err)
				}
			}
			switch reasonsSpeciesExtinct[classEmptied] {
			case extinctionReasonOverStagnantGlobal:
				log.Println(len(expelledOrgs), "creature(s) of", classEmptied.Niche(), "got annihilated for this entire universe being stagnant over", univ.Config.MaxCntApocalypse, "generation(s) (extinction)") //
			case extinctionReasonOverStagnantLocal:
				log.Println(len(expelledOrgs), "creature(s) of", classEmptied.Niche(), "got annihilated for being locally stagnant over", univ.Config.MaxCntArmageddon, "generation(s) (extinction)") //
			case extinctionReasonTooWeakAvgFitness:
				log.Println(len(expelledOrgs), "creature(s) of", classEmptied.Niche(), "got annihilated for being too weak in avg-fitness (extinction)") //
			case extinctionReasonTooWeakTopFitness:
				log.Println(len(expelledOrgs), "creature(s) of", classEmptied.Niche(), "got annihilated for being too weak in top-fitness (extinction)") //
			case extinctionReasonTooWeakHomeAlone:
				log.Println(len(expelledOrgs), "creature(s) of", classEmptied.Niche(), "got annihilated for being too weak with only a single population (extinction)") //
			default:
				panic("extinction reason unjustified")
			}
		}
		log.Println(len(classesRemovedAndEmpty), "classification(s) got annihilated for being stagnant or being too weak (extinction)") //
	}
	cull := func(univ *Universe) {
		// Wiki: In biology, culling is the process of segregating organisms from
		// a group according to desired or undesired characteristics.
		univ.MutexClasses.Lock()
		for _, zoo := range univ.Classes {
			eliminated := zoo.Cull(univ.Config.PercentageCulling)
			log.Println(len(eliminated), "creature(s) got culled off of", zoo.Niche()) //
			for _, organism := range eliminated {
				univ.RemoveOrganism(organism)
			}
		}
		univ.MutexClasses.Unlock()
	}
	reproduce := func(univ *Universe) (orphans []*Organism) {
		// fitnesses
		sumFitAvgAdj, fitAvgsAdj := univ.AdjAvgFitnessesOfSpecies()
		// orphans
		orphans = []*Organism{}
		nSizeOrphanageForNewbs := func() int {
			// assuming this is after the culling and the extinction.
			if univ.Config.IsProtectiveElitism {
				return univ.Config.SizePopulation - len(univ.Livings)
			}
			return univ.Config.SizePopulation - len(univ.Classes)
		}()
		// orphanage 1
		univ.MutexClasses.Lock()
		for i, s := range univ.Classes {
			if nOffsprings := int(math.Floor((fitAvgsAdj[i] / sumFitAvgAdj) * float64(nSizeOrphanageForNewbs))); nOffsprings > 0 {
				babies, err := univ.NewOrganismBrood(s, nOffsprings)
				if err != nil {
					// log.Println("fatal:", err) // debug //
					panic(err)
				}
				orphans = append(orphans, babies...)
			}
		}
		univ.MutexClasses.Unlock()
		log.Println("reproduced", len(orphans), "new offspring(s)", "and", nSizeOrphanageForNewbs-len(orphans), "filler(s) will be") //
		// orphanage 2
		for len(orphans) < nSizeOrphanageForNewbs {
			newFiller, err := univ.SpeciesRandom().Champion().GenoType().Copy()
			if err != nil {
				// log.Println("fatal:", err) // debug //
				panic(err)
			}
			univ.Mutate(newFiller)
			o, err := newFiller.NewOrganismSimple()
			if err != nil {
				// log.Println("fatal:", err) // debug //
				panic(err)
			}
			orphans = append(orphans, o)
		}
		log.Println("total of", len(orphans), "creature(s) reproduced") //
		if !univ.Config.IsProtectiveElitism {
			univ.MutexClasses.Lock()
			for _, zoo := range univ.Classes { // re-culling process in reproduce method
				eliminated := zoo.CullToSinglePopulation()
				log.Println(len(eliminated), "elite creature(s) of", zoo.Niche(), "are retired") //
				for _, organism := range eliminated {
					univ.RemoveOrganism(organism)
				}
			}
			univ.MutexClasses.Unlock()
		}
		// test
		if nSizeOrphanageForNewbs != len(orphans) {
			panic(fmt.Sprint(
				fmt.Sprintln("fatal miscalculation: your math is way off"),
				fmt.Sprintln(
					"nSizeOrphanageForNewbs ->", nSizeOrphanageForNewbs,
					"len(orphans) ->", len(orphans),
				),
				fmt.Sprintln(
					"sumFitAvgAdj ->", sumFitAvgAdj,
					"fitAvgsAdj ->", fitAvgsAdj,
				),
			))
		}
		// return
		return orphans
	}
	speciate := func(univ *Universe, orphans ...*Organism) {
		for _, orphan := range orphans {
			if orphan == nil {
				continue
			}
			if err := univ.AddOrganism(orphan); err != nil {
				// log.Println("fatal:", err) // debug //
				panic(err)
			}
			if err := univ.Speciate(orphan); err != nil {
				// log.Println("fatal:", err) // debug //
				panic(err)
			}
			// log.Println("speciated:", orphan) //
		}
	}
	forwardGeneration := func(univ *Universe) {
		univ.Generation++
		univ.Save("akashic." + univ.Config.ExperimentName + "." + strconv.Itoa(univ.Generation) + ".json")
		// test
		if univ.Config.SizePopulation != len(univ.Livings) {
			panic(fmt.Sprint(
				fmt.Sprintln("fatal miscalculation: your math is way off"),
				fmt.Sprintln(
					"univ.Config.SizePopulation ->", univ.Config.SizePopulation,
					"len(univ.Livings) ->", len(univ.Livings),
				),
			))
		}
	}
	epoch := func(univ *Universe) error {
		defer func() {
			if r := recover(); r != nil { //
				log.Println("epoch(): panic recovered")
				log.Println("reason for panic:", r)
				log.Println("stacktrace from panic: \r\n" + string(debug.Stack()))
			}
		}()

		var chReplyOnHold chan struct{}
		holdStopSign := func(chReplyOnHold chan struct{}) { // arg there to make this function look more explicit
			if chReplyOnHold == nil {
				select {
				case chReplyOnHold = <-univ.ChStopSign:
					// don't reply and throw that to outside procedure
					log.Println("")
					log.Println("NaNEAT Main: Shutting down... Please wait until this generation gets saved.")
					log.Println("")
				default:
					// Noop. Do nothing.
				}
			}
		}
		unholdStopSign := func(chReplyOnHold chan struct{}, msg string) error { // arg there to make this function look more explicit
			if chReplyOnHold != nil {
				log.Println("")
				log.Println(msg)
				log.Println("")
				// don't reply and throw that to other routine
				return newShutdownError(chReplyOnHold)
			}
			return nil
		}

		_, topFitness, population, numSpecies, generation, innovation, nicheCount := univ.Info()
		log.Println("")
		log.Println("NaNEAT Main:",
			"Gen", generation, "Population", population,
			"TopFitness", topFitness, "Innovation", innovation,
			"NicheCount", nicheCount, "Num.Species", numSpecies,
		) //
		log.Println("")

		if err := unholdStopSign(
			chReplyOnHold, "NaNEAT Main: Run stops. All the work is save to a json file.",
		); err != nil {
			return err // Stop running this epoch.
		}

		select { // resolve stop sign // neither hold nor unhold
		case chReply := <-univ.ChStopSign:
			log.Println("")
			log.Println("NaNEAT Main: Run interrupted and just stopped. All the work is save to a json file if Gen > 0.")
			log.Println("")
			// don't reply and throw that to other routine
			return newShutdownError(chReply) // Stop running this epoch.
		default:
			// Noop. Do nothing.
		}

		log.Println("")
		log.Println("NaNEAT Main: 1. Measure", univ.Generation) //
		log.Println("")
		if err := measure(univ); err != nil { // Measure fitnesses of our creatures.
			switch err.(type) { // resolve stop sign // neither hold nor unhold
			case *shutdownError: // catch thrown channel
				univ.Save("akashic." + univ.Config.ExperimentName + "." + strconv.Itoa(univ.Generation) + ".measuring.json")
				log.Println("")
				log.Println("NaNEAT Main: Run interrupted and just stopped. All the work is saved to a json file.")
				log.Println("")
				// don't reply and throw that to other routine
				return err // Stop running this epoch.
			default: // panic for unhandled errors
				panic(err)
			}
		}

		holdStopSign(chReplyOnHold)

		log.Println("")
		log.Println("NaNEAT Main: 2. Extinctify", univ.Generation) //
		log.Println("")
		extinctify(univ) // Stagnant species get obliterated and face their extinction. xD

		holdStopSign(chReplyOnHold)

		log.Println("")
		log.Println("NaNEAT Main: 3. Cull", univ.Generation) //
		log.Println("")
		cull(univ) // Cull off the weakers in each species.

		holdStopSign(chReplyOnHold)

		log.Println("")
		log.Println("NaNEAT Main: 4. Reproduce", univ.Generation) //
		log.Println("")
		kids := reproduce(univ) // Breed alive creatures in their mating season, involving genetic mutation and crossover.

		holdStopSign(chReplyOnHold)

		log.Println("")
		log.Println("NaNEAT Main: 5. Speciate", univ.Generation) //
		log.Println("")
		speciate(univ, kids...) // Classify all children of this new generation individually. Plus respeciate previous ones.

		holdStopSign(chReplyOnHold)

		log.Println("")
		log.Println("NaNEAT Main: 6. ForwardGeneration", univ.Generation) //
		log.Println("")
		forwardGeneration(univ)

		return nil
	} // func literal

	// Init cycles.
	for {
		if err := epoch(univ); err != nil {
			defer func() { // in case another stop sign came in while doing all these things
				select {
				case chReply := <-univ.ChStopSign:
					chReply <- struct{}{}
				default:
					// Noop. Do nothing.
				}
			}()
			switch vErr := err.(type) {
			case *shutdownError: // Reply at the end of Run() to prevent a new stop sign to be received while handling one of these cases.
				vErr.chReply <- struct{}{}
				return nil // Successfully stopping running this universe.
			default: // panic for unhandled errors
				panic(err)
			}
		}
	} // for

	// return nil // unreachable
}

// IsPumping thread-safely tells if this universe is running or not.
func (univ *Universe) IsPumping() bool {
	univ.MutexRun.Lock()
	defer univ.MutexRun.Unlock()

	return univ.IsRunning
}

// Shutdown stops running this universe.
// This blocks the routine this method is being called upon until the universe stops.
// Falls through if the universe isn't currently running (at the time this function gets called).
func (univ *Universe) Shutdown() {
	if univ.IsPumping() {
		chReplyWhenStopped := make(chan struct{})
		univ.ChStopSign <- chReplyWhenStopped
		<-chReplyWhenStopped
	}
}

// ----------------------------------------------------------------------------
// Universe - Getter
//
// Thread-safely get a universe state.
//

// Info peeks this universe's basic status. This should be thread-safe.
// The return is a nice summarized set of basic informative infomations.
//
// Usage:
// 	_, topFitness, population, numSpecies, generation, innovation, nicheCount := univ.Info()
//
func (univ *Universe) Info() (
	conf Configuration, topFitness float64,
	population, numSpecies, generation, innovation, nicheCount int,
) {
	return univ.Config, univ.TopFitness,
		len(univ.Livings), len(univ.Classes),
		univ.Generation, univ.Innovation, univ.NicheCount
}

// Measurers thread-safely returns all registered measurers and their work details we might want to take a look at.
// The returned map is a copy but its contents are pointers to original objects.
func (univ *Universe) Measurers() (ret map[Measurer]*Subscribed) {
	univ.MutexAgents.Lock()
	defer univ.MutexAgents.Unlock()

	ret = map[Measurer]*Subscribed{}
	for measurer, subbed := range univ.Agents {
		ret[measurer] = subbed
	}
	return ret
}

// Breeds is a getter thread-safely returning all species present in this universe.
// The returned slice is newly constructed and, comes with its contents that reference to original objects.
func (univ *Universe) Breeds() []*Species {
	univ.MutexClasses.Lock()
	defer univ.MutexClasses.Unlock()

	return append([]*Species{}, univ.Classes...)
}

// Organisms is a getter thread-safely returning all organisms present in this universe.
// The returned slice is newly constructed and, comes with its contents that reference to original objects.
func (univ *Universe) Organisms() []*Organism {
	univ.MutexLivings.Lock()
	defer univ.MutexLivings.Unlock()

	ret := make([]*Organism, len(univ.Livings))
	i := 0
	for creature := range univ.Livings {
		ret[i] = creature
		i++
	}
	return ret
}

// BiasGenes returns genes for bias input nodes. This should be thread-safe.
// The returned slice is newly constructed and, comes with its contents that reference to original objects.
func (univ *Universe) BiasGenes() []*NodeGene {
	return append([]*NodeGene{}, univ.InputGenes[:univ.Config.NumBiases]...)
}

// NonBiasInputGenes returns genes for non-bias input nodes. This should be thread-safe.
// The returned slice is newly constructed and, comes with its contents that reference to original objects.
func (univ *Universe) NonBiasInputGenes() []*NodeGene {
	return append([]*NodeGene{}, univ.InputGenes[univ.Config.NumBiases:]...)
}

// GetInputGenes is a getter. This should be thread-safe.
// The returned slice is newly constructed and, comes with its contents that reference to original objects.
func (univ *Universe) GetInputGenes() []*NodeGene {
	return append([]*NodeGene{}, univ.InputGenes...)
}

// GetOutputGenes is a getter. This should be thread-safe.
// The returned slice is newly constructed and, comes with its contents that reference to original objects.
func (univ *Universe) GetOutputGenes() []*NodeGene {
	return append([]*NodeGene{}, univ.OutputGenes...)
}

// ----------------------------------------------------------------------------
// Universe - Creator
//
// Note:
// What New~Basic() would return are so pre-defined, (all filled up with hypers)
// thus those functions do not take any argument other than the method's receiver.
//

// AncestorGenes is a getter returning a copy slice of all ancestral node-genes in this universe.
// The slices returned are new; newly constructed with the contents that reference to those original objects.
func (univ *Universe) AncestorGenes() (inputs []*NodeGene, outputs []*NodeGene, err error) {
	inputs = make([]*NodeGene, len(univ.InputGenes))
	if copy(inputs, univ.InputGenes) != len(inputs) {
		return nil, nil, errors.New("failed to copy those inputs node-genes")
	}
	outputs = make([]*NodeGene, len(univ.OutputGenes))
	if copy(outputs, univ.OutputGenes) != len(outputs) {
		return nil, nil, errors.New("failed to copy those outputs node-genes")
	}
	return inputs, outputs, nil
}

// NewChanceGeneBasic is a constructor.
func (univ *Universe) NewChanceGeneBasic() *ChanceGene {
	enabled := univ.Config.ChanceIsMutational
	addNode := univ.Config.ChanceAddNode
	addLink := univ.Config.ChanceAddLink
	addBias := univ.Config.ChanceAddBias
	perturbWeight := univ.Config.ChancePerturbWeight
	nullifyWeight := univ.Config.ChanceNullifyWeight
	turnOn := univ.Config.ChanceTurnOn
	turnOff := univ.Config.ChanceTurnOff
	bump := univ.Config.ChanceBump
	return NewChanceGene(enabled, addNode, addLink, addBias, perturbWeight, nullifyWeight, turnOn, turnOff, bump)
}

// NewChanceGeneMix is a constructor.
// All parameters are mandatory and should not be nil. This function panics otherwise.
// The return is the result of a crossover. nil is returned if there is a logical error.
func (univ *Universe) NewChanceGeneMix(vParent1, vParent2 ChanceGene, isAvgNotRnd bool) *ChanceGene {
	if vParent1.IsEnabled() && vParent2.IsEnabled() {
		const left, right int = 0, 1
		pair := [2][]float64{
			func(n ...float64) []float64 { return n }(vParent1.Unpack()), // left
			func(n ...float64) []float64 { return n }(vParent2.Unpack()), // right
		}
		if isAvgNotRnd {
			return NewChanceGeneEnabled(
				(pair[left][0]+pair[right][0])/2, // addNode
				(pair[left][1]+pair[right][1])/2, // addLink
				(pair[left][2]+pair[right][2])/2, // addBias
				(pair[left][3]+pair[right][3])/2, // perturbWeight
				(pair[left][4]+pair[right][4])/2, // nullifyWeight
				(pair[left][5]+pair[right][5])/2, // turnOn
				(pair[left][6]+pair[right][6])/2, // turnOff
				(pair[left][7]+pair[right][7])/2, // bump
			) // averaged
		}
		// Rnd
		return NewChanceGeneEnabled(
			pair[rand.Intn(2)][0], // addNode
			pair[rand.Intn(2)][1], // addLink
			pair[rand.Intn(2)][2], // addBias
			pair[rand.Intn(2)][3], // perturbWeight
			pair[rand.Intn(2)][4], // nullifyWeight
			pair[rand.Intn(2)][5], // turnOn
			pair[rand.Intn(2)][6], // turnOff
			pair[rand.Intn(2)][7], // bump
		) // 50% chance each
	} else if !vParent1.IsEnabled() && !vParent2.IsEnabled() {
		return univ.NewChanceGeneBasic()
	} else if vParent1.IsEnabled() && !vParent2.IsEnabled() {
		return vParent1.Copy()
	} else if !vParent1.IsEnabled() && vParent2.IsEnabled() {
		return vParent2.Copy()
	}
	return nil
}

// NewInnovation pulls out a new historical marker from this (current) NEAT context.
// The return is an identifiable innovation number, which is unique in this universe and is considered a component of the LinkGene.
func (univ *Universe) NewInnovation() (innovation *Innovation) {
	v := Innovation(univ.Innovation)
	innovation = &v
	univ.Innovation++
	return innovation
}

// NewLinkGene is a constructor.
// The link gene comes with a historical marker unique in this universe. (an identifiable innovation number)
func (univ *Universe) NewLinkGene(from, to *NodeGene, weight float64, enabled bool) *LinkGene {
	return NewLinkGene(*univ.NewInnovation(), from, to, weight, enabled)
}

// NewLinkGeneSimple is a constructor.
// The link gene comes with an identifiable innovation number.
// A perturbed random weight is assigned to the gene created.
func (univ *Universe) NewLinkGeneSimple(from, to *NodeGene, enabled bool) *LinkGene {
	ret := NewLinkGene(*univ.NewInnovation(), from, to, float64(NeuralValueMid), enabled)
	univ.PerturbLinkGene(ret)
	return ret
}

// NewLinkGeneFromLinkGene is a constructor.
// The link gene comes with an identifiable innovation number.
// The rest of its fields are filled up with the parent's.
//
// Parameter:
//  - `vParent`: The parent as value.
//
func (univ *Universe) NewLinkGeneFromLinkGene(vParent LinkGene) (daughter *LinkGene) {
	ret := NewLinkGene(*univ.NewInnovation(), vParent.Topo.From, vParent.Topo.To, vParent.Weight, vParent.Enabled)
	return ret
}

// NewLinkGeneSimpleFromLinkGene is a constructor.
// The link gene comes with an identifiable innovation number.
// A perturbed random weight is assigned to the gene created.
// The rest of its fields are filled up with the parent's.
//
// Parameter:
//  - `vParent`: The parent as value.
//
func (univ *Universe) NewLinkGeneSimpleFromLinkGene(vParent LinkGene) (daughter *LinkGene) {
	ret := univ.NewLinkGeneFromLinkGene(vParent)
	univ.PerturbLinkGene(ret)
	return ret
}

// NewNodeGeneBasic is a constructor.
// The return is a new unique identifiable hidden node.
func (univ *Universe) NewNodeGeneBasic() *NodeGene {
	return NewNodeGene("", HiddenNode, -1)
}

// NewChromosome is a constructor.
func (univ *Universe) NewChromosome(linkGenes []*LinkGene) (*Chromosome, error) {
	input, output, err := univ.AncestorGenes()
	if err != nil {
		return nil, err
	}
	return NewChromosome(linkGenes, append(input, output...), univ.NewChanceGeneBasic())
}

// NewChromosomeBasic is a constructor.
func (univ *Universe) NewChromosomeBasic() (*Chromosome, error) {
	return univ.NewChromosome(nil)
}

// NewGenomeBasic is a constructor.
func (univ *Universe) NewGenomeBasic() (g *Genome, err error) {
	chromes := make([]*Chromosome, univ.Config.NumChromosomes)
	for i := range chromes {
		chromes[i], err = univ.NewChromosomeBasic()
		if err != nil {
			return nil, err
		}
	}
	return NewGenome(chromes)
}

// NewOrganismBasic is a constructor.
// This returns a basic mutant that's unmeasured and unrelated with any species.
func (univ *Universe) NewOrganismBasic() (o *Organism, err error) {
	g, err := univ.NewGenomeBasic()
	if err != nil {
		return nil, err
	}
	univ.Mutate(g)
	return g.NewOrganismSimple()
}

// NewOrganismBrood returns N new offsprings (re)produced from a designated species,
// where N is given by the parameter `nChildren` >= 1.
// The hermaphroditism is strongly avoided unless there is only one creature left in the species.
// This function can return error for bad genetics issue. Panics upon runtime violation.
func (univ *Universe) NewOrganismBrood(s *Species, nChildren int) (children []*Organism, err error) {
	if s == nil {
		return nil, errors.New("nil species")
	}
	if s.Size() < 1 {
		return nil, errors.New("empty species")
	}
	if nChildren < 1 {
		return nil, errors.New("nChildren < 1")
	}
	const (
		CrossoverMultipointRnd  = iota // Sexual reproduction 1
		CrossoverMultipointAvg         // Sexual reproduction 2
		CrossoverSinglepointRnd        // Sexual reproduction 3
		CrossoverSinglepointAvg        // Sexual reproduction 4
		Fission                        // Asexual breeding (Binary fission)
		Sterile                        // Birth control (Contraception)
	)
	// local return
	children = make([]*Organism, nChildren)
	populate := func(iBaby int, genomeBaby *Genome) error {
		if genomeBaby == nil {
			children[iBaby] = nil
		}
		univ.Mutate(genomeBaby)
		children[iBaby], err = genomeBaby.NewOrganismSimple()
		return err
	}
	for i := range children {
		// 1. get newGenomeBaby
		var (
			newGenomeBaby *Genome
			mom, dad      *Organism
		)
		if nCreatures := s.Size(); nCreatures == 1 {
			mom = s.OrganismRandom()
			dad = mom // hermaphrodite
		} else if nCreatures > 1 {
			parents := s.RandomOrganisms(2)
			mom = parents[0]
			dad = parents[1]
		} else {
			panic("runtime violation - empty species")
		}
		switch Roulette(univ.Config.BirthRatioSimplified()) {
		case CrossoverMultipointRnd: // 1
			if mom == dad { // hermaphrodite
				newGenomeBaby, err = mom.GenoType().Copy()
			} else { // sexual
				newGenomeBaby, err = univ.CrossoverMultipoint(mom, dad, false)
			}
			if err != nil {
				return nil, err
			}
		case CrossoverMultipointAvg: // 2
			if mom == dad { // hermaphrodite
				newGenomeBaby, err = mom.GenoType().Copy()
			} else { // sexual
				newGenomeBaby, err = univ.CrossoverMultipoint(mom, dad, true)
			}
			if err != nil {
				return nil, err
			}
		case CrossoverSinglepointRnd: // 3
			if mom == dad { // hermaphrodite
				newGenomeBaby, err = mom.GenoType().Copy()
			} else { // sexual
				newGenomeBaby, err = univ.CrossoverSinglepoint(mom.GenoType(), dad.GenoType(), false)
			}
			if err != nil {
				return nil, err
			}
		case CrossoverSinglepointAvg: // 4
			if mom == dad { // hermaphrodite
				newGenomeBaby, err = mom.GenoType().Copy()
			} else { // sexual
				newGenomeBaby, err = univ.CrossoverSinglepoint(mom.GenoType(), dad.GenoType(), true)
			}
			if err != nil {
				return nil, err
			}
		case Fission: // 5
			newGenomeBaby, err = mom.GenoType().Copy()
			if err != nil {
				return nil, err
			}
		default: // This can't be taken lightly. xD
			return nil, errors.New("undefined reproduction method")
		} // switch case
		//
		// 2. populate newGenomeBaby
		if err := populate(i, newGenomeBaby); err != nil {
			return nil, err
		}
	} // for
	return children, nil
}

// NewNiche returns an identifiable niche that's unique in this universe.
// Niche is considered a component of the Species.
func (univ *Universe) NewNiche() (habitat *Niche) {
	v := Niche(univ.NicheCount)
	habitat = &v
	univ.NicheCount++
	return habitat
}

// NewSpecies is a constructor. This creates a species with its unique identifier.
func (univ *Universe) NewSpecies(livings []*Organism) *Species {
	return NewSpecies(*univ.NewNiche(), livings)
}

// NewSpeciesBasic is a constructor. This simply creates an empty species with its own niche.
func (univ *Universe) NewSpeciesBasic() *Species {
	return univ.NewSpecies(nil)
}

// ----------------------------------------------------------------------------
// Universe - Modifier

// PerturbChanceGene perturbs the weight of a dice-gene, in a strength constant and statically defined in this universe.
func (univ *Universe) PerturbChanceGene(perturbed *ChanceGene) {
	perturbed.Perturb(
		univ.Config.TendencyPerturbDice,
		univ.Config.StrengthPerturbDice,
	)
}

// PerturbLinkGene perturbs the weight of a link-gene, in a strength defined in the dice.
// Argument dice is allowed to be nil. The other arguments are not.
func (univ *Universe) PerturbLinkGene(perturbed *LinkGene) {
	perturbed.Perturb(
		univ.Config.TendencyPerturbWeight,
		univ.Config.StrengthPerturbWeight,
	)
}

// ----------------------------------------------------------------------------
// Universe - Mutator

// Mutate a genome under this universe's context.
// To avoid confusion make sure that you don't mutate an existing organism's genome.
// It is strongly recommended that you get a copy of a genome you want to mutate before this.
func (univ *Universe) Mutate(g *Genome) {
	for _, chrome := range g.Chromosomes() {
		univ.MutateDice(chrome)
		univ.MutatePerturbWeight(chrome)
		univ.MutateNullifyWeight(chrome)
		if err := univ.MutateAddLinkNoBias(chrome); err != nil {
			switch err.Error() {
			case "nowhere": // handeled error
				// log.Println("Cannot MutateAddLink() with this network. It might be fully connected.") // debug //
			default: // unhandled errors
				panic(err)
				// log.Println("fatal:", err) // debug //
			}
		}
		if err := univ.MutateAddBias(chrome); err != nil {
			switch err.Error() {
			case "nowhere": // handeled error
				// log.Println("Cannot mutate add bias: either bias is not present or fully connected") // debug //
			default: // unhandled errors
				panic(err)
				// log.Println("fatal:", err) // debug //
			}
		}
		univ.MutateAddNode(chrome)
		univ.MutateBump(chrome)
		univ.MutateEnableOnOff(chrome)
	}
}

// MutateDice noises all the dynamic chances of a genetic encoder Gaussian distributed.
// The way this works is similar to the weight mutation.
func (univ *Universe) MutateDice(chrome *Chromosome) {
	if chrome.IsDiceEnabled() {
		// Chance(probability) for mutating a dice is fixed to a constant 100%.
		univ.PerturbChanceGene(chrome.DiceGene)
	}
}

// MutatePerturbWeight noises all the weights of a genetic encoder Gaussian distributed.
func (univ *Universe) MutatePerturbWeight(chrome *Chromosome) {
	for _, linkGene := range chrome.LinkGenes {
		for chance := func() float64 { // the initial value
			p := 0.0
			if chrome.IsDiceEnabled() {
				p = chrome.DiceGene.PerturbWeight
			} else if !chrome.IsDiceEnabled() {
				p = univ.Config.ChancePerturbWeight
			} else {
				panic("runtime violation")
			}
			return p
		}(); chance > 0.0; chance -= 1.0 {
			if rand.Float64() < chance {
				univ.PerturbLinkGene(linkGene)
			}
		}
	}
}

// MutateNullifyWeight gives a chance for synaptic weights to become zero.
func (univ *Universe) MutateNullifyWeight(chrome *Chromosome) {
	for _, linkGene := range chrome.LinkGenes {
		for chance := func() float64 { // the initial value
			p := 0.0
			if chrome.IsDiceEnabled() {
				p = chrome.DiceGene.NullifyWeight
			} else if !chrome.IsDiceEnabled() {
				p = univ.Config.ChanceNullifyWeight
			} else {
				panic("runtime violation")
			}
			return p
		}(); chance > 0.0; chance -= 1.0 {
			if rand.Float64() < chance {
				linkGene.Weight = 0.0
			}
		}
	}
}

// MutateAddLinkNoBias creates a structural innovation between existing non-bias nodes.
func (univ *Universe) MutateAddLinkNoBias(chrome *Chromosome) error {
	if chrome == nil {
		return errors.New("nil chromosome")
	}
	for chance := func() float64 { // the initial value
		p := 0.0
		if chrome.IsDiceEnabled() {
			p = chrome.DiceGene.AddLink
		} else if !chrome.IsDiceEnabled() {
			p = univ.Config.ChanceAddLink
		} else {
			panic("runtime violation")
		}
		return p
	}(); chance > 0.0; chance -= 1.0 {
		if rand.Float64() < chance {
			proto, err := chrome.NewNeuralNetworkProto(true, true, false)
			if err != nil {
				return err
			}
			from, to, err := proto.RandomNicePlaceForNewAcyclicLink()
			if err != nil {
				return err
			}
			// mutate genetics
			newLinkGene := univ.NewLinkGeneSimple(from.Gene(), to.Gene(), true)
			chrome.AddLinkGenes(newLinkGene) // This will/must not create any cycle in the graph.
		}
	}
	return nil
}

// MutateAddBias creates a link that connects to a bias.
func (univ *Universe) MutateAddBias(chrome *Chromosome) error {
	if chrome == nil {
		return errors.New("nil chromosome")
	}
	for chance := func() float64 { // the initial value
		p := 0.0
		if chrome.IsDiceEnabled() {
			p = chrome.DiceGene.AddBias
		} else if !chrome.IsDiceEnabled() {
			p = univ.Config.ChanceAddBias
		} else {
			panic("runtime violation")
		}
		return p
	}(); chance > 0.0; chance -= 1.0 {
		if rand.Float64() < chance {
			proto, err := chrome.NewNeuralNetworkProto(true, false, true)
			if err != nil {
				return err
			}
			nodes, err := proto.Sort()
			if err != nil {
				return err
			}
			from, to := func() (fromNode, toNode *NeuralNode) {
				if biases := proto.GetInputNodes(); len(biases) > 0 {
					fromNode = biases[rand.Intn(len(biases))]
					toNode = proto.FromTo(nodes, fromNode)
					return fromNode, toNode
				}
				return nil, nil
			}()
			if from == nil || to == nil {
				return errors.New("nowhere") // cannot mutate add bias: either bias is not present or fully connected
			}
			// mutate genetics
			newLinkGene := univ.NewLinkGeneSimple(from.Gene(), to.Gene(), true)
			chrome.AddLinkGenes(newLinkGene) // This will/must not create any cycle in the graph.
		}
	}
	return nil
}

// MutateAddNode of a genetic encoder creates a structural innovation.
// This splits a link resulting in two links and a node.
// The function will panic if nil parameter is provided.
// In theory this shouldn't raise any error as long as the Mutate-Add-Link operation works fine.
func (univ *Universe) MutateAddNode(chrome *Chromosome) {
	// if chrome == nil {
	// 	return errors.New("nil chromosome")
	// }
MutateAddNodeLoop:
	for chance := func() float64 { // the initial value
		p := 0.0
		if chrome.IsDiceEnabled() {
			p = chrome.DiceGene.AddNode
		} else if !chrome.IsDiceEnabled() {
			p = univ.Config.ChanceAddNode
		} else {
			panic("runtime violation")
		}
		return p
	}(); chance > 0.0; chance -= 1.0 {
		links := func(chrome *Chromosome) (enabledLinks []*LinkGene) {
			enabledLinks = []*LinkGene{}
			for _, linkGene := range chrome.LinkGenes {
				if linkGene.Enabled {
					enabledLinks = append(enabledLinks, linkGene)
				}
			}
			return enabledLinks
		}(chrome)
		if len(links) <= 0 {
			break MutateAddNodeLoop
		}
		if rand.Float64() < chance {
			linkSplit := func(enabledLinks []*LinkGene) (randomSplittableLink *LinkGene) {
				return enabledLinks[rand.Intn(len(enabledLinks))]
			}(links)
			// Do not ruin the order below.
			vParentFrom := *linkSplit
			vParentTo := *linkSplit
			if NeuralValue(vParentFrom.Weight).Kind() == NVKindNegative &&
				NeuralValue(vParentTo.Weight).Kind() == NVKindNegative {
				vParentTo.Weight *= -1
			}
			linkSplit.Enabled = false
			newNodeGene := univ.NewNodeGeneBasic()
			vParentFrom.Topo.To = newNodeGene
			vParentTo.Topo.From = newNodeGene
			chrome.AddLinkGenes(
				univ.NewLinkGeneSimpleFromLinkGene(vParentFrom),
				univ.NewLinkGeneSimpleFromLinkGene(vParentTo),
			)
			// Do not ruin the order above.
		}
	} // for
	// Noop. Do nothing.
	return // Return nothing.
}

// MutateBump re-enables the oldest disabled gene.
// This method is only an alias for `(*Chromosome).Bump()`.
//
// Mutations that fiddle around with the enable flag of genes:
//  1. Re-enable the oldest disabled gene. (BUMP)
//  2. Toggle on or off randomly at a rate.
//
func (univ *Universe) MutateBump(chrome *Chromosome) {
	for chance := func() float64 { // the initial value
		p := 0.0
		if chrome.IsDiceEnabled() {
			p = chrome.DiceGene.Bump
		} else if !chrome.IsDiceEnabled() {
			p = univ.Config.ChanceBump
		} else {
			panic("runtime violation")
		}
		return p
	}(); chance > 0.0; chance -= 1.0 {
		if rand.Float64() < chance {
			chrome.Bump()
		}
	}
}

// MutateEnableOnOff of a genetic encoder mutates(toggles) the link-enable flag of one randomly chosen gene.
// Toggle genes from enable on to enable off or vice versa.
//
// Mutations that fiddle around with the enable flag of genes:
//  1. Re-enable the oldest disabled gene. (BUMP)
//  2. Toggle on or off randomly at a rate.
//
func (univ *Universe) MutateEnableOnOff(chrome *Chromosome) {
	enabled, disabled := []*LinkGene{}, []*LinkGene{}
	for _, linkGene := range chrome.LinkGenes {
		if linkGene.Enabled {
			enabled = append(enabled, linkGene)
		} else if !linkGene.Enabled {
			disabled = append(disabled, linkGene)
		} else {
			panic("runtime violation")
		}
	}
	// ON
	for chance := func() float64 { // the initial value
		p := 0.0
		if chrome.IsDiceEnabled() {
			p = chrome.DiceGene.TurnOn
		} else if !chrome.IsDiceEnabled() {
			p = univ.Config.ChanceTurnOn
		} else {
			panic("runtime violation")
		}
		return p
	}(); chance > 0.0 && len(disabled) > 0; chance -= 1.0 {
		if rand.Float64() < chance {
			i := rand.Int63n(int64(len(disabled)))
			disabled[i].Enabled = true
			disabled = append(disabled[:i], disabled[i+1:]...)
		}
	}
	// OFF
	for chance := func() float64 { // the initial value
		p := 0.0
		if chrome.IsDiceEnabled() {
			p = chrome.DiceGene.TurnOff
		} else if !chrome.IsDiceEnabled() {
			p = univ.Config.ChanceTurnOff
		} else {
			panic("runtime violation")
		}
		return p
	}(); chance > 0.0 && len(enabled) > 0; chance -= 1.0 {
		if rand.Float64() < chance {
			i := rand.Int63n(int64(len(enabled)))
			enabled[i].Enabled = false
			enabled = append(enabled[:i], enabled[i+1:]...)
		}
	}
}

// ----------------------------------------------------------------------------
// Universe - Crossover

// CrossoverMultipoint mates two fitness-measured organisms.
// The return is their new obvious child, with mixed up genes, inheriting all
// the structural innovation from the dominant(more-fit) of either parents.
// This allows the topologies to grow incrementally maintaining their minimal
// structure, helping those evolving structures to find out a better solution
// whilst keeping themselves as small as possible.
// This operation alone can interbreed two organisms of different species,
// as it doesn't care the compatibility distance of them.
//
// - - -
//
// Parameters:
//  - `o1`, `o2`: The parents. (can be given in any order)
//  - `isAvgNotRnd`: This option determines the mating method either
//   the matching genes are average weighted or are randomly chosen from either of the parents.
//
// Returns:
//  - `offspring`: The baby as a genetic encoder.
//  - `err`: Not nil if any kind of invalid value was found.
//
// - - -
//
// Multipoint mating methods:
//
//  1. `Rnd`;
//   For every point in each chromosome where each chromosome shares the innovation number,
//   the gene is chosen randomly from either parent.
//   If one parent has an innovation absent in the other,
//   the baby always inherits the innovation from the more fit parent.
//
//  2. `Avg`;
//   Instead of selecting one or the other when the innovation numbers match,
//   this averages the weight at each matching gene.
//
// - - -
//
func (univ *Universe) CrossoverMultipoint(o1, o2 *Organism, isAvgNotRnd bool) (offspring *Genome, err error) {
	// CrossoverableO checks if two chosen organisms are okay to be crossovered,
	// without considering their compatibility.
	// The return provides the reason why the two are unable to reproduce.
	if err = func(o1, o2 *Organism, isFitnessRequired bool) (err error) { // CrossoverableO
		if o1 == nil {
			return errors.New(fmt.Sprint("nil organism: ", o1))
		}
		if o2 == nil {
			return errors.New(fmt.Sprint("nil organism: ", o2))
		}
		if isFitnessRequired {
			if !o1.IsFitnessMeasured() {
				return errors.New(fmt.Sprint("fitness unmeasured: ", o1))
			}
			if !o2.IsFitnessMeasured() {
				return errors.New(fmt.Sprint("fitness unmeasured: ", o2))
			}
		}
		if o1.GenoType().Length() != o2.GenoType().Length() {
			return errors.New(fmt.Sprint("the number of chromosome(s) didn't match: ", o1, o2))
		}
		return nil
	}(o1, o2, true); err != nil {
		return nil, err
	}
	fitness1, err := o1.GetFitness()
	if err != nil {
		return nil, err
	}
	fitness2, err := o2.GetFitness()
	if err != nil {
		return nil, err
	}
	// type local; a full set of what (*Chromosome).Genetics() returns.
	type genetics struct {
		*Chromosome  // inheritance
		mapLinkGenes map[Innovation]*LinkGene
		nLinkGenes   int
		historical   Innovation
	}
	// recombination method
	recombination := /*switch case*/ func() func(superior, inferior /*in*/ genetics, newChromeBeingConstructed /*out*/ *Chromosome) {
		if isAvgNotRnd {
			return func(superior, inferior genetics, newChrome *Chromosome) {
				for innovation, linkGeneBetter := range superior.mapLinkGenes { // for genes
					if linkGenePoorer, isGeneMatched := inferior.mapLinkGenes[innovation]; isGeneMatched {
						// When matched, the weight is averaged.
						newChrome.AddLinkGenes(func() *LinkGene {
							newLinkGeneAvg := linkGenePoorer.Copy()
							newLinkGeneAvg.Weight = (linkGenePoorer.Weight + linkGeneBetter.Weight) / 2
							return newLinkGeneAvg
						}())
					} else { // The rest are from the fitter.
						newChrome.AddLinkGenes(linkGeneBetter.Copy())
					}
				}
			}
		}
		return func(superior, inferior genetics, newChrome *Chromosome) {
			for innovation, linkGeneBetter := range superior.mapLinkGenes { // for genes
				if linkGenePoorer, isGeneMatched := inferior.mapLinkGenes[innovation]; isGeneMatched &&
					linkGenePoorer.Enabled && rand.Float64() < 0.5 {
					// When matched, enabled inferior genes have 50% chance of replacing the opposite for each.
					newChrome.AddLinkGenes(linkGenePoorer.Copy())
				} else { // The rest are from the fitter.
					newChrome.AddLinkGenes(linkGeneBetter.Copy())
				}
			}
		}
	}() // end of switch case
	// create & fill newChromes
	newChromes := make([]*Chromosome, o1.GenoType().Length())
	if err = o1.GenoType().ForEachMatchingChromosome(o2.GenoType(), func(
		i int, // iChrome
		// 1
		chrome1 *Chromosome,
		linkGenes1 map[Innovation]*LinkGene,
		nLinkGenes1 int,
		historical1 Innovation,
		// 2
		chrome2 *Chromosome,
		linkGenes2 map[Innovation]*LinkGene,
		nLinkGenes2 int,
		historical2 Innovation,
	) error {
		// tell structural baggage
		superior, inferior, err := func() (topologicalDominant, topologicalRecessive genetics, err error) {
			// second closure
			genetics1 := genetics{chrome1, linkGenes1, nLinkGenes1, historical1}
			genetics2 := genetics{chrome2, linkGenes2, nLinkGenes2, historical2}
			if fitness1 > fitness2 {
				topologicalDominant = genetics1
				topologicalRecessive = genetics2
			} else if fitness1 < fitness2 {
				topologicalDominant = genetics2
				topologicalRecessive = genetics1
			} else if fitness1 == fitness2 {
				if nLinkGenes1 >= nLinkGenes2 {
					topologicalDominant = genetics2
					topologicalRecessive = genetics1
				} else if nLinkGenes1 < nLinkGenes2 {
					topologicalDominant = genetics1
					topologicalRecessive = genetics2
				} else {
					panic("runtime violation")
				}
			} else {
				panic("runtime violation")
			}
			return topologicalDominant, topologicalRecessive, nil
		}() // closure call
		if err != nil {
			return err
		}
		// fill me
		newChrome, err := univ.NewChromosomeBasic()
		if err != nil {
			return err
		}
		// for range genes
		recombination(superior, inferior, newChrome)
		// dice gene by val
		*newChrome.DiceGene = *univ.NewChanceGeneMix(
			*superior.DiceGene, *inferior.DiceGene, isAvgNotRnd,
		)
		// local return
		newChromes[i] = newChrome
		return nil
	}); err != nil { // This is after the closure above returned.
		return nil, err
	}
	return NewGenome(newChromes)
}

// CrossoverSinglepoint mates two organisms.
// This method does not require the fitnesses to be measured. The return is their new child.
// A single point is chosen in the smaller chromosome to split and cross with the bigger one.
// Genes to the left of that point in the smaller chromosome is connected to the other where
// every single gene is taken from the larger chromosome.
// This operation alone can interbreed two organisms of different species,
// as it doesn't care the compatibility distance of the parents.
//
// - - -
//
// Singlepoint mating methods:
//  1. `Avg`; The gene at crosspoint is averaged with the matching gene from the larger, iif there is one.
//  2. `Rnd`; The gene is randomly chosen from either of the parents' that's not absent.
//
// Parameters:
//  - `g1`, `g2`: The parents. (can be given in any order)
//  - `isAvgNotRnd`: This option determines the mating method either
//   the crosspoint gene is average weighted or is randomly chosen from either of the parents.
//
// Returns:
//  - `offspring`: The baby genome.
//  - `err`: Not nil if any kind of invalid value was found.
//
// - - -
//
func (univ *Universe) CrossoverSinglepoint(g1, g2 *Genome, isAvgNotRnd bool) (offspring *Genome, err error) {
	// CrossoverableG checks if two chosen organisms are okay to be crossovered,
	// without considering their compatibility.
	// The return provides the reason why the two are unable to reproduce.
	if err = func(g1, g2 *Genome) (err error) { // CrossoverableG
		if g1 == nil {
			return errors.New(fmt.Sprint("nil genome: ", g1))
		}
		if g2 == nil {
			return errors.New(fmt.Sprint("nil genome: ", g2))
		}
		if g1.Length() != g2.Length() {
			return errors.New(fmt.Sprint("the number of chromosome(s) didn't match: ", g1, g2))
		}
		return nil
	}(g1, g2); err != nil {
		return nil, err
	}
	// create & fill newChromes
	newChromes := make([]*Chromosome, g1.Length())
	if err = g1.ForEachMatchingChromosome(g2, func(
		i int, // iChrome
		// 1
		chrome1 *Chromosome,
		linkGenes1 map[Innovation]*LinkGene,
		nLinkGenes1 int,
		historical1 Innovation,
		// 2
		chrome2 *Chromosome,
		linkGenes2 map[Innovation]*LinkGene,
		nLinkGenes2 int,
		historical2 Innovation,
	) error {
		// type local; a full set of what (*Chromosome).Genetics() returns.
		type genetics struct {
			*Chromosome  // inheritance
			mapLinkGenes map[Innovation]*LinkGene
			nLinkGenes   int
			historical   Innovation
		}
		// play a role
		smaller, bigger, err := func() (shorter, longer genetics, err error) {
			// second closure
			genetics1 := genetics{chrome1, linkGenes1, nLinkGenes1, historical1}
			genetics2 := genetics{chrome2, linkGenes2, nLinkGenes2, historical2}
			if nLinkGenes1 > nLinkGenes2 {
				shorter = genetics2
				longer = genetics1
			} else if nLinkGenes1 < nLinkGenes2 {
				shorter = genetics1
				longer = genetics2
			} else if nLinkGenes1 == nLinkGenes2 {
				switch rand.Intn(2) { // just flip a two-sided coin
				case 0:
					shorter = genetics2
					longer = genetics1
				case 1:
					shorter = genetics1
					longer = genetics2
				default:
					panic("runtime violation")
				}
				// or decided depending on the innovation number
				// if historical1 >= historical2 {
				// 	shorter = genetics2
				// 	longer = genetics1
				// } else if historical1 < historical2 {
				// 	shorter = genetics1
				// 	longer = genetics2
				// } else {
				// 	panic("runtime violation")
				// }
			} else {
				panic("runtime violation")
			}
			return shorter, longer, nil
		}() // closure call
		if err != nil {
			return err
		}
		// fill me
		newChrome, err := univ.NewChromosomeBasic()
		if err != nil {
			return err
		}
		// cross
		iCrosspoint := rand.Intn(smaller.nLinkGenes) - 1 // where the last element of the left is in
		// -1 can be returned for sizeBuilding 0 chrome
		if iCrosspoint != -1 {
			{ // left
				smaller.Sort()
				for iLeft := 0; iLeft < iCrosspoint; iLeft++ { // excluding the crosspoint
					newChrome.AddLinkGenes(smaller.LinkGenes[iLeft].Copy())
				}
			}
			{ // crosspoint
				linkGeneLeft := smaller.LinkGenes[iCrosspoint]
				if linkGeneRight, isGeneMatched := bigger.mapLinkGenes[linkGeneLeft.Topo.Innov]; isGeneMatched {
					linkGeneMix := linkGeneLeft.Copy()
					if isAvgNotRnd {
						linkGeneMix.Weight = (linkGeneLeft.Weight + linkGeneRight.Weight) / 2
						newChrome.AddLinkGenes(linkGeneMix.Copy())
					} else { // toss a coin
						if rand.Float64() < 0.5 { // head
							newChrome.AddLinkGenes(linkGeneLeft.Copy())
						} else { // tail
							newChrome.AddLinkGenes(linkGeneRight.Copy())
						}
					}
				} else {
					newChrome.AddLinkGenes(linkGeneLeft.Copy())
				}
			}
			{ // right
				bigger.Sort()
				if iLower := sort.Search(bigger.nLinkGenes, func(i int) bool {
					return bigger.LinkGenes[i].Topo.Innov > smaller.historical
				}); iLower != -1 {
					for iRight := iLower; iRight < bigger.nLinkGenes; iRight++ {
						newChrome.AddLinkGenes(bigger.LinkGenes[iRight].Copy())
					}
				}
			}
		} else { // right
			for _, linkGeneRight := range bigger.mapLinkGenes {
				newChrome.AddLinkGenes(linkGeneRight.Copy())
			}
		}
		// dice gene by val
		*newChrome.DiceGene = *univ.NewChanceGeneMix(
			*smaller.DiceGene, *bigger.DiceGene, isAvgNotRnd,
		)
		// local return
		newChromes[i] = newChrome
		return nil
	}); err != nil { // This is after the closure above returned.
		return nil, err
	}
	return NewGenome(newChromes)
}

// ----------------------------------------------------------------------------
// Universe - Genetic Analysis

// ChanceDiff returns mutational differences between two die.
func (univ *Universe) ChanceDiff(dice1, dice2 *ChanceGene) (
	addNode, addLink, addBias, perturbWeight, nullifyWeight, turnOn, turnOff, bump float64,
) {
	if dice1 == nil || dice1.Enabled == false {
		dice1 = univ.NewChanceGeneBasic()
	}
	if dice2 == nil || dice2.Enabled == false {
		dice2 = univ.NewChanceGeneBasic()
	}
	// Unpack
	addNode1, addLink1, addBias1, perturbWeight1, nullifyWeight1, turnOn1, turnOff1, bump1 := dice1.Unpack()
	addNode2, addLink2, addBias2, perturbWeight2, nullifyWeight2, turnOn2, turnOff2, bump2 := dice2.Unpack()
	// Return
	addNode = math.Abs(addNode1 - addNode2)
	addLink = math.Abs(addLink1 - addLink2)
	addBias = math.Abs(addBias1 - addBias2)
	perturbWeight = math.Abs(perturbWeight1 - perturbWeight2)
	nullifyWeight = math.Abs(nullifyWeight1 - nullifyWeight2)
	turnOn = math.Abs(turnOn1 - turnOn2)
	turnOff = math.Abs(turnOff1 - turnOff2)
	bump = math.Abs(bump1 - bump2)
	return
}

// ChanceDiffInPercent returns mutational differences between two die, normalized to [0, 1].
func (univ *Universe) ChanceDiffInPercent(dice1, dice2 *ChanceGene) (
	addNode, addLink, addBias, perturbWeight, nullifyWeight, turnOn, turnOff, bump float64,
) {
	if dice1 == nil || dice1.Enabled == false {
		dice1 = univ.NewChanceGeneBasic()
	}
	if dice2 == nil || dice2.Enabled == false {
		dice2 = univ.NewChanceGeneBasic()
	}
	// Unpack
	addNode1, addLink1, addBias1, perturbWeight1, nullifyWeight1, turnOn1, turnOff1, bump1 := dice1.Unpack()
	addNode2, addLink2, addBias2, perturbWeight2, nullifyWeight2, turnOn2, turnOff2, bump2 := dice2.Unpack()
	// Return
	addNode = RelativeDifferenceNormalized(addNode1, addNode2)
	addLink = RelativeDifferenceNormalized(addLink1, addLink2)
	addBias = RelativeDifferenceNormalized(addBias1, addBias2)
	perturbWeight = RelativeDifferenceNormalized(perturbWeight1, perturbWeight2)
	nullifyWeight = RelativeDifferenceNormalized(nullifyWeight1, nullifyWeight2)
	turnOn = RelativeDifferenceNormalized(turnOn1, turnOn2)
	turnOff = RelativeDifferenceNormalized(turnOff1, turnOff2)
	bump = RelativeDifferenceNormalized(bump1, bump2)
	return
}

// Compatible checks if two genomes are compatible for crossover and in addition returns a measure of
// how distant two genomes are. Those genome parameters can be given in any order.
// This is where the compatibility distance gets in in order to speciate different organisms.
//
// The compatibility distance gives a measure of compatibility between two Genomes,
// by computing a linear combination of 3 characterizing variables of their compatibility.
// The 3 variables represent:
// PERCENT DISJOINT GENES, PERCENT EXCESS GENES, and MUTATIONAL DIFFERENCE WITHIN MATCHING GENES.
// So the formula for compatibility is: `th <= c1*pdg + c2*peg + c3*mdmg`.
// The 3 coefficients are global system parameters. (NEAT context of Universe here.)
// For the mutational difference, the weight of a gene can give a rough sense of
// how much mutation the gene has experienced since it originally appeared.
//
// This function returns an error if given parameters are not valid.
//
func (univ *Universe) Compatible(g1 *Genome, g2 *Genome) (isFertile bool, incompatibility float64, err error) {
	// vars
	nChromes := g1.Length()
	analysis1 := make([]bool, nChromes)    // isFertile for each chromosome
	analysis2 := make([]float64, nChromes) // incompatibility for each chromosome
	// iterate over
	if err = g1.ForEachMatchingChromosome(g2, func(
		i int, // iChrome
		// 1
		chrome1 *Chromosome,
		linkGenes1 map[Innovation]*LinkGene,
		nLinkGenes1 int,
		historical1 Innovation,
		// 2
		chrome2 *Chromosome,
		linkGenes2 map[Innovation]*LinkGene,
		nLinkGenes2 int,
		historical2 Innovation,
	) error {
		var ( // Divide two chosen chromosomes into Seme and Uke.
			mapExcessive, mapDeficient map[Innovation]*LinkGene
			minHistorical              Innovation
		) // Play your role...
		switch maxHistorical := math.Max(float64(historical1), float64(historical2)); maxHistorical {
		case float64(historical1):
			mapExcessive, mapDeficient = linkGenes1, linkGenes2
			minHistorical = historical2
		case float64(historical2):
			mapExcessive, mapDeficient = linkGenes2, linkGenes1
			minHistorical = historical1
		default:
			panic("runtime violation found in switch case")
		}
		// Two chosen chromosomes: Calculating the characterizing variables.
		var (
			nMatched, nDisjoint, nExcess               int
			sumPercentWeightDiff, sumPercentChanceDiff float64 // sum of those normalized
			sumWeightDiff, sumChanceDiff               float64 // sum of raw value differences
		)
		if univ.Config.CompatIsNormalizedForSize {
			sumPercentChanceDiff = Sum(univ.ChanceDiffInPercent(chrome1.DiceGene, chrome2.DiceGene))
			for innovation, geneFrom := range mapExcessive {
				if geneTo, available := mapDeficient[innovation]; available { // 1. matching gene
					nMatched++
					sumPercentWeightDiff += RelativeDifferenceNormalized(geneFrom.Weight, geneTo.Weight)
				} else if innovation <= minHistorical { // 2. disjoint gene
					nDisjoint++
				} else if innovation > minHistorical { // 3. excess gene
					nExcess++
				} else {
					panic("runtime violation found in an excessive chromosome")
				}
			}
		} else if !univ.Config.CompatIsNormalizedForSize {
			sumChanceDiff = Sum(univ.ChanceDiff(chrome1.DiceGene, chrome2.DiceGene))
			for innovation, geneFrom := range mapExcessive {
				if geneTo, available := mapDeficient[innovation]; available { // 1. matching gene
					nMatched++
					sumWeightDiff += math.Abs(geneFrom.Weight - geneTo.Weight)
				} else if innovation <= minHistorical { // 2. disjoint gene
					nDisjoint++
				} else if innovation > minHistorical { // 3. excess gene
					nExcess++
				} else {
					panic("runtime violation found in an excessive chromosome")
				}
			}
		} else {
			panic("runtime violation in hyper param")
		}
		for innovation /*, geneFrom*/ := range mapDeficient {
			if _, available := mapExcessive[innovation]; available { // 1. matching gene
				// Noop. Do nothing.
			} else if innovation <= minHistorical { // 2. disjoint gene
				nDisjoint++
			} else if innovation > minHistorical { // 3. excess gene
				panic("an excess gene was found in a deficient chromosome")
			} else {
				panic("runtime violation found in a deficient chromosome")
			}
		}
		// Local return of this chromosome.
		compatDistChromosome := func() (retCompatDistChromosome float64) {
			// coefficients
			c1 := univ.Config.CompatCoeffDisjoint
			c2 := univ.Config.CompatCoeffExcess
			c3 := univ.Config.CompatCoeffWeight
			c4 := univ.Config.CompatCoeffChance
			if univ.Config.CompatIsNormalizedForSize { // Normalize each of the characteristics to [0, 1].
				var (
					normalizeFactor, percentDisjoint, percentExcess float64
					avgWeightDiffInPercent, avgChanceDiffInPercent  float64
				)
				// pdg and peg
				normalizeFactor = Sum(float64(nLinkGenes1), float64(nLinkGenes2))
				// normalizeFactor = math.Max(float64(nLinkGenes1), float64(nLinkGenes2)) // normalize for size bigger of either
				if normalizeFactor != 0.0 { // handles divide by zero
					percentDisjoint = float64(nDisjoint) / normalizeFactor // in percatage
					percentExcess = float64(nExcess) / normalizeFactor     // in percatage
				}
				// mdmg
				if float64(nMatched) != 0.0 { // handles divide by zero
					avgWeightDiffInPercent = sumPercentWeightDiff / float64(nMatched) // in percatage
				}
				if float64(NumKindChance) != 0.0 { // handles divide by zero
					avgChanceDiffInPercent = sumPercentChanceDiff / float64(NumKindChance) // in percatage
				}
				// The compatibility formula. (for normalized characteristics)
				retCompatDistChromosome = (c1 * percentDisjoint) + (c2 * percentExcess) + (c3 * avgWeightDiffInPercent) + (c4 * avgChanceDiffInPercent)
				// if retCompatDistChromosome > c1+c2+c3+c4 || retCompatDistChromosome < 0 { // debug //
				// 	log.Println("len(mapExcessive)", len(mapExcessive))             // debug //
				// 	log.Println("len(mapDeficient)", len(mapDeficient))             // debug //
				// 	log.Println("minHistorical", minHistorical)                     // debug //
				// 	log.Println("/")                                                // debug //
				// 	log.Println("nDisjoint", nDisjoint)                             // debug //
				// 	log.Println("nExcess", nExcess)                                 // debug //
				// 	log.Println("nMatched", nMatched)                               // debug //
				// 	log.Println("NumKindChance", NumKindChance)                     // debug //
				// 	log.Println("sumWeightDiff", sumWeightDiff)                     // debug //
				// 	log.Println("sumChanceDiff", sumChanceDiff)                     // debug //
				// 	log.Println("normalizeFactor", normalizeFactor)                 // debug //
				// 	log.Println("/")                                                // debug //
				// 	log.Println("percentDisjoint", percentDisjoint)                 // debug //
				// 	log.Println("percentExcess", percentExcess)                     // debug //
				// 	log.Println("avgWeightDiffInPercent", avgWeightDiffInPercent)   // debug //
				// 	log.Println("avgChanceDiffInPercent", avgChanceDiffInPercent)   // debug //
				// 	log.Println("/")                                                // debug //
				// 	log.Println("retCompatDistChromosome", retCompatDistChromosome) // debug //
				// 	log.Println()                                                   // debug //
				// } // debug //
				return retCompatDistChromosome
			} else if !univ.Config.CompatIsNormalizedForSize {
				// mdmg
				var avgWeightDiff, avgChanceDiff float64
				if float64(nMatched) != 0.0 { // handles divide by zero
					avgWeightDiff = sumWeightDiff / float64(nMatched) // in percatage
				}
				if float64(NumKindChance) != 0.0 { // handles divide by zero
					avgChanceDiff = sumChanceDiff / float64(NumKindChance) // in percatage
				}
				// The compatibility formula. (for absolute characteristics)
				retCompatDistChromosome = (c1 * float64(nDisjoint)) + (c2 * float64(nExcess)) + (c3 * avgWeightDiff) + (c4 * avgChanceDiff)
				// log.Println("len(mapExcessive)", len(mapExcessive))             // debug //
				// log.Println("len(mapDeficient)", len(mapDeficient))             // debug //
				// log.Println("minHistorical", minHistorical)                     // debug //
				// log.Println("/")                                                // debug //
				// log.Println("nMatched", nMatched)                               // debug //
				// log.Println("NumKindChance", NumKindChance)                     // debug //
				// log.Println("sumWeightDiff", sumWeightDiff)                     // debug //
				// log.Println("sumChanceDiff", sumChanceDiff)                     // debug //
				// log.Println("/")                                                // debug //
				// log.Println("nDisjoint", nDisjoint)                             // debug //
				// log.Println("nExcess", nExcess)                                 // debug //
				// log.Println("avgWeightDiff", avgWeightDiff)                     // debug //
				// log.Println("avgChanceDiff", avgChanceDiff)                     // debug //
				// log.Println("/")                                                // debug //
				// log.Println("retCompatDistChromosome", retCompatDistChromosome) // debug //
				// log.Println()                                                   // debug //
				return retCompatDistChromosome
			}
			panic("runtime violation in hyper param")
		}()
		isChromosomeFertile := (compatDistChromosome <= univ.Config.CompatThreshold)
		// Local return.
		analysis1[i] = isChromosomeFertile
		analysis2[i] = compatDistChromosome
		// if i == nChromes-1 { // debug //
		// 	log.Println("univ.Compatible(g1, g2):", i+1, "matching chromosomes were found in this operation.") // debug //
		// 	log.Println("isFertile", analysis1)                                                                  // debug //
		// 	log.Println("incompatibility", analysis2)                                                            // debug //
		// 	log.Println()                                                                                        // debug //
		// } // debug //
		return nil
	}); err != nil {
		return false, math.NaN(), err
	} // for
	// Return
	isFertile = And(analysis1...)                           // important
	incompatibility = Sum(analysis2...) / float64(nChromes) // important
	err = nil                                               // nah
	return isFertile, incompatibility, err
}

// ----------------------------------------------------------------------------
// Universe - Speciation
//
// These methods does not care the thread-safety. You might want to use `univ.MutexClasses` with these.
//

// GetSpeciesOrderedByProminence returns a copy slice of all available classes in this universe.
// This method is used to tell the champion species of this universe.
// The returned slice here is in...
//  - 1) descending order of the top fitness of each species,
//  - 2) ascending order of the stagnancy of each species,
//  - 3) descending order of the average fitness of each species,
//  - 4) and ascending order of the number that denotes how early the niche of each species has appeared in this universe.
// `univ.MutexClasses` is required in order to get this operation guarantee the thread-safety as it accesses `univ.Classes` directly.
// This function panics if there are one or more organisms have not been measured.
func (univ *Universe) GetSpeciesOrderedByProminence() (sortByProminence []*Species, err error) {
	sortByProminence = make([]*Species, len(univ.Classes))
	if nCopied := copy(sortByProminence, univ.Classes); nCopied != len(univ.Classes) {
		return nil, errors.New("failed to copy given slice")
	}
	sort.Slice(sortByProminence, func(i, j int) bool {
		if sortByProminence[i].TopFitness != sortByProminence[j].TopFitness {
			return sortByProminence[i].TopFitness > sortByProminence[j].TopFitness // descending
		}
		if sortByProminence[i].Stagnancy != sortByProminence[j].Stagnancy {
			return sortByProminence[i].Stagnancy < sortByProminence[j].Stagnancy // ascending
		}
		{ // elif
			avgFitness1, err1 := sortByProminence[i].AverageFitness()
			if err1 != nil {
				panic(err1)
			}
			avgFitness2, err2 := sortByProminence[j].AverageFitness()
			if err2 != nil {
				panic(err2)
			}
			if avgFitness1 != avgFitness2 {
				return avgFitness1 > avgFitness2 // descending
			}
		}
		return sortByProminence[i].Niche() < sortByProminence[j].Niche() // ascending
	})
	return sortByProminence, nil
}

// AdjAvgFitnessesOfSpecies evaluates adjusted average-fitnesses of all competing species.
// Adjusted here only means that those values are made non-negative. The order for iSpecies is preserved.
func (univ *Universe) AdjAvgFitnessesOfSpecies() (sumFitAvgAdj float64, fitAvgsAdj []float64) {
	return func(sumFitAvg float64, fitAvgs []float64) (
		sumFitAvgAdj float64, fitAvgsAdj []float64,
	) {
		fitAvgsAdj = make([]float64, len(fitAvgs))
		minFitAvg := 0.0 // won't be bigger than 0
		for i, fitAvg := range fitAvgs {
			fitAvgsAdj[i] = fitAvg                  // copy
			minFitAvg = math.Min(minFitAvg, fitAvg) // find min
		}
		offset := math.Abs(minFitAvg)
		for i, fitAvg := range fitAvgsAdj {
			fitAvgsAdj[i] = fitAvg + offset // adjust
			if fitAvgsAdj[i] < 0 {
				panic(fmt.Sprint(
					fmt.Sprintln("terrible miscalculation:"),
					fmt.Sprintln(
						"sumFitAvg ->", sumFitAvg,
						"fitAvgs ->", fitAvgs,
					),
					fmt.Sprintln(
						"sumFitAvgAdj ->", sumFitAvgAdj,
						"fitAvgsAdj ->", fitAvgsAdj,
					),
				))
			}
		}
		sumFitAvgAdj = sumFitAvg + (offset * float64(len(fitAvgs)))
		return sumFitAvgAdj, fitAvgsAdj
	}(func(company []*Species) (sum float64, fits []float64) {
		fits = make([]float64, len(company))
		for i, team := range company {
			fitAvg, err := team.AverageFitness()
			if err != nil {
				// log.Println("fatal:", err) // debug //
				panic(err)
			}
			sum += fitAvg
			fits[i] = fitAvg
		}
		return sum, fits
	}(univ.Classes))
}

// AdjTopFitnessesOfSpecies evaluates adjusted top-fitnesses of all competing species.
// Adjusted here only means that those values are made non-negative. The order for iSpecies is preserved.
func (univ *Universe) AdjTopFitnessesOfSpecies() (sumFitTopAdj float64, fitTopsAdj []float64) {
	return func(sumFitTop float64, fitTops []float64) (
		sumFitTopAdj float64, fitTopsAdj []float64,
	) {
		fitTopsAdj = make([]float64, len(fitTops))
		minFitTop := 0.0 // won't be bigger than 0
		for i, fitTop := range fitTops {
			fitTopsAdj[i] = fitTop                  // copy
			minFitTop = math.Min(minFitTop, fitTop) // find min
		}
		offset := math.Abs(minFitTop)
		for i, fitTop := range fitTopsAdj {
			fitTopsAdj[i] = fitTop + offset // adjust
			if fitTopsAdj[i] < 0 {
				panic(fmt.Sprint(
					fmt.Sprintln("terrible miscalculation:"),
					fmt.Sprintln(
						"sumFitTop ->", sumFitTop,
						"fitTops ->", fitTops,
					),
					fmt.Sprintln(
						"sumFitTopAdj ->", sumFitTopAdj,
						"fitTopsAdj ->", fitTopsAdj,
					),
				))
			}
		}
		sumFitTopAdj = sumFitTop + (offset * float64(len(fitTops)))
		return sumFitTopAdj, fitTopsAdj
	}(func(company []*Species) (sum float64, fits []float64) {
		fits = make([]float64, len(company))
		for i, team := range company {
			fitTop := team.TopFitness
			sum += fitTop
			fits[i] = fitTop
		}
		return sum, fits
	}(univ.Classes))
}

// SortSpecies of this universe.
func (univ *Universe) SortSpecies() bool {
	sort.Slice(univ.Classes, func(i, j int) bool {
		return univ.Classes[i].Niche() < univ.Classes[j].Niche()
	})
	return false
}

// HasSpecies in this universe.
// This takes linear time O(N) searching for a species in a list.
func (univ *Universe) HasSpecies(s *Species) bool {
	if s == nil {
		return false
	}
	for _, species := range univ.Classes {
		if species == s {
			return true
		}
	}
	return false
}

// AddSpecies to this universe.
// The parameter should not be nil.
func (univ *Universe) AddSpecies(s *Species) error {
	if s == nil {
		return errors.New("nil species")
		// s = NewSpecies(nil) // empty species // This creates a default object (empty species) when the parameter is nil.
	}
	univ.Classes = append(univ.Classes, s)
	return nil
}

// RemoveClasses of this universe. It means extinction of species.
// This unspeciates all creatures of those species taking only constant time for each,
// and returns them (unspeciated) so they can be killed or respeciated or recycled however etc.
func (univ *Universe) RemoveClasses(indicesSpecies ...int) (
	removed []*Species, segregated [][]*Organism, err error,
) {
	// Just in case...
	unique.Slice(&indicesSpecies, func(i, j int) bool { return indicesSpecies[i] < indicesSpecies[j] })
	for _, argIndex := range indicesSpecies {
		if argIndex < 0 || argIndex >= len(univ.Classes) {
			return nil, nil, errors.New("index out of bound: " + strconv.Itoa(argIndex))
		}
	}
	// After those arguments are set...
	nRemoved := len(indicesSpecies)
	copyClasses := make([]*Species, len(univ.Classes)) // tmp copy
	if nCopied := copy(copyClasses, univ.Classes); nCopied != len(univ.Classes) {
		return nil, nil, errors.New("failed to copy univ species")
	}
	for i := 0; i < nRemoved; i++ { // move them to front
		j := indicesSpecies[i]
		copyClasses[i], copyClasses[j] = copyClasses[j], copyClasses[i]
	}
	removed = make([]*Species, nRemoved)
	segregated = make([][]*Organism, nRemoved)
	for iSpecies, s := range copyClasses[:nRemoved] {
		removed[iSpecies] = s
		segregated[iSpecies] = s.Cull(1.00)
	}
	// Notice that univ.Classes is updated in this function without mutex or any sync tool.
	univ.Classes = copyClasses[nRemoved:]
	univ.SortSpecies()
	return removed, segregated, nil
}

// SpeciesRandom returns a random species in this universe.
func (univ *Universe) SpeciesRandom() *Species {
	if len(univ.Classes) < 1 {
		return nil
	}
	return univ.Classes[rand.Intn(len(univ.Classes))]
}

// TellSpecies of an organism without touching it. This returns the closest species of that organism.
// If it turns out that there are two or more species of the same compatibility distance available
// in this universe, the return is the more earlier in the list in which species are sorted.
// This function returns error if the organism's genome is told corrupted.
func (univ *Universe) TellSpecies(o /*read-only*/ *Organism) (ret *Species, err error) {
	min := math.MaxFloat64
	for _, species := range univ.Classes {
		if president := species.Champion(); president != nil {
			if isFertile, incompatibility, err := univ.Compatible(o.GenoType(), president.GenoType()); err != nil {
				return nil, err
			} else if isFertile && incompatibility < min {
				ret = species
				min = incompatibility
			} else {
				// Noop. Do nothing.
			}
			continue
		}
	}
	return ret, nil
}

// Speciate an unspeciated organism.
// This relates both organism and species.
func (univ *Universe) Speciate(o *Organism) (err error) {
	if o == nil {
		return errors.New("nil organism")
	}
	if o.IsSpeciated() {
		return errors.New("you have to unspeciate this organism")
	}
	if o.Breed, err = univ.TellSpecies(o); err != nil {
		return err
	}
	if !univ.HasSpecies(o.Species()) {
		newClass := univ.NewSpeciesBasic()
		if err = univ.AddSpecies(newClass); err != nil {
			return err
		}
		newClass.AddOrganisms(o)
	} else {
		o.Species().AddOrganisms(o)
	}
	return nil
}

// AddOrganism to this universe. The organism does not get speciated here.
// The parameter should not be nil and it should be new in this universe.
func (univ *Universe) AddOrganism(o *Organism) (err error) {
	if o == nil {
		// o = univ.NewOrganismBasic() // This; AddOrganism, creates a default organism when the parameter is nil.
		return errors.New("nil organism")
	}
	if _, isAlreadyThere := univ.Livings[o]; isAlreadyThere {
		return errors.New("the organism already exists")
	}
	univ.Livings[o] = struct{}{}
	return nil
}

// RemoveOrganism of this universe. This unspeciates the organism.
// The time complexity is O(1), except for the case the organism is speciated,
// for which this operation takes linear time O(N) searching it as an element in a species in order to unspeciate it.
func (univ *Universe) RemoveOrganism(o *Organism) error {
	if o == nil {
		return errors.New("cannot remove nil organism")
	}
	if o.IsSpeciated() {
		if err := o.Unspeciate(); err != nil {
			return err
		}
	}
	delete(univ.Livings, o)
	return nil
}

// ----------------------------------------------------------------------------
// Species

// Niche is an identifier and a historical marker for species.
// This is unique for each of species under a NEAT context.
// And the bigger the number is, the later the Niche is.
// It basically is considered an integer starting from zero
// but could be used in other way depending on the context.
type Niche int64

// String callback of Niche.
func (niche /* not ptr */ Niche) String() string {
	return "<" + strconv.Itoa(int(niche)) + ">"
}

// Species is a classification of living organisms in our simulated world.
type Species struct {
	ID      Niche       // Identifier of the habitat of this species.
	Livings []*Organism // Creatures of this species.
	// What tracks for the N how long this species has been stagnant.
	// This species has been stagnant over N generations in a row.
	// It's named stagnancy instead of stagnation for it sounding not like an economics vocabulary.
	Stagnancy int
	// We keep track of this so we can evaluate Stagnancy of this Species in each epoch.
	TopFitness float64
}

// NewSpecies is a constructor.
// Arg livings can be nil to set by default.
func NewSpecies(id Niche, livings []*Organism) *Species {
	if livings == nil {
		livings = []*Organism{}
	}
	return &Species{
		ID:         id,
		Livings:    livings,
		Stagnancy:  0,
		TopFitness: 0,
	}
}

// String callback of Species.
func (s *Species) String() string {
	return fmt.Sprint(
		"{",
		"Species ", s.ID, " of ", len(s.Livings), " Organisms", "/",
		"Stagnancy:", s.Stagnancy, "/",
		"TopFitness:", s.TopFitness,
		"}",
	)
}

// Niche is a getter returning this species' identifier
// that could also be considered the habitat of creatures of this species.
func (s *Species) Niche() Niche {
	return s.ID
}

// Size returns the number of organisms in this species.
// Population of this species.
func (s *Species) Size() int {
	return len(s.Livings)
}

// Sort sorts organisms of this species by their fitness, in descending order.
func (s *Species) Sort() {
	sort.Slice(s.Livings, func(i, j int) bool {
		return s.Livings[i].Fitness > s.Livings[j].Fitness
	})
}

// AddOrganisms (organism) in this Species.
// This updates the organism(s) added as well.
func (s *Species) AddOrganisms(orgs ...*Organism) {
	s.Livings = append(s.Livings, orgs...)
	for _, o := range orgs {
		if o.Species() != s {
			o.Breed = s
		}
	}
	s.Sort()
}

// Cull off bottom `percentage` percent of organisms.
// The range of `percentage` is [0, 1].
// This function returns all organisms culled off of this zoo.
func (s *Species) Cull(percentage float64) (segregated []*Organism) {
	s.Sort()
	percentageFromTop := 1 - percentage
	percentileFrom := int(math.Ceil(float64(s.Size()) * percentageFromTop))
	segregated = s.Livings[percentileFrom:] // Panics if s[len(s)+1:] but s[len(s):] will give an empty slice without any error.
	s.Livings = s.Livings[:percentileFrom]
	for _, o := range segregated {
		o.Unbreed()
	}
	return segregated
}

// CullToSinglePopulation culls off all creatures but the champion of this species.
// The species must consist of one population at least. Otherwise this function panics.
func (s *Species) CullToSinglePopulation() (segregated []*Organism) {
	s.Sort()
	segregated = s.Livings[1:]
	s.Livings = s.Livings[:1]
	for _, o := range segregated {
		o.Unbreed()
	}
	return segregated
}

// Champion of this species.
// Returns nil if there is no organism in this species.
func (s *Species) Champion() *Organism {
	s.Sort()
	if len(s.Livings) > 0 {
		return s.Livings[0]
	}
	return nil
}

// EvaluateGeneration checks and updates the stagnancy of this species.
// This returns the updated stagnancy, and the top fitness of this species evaluated.
// An error is returned when the fitness of the champion is yet evaluated.
func (s *Species) EvaluateGeneration() (stagnancy int, topFitness float64, err error) {
	fitness, err := s.Champion().GetFitness()
	if err != nil {
		return s.Stagnancy, topFitness, err
	}
	if fitness > s.TopFitness {
		s.TopFitness = fitness
		s.Stagnancy = 0
		return s.Stagnancy, s.TopFitness, nil
	}
	s.Stagnancy++
	return s.Stagnancy, s.TopFitness, nil
}

// Unspeciate kicks out only a single organism of this species.
// Consider using (*Species).Cull() instead if you want a massacre.
// This operation takes O(N) time searching a given organism.
// The return is about whether it was successful or not.
func (s *Species) Unspeciate(o *Organism) (expelled bool) {
	for i, found := range s.Livings {
		if found == o {
			found.Unbreed()
			s.Livings = append(s.Livings[:i], s.Livings[i+1:]...)
			return true
		}
	}
	return false
}

// AverageFitness of this species.
// This returns an error when it turns out that there is at least one
// organism with its fitness unevaluated in this species.
func (s *Species) AverageFitness() (averageFitness float64, err error) {
	sum := 0.0
	for _, organism := range s.Livings {
		fitness, err := organism.GetFitness()
		if err != nil {
			return averageFitness,
				errors.New("unable to calculate the average fitness of this species: " + err.Error())
		}
		sum += fitness
	}
	return sum / float64(len(s.Livings)), nil
}

// OrganismRandom returns an organism chosen randomly of this species.
// The return won't and shouldn't be nil.
func (s *Species) OrganismRandom() *Organism {
	return s.Livings[rand.Intn(len(s.Livings))]
}

// RandomOrganisms returns N organisms randomly chosen from this species.
// The given parameter N must be less than or equal to the number of
// organisms of this species. Otherwise this function returns nil.
func (s *Species) RandomOrganisms(n int) (creatures []*Organism) {
	if 0 > n || n > len(s.Livings) {
		return nil
	}
	creatures = make([]*Organism, n)
	for i, v := range rand.Perm(len(s.Livings))[:n] {
		creatures[i] = s.Livings[v]
	}
	return creatures
}

// ----------------------------------------------------------------------------
// Organism

// Organism is an individual living thing born to compete with others.
type Organism struct {
	Genotype   *Genome
	Phenotype  []*NeuralNetwork
	Breed      *Species
	Fitness    float64 // private. Do not access this directly without a public method.
	IsMeasured bool    // private. Do not access this directly without a public method.
}

// String callback of Organism.
func (o *Organism) String() string {
	return fmt.Sprint(
		"{",
		"Organism", "/",
		// "Genotype:", o.Genotype, "/",
		// "Phenotype:", o.Phenotype, "/",
		"Breed:", o.Breed, "/",
		"Fitness:", o.Fitness, "/",
		"IsMeasured:", o.IsMeasured,
		"}",
	)
}

// IsFitnessMeasured returns true iff an organism is already measured, no not for being measured.
// Returns false when this organism needs its fitness evaluated.
func (o *Organism) IsFitnessMeasured() bool {
	return o.IsMeasured
}

// UpdateFitness is a setter with a positive side effect.
func (o *Organism) UpdateFitness(fitness float64) {
	o.Fitness = fitness
	o.IsMeasured = true
}

// GetFitness is a getter.
// Returns error when the fitness of this organism is yet measured.
func (o *Organism) GetFitness() (fitness float64, err error) {
	if o.IsFitnessMeasured() {
		return o.Fitness, nil
	}
	return o.Fitness, errors.New("fitness of this organism needs to be measured")
}

// GenoType returns the genotype of this organism.
// Uppercase T of this function name is to avoid naming conflicts. :P
func (o *Organism) GenoType() *Genome {
	return o.Genotype
}

// PhenoType returns the phenotype of this organism.
// Uppercase T of this function name is to avoid naming conflicts. :P
func (o *Organism) PhenoType() []*NeuralNetwork {
	return o.Phenotype
}

// Species returns the breed of this organism.
// Returns nil when this creature is yet classified.
func (o *Organism) Species() *Species {
	return o.Breed
}

// IsSpeciated of this organism.
func (o *Organism) IsSpeciated() bool {
	return o.Breed != nil
}

// Unbreed this organism. This method is different from Unspeciate().
//  - Unspeciate() **calls** the species the organism is in to dislink(unassociate;unrelate) the two. `o.Species().Unspeciate(o)`
//  - Unbreed() is **called** by the species the organism is in to safely access the struct's field. `o.Breed = nil`
// This function is a callback. Generally you don't want to use this.
func (o *Organism) Unbreed() {
	o.Breed = nil
}

// Unspeciate this single organism. This method is different from Unbreed().
//  - Unspeciate() **calls** the species the organism is in to dislink(unassociate;unrelate) the two. `o.Species().Unspeciate(o)`
//  - Unbreed() is **called** by the species the organism is in to safely access the struct's field. `o.Breed = nil`
// This function results in calling Unbreed() and so is preferred over the other most of the times.
// Also this operation may take up to O(N) time searching a given organism.
// Consider using (*Species).Cull() or (*Universe).RemoveSpecies() instead if you just want a massacre.
func (o *Organism) Unspeciate() error {
	if o.Species() == nil {
		return errors.New("cannot unspeciate an unspeciated organism")
	}
	if !o.Species().Unspeciate(o) {
		return errors.New("the species couldn't kick this guy out")
	}
	return nil
}

// Copy is a constructor. Notice: this method is no use in GA.
// The returned daughter is a genetic copy of this organism unspeciated, unevaluated, and unmeasured.
// This does not mutate the organism so you might want to use (*Genome).Copy() instead.
func (o *Organism) Copy() (daughter *Organism, err error) {
	newGenome, err := o.GenoType().Copy()
	if err != nil {
		return nil, err
	}
	return newGenome.NewOrganismSimple()
}

// OrganismMarshal is JSON export of Organism.
type OrganismMarshal struct {
	Genotype   *Genome
	Fitness    float64
	IsMeasured bool
}

// MarshalJSON of json.Marshaler interface.
func (o *Organism) MarshalJSON() ([]byte, error) {
	return json.Marshal(&OrganismMarshal{
		Genotype:   o.Genotype,
		Fitness:    o.Fitness,
		IsMeasured: o.IsMeasured,
	})
}

// UnmarshalJSON of json.Unmarshaler interface.
// The organism is unspeciated.
func (o *Organism) UnmarshalJSON(jsonOrganismMarshal []byte) (err error) {
	om := &OrganismMarshal{}
	if err = json.Unmarshal(jsonOrganismMarshal, om); err != nil {
		return err
	}
	//
	phenome, err := om.Genotype.NewPhenotype()
	if err != nil {
		return err
	}
	*o = Organism{
		Genotype:   om.Genotype,
		Phenotype:  phenome,
		Breed:      nil,
		Fitness:    om.Fitness,
		IsMeasured: om.IsMeasured,
	}
	return nil
}

// ----------------------------------------------------------------------------
// Genome

// Genome represents a genotype of a single living organism, from which, in general meaning,
// we can derive a neural network as a phenotype.
// This Genome designed here in specific can encode of multiple
// neural networks taking a separate set of inputs for each.
type Genome struct { // This would be a factory for an organism.
	Chromes []*Chromosome // Each of these chromosomes is for a network.
}

// NewGenome is a constructor.
// Arg chromes can be nil to set by default.
func NewGenome(chromes []*Chromosome) (g *Genome, err error) {
	g = &Genome{}
	//
	if chromes == nil {
		chrome, err := NewChromosome(nil, nil, nil)
		if err != nil {
			return nil, err
		}
		g.Chromes = []*Chromosome{chrome}
		return g, nil
	}
	//
	if len(chromes) < 1 {
		return nil, errors.New("the number of chromosome(s) should be at least one or more")
	}
	//
	g.Chromes = make([]*Chromosome, len(chromes))
	for iChrome, chromosome := range chromes {
		if chromosome == nil {
			g.Chromes[iChrome], err = NewChromosome(nil, nil, nil)
			if err != nil {
				return nil, err
			}
			continue
		}
		g.Chromes[iChrome] = chromosome
	}
	//
	return g, nil
}

// String of Genome.
func (g *Genome) String() string {
	const left, right = 0, 1
	switch len(g.Chromes) {
	case 0:
		return "{Genome empty}"
	case 1:
		return "{Genome of " + g.Chromes[left].String() + "}"
	case 2:
		return "{Genome of Left: " + g.Chromes[left].String() + " and Right: " + g.Chromes[right].String() + "}"
	default:
		return "{Genome of " + strconv.Itoa(len(g.Chromes)) + " Chromosomes with Left: " + g.Chromes[left].String() + " and Right: " + g.Chromes[right].String() + "}"
	}
	// return "{Genome unknown}" // unreachable
}

// Length returns the number of chromosomes this genome consist of.
func (g *Genome) Length() int {
	return len(g.Chromes)
}

// Chromosomes returns a list of all chromosomes of this genome.
// The slice returned is an original reference not a copy.
func (g *Genome) Chromosomes() []*Chromosome {
	return g.Chromes
}

// IsValid tests this genome.
func (g *Genome) IsValid() bool {
	if g.Chromosomes() == nil {
		return false
	}
	for _, chromosome := range g.Chromosomes() {
		if chromosome == nil {
			return false
		}
	}
	return true
}

// NodeGenesByUUID returns a map of all node genes available in this Genome.
// The parameter can be nil.
func (g *Genome) NodeGenesByUUID(fillMe map[uuid.UUID]*NodeGene) (filledOut map[uuid.UUID]*NodeGene) {
	if fillMe == nil {
		filledOut = map[uuid.UUID]*NodeGene{}
	} else {
		filledOut = fillMe
	}
	for _, chromosome := range g.Chromosomes() {
		if chromosome != nil {
			filledOut = chromosome.NodeGenesByUUID(filledOut)
		}
	}
	return filledOut
}

// ForEachMatchingChromosome of two genomes.
// Use this method to line up chromosomes of a genome beside one another.
// Genetic recombination may happen upon each genetic analysis of homologous chromosomes.
// The callback provides parental chromosomes that pair up with each other.
// This function returns error if each genome has a distinct number of chromosomes.
// Mandatory: All arguments are required and cannot be nil parameter.
func (g *Genome) ForEachMatchingChromosome(
	matchedPartner *Genome,
	forEachGeneticAnalysis func(
		i int, // iChrome
		chrome1 *Chromosome,
		linkGenes1 map[Innovation]*LinkGene,
		nLinkGenes1 int, historical1 Innovation,
		chrome2 *Chromosome,
		linkGenes2 map[Innovation]*LinkGene,
		nLinkGenes2 int, historical2 Innovation,
	) error,
) error {
	// mate g1 and g2
	if g1, g2 := g, matchedPartner; g1 != nil && g2 != nil && forEachGeneticAnalysis != nil {
		if chromes1, chromes2 := g1.Chromosomes(), g2.Chromosomes(); g1.Length() == g2.Length() {
			for i, chrome1 := range chromes1 { // for each matching chromosomes
				chrome2 := chromes2[i] // depicted chrome1 and chrome2
				links1, size1, innov1 := chrome1.Genetics()
				links2, size2, innov2 := chrome2.Genetics()
				if err := forEachGeneticAnalysis(
					i,
					chrome1, links1, size1, innov1,
					chrome2, links2, size2, innov2,
				); err != nil {
					return err
				}
			}
		} else {
			return errors.New("bad genome: the number of chromosome(s) does not match")
		}
	} else {
		return errors.New("nil parameter")
	}
	return nil
}

// NewPhenotype from this genotype.
func (g *Genome) NewPhenotype() (phenome []*NeuralNetwork, err error) {
	phenome = make([]*NeuralNetwork, g.Length())
	for i, chrome := range g.Chromosomes() {
		phenome[i], err = chrome.NewNeuralNetwork()
		if err != nil {
			return nil, err
		}
	}
	return phenome, nil
}

// NewOrganism is born. A constructor.
// The returned organism inherits the reference of this genome.
func (g *Genome) NewOrganism(breed *Species, fitness float64, isMeasured bool) (o *Organism, err error) {
	phenome, err := g.NewPhenotype()
	if err != nil {
		return nil, err
	}
	return &Organism{
		Genotype:   g,
		Phenotype:  phenome,
		Breed:      breed,
		Fitness:    fitness,
		IsMeasured: isMeasured,
	}, nil
}

// NewOrganismSimple is a constructor.
// The return is a default organism unmeasured and nil breed.
func (g *Genome) NewOrganismSimple() (o *Organism, err error) {
	return g.NewOrganism(nil, 0, false)
}

// Copy is a constructor.
func (g *Genome) Copy() (ret *Genome, err error) {
	newChromes := make([]*Chromosome, len(g.Chromosomes()))
	for i, chrome := range g.Chromosomes() {
		newChromes[i], err = chrome.Copy()
		if err != nil {
			return nil, err
		}
	}
	return NewGenome(newChromes)
}

// ----------------------------------------------------------------------------
// Chromosome

// Chromosome is a part of Genome the struct type.
// This encodes an NN and thus is a factory of a neural network.
// Chromosome may implement the genetic scheme of NEAT; a linear representation of
// network connectivity where its sequence alignment is significant.
type Chromosome struct {
	LinkGenes   []*LinkGene // Hidden nodes are inferred out of this. Sort by innov.
	IONodeGenes []*NodeGene `json:"-"` // This does not contain hidden nodes. Inputs and Outputs only.
	DiceGene    *ChanceGene
}

// NewChromosome is a constructor.
// Arg links, nodes, and dice can be nil to set them empty by default.
//
// Parameters
//  - `links`: Link-genes. The innovation number must not overlap with any. A link may be connected to the Input, Hidden, or Output nodes.
//  - `nodes`: Node-genes. Only Input/Output nodes are allowed. Hidden nodes etc. are forbidden.
//  - `dice`: A set of mutation probabilities.
//
func NewChromosome(links []*LinkGene, ioNodes []*NodeGene, dice *ChanceGene) (chrome *Chromosome, err error) {
	chrome = &Chromosome{}
	// 1. LinkGenes
	if links == nil { // new empty
		chrome.LinkGenes = []*LinkGene{}
	} else { // or copy
		chrome.LinkGenes = make([]*LinkGene, len(links))
		if len(chrome.LinkGenes) != copy(chrome.LinkGenes, links) {
			return nil, errors.New("failed to copy link-genes while constructing chromosome")
		}
	}
	// 2. NodeGenes
	if ioNodes == nil { // new empty
		chrome.IONodeGenes = []*NodeGene{}
	} else { // or copy
		chrome.IONodeGenes = func(nodes []*NodeGene) (filtered []*NodeGene) {
			filtered = []*NodeGene{} // nodeGenesWithoutHidden
			for _, nodeGene := range nodes {
				switch nodeGene.TypeInInt() {
				case InputNodeBias:
					fallthrough
				case InputNodeNotBias:
					fallthrough
				case OutputNode:
					filtered = append(filtered, nodeGene)
				}
			}
			return filtered
		}(ioNodes)
	} // NodeGenes are copied only for safety.
	// 3. DiceGene
	if dice == nil { // new empty
		chrome.DiceGene = NewChanceGeneDisabled()
	} else { // or copy
		chrome.DiceGene = func(dice *ChanceGene) (copy *ChanceGene) {
			v := *dice
			return &v
		}(dice)
	}
	// Finalize
	if !chrome.IsValid() {
		return nil, errors.New("GIGO: bad chromosome is being constructed")
	}
	chrome.Sort()
	return chrome, nil
}

// IsValid tests this chromosome.
func (chrome *Chromosome) IsValid() bool {
	if !chrome.IsValidForLinks() || !chrome.IsValidForNodes() || !chrome.IsValidForDice() {
		return false
	}
	return true
}

// IsValidForLinks tests links of this chromosome.
func (chrome *Chromosome) IsValidForLinks() bool {
	if chrome.LinkGenes == nil {
		return false
	}
	{ // check dup innovations in LinkGenes
		innovations := map[int]struct{}{}
		for _, linkGene := range chrome.LinkGenes {
			if _, isDup := innovations[int(linkGene.Topo.Innov)]; isDup {
				return false
			}
			innovations[int(linkGene.Topo.Innov)] = struct{}{}
		}
	}
	return true
}

// IsValidForNodes tests nodes of this chromosome.
func (chrome *Chromosome) IsValidForNodes() bool {
	if chrome.IONodeGenes == nil {
		return false
	}
	{ // check NodeGenes
		if !(len(chrome.IONodeGenes) > 0) {
			return false
		}
		hasInputNode, hasOutputNode := false, false
		for _, nodeGene := range chrome.IONodeGenes {
			if (nodeGene.TypeInInt() != InputNodeBias &&
				nodeGene.TypeInInt() != InputNodeNotBias &&
				nodeGene.TypeInInt() != OutputNode) ||
				!nodeGene.IsValid() {
				return false
			}
			{ // check has
				if !hasInputNode && nodeGene.TypeInInt() == InputNodeNotBias {
					hasInputNode = true
				}
				if !hasOutputNode && nodeGene.TypeInInt() == OutputNode {
					hasOutputNode = true
				}
			}
		}
		if !hasInputNode && !hasOutputNode {
			return false
		}
	}
	return true
}

// IsValidForDice tests the dice of this chromosome.
func (chrome *Chromosome) IsValidForDice() bool {
	if chrome.DiceGene == nil {
		return false
	}
	return true
}

// Sort of Chromosome sorts slice(s) of a chromosome itself.
// For nodes, the order is important for inputs & outputs.
// The link genes are sorted in ascending order of the innovation number.
func (chrome *Chromosome) Sort() {
	// Sort LinkGenes.
	sort.Slice(chrome.LinkGenes, func(i, j int) bool {
		return chrome.LinkGenes[i].Topo.Innov < chrome.LinkGenes[j].Topo.Innov
	})
	// Sort NodeGenes.
	sort.Slice(chrome.IONodeGenes, func(i, j int) bool {
		if chrome.IONodeGenes[i].TypeInInt() != chrome.IONodeGenes[j].TypeInInt() {
			return chrome.IONodeGenes[i].TypeInInt() < chrome.IONodeGenes[j].Type
		}
		return chrome.IONodeGenes[i].Idx < chrome.IONodeGenes[j].Idx
	})
}

// AddLinkGenes (gene) to this Chromosome.
// This does not verify the added gene(s) for its efficiency.
// You could test it with (*Chromosome).IsValidLinks().
func (chrome *Chromosome) AddLinkGenes(pleasePassInDeepCopiesHere ...*LinkGene) {
	chrome.LinkGenes = append(chrome.LinkGenes, pleasePassInDeepCopiesHere...)
	// chrome.Sort() // Lazy. Not necessary I think.
}

// Bump re-enables the oldest disabled structure a single gene represents.
// (It's like bringing up a post in message board terms.) Time complexity is O(N).
// If there ain't one disabled then this operation does nothing other than consuming some time searching for it.
func (chrome *Chromosome) Bump() {
	chrome.Sort()
	for _, linkGene := range chrome.LinkGenes {
		if !linkGene.Enabled {
			linkGene.Enabled = true
			return
		}
	}
}

// String callback of Chromosome.
func (chrome *Chromosome) String() string {
	const tooBig = 5
	if chrome.SizeBuilding() < tooBig {
		return fmt.Sprint(
			"{Chromosome of ",
			len(chrome.IONodeGenes), " NodeGenes", "/",
			"LinkGenes:", chrome.LinkGenes, "/",
			"DiceGene:", chrome.DiceGene, "}")
	}
	return fmt.Sprint(
		"{Chromosome of ",
		len(chrome.IONodeGenes), " NodeGenes and ",
		chrome.SizeBuilding(), " LinkGenes and ",
		chrome.DiceGene, "}")
}

// IsDiceEnabled tells if this chromosome has its own dynamic mutation rate other than the fixed probability.
func (chrome *Chromosome) IsDiceEnabled() bool {
	// DiceGene is not nil-able and it shouldn't. This function panics when it is nil.
	return chrome.DiceGene.Enabled
}

// NodeGenesByUUID returns a map of all node genes available in this Chromosome.
// The parameter can be nil.
func (chrome *Chromosome) NodeGenesByUUID(fillMe map[uuid.UUID]*NodeGene) (filledOut map[uuid.UUID]*NodeGene) {
	if fillMe == nil {
		filledOut = map[uuid.UUID]*NodeGene{}
	} else {
		filledOut = fillMe
	}
	for _, linkGene := range chrome.LinkGenes {
		if from := linkGene.Topo.From; from != nil {
			filledOut[from.UUID] = from
		}
		if to := linkGene.Topo.To; to != nil {
			filledOut[to.UUID] = to
		}
	}
	return filledOut
}

// Genetics provides the essentials of this chromosome's genetics; all the link-gene stuff that is.
//
// Returns:
//  - `linkGenesByInnovation`: A map of all link genes available in this Chromosome.
//  - `sizeLinkGenes`: The normalize size factor for the compatibility distance measure. Refer to (*Chromosome).SizeBuilding().
//  - `innovationLatest`: The biggest (latest) innovation number this chromosome has. This is a historical bound used to divide unmatched genes into disjoint and excess.
//
func (chrome *Chromosome) Genetics() (linkGenesByInnovation map[Innovation]*LinkGene, sizeLinkGenes int, innovationLatest Innovation) {
	linkGenesByInnovation = map[Innovation]*LinkGene{}
	for _, linkGene := range chrome.LinkGenes {
		linkGenesByInnovation[linkGene.Topo.Innov] = linkGene
	}
	sizeLinkGenes = chrome.SizeBuilding()
	if sizeLinkGenes > 0 {
		innovationLatest = chrome.LinkGenes[len(chrome.LinkGenes)-1].Topo.Innov
	}
	return
}

// SizeBuilding returns the number of all link genes this genetic encoder has.
// This allows us to measure the scale of structural innovations a topology gets made up with.
func (chrome *Chromosome) SizeBuilding() (sizeTopologicalInnovation int) {
	return len(chrome.LinkGenes)
}

// NewNeuralNetwork (verbed) from a Chromosome.
func (chrome *Chromosome) NewNeuralNetwork() (network *NeuralNetwork, err error) {
	// What we create and return here.
	network = &NeuralNetwork{
		DirectedGraph:   simple.NewDirectedGraph(),
		InputNodes:      []*NeuralNode{},
		OutputNodes:     []*NeuralNode{},
		NodeByGene:      map[*NodeGene]*NeuralNode{},
		EdgeByGene:      map[*LinkGene]*NeuralEdge{},
		NumBiasNodes:    0,
		NumInputNodes:   0,
		NumHiddenNodes:  0,
		NumOutputNodes:  0,
		NumLayers:       0,
		IsEvaluatedWell: false,
	}

	// Not needed since we've probably already sorted this,
	// but just to make sure...
	chrome.Sort()
	// Decode NodeGene(s) of a Chromosome.
	for _, verticeGene := range chrome.IONodeGenes {
		network.AddNewNeuralNode(verticeGene)
	}
	// Decode LinkGene(s) of a Chromosome.
	for _, edgeGene := range chrome.LinkGenes {
		if !edgeGene.Enabled {
			continue
		}

		// I play Pot of Greed.
		nodeFrom := network.FindNodeByGene(edgeGene.Topo.From)
		if nodeFrom == nil {
			nodeFrom = network.AddNewNeuralNode(edgeGene.Topo.From)
		}
		nodeTo := network.FindNodeByGene(edgeGene.Topo.To)
		if nodeTo == nil {
			nodeTo = network.AddNewNeuralNode(edgeGene.Topo.To)
		}
		// This allows me to draw 2 cards from my deck and add them to my hand.
		err := network.AddNewNeuralEdge(nodeFrom, nodeTo, edgeGene)
		if err != nil {
			return nil, errors.New("bad chromosome: " + err.Error())
		}
	}

	// Validate this network's nodes.
	{ // Validate forward.
		visitedForward := map[*NeuralNode]struct{}{}
		for _, inputNode := range network.GetInputNodes() {
			// Validates nodes in this network floodfilling forward in DFS order.
			network.TraverseForward(inputNode, func(onNodeVisit *NeuralNode) {
				if !onNodeVisit.IsValidForward {
					onNodeVisit.IsValidForward = true
				}
			}, visitedForward)
		}
	}
	{ // Validate backward.
		visitedBackward := map[*NeuralNode]struct{}{}
		for _, outputNode := range network.GetOutputNodes() {
			// Validates nodes in this network floodfilling backward in DFS order.
			network.TraverseBackward(outputNode, func(onNodeVisit *NeuralNode) {
				if !onNodeVisit.IsValidBackward {
					onNodeVisit.IsValidBackward = true
				}
			}, visitedBackward)
		}
	}
	{ // Then cull off disconnected (invalid) nodes, as well as any edges attached to it.
		nodes, err := network.Sort()
		if err != nil {
			return nil, errors.New("bad chromosome: " + err.Error())
		}
		for _, node := range nodes {
			if !node.IsValid() {
				// One gene must always match to a single neural node in a neural network.
				err = network.RemoveNeuralNode(node.NodeGene)
				if err != nil {
					return nil, errors.New("bad chromosome: " + err.Error())
				}
			}
		}
	}

	{ // Update the layer level.
		err := network.updateLayerLevel()
		if err != nil {
			return nil, errors.New("bad chromosome: " + err.Error())
		}
	}

	return network, nil
}

// Unexported, but defined here for unit-tests.
func (network *NeuralNetwork) updateLayerLevel() error {
	// Everything should be in initial state.
	nodes, err := network.Sort()
	if err != nil {
		return err
	}
	// FF in BFS-like order.
	for _, toNode := range nodes {
		level := float64(LayerLevelInit)
		{
			froms := network.NeuralNodesTo(toNode)
			for froms.Next() {
				fromNode, err := froms.NeuralNode().Level()
				if err != nil {
					return err
				}
				level = math.Max(float64(fromNode), float64(level))
			}
		}
		toNode.SetLevel(int(level + 1))
	}
	// OutputNodes get a special treat.
	maxLevel := int(1) // It always has to be bigger than 0.
	for _, outputNode := range network.OutputNodes {
		lvl, err := outputNode.Level()
		if err != nil {
			return err
		}
		maxLevel = int(math.Max(float64(maxLevel), float64(lvl)))
	}
	for _, outputNode := range network.OutputNodes {
		outputNode.SetLevel(maxLevel)
	}
	// Return
	network.NumLayers = maxLevel + 1
	return nil
}

// NewNeuralNetworkProto gets us a topological analysis of the network.
func (chrome *Chromosome) NewNeuralNetworkProto(
	includeDisabledLinks, includeNonBiasInputNodes, includeBiasNodes bool,
) (proto *NeuralNetworkProto, err error) {
	// What we create and return here.
	proto = &NeuralNetworkProto{
		&NeuralNetwork{
			DirectedGraph:   simple.NewDirectedGraph(),
			InputNodes:      []*NeuralNode{},
			OutputNodes:     []*NeuralNode{},
			NodeByGene:      map[*NodeGene]*NeuralNode{},
			EdgeByGene:      map[*LinkGene]*NeuralEdge{},
			NumBiasNodes:    0,
			NumInputNodes:   0,
			NumHiddenNodes:  0,
			NumOutputNodes:  0,
			NumLayers:       0,
			IsEvaluatedWell: false,
		},
	}

	// Not needed since we've probably already sorted this,
	// but just to make sure...
	chrome.Sort()
	// Decode NodeGene(s) of a Chromosome.
	for _, verticeGene := range chrome.IONodeGenes {
		// Exclusive...
		if (!includeBiasNodes && verticeGene.TypeInInt() == InputNodeBias) || // excluded links to bias input
			(!includeNonBiasInputNodes && verticeGene.TypeInInt() == InputNodeNotBias) { // excluded links to non-bias input
			continue
		}
		proto.AddNewNeuralNode(verticeGene)
	}
	// Decode LinkGene(s) of a Chromosome.
	for _, edgeGene := range chrome.LinkGenes {
		// Exclusive...
		if !includeDisabledLinks && !edgeGene.Enabled { // excluded disabled links
			continue
		}
		if !includeBiasNodes &&
			(edgeGene.Topo.From.TypeInInt() == InputNodeBias ||
				edgeGene.Topo.To.TypeInInt() == InputNodeBias) { // excluded links to bias input
			continue
		}
		if !includeNonBiasInputNodes &&
			(edgeGene.Topo.From.TypeInInt() == InputNodeNotBias ||
				edgeGene.Topo.To.TypeInInt() == InputNodeNotBias) { // excluded links to non-bias input
			continue
		}
		// I play Pot of Greed.
		nodeFrom := proto.FindNodeByGene(edgeGene.Topo.From)
		if nodeFrom == nil {
			nodeFrom = proto.AddNewNeuralNode(edgeGene.Topo.From)
		}
		nodeTo := proto.FindNodeByGene(edgeGene.Topo.To)
		if nodeTo == nil {
			nodeTo = proto.AddNewNeuralNode(edgeGene.Topo.To)
		}
		// This allows me to draw 2 cards from my deck and add them to my hand.
		err := proto.AddNewNeuralEdge(nodeFrom, nodeTo, edgeGene)
		if err != nil {
			return nil, errors.New("bad chromosome: " + err.Error())
		}
	}

	// Validate this network's nodes.
	{ // Validate forward.
		visitedForward := map[*NeuralNode]struct{}{}
		for _, inputNode := range proto.GetInputNodes() {
			// Validates nodes in this network floodfilling forward in DFS order.
			proto.TraverseForward(inputNode, func(onNodeVisit *NeuralNode) {
				if !onNodeVisit.IsValidForward {
					onNodeVisit.IsValidForward = true
				}
			}, visitedForward)
		}
	}
	{ // Validate backward.
		visitedBackward := map[*NeuralNode]struct{}{}
		for _, outputNode := range proto.GetOutputNodes() {
			// Validates nodes in this network floodfilling backward in DFS order.
			proto.TraverseBackward(outputNode, func(onNodeVisit *NeuralNode) {
				if !onNodeVisit.IsValidBackward {
					onNodeVisit.IsValidBackward = true
				}
			}, visitedBackward)
		}
	}

	{ // Update the layer level.
		err := proto.updateLayerLevel()
		if err != nil {
			return nil, errors.New("bad chromosome: " + err.Error())
		}
	}

	return proto, nil
}

// Copy is a constructor.
// Deep-copies everything of this chromosome except node-genes.
func (chrome *Chromosome) Copy() (*Chromosome, error) {
	return NewChromosome(func(oldLinks []*LinkGene) []*LinkGene {
		links := make([]*LinkGene, len(oldLinks))
		for i, oldLink := range oldLinks {
			links[i] = oldLink.Copy()
		}
		return links
	}(chrome.LinkGenes), func(oldSlice []*NodeGene) []*NodeGene {
		// node genes are not copied but the slice holding them needs to.
		nodes := make([]*NodeGene, len(oldSlice))
		nCopied := copy(nodes, oldSlice)
		if nCopied != len(oldSlice) || nCopied != len(nodes) {
			// log.Println("fatal: failed to copy a slice of nodes") // debug //
			panic("failed to copy a slice of nodes")
		}
		return nodes
	}(chrome.IONodeGenes), chrome.DiceGene.Copy())
}

// ----------------------------------------------------------------------------
// NeuralNetwork & its stuff

// Constants regarding the neural node's inputs & outputs.
const (
	// Defines the range of a value an axon outputs, (domain of activation)
	// which (state) is right before it gets multiplied by the synaptic weight of its axon.
	// NeuralValue is designed to be in range of [-128, +128].
	// The reason is because we might lose our precisions if the value was too small.
	// Also this is not to be confused with the signed byte integer range of [-128, +127].
	NeuralValueMin   NeuralValue = -128.0
	NeuralValueMax   NeuralValue = +128.0
	NeuralValueMid   NeuralValue = 0.0   // Denotes the neutral neural value.
	NeuralValueWidth NeuralValue = 256.0 // Const def of Abs(Min) + Abs(Max).
	// Constant of the Nth neural layer.
	LayerLevelInit int = -1 // An integer smaller than the smallest value of the layer N.
)

// NeuralValue is the value an axon outputs, which is *about* to be multiplied by the synaptic weight of its axon.
// Or it may also be considered the input a neural node receives, summing up all those outputs multiplied by their weights.
// Note that NeuralValue as an input and as an output are two totally different things. In/Out should be distinguishable from each other.
// NeuralValue as an input can go outside the range of an output which is the domain of our activation function.
type NeuralValue float64

// NV is a shorthand for `NeuralValue()`.
//  - NV(): NeuralValue from float64.
//  - NVU(): NeuralValue from uint8.
//  - NVS(): NeuralValue from int8.
// Note that `NeuralValue()` may perform slightly faster than `NV()` by like 0.02ns - that at least on my laptop.
// `NeuralValue()` is a static type cast without any cost whilst `NV()` being a function that runs at runtime.
func NV(from float64) (to NeuralValue) {
	return NeuralValue(from)
}

// NVU is a shorthand for `NeuralValueFromUint8()`.
//  - NV(): NeuralValue from float64.
//  - NVU(): NeuralValue from uint8.
//  - NVS(): NeuralValue from int8.
func NVU(from uint8) (to NeuralValue) {
	return NeuralValueFromUint8(from)
}

// NVS is a shorthand for `NeuralValueFromInt8()`.
//  - NV(): NeuralValue from float64.
//  - NVU(): NeuralValue from uint8.
//  - NVS(): NeuralValue from int8.
func NVS(from int8) (to NeuralValue) {
	return NeuralValueFromInt8(from)
}

// NeuralValueFromUint8 converts a raw uint8 value to a NeuralValue.
//
// Interpreted:
//  - uint8(0) <---> uint8(255)
//  - uint8(255) == uint8(math.MaxUint8)
//  - uint8(0) <---> uint8(math.MaxUint8)
//  - float64(NeuralValueMin) <---> float64(NeuralValueMax)
//  - float64(NeuralValueMin):uint8(0) <---> uint8(math.MaxUint8):float64(NeuralValueMax)
//  - float64(NeuralValueMin):MaxNegative:uint8(0) == float64(NeuralValueMax):MaxPositive:uint8(math.MaxUint8)
//  - uint8(0):MaxNegative:float64(NeuralValueMin) == uint8(math.MaxUint8):MaxPositive:float64(NeuralValueMax)
//
func NeuralValueFromUint8(from uint8) (to NeuralValue) {
	percent := float64(from) / math.MaxUint8
	absMaxNV := float64(NeuralValueWidth)
	offsetNegative := math.Abs(float64(NeuralValueMin))
	return NeuralValue(percent*absMaxNV - offsetNegative)
}

// NeuralValueFromInt8 converts a raw int8 value to a NeuralValue.
//
// Interpreted:
//  - int8(-128) <---> int8(+127)
//  - NV(-128.0) <---> NV(+128.0)
//  - 空; nil; NaN <==> NV(NeuralValueMid) --> NVKindNeutral
//  - int8(nil) <==> NV(0.0)
//  - int8(NaN) <==> NV(0.0)
//  - int8(0) <==> NV(1.0)
//  - int8(+127) <==> NV(128.0)
//  - int8(-128):MaxNegative:float64(NeuralValueMin) == int8(+127):MaxPositive:float64(NeuralValueMax)
//
func NeuralValueFromInt8(from int8) (to NeuralValue) {
	if from < 0 {
		return NeuralValue(from)
	}
	return NeuralValue(float64(from) + 1.0)
}

// Float64 of this NeuralValue.
// This method is defined for convenience, although it performs a bit slower than the type cast.
// `(NeuralValue).Float64()` is less performant than the `float64(NeuralValue)` static type cast which almost has the same effect.
func (nv NeuralValue) Float64() float64 {
	return float64(nv)
}

// Concentration of this NeuralValue. Inverse operation of `NeuralValueFromUint8()`.
// The return is a concentration; uint8 diluteness in range [0x0, 0xFF] equal to [0, 255];
// what tells whether this neural value is more closer to the positive side or to the negative side.
// This function also tests the range of this neural value.
// An error is returned iif the result is overflowed/underflowed.
func (nv NeuralValue) Concentration() (diluteness uint8, err error) {
	if nv > NeuralValueMax {
		err = errors.New("overflowed uint8")
	}
	if nv < NeuralValueMin {
		err = errors.New("underflowed uint8")
	}
	offsetNegative := math.Abs(float64(NeuralValueMin))
	absMaxNV := float64(NeuralValueWidth)
	percent := (float64(nv) + offsetNegative) / absMaxNV
	return uint8(percent * math.MaxUint8), err
}

// Strength returns a uint8 in range [0x0, 0xFF] equal to [0, 255],
// which denotes how strong this negative/positive signal is.
// 0 is returned if this neural value is determined neither negative nor positive kind.
// An error is returned iif this neural value is not in its range valid.
// The function panics upon runtime error.
func (nv NeuralValue) Strength() (strength uint8, err error) {
	switch nv.Kind() {
	case NVKindNeutral:
		return uint8(NeuralValueMid), nil // 0 is returned.
	case NVKindNegative:
		percent := math.Abs(float64(nv)) / math.Abs(float64(NeuralValueMin))
		return uint8(percent * math.MaxUint8), nil
	case NVKindPositive:
		percent := math.Abs(float64(nv)) / math.Abs(float64(NeuralValueMax))
		return uint8(percent * math.MaxUint8), nil
	case NVKindExceptional:
		var percent float64
		if nv > NeuralValueMax {
			percent = math.Abs(float64(nv)) / math.Abs(float64(NeuralValueMax))
		} else if nv < NeuralValueMin {
			percent = math.Abs(float64(nv)) / math.Abs(float64(NeuralValueMin))
		} else {
			panic("runtime violation")
		}
		return uint8(percent * math.MaxUint8), errors.New("overflowed uint8")
	default:
		return uint8(math.NaN()), errors.New("unknown value kind")
	}
	// panic("runtime violation") // unreachable
}

// IsEvaluated tests if this neural value as an *output* is in its valid range.
//
// Returns:
//  - True if the value is within the range, that's a necessary condition for the value to be an output value,
//   thus considered equivalent to that of a neural value filtered out of the activation, while not being a
//   sufficient condition to be an output. So it could be an input or an output but at least not a complete raw input.
//  - False if outside the range, or this value is turned out to be NaN; which means
//   that the value is in the initial state of a neural node and is so unevaluated.
//
func (nv NeuralValue) IsEvaluated() bool {
	return nv >= NeuralValueMin && nv <= NeuralValueMax
}

// Kind tells the kind of this NeuralValue and also tests if it is in its valid range while doing so.
//
// Returns:
//  - NVKindPositive: (NV-Mid, NV-Max]
//  - NVKindNegative: [NV-Min, NV-Mid)
//  - NVKindNeutral: [NV-Mid, NV-Mid]
//  - NVKindExceptional: Outside [NV-Min, NV-Max]
//
func (nv NeuralValue) Kind() NeuralValueKind {
	if nv > NeuralValueMid && nv <= NeuralValueMax {
		return NVKindPositive // (Mid, Max]
	}
	if nv < NeuralValueMid && nv >= NeuralValueMin {
		return NVKindNegative // [Min, Mid)
	}
	if nv == NeuralValueMid {
		return NVKindNeutral // [Mid, Mid]
	}
	return NVKindExceptional // Outside [Min, Max]
}

// NeuralValueKind enum for Neutral, Negative, and Positive.
type NeuralValueKind int

// const Neutral, Negative, and Positive of NeuralValueKind.
const (
	NVKindExceptional = iota // catch me
	NVKindNeutral
	NVKindNegative
	NVKindPositive
)

// Sigmoid our classic differentiable activation function.
//
// Formula:
//  y = 256/(1+Exp(-x*1.0)) - 128
//
func Sigmoid(x NeuralValue) (y NeuralValue) {
	const height = 256
	const offsetY = -128
	const slope = 1.0 // used to be 4.924273
	return NeuralValue(height/(1+math.Exp(float64(-x)*slope)) + offsetY)
}

// Ramp is a kind of rectifier and a pseudo-activation.
func Ramp(x NeuralValue) (y NeuralValue) {
	const (
		// Be considerate if you're going to change these.
		// This set of consts is a dependency all over this package.
		slope        = 1.0
		minInputRamp = NeuralValueMin / slope
		maxInputRamp = NeuralValueMax / slope
		minOutput    = NeuralValueMin
		maxOutput    = NeuralValueMax
	)
	if x >= maxInputRamp {
		return maxOutput
	}
	if x > minInputRamp {
		return x * slope
	}
	return minOutput
}

// Activation function is that of a node which takes the sum of weighted inputs as its argument, then returns
// a value a neural node holds(outputs). The function is defined statically here for the efficiency.
func Activation(x NeuralValue) (y NeuralValue) {
	return Sigmoid(x)
}

// NeuralNetworkProto is a subclass of NeuralNetwork.
// It can provide a rough topological analysis of a neural network that might be constructed with
// incomplete structures so we can get an idea how it should/can be mutated and etc.
// How it really is different from NeuralNetwork is dynamically (at runtime) defined by a constructor.
type NeuralNetworkProto struct {
	*NeuralNetwork
}

// NewNeuralEdge of NeuralNetworkProto is an override.
// The only difference with its super is that this method allows this neural network to be constructed with disabled link genes.
func (nnp *NeuralNetworkProto) NewNeuralEdge(from, to graph.Node, gene *LinkGene) *NeuralEdge {
	if gene == nil {
		// log.Println("The gene cannot be nil:", "while adding a neural edge", "from", from, "to", to) // debug //
		panic("The gene cannot be nil.")
	}
	return &NeuralEdge{
		Edge: nnp.NewEdge(from, to),
		Gene: gene,
	}
}

// AddNewNeuralEdge of NeuralNetworkProto is an override.
// The only difference with its super is that this method allows this neural network to be constructed with disabled link genes.
func (nnp *NeuralNetworkProto) AddNewNeuralEdge(from, to graph.Node, gene *LinkGene) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = errors.New(fmt.Sprint(r.(string), " from:", from, " to:", to, " gene:", gene))
		}
	}()
	edge := nnp.NewNeuralEdge(from, to, gene)
	nnp.SetEdge(edge)
	nnp.EdgeByGene[gene] = edge
	return err
}

// GetNeuralNetwork is a getter.
func (nnp *NeuralNetworkProto) GetNeuralNetwork() *NeuralNetwork {
	return nnp.NeuralNetwork
}

// ToFrom finds a random `fromNode` that connects to `toNode`. The return can be nil if none is found.
func (nnp *NeuralNetworkProto) ToFrom(topoNodes []*NeuralNode, toNode *NeuralNode) (fromNode *NeuralNode) {
	// check for the only condition that could produce a cycle
	visitedForward := map[*NeuralNode]struct{}{} // subtree
	nnp.TraverseForward(toNode, nil, visitedForward)
	// collect & exclude candidates
	fromNodeCandidates := map[*NeuralNode]struct{}{}
	for _, node := range topoNodes {
		if node.TypeInInt() != OutputNode {
			fromNodeCandidates[node] = struct{}{}
		}
	}
	// cyclic ones can't be included in candidates
	// (!) Also this does not allow the recurrent link with which a node gets linked back to itself.
	for node := range visitedForward {
		delete(fromNodeCandidates, node)
	}
	// occupied ones excluded (direct connections)
	froms := nnp.NeuralNodesTo(toNode)
	for froms.Next() {
		delete(fromNodeCandidates, froms.NeuralNode())
	}
	// 0 candidates for our chosen gene
	if len(fromNodeCandidates) <= 0 {
		return nil
	}
	// choose one of the candidates randomly
	iFrom := rand.Intn(len(fromNodeCandidates))
	for node := range fromNodeCandidates {
		if iFrom == 0 {
			return node
		}
		iFrom--
	}
	return nil // This must be an error. Something went wrong.
}

// FromTo finds a random `toNode` where `fromNode` is connected to. The return can be nil if none is found.
func (nnp *NeuralNetworkProto) FromTo(topoNodes []*NeuralNode, fromNode *NeuralNode) (toNode *NeuralNode) {
	// check for the only condition that could produce a cycle
	visitedBackward := map[*NeuralNode]struct{}{} // parent-tree
	nnp.TraverseBackward(fromNode, nil, visitedBackward)
	// collect & exclude candidates
	toNodeCandidates := map[*NeuralNode]struct{}{}
	for _, node := range topoNodes {
		if node.TypeInInt() != InputNodeBias && node.TypeInInt() != InputNodeNotBias {
			toNodeCandidates[node] = struct{}{}
		}
	}
	// cyclic ones can't be included in candidates
	// (!) Also this does not allow the recurrent link with which a node gets linked back to itself.
	for node := range visitedBackward {
		delete(toNodeCandidates, node)
	}
	// occupied ones excluded (direct connections)
	dests := nnp.NeuralNodesFrom(fromNode)
	for dests.Next() {
		delete(toNodeCandidates, dests.NeuralNode())
	}
	// 0 candidates for our chosen gene
	if len(toNodeCandidates) <= 0 {
		return nil
	}
	// choose one of the candidates randomly
	iTo := rand.Intn(len(toNodeCandidates))
	for node := range toNodeCandidates {
		if iTo == 0 {
			return node
		}
		iTo--
	}
	return nil // This must be an error. Something went wrong.
}

// RandomNicePlaceForNewAcyclicLink gets a random nice place where a new link can be put without creating a cycle in this graph.
// So we can ensure that the graph stays a DAG after insertion of a link in between two existing nodes.
// The returns are the two existing nodes of this graph a new link can bridge from and to.
// This function does a bit of calculations for finding the nice place. Time complexity is guessed O(V*E).
// To test this out, you may connect the two returned nodes and perform a topological sort of the graph to see if there is a cycle.
func (nnp *NeuralNetworkProto) RandomNicePlaceForNewAcyclicLink() (fromNode, toNode *NeuralNode, err error) {
	topoNodes, err := nnp.Sort()
	if err != nil {
		return nil, nil, err
	} // It is verified to be DAG by now, because the topological sort above returned without any error.
	for len(topoNodes) > 0 {
		i := rand.Intn(len(topoNodes))
		randomChosenNode := topoNodes[i]
		topoNodes = append(topoNodes[:i], topoNodes[i+1:]...)
		//
		switch randomChosenNode.TypeInInt() {
		case InputNodeBias:
			fallthrough
		case InputNodeNotBias:
			fromNode = randomChosenNode
			toNode = nnp.FromTo(topoNodes, fromNode)
		case OutputNode:
			toNode = randomChosenNode
			fromNode = nnp.ToFrom(topoNodes, toNode)
		case HiddenNode:
			if rand.Intn(2) < 1 { // 50% chance of
				fromNode = randomChosenNode
				toNode = nnp.FromTo(topoNodes, fromNode)
			} else {
				toNode = randomChosenNode
				fromNode = nnp.ToFrom(topoNodes, toNode)
			}
		default:
			return nil, nil, errors.New("unhandled error")
		}
		//
		if fromNode != nil && toNode != nil {
			break
		} else {
			fromNode = nil
			toNode = nil
			continue
		}
	} // for topoNodes
	if fromNode == nil || toNode == nil {
		// RandomNicePlaceForNewAcyclicLink() is nowhere.
		// This may mean that all available nodes in this NN are already fully connected.
		return nil, nil, errors.New("nowhere")
	}
	return fromNode, toNode, nil
}

// NeuralNetwork is a part of an artificial brain structure of an individual living organism.
// The basic form of it should be evaluatable and be constructed with valid structures only.
// The data structure of a neural network is represented as a DAG(Directed Acyclic Graph).
// All neural nodes in a valid neural network is connected directly or undirectly to
// both Inputs and Outputs layer but just except for those input nodes and output nodes.
// It is strongly recommended not to access any of these members directly outside this package,
// unless you understand what those really are.
// Otherwise use this struct's methods and its constructor instead.
type NeuralNetwork struct {
	// Inheritance. (super class)
	*simple.DirectedGraph

	// Nodes in order.
	InputNodes  []*NeuralNode
	OutputNodes []*NeuralNode

	// Phonebooks.
	NodeByGene map[*NodeGene]*NeuralNode // Nodes indexed by NodeGene. // private use (member could be unexported)
	EdgeByGene map[*LinkGene]*NeuralEdge // Edges indexed by LinkGene. // private use (member could be unexported)

	// Counts up whenever a node is added to this network.
	// Always equal to the number of nodes this neural network has.
	NumBiasNodes   int
	NumInputNodes  int // including biases
	NumHiddenNodes int
	NumOutputNodes int
	// NumLayers 0 by default.
	// This updates only when the layer level is evaluated.
	NumLayers int

	// Set to false when the values are being updated or yet updated.
	IsEvaluatedWell bool
}

// GetInputNodes returns a copy slice of InputNodes.
// The ptrs a slice returned here holds are to originals not to copies.
func (network *NeuralNetwork) GetInputNodes() []*NeuralNode {
	ret := make([]*NeuralNode, len(network.InputNodes))
	for i, v := range network.InputNodes {
		ret[i] = v
	}
	return ret
}

// GetOutputNodes returns a copy slice of OutputNodes.
// The ptrs a slice returned here holds are to originals not to copies.
func (network *NeuralNetwork) GetOutputNodes() []*NeuralNode {
	ret := make([]*NeuralNode, len(network.OutputNodes))
	for i, v := range network.OutputNodes {
		ret[i] = v
	}
	return ret
}

// FindNodeByGene finds (returns) a NeuralNode in this NeuralNetwork by
// the NeuralNode's NodeGene, costing only constant time complexity.
// Returns nil if not present.
// Each NeuralNode in a NeuralNetwork is/should be matched to a single NodeGene.
func (network *NeuralNetwork) FindNodeByGene(gene *NodeGene) *NeuralNode {
	return network.NodeByGene[gene]
}

// FindEdgeByGene finds (returns) a NeuralEdge in this NeuralNetwork by the NeuralEdge's LinkGene,
// costing only constant time complexity.
// Returns nil if not present.
// Each NeuralEdge in a NeuralNetwork is/should be matched to a single LinkGene.
func (network *NeuralNetwork) FindEdgeByGene(gene *LinkGene) *NeuralEdge {
	return network.EdgeByGene[gene]
}

// Evaluate of a NeuralNetwork performs a feedforward of a network.
// The list (slice) returned by this function keeps copies of all output nodes,
// from which we get the output values of a network evaluated.
func (network *NeuralNetwork) Evaluate(inputs []NeuralValue) (outputs []NeuralNode, err error) {
	// Validate the input.
	if len(inputs) != network.NumInputNodes-network.NumBiasNodes {
		return nil, errors.New("the number of inputs does not match to that of this neural network")
	}

	// Init.
	nodes, err := network.Clear() // IsEvaluated set to false.
	if err != nil {
		return nil, err
	}

	// All input nodes are evaluated here.
	for _, node := range network.InputNodes[:network.NumBiasNodes] {
		node.EvaluateWithoutActivation(NeuralValueMax)
	}
	// Pass inputs.
	for i, node := range network.InputNodes[network.NumBiasNodes:] {
		// log.Println(node) // debug //
		if !inputs[i].IsEvaluated() {
			return nil, errors.New(
				fmt.Sprint(
					"each input value should be in range",
					" [", NeuralValueMin, ", ", NeuralValueMax, "]",
					// It doesn't actually have to be.
					// Generating error like this is just to help the user know what he/she is doing.
				),
			)
		}
		node.EvaluateWithoutActivation(inputs[i])
		// log.Println(inputs[i], "->", node.Value) // debug //
	}

	// FF in BFS-like order.
	for _, toNode := range nodes {
		// log.Println(toNode) // debug //
		if toNode.IsEvaluated() { // Skip when toNode is an input node.
			continue
		}
		sum := 0.0 // 1
		{
			froms := network.NeuralNodesTo(toNode)
			for froms.Next() {
				fromNode := froms.NeuralNode()
				w := network.NeuralEdge(fromNode, toNode).Weight()
				v, err := fromNode.Output()
				if err != nil {
					return nil, err
				}
				sum += (float64(v) * w) // 1
			}
		}
		toNode.Evaluate(NeuralValue(sum)) // 1
		// log.Println(sum, "->", toNode.Value) // debug //
	}

	// Return the output nodes.
	outputs = make([]NeuralNode, len(network.OutputNodes))
	for i := range outputs {
		outputs[i] = *network.OutputNodes[i]
	}
	network.IsEvaluatedWell = true // back at it again.
	return outputs, nil
}

// NLayers returns the number of layers in this network.
func (network *NeuralNetwork) NLayers() int {
	return network.NumLayers
}

// IsEvaluated tells whether neural values of this neural network has been evaluated or not.
func (network *NeuralNetwork) IsEvaluated() bool {
	return network.IsEvaluatedWell
}

// Clear values of all neural nodes of this NeuralNetwork.
// Sets IsEvaluated to false.
// This function returns the topological sort of this graph.
func (network *NeuralNetwork) Clear() (nodes []*NeuralNode, err error) {
	nodes, err = network.Sort()
	if err != nil {
		return nil, err
	}
	network.IsEvaluatedWell = false
	for _, node := range nodes {
		node.ClearValue()
	}
	return nodes, nil
}

// Sort does the topological sort of this graph, (often abbreviated to topo-sort)
// so that you can iterate over all the nodes of a neural network in a feedforward order. The order is *similar* to that of BFS.
// The topological sort can also be used as a cycle detection of the graph.
// This sort is stable, as it follows the sequence alignment of the genetic encoder (chromosome) as well.
// The returned list; slice is a copy and holds the actual (pointing not to a copy) pointers to all the neural nodes of this network.
func (network *NeuralNetwork) Sort() (nodes []*NeuralNode, err error) {
	raw, err := topo.SortStabilized(network.DirectedGraph, nil)
	// convert ([]graph.Node) to ([]*NeuralNode)
	nodes = make([]*NeuralNode, len(raw))
	for i, v := range raw {
		nodes[i] = v.(*NeuralNode)
	}
	return nodes, err
}

// TraverseForward DFSes a subtree of a node.
//
// Parameters:
//  - `node`: The node where you want to start DFSing.
//  - `onNodeVisit`: Callback on each node visit. It's allowed to be nil.
//  - `visited`: DFS context. This argument cannot be nil. The function panics when it is.
//
func (network *NeuralNetwork) TraverseForward(
	node *NeuralNode,
	onNodeVisit func(nodeVisit *NeuralNode),
	visited map[*NeuralNode]struct{},
) {
	if _, ok := visited[node]; !ok {
		visited[node] = struct{}{}
		if onNodeVisit != nil {
			onNodeVisit(node)
		}
		toNodes := network.NeuralNodesFrom(node)
		for toNodes.Next() {
			network.TraverseForward(toNodes.NeuralNode(), onNodeVisit, visited)
		}
	}
	// Do nothing here. Return right away if it's the case we've already visited the node.
	return
}

// TraverseBackward DFSes a parent-tree(reverse-subtree) of a node.
//
// Parameters:
//  - `node`: The node where you want to start DFSing.
//  - `onNodeVisit`: Callback on each node visit. It's allowed to be nil.
//  - `visited`: DFS context. This argument cannot be nil. The function panics when it is.
//
func (network *NeuralNetwork) TraverseBackward(
	node *NeuralNode,
	onNodeVisit func(nodeVisit *NeuralNode),
	visited map[*NeuralNode]struct{},
) {
	if _, ok := visited[node]; !ok {
		visited[node] = struct{}{}
		if onNodeVisit != nil {
			onNodeVisit(node)
		}
		froms := network.NeuralNodesTo(node)
		for froms.Next() {
			network.TraverseBackward(froms.NeuralNode(), onNodeVisit, visited)
		}
	}
	// Do nothing here. Return right away if it's the case we've already visited the node.
	return
}

// NewNeuralNode is a basic constructor for a NeuralNode.
// The node created here inherits a NodeGene.
// The returned node's identifier is not valid,
// until the node is added to the NeuralNetwork that created it.
// You might want to use AddNewNeuralNode() instead.
func (network *NeuralNetwork) NewNeuralNode(gene *NodeGene) *NeuralNode {
	if gene == nil {
		// log.Println("The gene cannot be nil:", "while creating a neural node") // debug //
		panic("Cannot create a neural node from a nil node gene")
	}
	return &NeuralNode{
		Identifier:      network.DirectedGraph.NewNode().ID(), // The Node's ID does not become valid in a graph until the Node is added to a graph.
		NodeGene:        gene,
		Value:           NeuralValue(math.NaN()),
		LayerLevel:      LayerLevelInit,
		IsValidForward:  false,
		IsValidBackward: false,
	}
}

// AddNewNeuralNode creates a unique identifiable neural node within this neural network, along with that node's genetics.
// Returns the node added to this network.
func (network *NeuralNetwork) AddNewNeuralNode(gene *NodeGene) (node *NeuralNode) {
	node = network.NewNeuralNode(gene)
	network.AddNode(node)
	network.NodeByGene[gene] = node
	switch gene.TypeInInt() {
	case InputNodeBias:
		network.NumBiasNodes++
		fallthrough
	case InputNodeNotBias:
		network.NumInputNodes++
		network.InputNodes = append(network.InputNodes, node)
	case HiddenNode:
		network.NumHiddenNodes++
	case OutputNode:
		network.NumOutputNodes++
		network.OutputNodes = append(network.OutputNodes, node)
	}
	return node
}

// RemoveNeuralNode deletes a neural node of this network, as well as any neural edges attached to it.
// This does the inverse of AddNewNeuralNode().
// Returns an error if the neural node to be removed is not present in this neural network.
func (network *NeuralNetwork) RemoveNeuralNode(gene *NodeGene) error {
	node := network.FindNodeByGene(gene)
	if node == nil {
		return errors.New("the node is not present")
	}
	network.RemoveNode(node.ID())
	delete(network.NodeByGene, gene)
	switch gene.TypeInInt() {
	case InputNodeBias:
		network.NumBiasNodes--
		fallthrough
	case InputNodeNotBias:
		network.NumInputNodes--
		for i, v := range network.InputNodes {
			if v == node {
				network.InputNodes = append(network.InputNodes[:i], network.InputNodes[i+1:]...)
				break
			}
		}
	case HiddenNode:
		network.NumHiddenNodes--
	case OutputNode:
		network.NumOutputNodes--
		for i, v := range network.OutputNodes {
			if v == node {
				network.OutputNodes = append(network.OutputNodes[:i], network.OutputNodes[i+1:]...)
				break
			}
		}
	}
	return nil
}

// NewNeuralEdge creates an edge linked between nodes given as arguments, the from and the to.
// The gene must be enabled to create a valid evaluatable network.
// This edge is not valid until it is added to the NeuralNetwork that created it.
// You might want to use AddNewNeuralEdge() instead.
func (network *NeuralNetwork) NewNeuralEdge(from, to graph.Node, gene *LinkGene) *NeuralEdge {
	if gene == nil {
		// log.Println("The gene cannot be nil:", "while adding a neural edge", "from", from, "to", to) // debug //
		panic("The gene cannot be nil.")
	}
	if !gene.Enabled {
		// log.Println("You cannot add a disabled gene to this network.") // debug //
		panic("You cannot add a disabled gene to this network.")
	}
	return &NeuralEdge{
		Edge: network.NewEdge(from, to),
		Gene: gene,
	}
}

// AddNewNeuralEdge creates a neural edge and adds it to this network.
// It replaces this network's previous edge connected from from to to if there is any.
// Returns an error if the IDs of the from and to are equal.
func (network *NeuralNetwork) AddNewNeuralEdge(from, to graph.Node, gene *LinkGene) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = errors.New(fmt.Sprint(r.(string), " from:", from, " to:", to, " gene:", gene))
		}
	}()
	edge := network.NewNeuralEdge(from, to, gene)
	network.SetEdge(edge)
	network.EdgeByGene[gene] = edge
	return err
}

// NeuralEdge returns the neural edge from from to to. The return can be nil.
func (network *NeuralNetwork) NeuralEdge(from, to graph.Node) *NeuralEdge {
	ret := network.Edge(from.ID(), to.ID())
	if ret == nil {
		return nil
	}
	return ret.(*NeuralEdge)
}

// NeuralEdges returns all neural edges of this neural network.
//
// Usage:
//
//  neuralEdges := network.NeuralEdges()
//  for neuralEdges.Next() {
//  	neuralEdge := neuralEdges.NeuralEdge()
//  	// do whatever
//  }
//
func (network *NeuralNetwork) NeuralEdges() *NeuralEdges {
	ret := network.Edges()
	if ret == nil {
		return nil
	}
	return &NeuralEdges{ret}
}

// NeuralNodesTo returns all nodes in this neural network that can directly reach to the to node.
// The returned nodes are those that our dendrites are receiving signals from.
//
// Usage:
//
//  neuralNodes := network.NeuralNodesTo(to)
//  for neuralNodes.Next() {
//  	neuralNode := neuralNodes.NeuralNode()
//  	// do whatever
//  }
//
func (network *NeuralNetwork) NeuralNodesTo(to graph.Node) *NeuralNodes {
	return &NeuralNodes{network.To(to.ID())}
}

// NeuralNodesFrom of this neural network returns all nodes that can be reached directly from the from node.
// The returned nodes are what our axons linked to.
//
// Usage:
//
//  neuralNodes := network.NeuralNodesFrom(from)
//  for neuralNodes.Next() {
//  	neuralNode := neuralNodes.NeuralNode()
//  	// do whatever
//  }
//
func (network *NeuralNetwork) NeuralNodesFrom(from graph.Node) *NeuralNodes {
	return &NeuralNodes{network.From(from.ID())}
}

// NeuralNodes is a NeuralNode iterator.
//
// Usage:
//
//  for neuralNodes.Next() {
//  	neuralNode := neuralNodes.NeuralNode()
//  	// do whatever
//  }
//
type NeuralNodes struct {
	graph.Nodes
}

// NeuralNode returns the NeuralNode of this iterator.
//
// Usage:
//
//  for neuralNodes.Next() {
//  	neuralNode := neuralNodes.NeuralNode()
//  	// do whatever
//  }
//
func (nodes *NeuralNodes) NeuralNode() *NeuralNode {
	return nodes.Node().(*NeuralNode)
}

// ToSlice of this NeuralNodes.
func (nodes *NeuralNodes) ToSlice() []*NeuralNode {
	ret, i := make([]*NeuralNode, nodes.Len()), 0
	for nodes.Next() {
		ret[i] = nodes.NeuralNode()
		i++
	}
	return ret
}

// String callback of this NeuralNodes.
func (nodes *NeuralNodes) String() string {
	return fmt.Sprint(nodes.ToSlice())
}

// NeuralEdges is a NeuralEdge iterator.
//
// Usage:
//
//  for neuralEdges.Next() {
//  	neuralEdge := neuralEdges.NeuralEdge()
//  	// do whatever
//  }
//
type NeuralEdges struct {
	graph.Edges
}

// NeuralEdge returns the NeuralEdge of this iterator.
//
// Usage:
//
//  for neuralEdges.Next() {
//  	neuralEdge := neuralEdges.NeuralEdge()
//  	// do whatever
//  }
//
func (edges *NeuralEdges) NeuralEdge() *NeuralEdge {
	return edges.Edge().(*NeuralEdge)
}

// ToSlice of this NeuralEdges.
func (edges *NeuralEdges) ToSlice() []*NeuralEdge {
	ret, i := make([]*NeuralEdge, edges.Len()), 0
	for edges.Next() {
		ret[i] = edges.NeuralEdge()
		i++
	}
	return ret
}

// String callback of this NeuralEdges.
func (edges *NeuralEdges) String() string {
	return fmt.Sprint(edges.ToSlice())
}

// NeuralNode of an individual living organism.
// Use (*NeuralNetwork).NewNeuralNode() to create a new instance.
// NeuralNode is also considered graph.Node.
type NeuralNode struct {
	Identifier int64 // What identifies this node, given by the network this node belongs to. This ID is only unique within a graph.
	*NodeGene        // Genetics it inherits from.
	// Use only for the memoization of dynamic programming.
	Value NeuralValue // The value this single node holds(outputs).
	// The Nth layer of the neural network this node is located in.
	// N, where N is an integer >= 0, which denotes how the layer is closer to the output layer.
	// E.g. LayerLevel 0 points to the input layer of the neural network.
	LayerLevel int
	// This is set false by default and always true for outputs nodes recognized by a neural network,
	// otherwise it tells if this input/hidden node's synaptic influence can reach to the output layer or not.
	IsValidBackward bool
	// This is set false by default and always true for inputs nodes recognized by a neural network,
	// otherwise it tells if this output/hidden node's synaptic influence can reach to the input layer or not.
	IsValidForward bool
}

// ID of NeuralNode implements the graph.Node interface,
// so that the node can be added to a simple.DirectedGraph with its unique identifier.
func (neuralNode *NeuralNode) ID() int64 {
	return neuralNode.Identifier
}

// Gene of this NeuralNode.
func (neuralNode *NeuralNode) Gene() *NodeGene {
	return neuralNode.NodeGene
}

// String callback of NeuralNode.
func (neuralNode *NeuralNode) String() string {
	return fmt.Sprint(
		"{",
		"NeuralNode", "/",
		"ID:", neuralNode.Identifier, "/",
		"Val:", neuralNode.Value, "/",
		"LayerLevel:", neuralNode.LayerLevel, "/",
		"PathToOutputs:", neuralNode.IsValidBackward, "/",
		"PathToInputs:", neuralNode.IsValidForward, "/",
		"Gene:", neuralNode.NodeGene.String(),
		"}",
	)
}

// ClearAll resets all memos of this.
func (neuralNode *NeuralNode) ClearAll() {
	neuralNode.ClearValue()
	neuralNode.ClearLevel()
}

// ClearValue resets the value of this.
func (neuralNode *NeuralNode) ClearValue() {
	neuralNode.Value = NeuralValue(math.NaN())
}

// ClearLevel resets the layer level of this.
func (neuralNode *NeuralNode) ClearLevel() {
	neuralNode.LayerLevel = LayerLevelInit
}

// Evaluate the value of NeuralNode.
// The return gets stored (memoized) in this;self;receiver as well.
func (neuralNode *NeuralNode) Evaluate(input NeuralValue) (output NeuralValue) {
	output = Activation(input)
	neuralNode.Value = output
	return output
}

// EvaluateWithoutActivation of NeuralNode.
// The return gets stored (memoized) in this;self;receiver as well.
func (neuralNode *NeuralNode) EvaluateWithoutActivation(input NeuralValue) (output NeuralValue) {
	// output = input // without filter
	output = Ramp(input) // with filter making the neural input always valid
	neuralNode.Value = output
	return output
}

// SetLevel memoizes the Nth layer this node is located in its network.
func (neuralNode *NeuralNode) SetLevel(layerLevel int) {
	neuralNode.LayerLevel = layerLevel
}

// IsValid tells whether this node has been validated in a neural network.
// Returns true if this node is an input/output node recognized by a neural network,
// or it at least has a path (direct or not) open to both the output layer and the input layer.
func (neuralNode *NeuralNode) IsValid() bool {
	switch neuralNode.TypeInInt() {
	case OutputNode:
		return neuralNode.IsValidBackward
	case HiddenNode:
		return neuralNode.IsValidForward && neuralNode.IsValidBackward
	case InputNodeBias:
		fallthrough
	case InputNodeNotBias:
		return neuralNode.IsValidForward
	}
	return false
}

// IsEvaluated tells whether this NeuralNode's value has been updated or not.
func (neuralNode *NeuralNode) IsEvaluated() bool {
	return neuralNode.Value.IsEvaluated()
}

// IsInactive is not the inversion of this node's IsEvaluated().
// This function tells if this node has been evaluated as inactive.
func (neuralNode *NeuralNode) IsInactive() bool {
	const inactive = NeuralValueMid // The kind of this value is probably told to be neutral.
	// And what's considered inactive is the value, not the kind of the value.
	return neuralNode.Value == inactive // tells [0, 0]
}

// Output returns the value this neuron's axon outputs.
// Returns error when the value is determined as uninitialized(invalid or not evaluated).
func (neuralNode *NeuralNode) Output() (value NeuralValue, err error) {
	if !neuralNode.IsEvaluated() {
		return NeuralValue(math.NaN()), errors.New("this node needs to be evaluated: " + neuralNode.String())
	}
	return neuralNode.Value, nil
}

// Level returns the Nth layer this node is located in its network.
// N is an integer >= 0, which denotes how the layer is closer to the output layer.
// E.g. Level 0 points to the input layer of the neural network.
// Returns error when the value is determined as uninitialized(invalid or not evaluated).
// The returned N could be bigger than the N that points to the output layer, which case is not considered an error.
func (neuralNode *NeuralNode) Level() (level int, err error) {
	if neuralNode.LayerLevel == LayerLevelInit {
		return int(math.NaN()), errors.New("uninitialized value: " + neuralNode.String())
	} else if neuralNode.LayerLevel < 0 {
		return int(math.NaN()), errors.New("invalid value: " + neuralNode.String())
	}
	return neuralNode.LayerLevel, nil
}

// NeuralEdge is a neural connection in the neural network.
type NeuralEdge struct {
	graph.Edge
	Gene *LinkGene
}

// ToGene is a getter returning a gene of this NeuralEdge.
func (neuralEdge *NeuralEdge) ToGene() *LinkGene {
	return neuralEdge.Gene
}

// String callback of NeuralEdge.
func (neuralEdge *NeuralEdge) String() string {
	return fmt.Sprint(
		"{",
		"NeuralEdge", "/",
		"NodeFrom:", neuralEdge.NeuralNodeFrom(), "/",
		"NodeTo:", neuralEdge.NeuralNodeTo(), "/",
		"Gene:", neuralEdge.Gene.String(),
		"}",
	)
}

// Weight of this synaptic link.
func (neuralEdge *NeuralEdge) Weight() float64 {
	return neuralEdge.Gene.Weight
}

// IsValid tells whether this edge is valid or not.
func (neuralEdge *NeuralEdge) IsValid() bool {
	return neuralEdge.Gene.Enabled
}

// NeuralNodeFrom this neural edge.
func (neuralEdge *NeuralEdge) NeuralNodeFrom() *NeuralNode {
	return neuralEdge.Edge.From().(*NeuralNode)
}

// NeuralNodeTo this neural edge.
func (neuralEdge *NeuralEdge) NeuralNodeTo() *NeuralNode {
	return neuralEdge.Edge.To().(*NeuralNode)
}

// ----------------------------------------------------------------------------
// DiceGene

// ChanceGene (an alias for DiceGene here) is a set of mutation probabilities for a Chromosome.
// This allows our searching algorithm (the genetic algorithm) to have different
// exploration-exploitation ability in different stage of the search process.
// Also in case such dynamic mutation rate is not preferred, ChanceGene provides a way that
// it can be disabled to have constant probabilities given by hyper parameters.
type ChanceGene struct {
	// It is strongly recommended to use getter methods to retrieve these fields if possible.
	// IsEnabled() and Unpack() are/should be the only getter methods the struct provides.
	// That design simply simplifies stuff and may prevent lots of errors in the future.
	Enabled       bool    // What tells either the local(dynamic) M-Rate enabled or not.
	AddNode       float64 // Add-node structural mutation chance in percentage. (local M-Rate)
	AddLink       float64 // Add-link between non-bias nodes structural mutation chance in percentage. (local M-Rate)
	AddBias       float64 // Add-link to bias structural mutation chance in percentage. (local M-Rate)
	PerturbWeight float64 // Synaptic weight mutation chance in percentage. (local M-Rate)
	NullifyWeight float64 // Mutation chance of synaptic weight becoming zero. (local M-Rate)
	TurnOn        float64 // Enable-gene mutation chance in percentage. (local M-Rate)
	TurnOff       float64 // Disable-gene mutation chance in percentage. (local M-Rate)
	Bump          float64 // Bump-enable mutation chance in percentage. (local M-Rate)
}

// NumKindChance must tell the number of M-Rate members in a ChanceGene. Unit-test with NumKindChance.
//
// The purpose of this is to force a kind of dependency test all over the code and prevent bugs there.
// So this works as a filter bottlenecking possible mistakes.
//
const NumKindChance = 8

// NewChanceGene is a constructor.
func NewChanceGene(enabled bool, addNode, addLink, addBias, perturbWeight, nullifyWeight, turnOn, turnOff, bump float64) *ChanceGene {
	// dependent on const NumKindChance (unit-test defines it)
	return &ChanceGene{
		Enabled:       enabled,
		AddNode:       addNode,
		AddBias:       addBias,
		AddLink:       addLink,
		PerturbWeight: perturbWeight,
		NullifyWeight: nullifyWeight,
		TurnOn:        turnOn,
		TurnOff:       turnOff,
		Bump:          bump,
	}
}

// NewChanceGeneEnabled is a constructor.
func NewChanceGeneEnabled(addNode, addLink, addBias, nullifyWeight, perturbWeight, turnOn, turnOff, bump float64) *ChanceGene {
	// purposely dependent on NewChanceGene() constructor
	return NewChanceGene(true, addNode, addLink, addBias, nullifyWeight, perturbWeight, turnOn, turnOff, bump)
}

// NewChanceGeneDisabled is a constructor.
func NewChanceGeneDisabled() *ChanceGene {
	return &ChanceGene{
		Enabled: false,
	}
}

// String callback of ChanceGene.
func (diceGene *ChanceGene) String() string {
	addNode, addLink, addBias, perturbWeight, nullifyWeight, turnOn, turnOff, bump := diceGene.Unpack()
	return fmt.Sprint(
		"{",
		"DiceGene", "/",
		"AddNode:", addNode, "/",
		"AddLink:", addLink, "/",
		"AddBias:", addBias, "/",
		"PerturbW.:", perturbWeight, "/",
		"NullifyW.:", nullifyWeight, "/",
		"TurnOn:", turnOn, "/",
		"TurnOff:", turnOff, "/",
		"Bump:", bump, "/",
		"}",
	)
}

// IsEnabled tells either the local(dynamic) M-Rate is enabled or not.
func (diceGene *ChanceGene) IsEnabled() bool {
	return diceGene.Enabled
}

// Unpack returns all the chance values (defined probabilities; rates) of this dice-gene,
// regardless of whether it's enabled or not.
//
// The purpose of this is to force a kind of dependency-test all over the code and prevent bugs there.
// So this works as a filter bottlenecking possible mistakes.
//
// I use this method whenever two or more of the members of this struct are needed to be read at a time,
// because in those places the way this function works reduced lots of errors.
//
func (diceGene *ChanceGene) Unpack() (addNode, addLink, addBias, perturbWeight, nullifyWeight, turnOn, turnOff, bump float64) {
	addNode = diceGene.AddNode
	addLink = diceGene.AddLink
	addBias = diceGene.AddBias
	perturbWeight = diceGene.PerturbWeight
	nullifyWeight = diceGene.NullifyWeight
	turnOn = diceGene.TurnOn
	turnOff = diceGene.TurnOff
	bump = diceGene.Bump
	return
}

// Perturb mutates this dice-gene's weight with a random, normally distributed Gaussian noise.
// The way this works is similar to that of the LinkGene's.
//
// Parameters:
//  - `mean` - What tells how bias this operation is.
//  - `deviation` - That defines how various a distribution is.
//
func (diceGene *ChanceGene) Perturb(mean, deviation float64) {
	addNode, addLink, addBias, perturbWeight, nullifyWeight, turnOn, turnOff, bump := diceGene.Unpack()
	squared := deviation * deviation
	diceGene.AddNode += Gaussian(mean, Gaussian(addNode*deviation, squared))
	diceGene.AddLink += Gaussian(mean, Gaussian(addLink*deviation, squared))
	diceGene.AddBias += Gaussian(mean, Gaussian(addBias*deviation, squared))
	diceGene.PerturbWeight += Gaussian(mean, Gaussian(perturbWeight*deviation, squared))
	diceGene.NullifyWeight += Gaussian(mean, Gaussian(nullifyWeight*deviation, squared))
	diceGene.TurnOn += Gaussian(mean, Gaussian(turnOn*deviation, squared))
	diceGene.TurnOff += Gaussian(mean, Gaussian(turnOff*deviation, squared))
	diceGene.Bump += Gaussian(mean, Gaussian(bump*deviation, squared))
}

// Copy is a constructor.
// A deep copy of this object is returned.
func (diceGene *ChanceGene) Copy() *ChanceGene {
	v := *diceGene
	return &v
}

// ----------------------------------------------------------------------------
// LinkGene

// Innovation is the historical marker known as `innovation number`.
// The innovation number is unique for each of structural innovations under a NEAT context.
// It basically is considered an integer starting from zero
// but could be used in other way depending on the context.
type Innovation int64

// LinkGene is a gene that encodes of something like synapses, axons and dendrites, etc.
// Abstracted as simple as possible.
type LinkGene struct {
	Topo struct { // ByVal. The topological structure of the link.
		Innov Innovation // The innovation number; an identifier of the topological innovation, not a link-gene.
		From  *NodeGene  // Where this link is from. (in-node)
		To    *NodeGene  // Where this link is connected to. (out-node)
	}
	Weight float64 // Synaptic weight in neural network.
	// The 'Enabled' flag in a link gene.
	// It only determines whether this link gene's trait as a neural edge is expressed in a neural network or not.
	// Besides the crossover, this becomes false only when the link is split in half. But the link gene is still valid then,
	// even though having this false disallows the link gene itself to be instantiated to the neural edge.
	Enabled bool // Kinda like the gene's dominant or recessive.
}

// NewLinkGene is a constructor.
func NewLinkGene(innov Innovation, from, to *NodeGene, weight float64, enabled bool) *LinkGene {
	return &LinkGene{
		Topo: struct {
			Innov Innovation
			From  *NodeGene
			To    *NodeGene
		}{
			Innov: innov,
			From:  from,
			To:    to,
		},
		Weight:  weight,
		Enabled: enabled,
	}
}

// Perturb mutates this link gene's weight with a random, normally distributed Gaussian noise.
// The size of this perturbation - regarding the standard deviation of a Gaussian distribution - is given by the parameter.
//
// Parameters:
//  - `mean` - What tells how bias this operation is.
//  - `deviation` - That defines how various a distribution is.
//
func (linkGene *LinkGene) Perturb(mean, deviation float64) {
	linkGene.Weight += Gaussian(mean, float64(NeuralValueWidth/2)*deviation)
}

// String callback of LinkGene.
func (linkGene *LinkGene) String() string {
	return fmt.Sprint(
		"{",
		"LinkGene", "/",
		"Innov.:", linkGene.Topo.Innov, "/",
		"From:", linkGene.Topo.From, "/",
		"To:", linkGene.Topo.To, "/",
		"Weight:", linkGene.Weight, "/",
		"Enabled:", linkGene.Enabled, "/",
		"}",
	)
}

// Copy is a constructor.
// A deep copy of this object is returned.
func (linkGene *LinkGene) Copy() *LinkGene {
	v := *linkGene
	return &v
}

// LinkGeneMarshal is JSON export of LinkGene.
// We use this struct because of the LinkGene's references. (members of pointer type)
type LinkGeneMarshal struct {
	Topo struct {
		Innov Innovation
		From  uuid.UUID
		To    uuid.UUID
	}
	Weight  float64
	Enabled bool
}

// ToLinkGene from LinkGeneMarshal.
// The parameter can be nil.
// When there's a field of reference type not provided with any actual reference,
// this function would allocate a dummy object of that UUID and populate it.
func (lgm *LinkGeneMarshal) ToLinkGene(referenceByUUID map[uuid.UUID]*NodeGene) *LinkGene {
	if referenceByUUID == nil {
		referenceByUUID = map[uuid.UUID]*NodeGene{}
	}
	ret := &LinkGene{
		Topo: struct {
			Innov Innovation
			From  *NodeGene
			To    *NodeGene
		}{
			Innov: lgm.Topo.Innov,
			From:  nil,
			To:    nil,
		},
		Weight:  lgm.Weight,
		Enabled: lgm.Enabled,
	}
	if lgm.Topo.From != uuid.Nil {
		if nodeGene := referenceByUUID[lgm.Topo.From]; nodeGene != nil {
			ret.Topo.From = nodeGene
		} else {
			ret.Topo.From = &NodeGene{
				UUID: lgm.Topo.From,
				Type: ExceptionalNode,
			}
		}
	}
	if lgm.Topo.To != uuid.Nil {
		if nodeGene := referenceByUUID[lgm.Topo.To]; nodeGene != nil {
			ret.Topo.To = nodeGene
		} else {
			ret.Topo.To = &NodeGene{
				UUID: lgm.Topo.To,
				Type: ExceptionalNode,
			}
		}
	}
	return ret
}

// MarshalJSON of json.Marshaler interface.
// References are lost marshaling into JSON but the UUIDs.
func (linkGene *LinkGene) MarshalJSON() ([]byte, error) {
	from := uuid.Nil
	if linkGene.Topo.From != nil {
		from = linkGene.Topo.From.UUID
	}
	to := uuid.Nil
	if linkGene.Topo.To != nil {
		to = linkGene.Topo.To.UUID
	}
	return json.Marshal(&LinkGeneMarshal{
		Topo: struct {
			Innov Innovation
			From  uuid.UUID
			To    uuid.UUID
		}{
			Innov: linkGene.Topo.Innov,
			From:  from,
			To:    to,
		},
		Weight:  linkGene.Weight,
		Enabled: linkGene.Enabled,
	})
}

// UnmarshalJSON of json.Unmarshaler interface.
// Notice that all references will point to dummy objects.
func (linkGene *LinkGene) UnmarshalJSON(jsonLinkGeneMarshal []byte) (err error) {
	lgm := &LinkGeneMarshal{}
	if err = json.Unmarshal(jsonLinkGeneMarshal, lgm); err != nil {
		return err
	}
	*linkGene = *lgm.ToLinkGene(nil)
	return nil
}

// ----------------------------------------------------------------------------
// NodeGene

// NodeGene is not a node what's in the NN.
// This just encodes information about how a possible instantiated node might look like.
type NodeGene struct {
	UUID uuid.UUID // Having this member might help identifying this specific gene.
	Name string    // An optional label; nickname for this.
	Type int       // An index. InputNode, HiddenNode, or OutputNode. There are constants defined for this.
	Idx  int       // Optional. Determines the order of NodeGenes in a chromosome, followed by the Type.
}

// const InputNode, OutputNode, and HiddenNode for NodeGene.
const (
	NeglectedNode    = iota // invalid && unhandled (initial value of int set by default)
	InputNodeBias           // valid && handled
	InputNodeNotBias        // valid && handled
	OutputNode              // valid && handled
	HiddenNode              // valid && handled
	ExceptionalNode         // invalid && handled
)

// NewNodeGene is a constructor.
func NewNodeGene(name string, kind int, index int) *NodeGene {
	return &NodeGene{
		UUID: uuid.Must(uuid.NewV4()),
		Name: name,
		Type: kind,
		Idx:  index,
	}
}

// NewNodeGene is a constructor.
// What returned is a new identifiable object resembling this object.
func (nodeGene *NodeGene) NewNodeGene() *NodeGene {
	return &NodeGene{
		UUID: uuid.Must(uuid.NewV4()),
		Name: nodeGene.Name,
		Type: nodeGene.Type,
		Idx:  nodeGene.Idx,
	}
}

// TypeInInt returns the node type as an int.
// There are constants defined for this. See def const InputNode, OutputNode, and HiddenNode for NodeGene.
func (nodeGene *NodeGene) TypeInInt() int {
	return nodeGene.Type
}

// TypeInString returns the node type as a string.
func (nodeGene *NodeGene) TypeInString() string {
	switch nodeGene.Type {
	case NeglectedNode:
		return "NeglectedNode"
	case InputNodeBias:
		return "InputNodeBias"
	case InputNodeNotBias:
		return "InputNodeNotBias"
	case OutputNode:
		return "OutputNode"
	case HiddenNode:
		return "HiddenNode"
	case ExceptionalNode:
		return "ExceptionalNode"
	}
	return "Unknown invalid node type"
}

// String callback of NodeGene.
func (nodeGene *NodeGene) String() string {
	return fmt.Sprint(
		"{",
		"NodeGene", "/",
		// nodeGene.UUID, // excluded
		"Alias:", nodeGene.Name, "/",
		"Type:", nodeGene.TypeInString(), "/",
		"Idx:", nodeGene.Idx,
		"}",
	)
}

// IsValid validates this NodeGene.
func (nodeGene *NodeGene) IsValid() bool {
	// UUID is not valid.
	if nodeGene.UUID == uuid.Nil {
		return false
	}
	// Type is not valid.
	switch nodeGene.Type {
	case InputNodeBias:
	case InputNodeNotBias:
	case OutputNode:
	case HiddenNode:
	default:
		return false
	}
	// All valid.
	return true
}

// SetName alias for this. Replaces the previous one if it already has its name.
func (nodeGene *NodeGene) SetName(nickname string) {
	nodeGene.Name = nickname
}

// ----------------------------------------------------------------------------
// Trivials

// And gate of booleans.
func And(bools ...bool) bool {
	for _, b := range bools {
		if !b {
			return false
		}
	}
	return true
}

// Sum takes multiple numbers as its parameters then returns the sum of those. (Duh.)
func Sum(nums ...float64) float64 {
	ret := 0.0
	for _, num := range nums {
		ret += num
	}
	return ret
}

// Avg takes multiple numbers as its parameters then returns the average of those.
func Avg(nums ...float64) float64 {
	if len(nums) <= 0 {
		panic("no argument passed")
	}
	return Sum(nums...) / float64(len(nums))
}

// GCD returns the greatest common divisor of all given integers.
func GCD(numbers ...int) int {
	// gcd returns the greatest common divisor of two given integers.
	gcd := func(a, b int) int {
		for b > 0 {
			a, b = b, a%b
		}
		return a
	}
	// iterate over
	ret := 0
	for _, n := range numbers {
		ret = gcd(ret, n)
	}
	return ret
}

// Roulette accepts weights of a ratio as its parameters,
// and returns a weighted random index number in [0, len(ratio)).
func Roulette(ratio ...float64) (iPick int) {
	n := rand.Float64() * Sum(ratio...)
	for i, scale := range ratio {
		if n < scale {
			return i
		}
		n -= scale
	}
	// log.Println("fatal: roulette error") // debug //
	panic("roulette error")
	// return // unreachable
}

// Gaussian returns a random number N in a given normal distribution.
// Don't forget the good old `N(0, 1)` which gets us the standard normal distribution.
func Gaussian(mean, deviation float64) float64 {
	return mean + (rand.NormFloat64() * deviation)
}

// RelativeDifferenceNormalized returns a value in [0, 1]; a percentage that tells how different two given numbers are.
//
// Idea: https://en.wikipedia.org/wiki/Relative_change_and_difference
// http://mathcentral.uregina.ca/QQ/database/QQ.09.06/s/carolyn1.html
// It tells us to divide the absolute distance by the average of the two given numbers.
// (The formula of this function is of my own - I just felt like this would work.
// `Divide by Sum(a, b)` makes more sense to me than `divide by Max(a, b)` or `divide by Avg(a, b)`
// because `Divide by Sum(a, b)` limits the range of its outcome to [0, 1] and the others are not.)
//
func RelativeDifferenceNormalized(a, b float64) float64 {
	if a == 0 && b == 0 { // handles divide by zero
		return 0
	}
	if a*b < 0 {
		offset := math.Abs(math.Min(a, b))
		a += offset
		b += offset
	}
	return math.Abs(a-b) / math.Abs(Sum(a, b)) // return dist/scale
}

// ----------------------------------------------------------------------------
// NRGBA

// NRGBA extends `color.NRGBA`.
// NRGBA is a pixel converter converting `color.NRGBA`.
// Get this from `PixelConverter()`.
type NRGBA color.NRGBA

// PixelConverter is a factory and a utility for processing inputs.
func PixelConverter(c color.NRGBA) NRGBA {
	return NRGBA(c)
}

// ToGrayscaleFloat converts an NRGBA to a plain grayscale value. The alpha is ignored.
func (c NRGBA) ToGrayscaleFloat() float64 {
	return (float64(c.R) + float64(c.G) + float64(c.B)) / 3.0
}

// ToGrayscaleColor converts an NRGBA to a struct Gray.
func (c NRGBA) ToGrayscaleColor() color.Gray {
	return color.Gray{uint8((float64(c.R) + float64(c.G) + float64(c.B)) / 3.0)}
}

// ToGrayscaleNRGBA converts a colorful NRGBA to a grayscale NRGBA pixel. The alpha is preserved.
func (c NRGBA) ToGrayscaleNRGBA() color.NRGBA {
	mean := c.ToGrayscaleFloat()
	return color.NRGBA{
		R: uint8(mean),
		G: uint8(mean),
		B: uint8(mean),
		A: c.A,
	}
}

// ToRGBA converts an NRGBA to a premultiplied RGBA pixel.
func (c NRGBA) ToRGBA() color.RGBA {
	r, g, b, a := color.NRGBA(c).RGBA()
	return color.RGBA{
		R: uint8(r & 0xFF),
		G: uint8(g & 0xFF),
		B: uint8(b & 0xFF),
		A: uint8(a & 0xFF),
	}
}

// ToGrayscaleRGBA converts an NRGBA to a premultiplied grayscale RGBA pixel.
func (c NRGBA) ToGrayscaleRGBA() color.RGBA {
	gray := c.ToGrayscaleColor()
	r, g, b, a := gray.RGBA()
	return color.RGBA{
		R: uint8(r & 0xFF),
		G: uint8(g & 0xFF),
		B: uint8(b & 0xFF),
		A: uint8(a & 0xFF),
	}
}

// ToNeuralValue converts an NRGBA to a grayscale naneat input.
func (c NRGBA) ToNeuralValue() NeuralValue {
	if c.A == 0 { // completely transparent
		return 0
	}
	return NeuralValueFromUint8(c.ToGrayscaleColor().Y)
}

// ----------------------------------------------------------------------------
// static class ImageConverter

// ImageConverter converts images.
type ImageConverter struct {
	// Nothing. Only for namespace.
}

// NewImageConverter is a constructor.
func _NewImageConverter() *ImageConverter {
	return &ImageConverter{}
}

// private (read-only)
var imgCvt = _NewImageConverter()

// ImageMan is a getter and what returned from it is considered a public singleton object.
func ImageMan() ImageConverter {
	return *imgCvt
}

// ----------------------------------------------------------------------------
// ImageConverter - from image.Image to linear

// FromImageToList combined with `(image.Image).At(x, y)` can convert an `image.Image` to a list(slice).
func (ImageConverter) FromImageToList(img image.Image, onGivenXY func(x, y int) interface{}) []interface{} {
	rect := img.Bounds()
	width := rect.Dx()
	height := rect.Dy()
	xFrom := rect.Min.X
	xTo := rect.Max.X
	yFrom := rect.Max.Y - 1
	yTo := rect.Min.Y - 1

	ret := make([]interface{}, width*height)
	for y := yFrom; y > yTo; y-- {
		for x := xFrom; x < xTo; x++ {
			ret[((yFrom-y)*width)+x] = onGivenXY(x, y)
		}
	}
	return ret
}

// FromImageToColors converts an image.Image to a color.Color list.
func (ImageConverter) FromImageToColors(img image.Image) []color.Color {
	rect := img.Bounds()
	width := rect.Dx()
	height := rect.Dy()
	xFrom := rect.Min.X
	xTo := rect.Max.X
	yFrom := rect.Max.Y - 1
	yTo := rect.Min.Y - 1

	ret := make([]color.Color, width*height)
	for y := yFrom; y > yTo; y-- {
		for x := xFrom; x < xTo; x++ {
			ret[((yFrom-y)*width)+x] = img.At(x, y)
		}
	}
	return ret
}

// FromImageNRGBAToRGBAs converts an image.NRGBA to a premultiplied color.RGBA list.
func (ImageConverter) FromImageNRGBAToRGBAs(img *image.NRGBA) []color.RGBA {
	rect := img.Bounds()
	width := rect.Dx()
	height := rect.Dy()
	xFrom := rect.Min.X
	xTo := rect.Max.X
	yFrom := rect.Max.Y - 1
	yTo := rect.Min.Y - 1

	ret := make([]color.RGBA, width*height)
	for y := yFrom; y > yTo; y-- {
		for x := xFrom; x < xTo; x++ {
			ret[((yFrom-y)*width)+x] = PixelConverter(img.NRGBAAt(x, y)).ToRGBA()
		}
	}
	return ret
}

// FromImageNRGBAToNeuralValues converts an image.NRGBA to a NeuralValue list.
// The return could be considered a list of inputs.
func (ImageConverter) FromImageNRGBAToNeuralValues(img *image.NRGBA) []NeuralValue {
	rect := img.Bounds()
	width := rect.Dx()
	height := rect.Dy()
	xFrom := rect.Min.X
	xTo := rect.Max.X
	yFrom := rect.Max.Y - 1
	yTo := rect.Min.Y - 1

	ret := make([]NeuralValue, width*height)
	for y := yFrom; y > yTo; y-- {
		for x := xFrom; x < xTo; x++ {
			ret[((yFrom-y)*width)+x] = PixelConverter(img.NRGBAAt(x, y)).ToNeuralValue()
		}
	}
	return ret
}

// FromImageGrayToNeuralValues converts an image.Gray to a NeuralNeNeuralValueuralValueInput list.
// The return could be considered a list of inputs.
func (ImageConverter) FromImageGrayToNeuralValues(img *image.Gray) []NeuralValue {
	rect := img.Bounds()
	width := rect.Dx()
	height := rect.Dy()
	xFrom := rect.Min.X
	xTo := rect.Max.X
	yFrom := rect.Max.Y - 1
	yTo := rect.Min.Y - 1

	ret := make([]NeuralValue, width*height)
	for y := yFrom; y > yTo; y-- {
		for x := xFrom; x < xTo; x++ {
			ret[((yFrom-y)*width)+x] = NeuralValueFromUint8(img.GrayAt(x, y).Y)
		}
	}
	return ret
}

// ----------------------------------------------------------------------------
// ImageConverter - from linear to image.Image

// FromColorsToImage converts a color.Color list to an image.NRGBA.
func (ImageConverter) FromColorsToImage(colors []color.Color, width, height int) *image.NRGBA {
	ret := image.NewNRGBA(image.Rect(0, 0, width, height))
	for i, color := range colors {
		col := i % width
		row := (height - 1) - (i-col)/width
		ret.Set(col, row, color)
	}
	return ret
}

// FromNRGBAsToImage converts a color.NRGBA list to an image.NRGBA.
func (ImageConverter) FromNRGBAsToImage(colors []color.NRGBA, width, height int) *image.NRGBA {
	ret := image.NewNRGBA(image.Rect(0, 0, width, height))
	for i, color := range colors {
		col := i % width
		row := (height - 1) - (i-col)/width
		ret.SetNRGBA(col, row, color)
	}
	return ret
}

// FromGraysToImage converts a color.Gray list to an image.Gray.
func (ImageConverter) FromGraysToImage(colors []color.Gray, width, height int) *image.Gray {
	ret := image.NewGray(image.Rect(0, 0, width, height))
	for i, color := range colors {
		col := i % width
		row := (height - 1) - (i-col)/width
		ret.SetGray(col, row, color)
	}
	return ret
}
