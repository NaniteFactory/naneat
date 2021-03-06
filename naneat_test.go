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

package naneat // white box

import (
	"errors"
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"testing"
)

// ----------------------------------------------------------------------------
// benchmark basic type casting of neural value

func BenchmarkNeuralValue(b *testing.B) {
	for i := 0.0; i < float64(b.N); i++ {
		func(NeuralValue) {}(NeuralValue(i))
	}
}

func BenchmarkNV(b *testing.B) {
	for i := 0.0; i < float64(b.N); i++ {
		func(NeuralValue) {}(NV(i))
	}
}

func BenchmarkNVFloat64TypeCast(b *testing.B) {
	for i := 0.0; i < float64(b.N); i++ {
		func(float64) {}(float64(NeuralValueMax))
	}
}

func BenchmarkNVFloat64Method(b *testing.B) {
	for i := 0.0; i < float64(b.N); i++ {
		func(float64) {}(NeuralValueMax.Float64())
	}
}

// ----------------------------------------------------------------------------
// activation function test

func TestNeuralValue(t *testing.T) {
	for x1 := NeuralValueMid.Float64() + 1; x1 < NeuralValueMax.Float64()+1; x1++ { // true (positive)
		if kind := NV(x1).Kind(); kind != NVKindPositive {
			t.Error("nv.Kind() error")
			t.Error(kind)
		}
	}
	for x2 := NeuralValueMid.Float64() - 1; x2 > NeuralValueMin.Float64()-1; x2-- { // false (negative)
		if kind := NV(x2).Kind(); kind != NVKindNegative {
			t.Error("nv.Kind() error")
			t.Error(kind)
		}
	}
}

func testNeuralValue(t *testing.T, nv NeuralValue) {
	if v := nv.Kind(); v == NVKindExceptional {
		t.Error("nv.Kind() error")
		t.Error(nv, v, "NVKindExceptional")
	}
	if v, err := nv.Concentration(); err != nil {
		t.Error("nv.ToConcentration() nv.ToUint8() error")
		t.Error(nv, v, err)
	}
	if v, err := nv.Strength(); err != nil {
		t.Error("nv.ToStrength() error")
		t.Error(nv, v, err)
	}
}

func TestActivationInputInt(t *testing.T) {
	for i := -10000; i < 10000; i++ {
		nv := Activation(NeuralValue(i))
		testNeuralValue(t, nv)
	}
}

func TestActivationInputUint8(t *testing.T) {
	for i := 0; i <= 0xFF; i++ {
		nv := Activation(NVU(uint8(i)))
		testNeuralValue(t, nv)
	}
}

func TestActivationInputColor(t *testing.T) {
	for r := 0; r <= 0xFF; r++ {
		for g := 0; g <= 0xFF; g++ {
			for b := 0; b <= 0xFF; b++ {
				for a := 1; a <= 0xFF; a += 127 {
					nv := Activation(PixelConverter(color.NRGBA{uint8(r), uint8(g), uint8(b), uint8(a)}).ToNeuralValue())
					testNeuralValue(t, nv)
				}
			}
		}
	}
}

// ----------------------------------------------------------------------------
// test neural network's layer level evaluat'ed'

// tests if all output nodes are in one same layer
func testNetworkOutputNodesLayerLevel(t *testing.T, network *NeuralNetwork) {
	// Validate the level of output nodes.
	maxLevel := LayerLevelInit
	for i, outputNode := range network.OutputNodes {
		levelValid, err := outputNode.Level()
		if err != nil {
			t.Error("error while validating the layer levels of output nodes")
			t.Error("outputNode.Level()")
			t.Error(err)
		}
		if maxLevel == LayerLevelInit {
			maxLevel = levelValid
			// Agnostic. Neither true nor false.
			continue
		}
		if maxLevel != levelValid {
			t.Error("the layer level is invalid for [" + string(i) + "]th output node")
		}
	}
}

// ----------------------------------------------------------------------------
// test net constructors

func TestNewChromosomeNewNetwork(t *testing.T) {
	// nodes genes
	node1 := NewNodeGene("inp1", InputNodeNotBias, 1)
	node2 := NewNodeGene("inp2", InputNodeNotBias, 2)
	node3 := NewNodeGene("inp3", InputNodeNotBias, 3)
	node4 := NewNodeGene("hid1", HiddenNode, 4)
	node5 := NewNodeGene("hid2", HiddenNode, 5)
	node6 := NewNodeGene("out1", OutputNode, 6)
	// nodes genes list
	nodeGenes1 := []*NodeGene{node1, node2, node3, node4, node5, node6}
	nodeGenes2 := []*NodeGene{node1, node5, node6, node2, node4, node3}

	// links
	link1 := NewLinkGene(Innovation(1), node1, node4, 0, true)
	link2 := NewLinkGene(Innovation(2), node2, node4, 0, true)
	link3 := NewLinkGene(Innovation(3), node3, node6, 700, true)
	link4 := NewLinkGene(Innovation(4), node4, node5, 160, true)
	link5 := NewLinkGene(Innovation(5), node5, node6, 100, true)
	links := []*LinkGene{link1, link2, link3, link4, link5}

	// mutes genes
	mute0 := NewChanceGeneDisabled()

	// factory of neural network
	factory1, err := NewChromosome(links, nodeGenes1, mute0) // param ordered
	if err != nil {
		t.Error("factory1, err := NewChromosome()")
		t.Error(err)
	}
	factory2, err := NewChromosome(links, nodeGenes2, mute0) // param unordered
	if err != nil {
		t.Error("factory2, err := NewChromosome()")
		t.Error(err)
	}

	// test chromes
	if !factory1.IsValid() {
		t.Error("!factory1.IsValid()")
		t.Error("bad chromosome")
	}
	if !factory2.IsValid() {
		t.Error("!factory2.IsValid()")
		t.Error("bad chromosome")
	}

	// new net
	nn1, err := factory1.NewNeuralNetwork()
	if err != nil {
		t.Error("nn1, err := factory1.NewNeuralNetwork()")
		t.Error(err)
	}
	nn2, err := factory2.NewNeuralNetwork()
	if err != nil {
		t.Error("nn2, err := factory2.NewNeuralNetwork()")
		t.Error(err)
	}

	// topo sort of net's nodes
	nodes1, err := nn1.Sort()
	if err != nil {
		t.Error("node1, err := nn1.Sort()")
		t.Error(err)
	}
	nodes2, err := nn2.Sort()
	if err != nil {
		t.Error("node2, err := nn2.Sort()")
		t.Error(err)
	}

	// validate layer level
	testNetworkOutputNodesLayerLevel(t, nn1)
	testNetworkOutputNodesLayerLevel(t, nn2)

	// comparison

	// topo nodes' len
	if len(nodes1) != len(nodes2) {
		t.Error("The topo sort of a net is wrong.")
		t.Error(nodes1)
		t.Error(nodes2)
		t.Error(len(nodes1), len(nodes2))
	}

	// must be same gene in same order
	for i, node1 := range nodes1 {
		if node1.NodeGene != nodes2[i].NodeGene {
			t.Error("Bad chromosome or network.")
			t.Error(factory1)
			t.Error(factory2)
			t.Error(nodes1)
			t.Error(nodes2)
		}
	}
	for i, node2 := range nodes2 {
		if node2.NodeGene != nodes1[i].NodeGene {
			t.Error("Bad chromosome or network.")
			t.Error(factory1)
			t.Error(factory2)
			t.Error(nodes1)
			t.Error(nodes2)
		}
	}
} // end of test

// NewTestNetFactoryBasic returns a factory that returns a net with only 3 inputs.
func NewTestNetFactoryBasic() (netFactory *Chromosome, err error) {
	// nodes genes
	node1 := NewNodeGene("inp1", InputNodeNotBias, 1)
	node2 := NewNodeGene("inp2", InputNodeNotBias, 2)
	node3 := NewNodeGene("inp3", InputNodeNotBias, 3)
	node4 := NewNodeGene("hid1", HiddenNode, 4)
	node5 := NewNodeGene("hid2", HiddenNode, 5)
	node6 := NewNodeGene("out1", OutputNode, 6)
	nodes := []*NodeGene{node1, node5, node6, node2, node4, node3}

	// links genes
	link1 := NewLinkGene(Innovation(1), node1, node4, 0, true)
	link2 := NewLinkGene(Innovation(2), node2, node4, 0, true)
	link3 := NewLinkGene(Innovation(3), node3, node6, 700, true)
	link4 := NewLinkGene(Innovation(4), node4, node5, 160, true)
	link5 := NewLinkGene(Innovation(5), node5, node6, 100, true)
	links := []*LinkGene{link1, link2, link3, link4, link5}

	// mutes genes
	mute0 := NewChanceGeneDisabled()

	// factory of neural network
	return NewChromosome(links, nodes, mute0)
}

// must (nInputs >= 1)
func NewTestNetFactoryMultipleInputs(nInputs int) (netFactory *Chromosome, err error) {
	// nodes genes
	hid1 := NewNodeGene("hid1", HiddenNode, 4)
	hid2 := NewNodeGene("hid2", HiddenNode, 5)
	out1 := NewNodeGene("out1", OutputNode, 6)
	nodes := []*NodeGene{hid1, hid2, out1}

	// links genes
	link1 := NewLinkGene(Innovation(1), hid1, hid2, 255*rand.Float64(), true)
	link2 := NewLinkGene(Innovation(2), hid2, out1, 255*rand.Float64(), true)
	links := []*LinkGene{link1, link2}

	// additional inputs
	plusNodes := []*NodeGene{}
	plusLinks := []*LinkGene{}
	for i := 0; i < nInputs; i++ {
		plusNodes = append(plusNodes, NewNodeGene("inp"+fmt.Sprint(i+1), InputNodeNotBias, i+1))
		plusLinks = append(plusLinks, NewLinkGene(
			Innovation(i+3 /*because there are two already*/),
			plusNodes[i], out1, 255*rand.Float64(), true),
		)
		// log.Println(plusNodes[i].Name, "->", out1) //
	}

	// merge
	links = append(links, NewLinkGene(Innovation(0), plusNodes[0], hid1, 255*rand.Float64(), true))
	links = append(links, plusLinks...)
	nodes = append(nodes, plusNodes...)
	// log.Println("NewTestNetFactoryMultipleInputs(): ", links) //
	// log.Println("NewTestNetFactoryMultipleInputs(): ", nodes) //

	// mutes genes
	mute0 := NewChanceGeneDisabled()

	// factory of neural network
	return NewChromosome(links, nodes, mute0)
}

func NewTestNetFactoryForest() (netFactory *Chromosome, err error) {
	// nodes genes
	node1 := NewNodeGene("inp1", InputNodeNotBias, 1)
	node2 := NewNodeGene("inp2", InputNodeNotBias, 2)
	node3 := NewNodeGene("inp3", InputNodeNotBias, 3)
	// node4 := NewNodeGene("hid1", HiddenNode, 4)
	// node5 := NewNodeGene("hid2", HiddenNode, 5)
	node6 := NewNodeGene("out1", OutputNode, 6)
	node7 := NewNodeGene("out2", OutputNode, 7)
	nodes := []*NodeGene{node1, node6, node2, node3, node7}

	// links genes
	link1 := NewLinkGene(Innovation(1), node1, node6, 0, true)
	link2 := NewLinkGene(Innovation(2), node2, node7, 0, true)
	links := []*LinkGene{link1, link2}

	// mutes genes
	mute0 := NewChanceGeneDisabled()

	// factory of neural network
	return NewChromosome(links, nodes, mute0)
}

func TestNewNet(t *testing.T) {
	// dumb bot for testing
	factory1, err := NewTestNetFactoryMultipleInputs(1000)
	if err != nil {
		t.Error("NewTestNetFactoryMultipleInputs(1000)")
		t.Error(err)
	}
	factory2, err := NewTestNetFactoryBasic()
	if err != nil {
		t.Error("NewTestNetFactoryBasic()")
		t.Error(err)
	}
	factory3, err := NewTestNetFactoryForest()
	if err != nil {
		t.Error("NewTestNetFactoryForest()")
		t.Error(err)
	}
	testNetLeft, err := factory1.NewNeuralNetwork()
	if err != nil {
		t.Error("factory1.NewNeuralNetwork()")
		t.Error(testNetLeft)
		t.Error(err)
	}
	testNetRight, err := factory2.NewNeuralNetwork()
	if err != nil {
		t.Error("factory2.NewNeuralNetwork()")
		t.Error(testNetRight)
		t.Error(err)
	}
	testNetThird, err := factory3.NewNeuralNetwork()
	if err != nil {
		t.Error("factory3.NewNeuralNetwork()")
		t.Error(testNetThird)
		t.Error(err)
	}
	testNetLeftNodes, err := testNetLeft.Sort()
	if err != nil {
		t.Error("testNetLeft.Sort()")
		t.Error(testNetLeftNodes)
		t.Error(err)
	}
	testNetRightNodes, err := testNetRight.Sort()
	if err != nil {
		t.Error("testNetRight.Sort()")
		t.Error(testNetRightNodes)
		t.Error(err)
	}
	testNetThirdNodes, err := testNetThird.Sort()
	if err != nil {
		t.Error("testNetThird.Sort()")
		t.Error(testNetThirdNodes)
		t.Error(err)
	}
	// t.Error(testNetThirdNodes) //
	testNetworkOutputNodesLayerLevel(t, testNetLeft)
	testNetworkOutputNodesLayerLevel(t, testNetRight)
	testNetworkOutputNodesLayerLevel(t, testNetThird)
}

// ----------------------------------------------------------------------------
// neural network methods test

// ClearValues of all neural nodes of this NeuralNetwork.
// This function returns the topological sort of this graph.
func testClearValues(t *testing.T, network *NeuralNetwork) (nodes []*NeuralNode, err error) {
	nodes, err = network.Sort()
	if err != nil {
		return nil, err
	}
	for _, node := range nodes {
		node.ClearValue()
	}
	return nodes, nil
}

// ClearLevels of all neural nodes of this NeuralNetwork.
// This function returns the topological sort of this graph.
func testClearLevels(t *testing.T, network *NeuralNetwork) (nodes []*NeuralNode, err error) {
	nodes, err = network.Sort()
	if err != nil {
		return nil, err
	}
	for _, node := range nodes {
		node.ClearLevel()
	}
	return nodes, nil
}

// ClearAll of all neural nodes of this NeuralNetwork.
// This function returns the topological sort of this graph.
func testClearAll(t *testing.T, network *NeuralNetwork) (nodes []*NeuralNode, err error) {
	nodes, err = network.Sort()
	if err != nil {
		return nil, err
	}
	for _, node := range nodes {
		node.ClearAll()
	}
	return nodes, nil
}

func TestClearNodeMemos(t *testing.T) {
	factory, err := NewTestNetFactoryMultipleInputs(100)
	if err != nil {
		t.Error("NewTestNetFactoryMultipleInputs()")
		t.Error(err)
	}
	{ // testClearAll
		net, err := factory.NewNeuralNetwork()
		if err != nil {
			t.Error("testClearAll")
			t.Error(net, err)
		}
		nodes, err := testClearAll(t, net)
		if err != nil {
			t.Error("testClearAll")
			t.Error(nodes, err)
		}
	}
	{ // testClearLevels
		net, err := factory.NewNeuralNetwork()
		if err != nil {
			t.Error("testClearLevels")
			t.Error(net, err)
		}
		nodes, err := testClearLevels(t, net)
		if err != nil {
			t.Error("testClearLevels")
			t.Error(nodes, err)
		}
	}
	{ // testClearValues
		net, err := factory.NewNeuralNetwork()
		if err != nil {
			t.Error("testClearValues")
			t.Error(net, err)
		}
		nodes, err := testClearValues(t, net)
		if err != nil {
			t.Error("testClearValues")
			t.Error(nodes, err)
		}
	}
}

// ----------------------------------------------------------------------------
// test neural network's layer level evaluation

func testNetIsNeuralLevelEvaluated(t *testing.T, network *NeuralNetwork) {
	// Getter.
	nodes, err := network.Sort()
	if err != nil {
		t.Error("network.Sort() did not work well.")
		t.Error("Agnostic. IsNeuralLevelEvaluated neither true nor false for this getter fail.")
	}
	// All nodes in BFS-like order.
	for i, toNode := range nodes {
		_, err = toNode.Level()
		if err != nil {
			t.Error("toNode.Level()")
			t.Error("this ", i, "th node was not evaluated well: ", toNode)
			t.Error(err)
		}
	}
	testNetworkOutputNodesLayerLevel(t, network) // Validate (test) the level of output nodes.
}

func TestNetEvaluateNeuralLayerLevel(t *testing.T) {
	// using constructor
	factory, err := NewTestNetFactoryMultipleInputs(100)
	if err != nil {
		t.Error("NewTestNetFactoryMultipleInputs()")
		t.Error(err)
	}
	network, err := factory.NewNeuralNetwork()
	if err != nil {
		t.Error("factory.NewNeuralNetwork()")
		t.Error(network, err)
	}
	testNetIsNeuralLevelEvaluated(t, network)

	// clear and update manually
	testClearAll(t, network)
	err = network.updateLayerLevel()
	if err != nil {
		t.Error("network.updateLayerLevel()")
		t.Error(err)
	}
	testNetIsNeuralLevelEvaluated(t, network)
}

// ----------------------------------------------------------------------------
// test neural network's neural value evaluation

func testNetIsNeuralValueEvaluated(t *testing.T, network *NeuralNetwork) {
	// Getter.
	nodes, err := network.Sort()
	if err != nil {
		t.Error("network.Sort() did not work well.")
		t.Error("Agnostic. IsNeuralValueEvaluated neither true nor false for this getter fail.")
	}
	// All nodes in BFS-like order.
	for i, toNode := range nodes {
		_, err := toNode.Output()
		if err != nil {
			t.Error("toNode.Output()")
			t.Error("this ", i, "th node was not evaluated well: ", toNode)
			t.Error(err)
		}
	}
}

func TestNetEvaluateNeuralValue(t *testing.T) {
	// using constructor
	factory, err := NewTestNetFactoryBasic()
	if err != nil {
		t.Error("NewTestNetFactoryBasic()")
		t.Error(err)
	}
	network, err := factory.NewNeuralNetwork()
	if err != nil {
		t.Error("factory.NewNeuralNetwork()")
		t.Error(network, err)
	}
	{ // clear ff test 1
		// testClearValues(t, network)
		out, err := network.Evaluate([]NeuralValue{NVU(0), NVU(0xFF), NVU(25)})
		if err != nil {
			t.Error("network.Evaluate([]NeuralValue{NVU(0), NVU(0xFF), NVU(25)})")
			t.Error(out, err)
		}
		testNetIsNeuralValueEvaluated(t, network)
	}
	{ // clear ff test 2
		testClearValues(t, network)
		out, err := network.Evaluate([]NeuralValue{NV(0), NV(0.1), NV(0.5)})
		if err != nil {
			t.Error("network.Evaluate([]NeuralValue{NV(0), NV(0.1), NV(0.5))")
			t.Error(out, err)
		}
		testNetIsNeuralValueEvaluated(t, network)
	}
	{ // clear ff test 3
		testClearValues(t, network)
		out, err := network.Evaluate([]NeuralValue{NeuralValue(0), NeuralValue(0), NeuralValue(0)})
		if err != nil {
			t.Error("network.Evaluate([]NeuralValue{NeuralValue(0), NeuralValue(0), NeuralValue(0)})")
			t.Error(out, err)
		}
		testNetIsNeuralValueEvaluated(t, network)
	}
	{ // clear ff test 4
		testClearValues(t, network)
		out, err := network.Evaluate([]NeuralValue{0, NeuralValueMax + 1, 25, 5})
		if err == nil {
			t.Error("this is supposed to raise an error but it did not.")
			t.Error(out, err)
		}
		t.SkipNow()
		// testNetIsNeuralValueEvaluated(t, network) // this always generates errors.
	}
	{ // clear ff test 5
		testClearValues(t, network)
		out, err := network.Evaluate([]NeuralValue{0, NeuralValueMax + 2, 25})
		if err == nil {
			t.Error("this is supposed to raise an error but it did not.")
			t.Error(out, err)
		}
		// testNetIsNeuralValueEvaluated(t, network) // this always generates errors.
	}
	{ // clear ff test 6
		testClearValues(t, network)
		out, err := network.Evaluate([]NeuralValue{0, NeuralValueMin - 1, 25})
		if err == nil {
			t.Error("this is supposed to raise an error but it did not.")
			t.Error(out, err)
		}
		// testNetIsNeuralValueEvaluated(t, network) // this always generates errors.
	}
}

// ----------------------------------------------------------------------------
// test neural network's (neural value evaluation && layer level evaluation)

func testNetIsBothNeuralValueAndNeuralLevelEvaluated(t *testing.T, network *NeuralNetwork) {
	// Getter.
	nodes, err := network.Sort()
	if err != nil {
		t.Error("network.Sort() did not work well.")
		t.Error("Agnostic. IsBothNeuralValueAndNeuralLevelEvaluated neither true nor false for this getter fail.")
	}
	// All nodes in BFS-like order.
	for i, toNode := range nodes {
		_, err = toNode.Output()
		if err != nil {
			t.Error("toNode.Output()")
			t.Error("this ", i, "th node was not evaluated well: ", toNode)
			t.Error(err)
		} // Notice that it doesn't examine IsNeuralLevelEvaled after this.
		_, err = toNode.Level()
		if err != nil {
			t.Error("toNode.Level()")
			t.Error("this ", i, "th node was not evaluated well: ", toNode)
			t.Error(err)
		}
	}
	testNetIsNeuralLevelEvaluated(t, network)
}

// Evaluate two things at a time (in a single traverse/iter). A method of (*NeuralNetwork) implemented from scratch.
// You would not need any of this function's return, if your purpose is just testing only.
func testNetEvaluateBothNeuralValueAndNeuralLevel(t *testing.T, network *NeuralNetwork, inputs []NeuralValue) (
	outputs []NeuralNode, maxLevelUpperBound int, err error,
) {
	const levelBase = LayerLevelInit + 1 // 0 is returned for (maxLevelUpperBound int) when there is an error.

	// Validate the input.
	if len(inputs) != network.NumInputNodes-network.NumBiasNodes {
		t.Error("testNetEvaluateTwo()")
		t.Error("the number of inputs does not match to that of this neural network")
		return nil, levelBase, errors.New("the number of inputs does not match to that of this neural network")
	}

	// Init.
	network.IsEvaluatedWell = false
	nodes, err := testClearAll(t, network)
	if err != nil {
		t.Error("testNetEvaluateTwo()")
		t.Error(err)
		return nil, levelBase, err
	}

	// All input nodes are evaluated here.
	for _, node := range network.InputNodes[:network.NumBiasNodes] {
		node.EvaluateWithoutActivation(NeuralValueMax)
	}
	// Pass inputs.
	for i, node := range network.InputNodes[network.NumBiasNodes:] {
		// log.Println(node) //
		if !inputs[i].IsEvaluated() {
			t.Error(
				fmt.Sprint(
					"each input value should be in range",
					" [", NeuralValueMin, ", ", NeuralValueMax, "]",
					// It doesn't actually have to be.
					// Generating error like this is just to help the user know what he/she is doing.
				),
			)
			return nil, levelBase, errors.New(
				fmt.Sprint(
					"each input value should be in range",
					" [", NeuralValueMin, ", ", NeuralValueMax, "]",
					// It doesn't actually have to be.
					// Generating error like this is just to help the user know what he/she is doing.
				),
			)
		}
		node.EvaluateWithoutActivation(inputs[i])
		node.SetLevel(0)
		// log.Println(inputs[i], "->", node.Value) //
	}

	// FF in BFS-like order.
	for _, toNode := range nodes {
		// log.Println(toNode) //
		if toNode.IsEvaluated() { // When toNode is not an input node.
			continue
		}
		sum := 0.0                       // 1
		level := float64(LayerLevelInit) // 2
		{
			froms := network.NeuralNodesTo(toNode)
			for froms.Next() {
				fromNode := froms.NeuralNode()
				w := network.NeuralEdge(fromNode, toNode).Weight()
				v, err := fromNode.Output()
				if err != nil {
					t.Error(err)
					return nil, levelBase, err
				}
				sum += (v.Float64() * w) // 1
				candidate, err := fromNode.Level()
				if err != nil {
					t.Error(err)
					return nil, levelBase, err
				}
				level = math.Max(float64(candidate), float64(level)) // 2
			}
		}
		toNode.Evaluate(NeuralValue(sum)) // 1
		toNode.SetLevel(int(level + 1))   // 2
		// log.Println(sum, "->", toNode.Value) //
	}

	// The output nodes to be returned.
	outputs = make([]NeuralNode, len(network.OutputNodes))
	for i := range outputs {
		outputs[i] = *network.OutputNodes[i]
	}
	// Validate (test) these output nodes and return.
	maxLevel := LayerLevelInit
	for i, outputNode := range outputs {
		levelValid, err := outputNode.Level()
		if err != nil {
			t.Error(err)
			return nil, LayerLevelInit, err
		}
		if maxLevel == LayerLevelInit {
			maxLevel = levelValid
			continue
		}
		if maxLevel != levelValid {
			t.Error(err)
			return nil, LayerLevelInit, errors.New("the layer level is invalid for [" + string(i) + "]th output node")
		}
	}
	network.IsEvaluatedWell = true // back at it again.
	return outputs, maxLevel + 1, nil
}

func TestNetEvaluateBothNeuralValueAndNeuralLevel(t *testing.T) {
	// using constructor
	factory, err := NewTestNetFactoryBasic()
	if err != nil {
		t.Error("NewTestNetFactoryBasic()")
		t.Error(err)
	}
	network, err := factory.NewNeuralNetwork()
	if err != nil {
		t.Error("factory.NewNeuralNetwork()")
		t.Error(network, err)
	}
	// ff1
	testNetEvaluateBothNeuralValueAndNeuralLevel(t, network, []NeuralValue{0, NeuralValueMax, 25})
	testNetIsBothNeuralValueAndNeuralLevelEvaluated(t, network)
	// ff2
	testNetEvaluateBothNeuralValueAndNeuralLevel(t, network, []NeuralValue{0, 0.1, 0.5})
	testNetIsBothNeuralValueAndNeuralLevelEvaluated(t, network)
	// ff3
	testNetEvaluateBothNeuralValueAndNeuralLevel(t, network, []NeuralValue{0, 0, 0})
	testNetIsBothNeuralValueAndNeuralLevelEvaluated(t, network)
}

// ----------------------------------------------------------------------------
// benchmark NN evaluation

func BenchmarkEvaluate(b *testing.B) {
	b.StopTimer()
	// constructor
	factory, err := NewTestNetFactoryBasic()
	if err != nil {
		b.Error("NewTestNetFactoryBasic()")
		b.Error(err)
	}
	network, err := factory.NewNeuralNetwork()
	if err != nil {
		b.Error("factory.NewNeuralNetwork()")
		b.Error(network, err)
	}
	b.StartTimer()

	// evaluate
	for i := 0; i < b.N; i++ {
		var err error
		_, err = network.Evaluate([]NeuralValue{0, NeuralValueMax, 25})
		_, err = network.Evaluate([]NeuralValue{0, 0.1, 0.5})
		_, err = network.Evaluate([]NeuralValue{0, 0, 0})
		_, err = network.Evaluate([]NeuralValue{2 / NeuralValueMax, 2 / NeuralValueMax, 2 / NeuralValueMax})
		_, err = network.Evaluate([]NeuralValue{2 / NeuralValueMax, NeuralValueMid, NeuralValueMax})
		if err != nil {
			b.Error(err)
		}
	}
}

// ----------------------------------------------------------------------------
// ChanceGene

func TestChancesHandled(t *testing.T) {
	dice := NewChanceGeneDisabled()
	slice := func(args ...float64) []float64 {
		return args
	}(dice.Unpack())
	if len(slice) != NumKindChance {
		t.Error("The number of returns returned from (*ChanceGene).Unpack() != NumKindChance")
		t.Error("(*ChanceGene).Unpack() and its relates need update")
	}
	if reflect.ValueOf(*dice).NumField()-1 /* -1 for (ChanceGene).Enabled */ != NumKindChance {
		t.Error("NewChanceGene() needs update - nargs:", reflect.ValueOf(*dice).NumField())
	}
	if reflect.TypeOf(NewChanceGene).NumIn()-1 /* -1 for (ChanceGene).Enabled */ != NumKindChance {
		t.Error("NewChanceGene() needs update - nargs:", reflect.TypeOf(NewChanceGene).NumIn())
	}
	if reflect.TypeOf(NewChanceGeneEnabled).NumIn() != NumKindChance {
		t.Error("NewChanceGene() needs update - nargs:", reflect.TypeOf(NewChanceGeneEnabled).NumIn())
	}
}

// ----------------------------------------------------------------------------
// Universe

func TestBasicUniverse(t *testing.T) {
	var err error
	// univ1 save
	univ1 := NewConfigurationSimple(2, 10, 7).NewUniverse()
	univ1.Config.ExperimentName = "test"
	filename := "akashic." + univ1.Config.ExperimentName + "." + strconv.Itoa(univ1.Generation) + ".json"
	err = univ1.Save(filename)
	if err != nil {
		t.Error(err)
	}
	// univ2 load
	dir, err := os.Getwd()
	if err != nil {
		t.Error(err)
	}
	ark, err := NewArkFromFile(filepath.Join(dir, filename))
	if err != nil {
		t.Error(err)
	}
	univ2, err := ark.NewUniverse()
	if err != nil {
		t.Error(err)
	}
	agent := Agent{}
	univ2.RegisterMeasurer(&agent)
	if len(univ2.Agents) != 1 {
		t.Error("univ2.RegisterMeasurer() error ", univ2.Agents)
	}
	univ2.UnregisterMeasurer(&agent)
	if len(univ2.Agents) != 0 {
		t.Error("univ2.UnregisterMeasurer() error ", univ2.Agents)
	}
}
