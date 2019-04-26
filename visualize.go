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
	"errors"
	"fmt"
	"image/color"
	"log"
	"math"
	"math/rand"
	"sort"

	"github.com/faiface/pixel"
	"github.com/faiface/pixel/imdraw"
	"github.com/faiface/pixel/text"
	"github.com/nanitefactory/visual"
	"golang.org/x/image/colornames"
)

// NeuralNetworkActor is a neural network as an actor.
type NeuralNetworkActor interface {
	visual.Actor
	Self() *NeuralNetworkVisual // Itf to struct.
	Bound() pixel.Rect
	PosOutNodes() []pixel.Vec
	Channel() chan []NeuralNode
}

// NeuralNetworkVisual implements the Actor interface.
// 해당 클래스(NNV)는 채널을 통해 신경원(NeuralNode)들의 값 사본의 리스트를 받아서 그리는 일을 한다.
// 별도의 쓰레드에서 전달되는 값을 매 비주얼라이저 프레임의 시작시에 채널에서 끄집어내어 읽어보고 확인된 변경사항만 다시 그린다.
// NNV의 기본 정의는 그림일 뿐이며 NN그래프를 상속 받지는 않는다. NNV의 인스턴스는 NN그래프에 대한 의존성이 없고 독립적이다.
// 생성시점에서 NN그래프를 한 번 인자로 받고 그 모양을 분석해서 그림을 그리기는 하지만 그렇다고 해서 이 타입의 인스턴스가 해당 객체를 소유하지는 않는다는 이야기다.
// 채널에 대해서는, 채널의 버퍼가 크면 안정적이지만 그에 따라 딜레이 때의 비주얼라이저(NNV)와 공급자(NN-Evaluater) 간의 시간차도 커진다는 점을 염두에 두자.
// 채널의 크기는 단지 1이면 족하다.
type NeuralNetworkVisual struct {
	ch chan []NeuralNode // The pump.
	// All collections here linear must be in stable sort,
	// so that we can access a node by its numeric integer index.
	prevNodes     []NeuralNode     // Nodes copy by value from the previous frame draw(update).
	posNodes      []pixel.Vec      // Positions of nodes that won't change after init.
	imdNodes      []*imdraw.IMDraw // Nodes drawings. nil imd for an invisible node.
	txtNodes      []*text.Text     // Nodes status displayed for debugging etc. nil for invisibility.
	axons         [][]axon         // Axons of nodes. I hope this is in stable sort.
	radiusNode    float64
	thicknessLink float64        // thicc
	bound         pixel.Rect     // rekt
	imdBoundFrame *imdraw.IMDraw // nil when not present.
}

const minAlpha, maxAlpha = 62, math.MaxUint8

// NewNeuralNetworkActor is a constructor.
//
// Arguments recommendation:
//  - `16.0` for `radiusNode`.
//  - `4.0` for `thicknessLink`.
//  - `1.0` strongly recommended for `nSizeBufChan`.
//  - `nil` for `posInputNodes`.
//  - `nil` for `posOutputNodes`.
//
func NewNeuralNetworkActor(
	network *NeuralNetwork, nSizeBufChan int,
	bound pixel.Rect, paddingTop, paddingRight, paddingBottom, paddingLeft float64,
	radiusNode, thicknessLink float64,
	fromRightToLeft, fromTopToBottom, isBoundVisible,
	isInputNodeVisible, isHiddenNodeVisible, isOutputNodeVisible, isTxtVisible bool,
	posInputNodes, posOutputNodes []pixel.Vec,
) (nna NeuralNetworkActor, err error) {

	// func literal
	analyzeNodes := func(nodesRaw []*NeuralNode, boundAdjusted pixel.Rect, nLayers int) (
		retNodesCopyDeep []NeuralNode, retPosNodes []pixel.Vec, retAxons [][]axon, err error,
	) {
		// what to return
		retNodesCopyDeep = make([]NeuralNode, len(nodesRaw))
		retPosNodes = make([]pixel.Vec, len(nodesRaw))
		retAxons = make([][]axon, len(nodesRaw))

		// iNodes
		iNodeOf := map[*NeuralNode]int{} // idx of a raw node
		for i, ptrNode := range nodesRaw {
			iNodeOf[ptrNode] = i
			retNodesCopyDeep[i] = *ptrNode // tie-in
		}

		// func literal
		positionSingleLayer := func(layer []*NeuralNode, iLayer int, posNodesCustom []pixel.Vec) error {

			// for nodes
			if posNodesCustom == nil {
				for j, nodeInLayer := range layer {
					// set position of this node in this layer.
					var x, y float64 // coords (x, y) of a node.
					// x
					if fromRightToLeft {
						x = boundAdjusted.Max.X - ((float64(boundAdjusted.W()) / float64(nLayers-1)) * float64(iLayer))
					} else if !fromRightToLeft {
						x = boundAdjusted.Min.X + (float64(boundAdjusted.W())/float64(nLayers-1))*float64(iLayer)
					} else {
						return errors.New(fmt.Sprint("positionSingleLayer(): critical runtime violation while positioning x of ", nodeInLayer, " in ", iLayer, "th layer"))
					}
					// y
					if len(layer) == 1 {
						y = boundAdjusted.Min.Y + (boundAdjusted.H() * rand.Float64())
					} else {
						if fromTopToBottom {
							y = boundAdjusted.Max.Y - ((float64(boundAdjusted.H()) / float64(len(layer)-1)) * float64(j))
						} else if !fromTopToBottom {
							y = boundAdjusted.Min.Y + (float64(boundAdjusted.H())/float64(len(layer)-1))*float64(j)
						} else {
							return errors.New(fmt.Sprint("positionSingleLayer(): critical runtime violation while positioning y of ", nodeInLayer, " in ", iLayer, "th layer"))
						}
					}
					// set (x, y) of this node.
					retPosNodes[iNodeOf[nodeInLayer]] = pixel.V(x, y)
				} // for nodeInLayer
			} else if len(posNodesCustom) == len(layer) {
				for j, nodeInLayer := range layer {
					retPosNodes[iNodeOf[nodeInLayer]] = posNodesCustom[j]
				}
			} else if len(posNodesCustom) != len(layer) {
				return errors.New(fmt.Sprint(
					"positionSingleLayer(): ",
					"the number of pre-defined positions of nodes does not match to that of this layer of this neural network: ",
					"this layer is ", iLayer, "th layer with ", len(layer), " nodes but we have ", len(posNodesCustom), " custom nodes",
				))
			} else {
				return errors.New("positionSingleLayer(): critical runtime violation: something went terribly wrong")
			}

			// for axons (link)
			for _, nodeInLayer := range layer {
				// iterate over all axons (link) of our currently selected node, setting their positions.
				retAxons[iNodeOf[nodeInLayer]] = []axon{}
				toNodes := network.NeuralNodesFrom(nodeInLayer) // I hope it is a stable sort. It must. Otherwise things are screwed up.
				for toNodes.Next() {
					outNode := toNodes.NeuralNode()
					retAxons[iNodeOf[nodeInLayer]] = append(retAxons[iNodeOf[nodeInLayer]],
						*newAxon(
							network.NeuralEdge(nodeInLayer, outNode),
							iNodeOf[outNode],
						))
				}
			}

			return nil
		} // func literal

		{ // Position nodes.
			nodesCandidates := make([]*NeuralNode, len(nodesRaw))
			if len(nodesRaw) != copy(nodesCandidates, nodesRaw) {
				log.Println("failed to copy nodes")
			}
			sort.Slice(nodesCandidates, func(i, j int) bool {
				if nodesCandidates[i].LayerLevel != nodesCandidates[j].LayerLevel {
					return nodesCandidates[i].LayerLevel < nodesCandidates[j].LayerLevel
				}
				return nodesCandidates[i].ID() < nodesCandidates[j].ID() // each layer sort by id
			})
			{ // 1. inputs
				i := sort.Search(len(nodesCandidates), func(i int) bool { return nodesCandidates[i].LayerLevel > 0 })
				if err := positionSingleLayer(nodesCandidates[:i], 0, posInputNodes); err != nil {
					return nil, nil, nil, err
				}
				nodesCandidates = nodesCandidates[i:]
			}
			for iLayer := 1; iLayer < nLayers-1; iLayer++ { // 2. hiddens
				i := sort.Search(len(nodesCandidates), func(i int) bool { return nodesCandidates[i].LayerLevel > iLayer })
				if err := positionSingleLayer(nodesCandidates[:i], iLayer, nil); err != nil {
					return nil, nil, nil, err
				}
				nodesCandidates = nodesCandidates[i:]
			}
			{ // 3. outputs
				if err := positionSingleLayer(nodesCandidates, nLayers-1, posOutputNodes); err != nil {
					return nil, nil, nil, err
				}
			}
		}

		return retNodesCopyDeep, retPosNodes, retAxons, nil
	} // func literal

	// Topo sort.
	nodesRaw, err := network.Sort()
	if err != nil {
		return nil, err
	}

	// Analyze raw nodes and get postions of all links and nodes.
	nodesCopyDeep, newPosNodes, newAxons, err := analyzeNodes(
		nodesRaw,
		BoundAdjust(bound, paddingTop, paddingRight, paddingBottom, paddingLeft),
		network.NLayers(),
	)
	if err != nil {
		return nil, err
	}

	newImdBoundFrame := func(bound pixel.Rect) *imdraw.IMDraw {
		if !isBoundVisible {
			return nil
		}
		imd := imdraw.New(nil)
		imd.EndShape = imdraw.RoundEndShape
		imd.Push(func(r pixel.Rect) (v1, v2, v3, v4 pixel.Vec) { // vertices of rect
			return r.Min, pixel.V(r.Max.X, r.Min.Y), r.Max, pixel.V(r.Min.X, r.Max.Y)
		}(bound))
		imd.Polygon(2)
		return imd
	}

	// What to return.
	nnv := NeuralNetworkVisual{
		ch:            make(chan []NeuralNode, nSizeBufChan),
		prevNodes:     nodesCopyDeep,
		posNodes:      newPosNodes,
		imdNodes:      make([]*imdraw.IMDraw, len(nodesRaw)),
		txtNodes:      make([]*text.Text, len(nodesRaw)),
		axons:         newAxons,
		radiusNode:    radiusNode,
		thicknessLink: thicknessLink,
		bound:         bound,
		imdBoundFrame: newImdBoundFrame(bound),
	}

	// Fill txt/imds.
	dilute, err := Ramp(0).Concentration()
	if err != nil {
		log.Println("NewNeuralNetworkActor(): cannot convert an invalid neural value to uint8") //
	}
	for i, node := range nodesRaw {
		// links
		for j, axon := range nnv.axons[i] {
			// imd; primitives drawer
			imd := imdraw.New(nil)
			imd.EndShape = imdraw.SharpEndShape
			imd.Clear()
			// draw a line
			imd.Color = color.NRGBA{axon.color, 64, 255 - axon.color, minAlpha}
			imd.Push(nnv.posNodes[i], nnv.posNodes[axon.to])
			imd.Line(thicknessLink)
			// save
			nnv.axons[i][j].imd = imd
		}
		// nodes
		switch node.TypeInInt() {
		case InputNodeBias:
			fallthrough
		case InputNodeNotBias:
			if !isInputNodeVisible {
				continue
			}
		case HiddenNode:
			if !isHiddenNodeVisible {
				continue
			}
		case OutputNode:
			if !isOutputNodeVisible {
				continue
			}
		case ExceptionalNode:
			log.Println("NewNeuralNetwork(): warning: ExceptionalNode node type: ", node)
			continue
		default:
			return nil, errors.New("NewNeuralNetwork(): Unknown node type: " + node.String())
		}
		// This initialization below is invoked only if this node is set to visible.
		{ // node imd
			imd := imdraw.New(nil)
			imd.Precision = 5
			imd.Color = color.NRGBA{dilute, 0, 0, minAlpha}
			imd.Push(nnv.posNodes[i])
			imd.Circle(nnv.radiusNode, 0)
			nnv.imdNodes[i] = imd
		}
		if isTxtVisible { // node txt
			txt := text.New(pixel.ZV, visual.AtlasASCII18())
			txt.Color = colornames.Black
			pos := nnv.posNodes[i].Add(pixel.V(nnv.radiusNode, nnv.radiusNode))
			txt.Orig = pos
			txt.Dot = pos
			txt.WriteString(fmt.Sprintf("%d: %s: %.2f", node.ID(), node.Name, node.Value))
			nnv.txtNodes[i] = txt
		}
	} // for nodesRaw

	return &nnv, nil
}

// Update obligatorily invoked by Visualizer on mainthread.
func (nnv *NeuralNetworkVisual) Update(dt float64) {
	select {
	case topoSort := <-nnv.ch:
		// Tell if this really is a topo sort.
		if len(topoSort) != len(nnv.prevNodes) {
			log.Println("nnv.Update(): warning: the size of this nodes list read from nnv chan is wrong.") //
			return
		}
		for i, node := range topoSort {
			if nnv.prevNodes[i].ID() != node.ID() {
				log.Println("nnv.Update(): warning: the order of this nodes list read from nnv chan is wrong.") //
				return
			}
		}
		// Now check values of nodes.
		isPrince := true // Set to false if this topoSort is revealed to be spoiled one.
		for i, node := range topoSort {
			// Update this node's txt.
			if txt := nnv.txtNodes[i]; txt != nil {
				txt.Clear()
				// txt.Color = colornames.Black
				pos := nnv.posNodes[i].Add(pixel.V(nnv.radiusNode, nnv.radiusNode))
				txt.Orig = pos
				txt.Dot = pos
				txt.WriteString(fmt.Sprintf("%d: %s: %.2f", node.ID(), node.Name, node.Value))
			}
			// Tell if there is a value change.
			nvNew, err := node.Output()
			if err != nil {
				log.Println("nnv.Update(): warning: invalid neural value: something went terribly wrong: ", err)             //
				log.Println("nnv.Update(): warning: ", i, "th node ", node, " of our new topo sort was not evaluated well.") //
				isPrince = false
				continue
			}
			nvPrev, _ := nnv.prevNodes[i].Output() // The error of this has already been handled. (otherwise it should've been.)
			if nvNew == nvPrev {
				continue
			}
			// Notice this is after that (nvNew != nvPrev).
			kindNew := nvNew.Kind()
			if kindNew == NVKindExceptional {
				log.Println("nnv.Update(): warning: invalid neural value: runtime violation: ", kindNew, err) //
				isPrince = false
				continue
			}
			strengthNew, err := nvNew.Strength()
			if err != nil {
				log.Println("nnv.Update(): warning: invalid neural value: runtime violation: ", strengthNew, err) //
				isPrince = false
				continue
			}
			// cut low
			// if alpha < minAlpha {
			// 	alpha = minAlpha
			// }
			//
			// cut high
			// alpha = uint8(math.Min(float64(int(alpha)+minAlpha), 0xFF))
			//
			// adjust range
			const percent = float64(float64(maxAlpha-minAlpha) / float64(maxAlpha))
			alpha := uint8((float64(strengthNew) * percent) + minAlpha)
			// alpha := uint8((float64(strengthNew*4) * percent) + minAlpha)
			// Update this node.
			if imd := nnv.imdNodes[i]; imd != nil {
				dilute, err := nvNew.Concentration()
				if err != nil {
					log.Println("nnv.Update(): warning: invalid neural value: runtime violation: ", dilute, err) //
					isPrince = false
					continue
				}
				// if float64(strengthNew) >= math.Abs(float64(NeuralValueMax))/4 ||
				// 	float64(strengthNew) >= math.Abs(float64(NeuralValueMin))/4 {
				// 	alpha = uint8(math.MaxUint8)
				// }
				imd.Clear()
				imd.Precision = 5
				imd.Color = color.NRGBA{dilute, 0, 0, alpha}
				imd.Push(nnv.posNodes[i])
				imd.Circle(nnv.radiusNode, 0)
			}
			// Update this node's axons. (links)
			for _, axon := range nnv.axons[i] {
				// imd update
				imd := axon.imd
				imd.Clear()
				// draw a line
				imd.Color = color.NRGBA{axon.color, 64, 255 - axon.color, alpha}
				imd.Push(nnv.posNodes[i], nnv.posNodes[axon.to])
				imd.Line(nnv.thicknessLink)
			} // for axons
		} // for range topoSort
		// Succeed the throne.
		if isPrince {
			nnv.prevNodes = topoSort
		}
	default:
		return
	}
} // func

// Draw obligatorily invoked by Visualizer on mainthread.
func (nnv *NeuralNetworkVisual) Draw(t pixel.Target) {
	if imd := nnv.imdBoundFrame; imd != nil {
		imd.Draw(t)
	}
	for iNode := range nnv.prevNodes {
		if imd := nnv.imdNodes[iNode]; imd != nil {
			imd.Draw(t)
		}
		if txt := nnv.txtNodes[iNode]; txt != nil {
			txt.Draw(t, pixel.IM)
		}
		for _, axon := range nnv.axons[iNode] {
			axon.imd.Draw(t) // These imds are never set to nil.
		}
	}
}

// ----------------------------------------------------------------------------

// Self of this returns itself as a struct.
func (nnv *NeuralNetworkVisual) Self() *NeuralNetworkVisual {
	return nnv
}

// Bound returns the bound of this. You get the size and its offset as well.
func (nnv *NeuralNetworkVisual) Bound() pixel.Rect {
	return nnv.bound
}

// PosOutNodes returns positions vectors of all output nodes.
func (nnv *NeuralNetworkVisual) PosOutNodes() []pixel.Vec {
	// get the first outNode index
	iBaseOutNodes := -1
	for i, node := range nnv.prevNodes {
		if node.TypeInInt() == OutputNode {
			iBaseOutNodes = i
			break
		}
	}
	if iBaseOutNodes == -1 {
		panic("cannot find any output node")
	}
	// get indice from there
	type refOutNode struct {
		index int
		id    int64
	}
	indiceOutNodes := []refOutNode{}
	for i, node := range nnv.prevNodes[iBaseOutNodes:] {
		if node.TypeInInt() == OutputNode {
			indiceOutNodes = append(indiceOutNodes, refOutNode{iBaseOutNodes + i, node.ID()})
		}
	}
	sort.Slice(indiceOutNodes, func(i, j int) bool {
		return indiceOutNodes[i].id < indiceOutNodes[j].id
	})
	// get positions from refs(indice)
	ret := []pixel.Vec{}
	for _, outNodeRef := range indiceOutNodes {
		ret = append(ret, nnv.posNodes[outNodeRef.index])
	}
	return ret
}

// Channel returns the channel this actor gets all the data from.
func (nnv *NeuralNetworkVisual) Channel() chan []NeuralNode {
	return nnv.ch
}

// ----------------------------------------------------------------------------

type axon struct {
	imd   *imdraw.IMDraw // The only thing in this struct that can vary. This is never set to nil.
	color uint8          // Color concentration in diluteness.
	to    int            // Idx of this axon's to node (out-node) in our nnv list.
}

func newAxon(neuralEdge *NeuralEdge, idxOutNode int) *axon {
	// get color
	const positiveMaxNeuralValue = float64(NeuralValueMax)
	dilute, err := Ramp(NeuralValue(
		positiveMaxNeuralValue * neuralEdge.Weight(),
	)).Concentration()
	if err != nil {
		panic(err)
	}
	// return
	return &axon{
		imd:   nil,
		color: dilute,
		to:    idxOutNode,
	}
}

// ----------------------------------------------------------------------------

// BoundAdjust returns the border without (inside) padding.
func BoundAdjust(
	bound pixel.Rect,
	paddingTop, paddingRight,
	paddingBottom, paddingLeft float64,
) pixel.Rect {
	boundAdjusted := bound
	boundAdjusted.Max.X -= paddingRight
	boundAdjusted.Max.Y -= paddingTop
	boundAdjusted.Min.X += paddingLeft
	boundAdjusted.Min.Y += paddingBottom
	return boundAdjusted
}

// PosPixelsOf returns all the positions of all pixels in a given rectangular bound, aligned in a linear order.
// This could be useful for positioning the input nodes of the neural network actor.
func PosPixelsOf(bound pixel.Rect, fromLeftToRight, fromTopToBottom bool) []pixel.Vec {
	width := int(bound.W())
	height := int(bound.H())
	base := bound.Min // Base what's prior to the offset.
	ret := make([]pixel.Vec, width*height)
	var thunk func(irow, icol int)
	if fromLeftToRight && fromTopToBottom {
		plus := base.Add(pixel.V(0.5, -0.5))
		thunk = func(irow, icol int) {
			ret[(irow*width)+icol] = plus.Add(pixel.V(float64(icol), float64(height-irow)))
		}
	} else if fromLeftToRight && !fromTopToBottom {
		plus := base.Add(pixel.V(0.5, 0.5))
		thunk = func(irow, icol int) {
			ret[(irow*width)+icol] = plus.Add(pixel.V(float64(icol), float64(irow)))
		}
	} else if !fromLeftToRight && fromTopToBottom {
		plus := base.Add(pixel.V(-0.5, -0.5))
		thunk = func(irow, icol int) {
			ret[(irow*width)+icol] = plus.Add(pixel.V(float64(width-icol), float64(height-irow)))
		}
	} else if !fromLeftToRight && !fromTopToBottom {
		plus := base.Add(pixel.V(-0.5, 0.5))
		thunk = func(irow, icol int) {
			ret[(irow*width)+icol] = plus.Add(pixel.V(float64(width-icol), float64(irow)))
		}
	} else {
		panic("PosPixelsOf(): runtime violation in its arguments")
	}
	// Include lower bound, exclude upper bound for integer indices.
	// +-0.5 pointing to the pixel center.
	for irow := 0; irow < height; irow++ {
		for icol := 0; icol < width; icol++ {
			thunk(irow, icol)
		}
	}
	return ret
}

// PosBoxesOf returns all the positions of all boxes in a given rectangular bound, aligned in a linear order.
// This could be useful for positioning the input nodes of the neural network actor.
// Args `widthBox` and `heightBox` should be exact otherwise outbound-errors.
func PosBoxesOf(bound pixel.Rect, fromLeftToRight, fromTopToBottom bool, widthBox, heightBox float64) []pixel.Vec {
	width := bound.W()
	height := bound.H()
	base := bound.Min // Base what's prior to the offset.
	ret := make([]pixel.Vec, int(width/widthBox)*int(height/heightBox))
	var thunk func(y, x float64)
	if fromLeftToRight && fromTopToBottom {
		plus := base.Add(pixel.V(0.5*widthBox, -0.5*heightBox))
		thunk = func(y, x float64) {
			ret[int((y/heightBox)*(width/widthBox))+int(x/widthBox)] = plus.Add(pixel.V(x, float64(height-y)))
		}
	} else if fromLeftToRight && !fromTopToBottom {
		plus := base.Add(pixel.V(0.5*widthBox, 0.5*heightBox))
		thunk = func(y, x float64) {
			ret[int((y/heightBox)*(width/widthBox))+int(x/widthBox)] = plus.Add(pixel.V(x, y))
		}
	} else if !fromLeftToRight && fromTopToBottom {
		plus := base.Add(pixel.V(-0.5*widthBox, -0.5*heightBox))
		thunk = func(y, x float64) {
			ret[int((y/heightBox)*(width/widthBox))+int(x/widthBox)] = plus.Add(pixel.V(float64(width-x), float64(height-y)))
		}
	} else if !fromLeftToRight && !fromTopToBottom {
		plus := base.Add(pixel.V(-0.5*widthBox, 0.5*heightBox))
		thunk = func(y, x float64) {
			ret[int((y/heightBox)*(width/widthBox))+int(x/widthBox)] = plus.Add(pixel.V(float64(width-x), y))
		}
	} else {
		panic("PosPixelsOf(): runtime violation in its arguments")
	}
	// Include lower bound, exclude upper bound for integer indices.
	// +-0.5*widthBox or +-0.5*heightBox pointing to the box center.
	for y := 0.0; y < height; y += heightBox {
		for x := 0.0; x < width; x += widthBox {
			// log.Println(width, height, width/widthBox, height/heightBox, int(width/widthBox)*int(height/heightBox), y, x, (y / heightBox), (x / widthBox)) //
			thunk(y, x)
		}
	}
	return ret
}
