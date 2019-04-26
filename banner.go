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
	"github.com/faiface/pixel"
	"github.com/faiface/pixel/imdraw"
	"github.com/faiface/pixel/text"
	"github.com/nanitefactory/visual"
	"golang.org/x/image/colornames"
)

// BannerActor implements Actor and HUD.
type BannerActor interface {
	Self() *Banner
	visual.HUD
}

// Banner implements Actor and HUD.
type Banner struct {
	// what to draw
	txt *text.Text
	imd *imdraw.IMDraw
	// anchor
	isAnchoredTop   bool
	isAnchoredRight bool
	// pos update
	pos   pixel.Vec
	chPos chan pixel.Vec
	// desc update
	desc   string
	chDesc chan string
}

// NewBanner is a constructor.
func NewBanner(isAnchoredTop, isAnchoredRight bool) BannerActor {
	return &Banner{
		txt: func() *text.Text {
			ret := text.New(pixel.ZV, visual.AtlasASCII18())
			ret.Color = colornames.Black
			return ret
		}(),
		imd: func() *imdraw.IMDraw {
			ret := imdraw.New(nil)
			ret.Color = colornames.Wheat
			return ret
		}(),
		isAnchoredTop:   isAnchoredTop,
		isAnchoredRight: isAnchoredRight,
		pos:             pixel.ZV,
		desc:            "",
		chPos:           make(chan pixel.Vec, 1),
		chDesc:          make(chan string, 1),
	}
}

// Self returns this object as a struct.
func (ban *Banner) Self() *Banner {
	return ban
}

// Draw implements Drawer.
func (ban *Banner) Draw(t pixel.Target) {
	ban.imd.Draw(t)
	ban.txt.Draw(t, pixel.IM)
}

// Update implements Updater.
func (ban *Banner) Update(_ float64) {
	select {
	case newPos := <-ban.chPos:
		ban.pos = newPos
		// Noop. Do not return here.
	case newDesc := <-ban.chDesc:
		if ban.desc == newDesc {
			return
		}
		ban.desc = newDesc
		// Noop. Do not return here.
	default:
		return
	}
	//
	ban.txt.Clear()
	ban.txt.Orig = ban.pos
	ban.txt.Dot = ban.pos
	if ban.isAnchoredTop { // aligned to top
		ban.txt.Dot.Y -= ban.txt.BoundsOf(ban.desc).H()
	}
	if ban.isAnchoredRight { // aligned to right
		ban.txt.Dot.X -= ban.txt.BoundsOf(ban.desc).W()
	}
	ban.txt.WriteString(ban.desc)
	//
	ban.imd.Clear()
	ban.imd.Push(func(r pixel.Rect) (v1, v2, v3, v4 pixel.Vec) { // vertices of rect
		return r.Min, pixel.V(r.Max.X, r.Min.Y), r.Max, pixel.V(r.Min.X, r.Max.Y)
	}(ban.txt.Bounds()))
	ban.imd.Polygon(0)
}

// PosOnScreen implements HUD. Async.
func (ban *Banner) PosOnScreen(width, height float64) {
	var x, y float64 = 5, 5
	if ban.isAnchoredTop { // aligned to top
		y = height
	}
	if ban.isAnchoredRight { // aligned to right
		x = width
	}
	go func() {
		ban.chPos <- pixel.V(x, y)
	}()
}

// UpdateDesc queues an update to the text of this.
// This function falls through when this Actor is working too busy.
func (ban *Banner) UpdateDesc(desc string) {
	select {
	case ban.chDesc <- desc:
		// Noop. Do nothing.
	default:
		// Noop. Do nothing.
	}
}
