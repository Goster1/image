"""Curated guide table for region cart310 (used by 10_edges_cart310_trace.py).

Each entry: id, guide anchor points (full-image px, ordered), pol (edgelib polarity for that
order, +1 = brighter on the left-hand normal), direction class, physical description, model line,
options (exclude = list of [x0, y0, x1, y1] boxes removed after tracing because the edge there is
not the named physical edge: marker cards, labels, image border ...; trim_gap = px trimmed at each
interruption by a label holder), and the verification note written after looking at the crop.
Guides came from LSD chains / row tracking in a rotated view (10_edges_cart310_chains/_rottrack.py).
"""

R = "cart310_"

FRONT_NOTE = ("front edge line of the shelf plate (Y~0); the height of the visible outline relative to the "
              "nominal shelf surface is unknown (bent-up lip / plate thickness), and the fronts of B..E appear "
              "~20-25 px closer to shelf A than the initial model predicts for the nominal surface fronts")
LIP_NOTE = ("divider cross flange edge spanning the cart depth (label holder at its front end); a straight "
            "sheet-metal edge (usable for plumb-line) but probably NOT exactly parallel to cart Y: the lower "
            "dividers deviate 4-6 deg from the cart-Y vanishing direction of the initial calibration (as if "
            "rising ~10 deg towards the back), so it is labelled 'unknown'")


def front(sh, z, kind, pol, guide, exclude, extra_note=""):
    side = ("bright->dark boundary (lip highlight -> thin dark outline line)" if kind == "fall"
            else "dark->bright boundary (thin dark outline line -> aisle side)")
    return dict(id=R + f"x_{sh}_front_{kind}", pol=pol, direction="cartX", guide=guide, exclude=exclude,
                trim_gap=12,
                what=f"shelf {sh} (surface Z={z}) front (Y=0) edge along the cart length: {side} of the "
                     f"shelf front lip, seen from the aisle side above",
                model_line={"fixed": {"Y": 0, "Z": z}, "free": "X", "range_mm": [0, 1600], "exact": False,
                            "note": FRONT_NOTE},
                note=("crop checked: follows the lip outline between the label holders (interruptions "
                      "removed, 12 px trimmed at each); parts along the marker cards at both ends excluded "
                      "(card border runs ~2 px from the lip). " + extra_note).strip())


EDGES = [
    # ---------------------------------------------------------------- top plate end boards (cart-Y)
    dict(id=R + "board0_outer", pol=+1, direction="cartY",
         guide=[[820.5, 953.8], [700.0, 978.0], [548.7, 1005.5]], exclude=[[0, 0, 577, 1080]],
         what="X=0 end white top board: outer (X=0) edge of its top face, silhouetted against the floor "
              "(bright board above, darker floor below); the X=0 end face is invisible from the camera",
         model_line={"fixed": {"X": 0, "Z": 0}, "free": "Y", "range_mm": [0, 450], "exact": True,
                     "note": "outer end edge of the top plate at the X=0 end (drawing: X=0, Z=0); edge slightly "
                             "rounded. Check: a plane homography from the 8 corners of markers 0 and 322 on this "
                             "board puts the traced edge at X = -3..-4 mm (~2 px from X=0) -> consistent"},
         note="clean white->floor step over the full board width; left end (x<577) excluded: there the grey "
              "downward flap of the board's back side and a thin light strip below it begin"),
    dict(id=R + "board1600_outer", pol=+1, direction="cartY", exclude=[[0, 0, 1920, 4]],
         guide=[[258.9, 77.2], [350.0, 40.0], [442.1, 1.1]],
         what="X=1600 end white top board: outer (X=1600) edge of its top face; bright board below, dark "
              "background above; leaves the image at the top border",
         model_line={"fixed": {"X": 1600, "Z": 0}, "free": "Y", "range_mm": [0, 450], "exact": True,
                     "note": "outer end edge of the top plate at X=1600 per the drawing (end face invisible, camera "
                             "X~1414); visible part ~Y 80..400, rounded back corner not included. SUSPECT: a plane "
                             "homography from the 8 corners of markers 1 and 2 on this board puts the traced edge "
                             "at X = 1618-1620 mm (12-15 px beyond the X=1600 line; same with/without a rough "
                             "undistortion) -> the board seems to overhang ~18 mm at this end"},
         note="clean edge; background changes (orange object) shift it by <~0.25 px; blue colour fringe "
              "(lateral chromatic aberration) on the edge; points within 4 px of the image border dropped"),
    dict(id=R + "board1600_inner", pol=+1, direction="cartY",
         guide=[[478.6, 113.4], [360.0, 163.0], [242.4, 212.7]], exclude=[[0, 0, 278, 1080], [466, 0, 1920, 1080]],
         what="X=1600 end white top board: inner edge (towards the cart centre); white board above, grey "
              "shelf area below",
         model_line={"fixed": {"X": 1435, "Z": 0}, "free": "Y", "range_mm": [0, 450], "exact": False,
                     "note": "board width is not in the drawing (165 mm only assumed from the centred markers "
                             "at X=1517.5); the camera (X~1414) sees the inner end face almost edge-on"},
         note="clean; both ends excluded (step at the front corner x>466, back corner / post x<278); "
              "shows a weak S-shaped +-0.5 px deviation (board edge not perfectly straight?)"),
    dict(id=R + "board0_inner_top", pol=+1, direction="cartY", verified="partly",
         guide=[[553.4, 924.1], [690.0, 892.0], [814.2, 863.2]],
         what="X=0 end white top board: inner edge of the white top face (white top face below, brown "
              "inner end face of the board above)",
         model_line={"fixed": {"X": 165, "Z": 0}, "free": "Y", "range_mm": [0, 450], "exact": False,
                     "note": "board width not in the drawing (165 mm assumed from the centred markers at X=82.5)"},
         note="follows the white/brown boundary over the full width; wavy by ~+-0.5 px (laminate edge / "
              "slightly warped board?)"),
    dict(id=R + "board0_inner_bottom", pol=+1, direction="cartY", hw=1.5, verified="partly",
         guide=[[610.0, 905.0], [700.0, 884.0], [805.0, 860.5]],
         what="X=0 end white top board: upper boundary of the brown inner end face (brown face below, "
              "darker cart interior above) = lower edge of the board's inner end face",
         model_line={"fixed": {"X": 165}, "free": "Y", "range_mm": [0, 450], "exact": False,
                     "note": "board thickness and width unknown"},
         note="low contrast (brown vs grey); see crop"),
    # ---------------------------------------------------------------- shelf front edges (cart-X, long)
    front("A", -195, "fall", +1, [[520.3, 19.3], [591.8, 233.6], [667.0, 446.5], [745.8, 658.0], [828.4, 868.2]],
          [[0, 0, 1920, 100], [0, 795, 1920, 1080]],
          "Straight to ~0.18 px rms after a rough reference undistortion; ~0.4 px local dips right after the "
          "holders (clip bends the lip?)."),
    front("B", -595, "fall", +1, [[577.3, 26.7], [638.2, 213.7], [701.5, 399.8], [767.1, 585.1], [835.3, 769.4]],
          [[0, 0, 1920, 92], [0, 618, 1920, 1080]]),
    front("B", -595, "rise", -1, [[599.3, 85.7], [654.8, 254.2], [712.3, 421.9], [771.8, 588.9], [833.4, 755.1]],
          [[0, 0, 1920, 92], [0, 585, 1920, 1080]], "Other side of the same ~3 px dark outline as x_B_front_fall."),
    front("C", -995, "fall", +1, [[620.8, 29.5], [675.7, 193.7], [731.8, 357.4], [789.2, 520.7], [847.9, 683.5]],
          [[0, 0, 1920, 95], [0, 578, 1920, 1080], [0, 262, 1920, 300]],
          "Section next to a purple label (y 262-300) removed."),
    front("C", -995, "rise", -1, [[623.8, 29.5], [678.6, 194.0], [734.8, 357.9], [792.3, 521.4], [851.3, 684.3]],
          [[0, 0, 1920, 95], [0, 578, 1920, 1080], [0, 228, 1920, 248], [0, 300, 1920, 342]],
          "Other side of the dark outline of x_C_front_fall; two short jumpy sections removed."),
    front("D", -1300, "fall", +1, [[649.0, 33.9], [697.7, 182.8], [747.1, 331.4], [797.3, 479.7], [848.2, 627.8]],
          [[0, 550, 1920, 1080]], "No marker cards on D; lower end (y>550, bright patch near the post) removed."),
    front("E", -1605, "fall", +1, [[677.3, 37.1], [721.6, 174.6], [766.7, 311.7], [812.6, 448.6], [859.3, 585.1]],
          [], "Outer silhouette of the cart front against the floor (dark floor on the aisle side); no cards on E."),
    # ---------------------------------------------------------------- back side / inner longitudinal lines
    dict(id=R + "x_back_inner", pol=-1, direction="cartX", verified="partly",
         exclude=[[0, 0, 1920, 370], [0, 640, 1920, 1080]],
         guide=[[362.4, 310.5], [403.2, 420.5], [446.0, 529.7], [490.6, 638.1], [537.2, 745.9]],
         what="back side (Y~450) longitudinal member between the divider brackets: left boundary of the light "
              "vertical strip (back panel edge / fold)",
         model_line={"fixed": {"Y": 450}, "free": "X", "range_mm": [0, 1600], "exact": False,
                     "note": "identity uncertain (back panel fold or rail), height unknown"},
         note="only the middle part kept (upper part and the part in front of the blue floor beam gave "
              "+-0.8 px S-shaped residuals); interrupted by the divider brackets"),
    dict(id=R + "x_B_inner_fall", pol=+1, direction="cartX", trim_gap=12, verified="partly",
         exclude=[[0, 492, 1920, 1080]],
         guide=[[657.9, 303.0], [677.2, 358.0], [696.4, 413.0], [715.7, 467.9], [735.1, 522.9]],
         what="shelf B front strip: inner longitudinal line (bright->dark) ~20 px towards shelf A, parallel to "
              "the B front (fold of the B front lip)",
         model_line={"fixed": {"Y": 0}, "free": "X", "range_mm": [0, 1600], "exact": False,
                     "note": "lip fold, height unknown"},
         note="soft, low-contrast fold/shading line (strength ~18); straight within ~0.2 px but lower reliability"),
    # ---------------------------------------------------------------- short cart-X / unknown board side edges
    dict(id=R + "x_board0_front", pol=+1, direction="unknown",
         guide=[[797.2, 870.6], [810.0, 911.0], [822.3, 952.1]],
         what="X=0 end white top board: front (Y=0, aisle) side boundary; white board on the left, darker "
              "post / shelf area on the right",
         model_line={"fixed": {"Y": 0}, "free": "X", "range_mm": [0, 165], "exact": False,
                     "note": "expected to be the lower edge of the visible board front face (a cart-X line) but it "
                             "deviates 4 deg from the initial cart-X vanishing direction -> 'unknown'"},
         note="clean white->dark step, 84 px"),
    dict(id=R + "x_board1600_front", pol=+1, direction="cartX", exclude=[[0, 75, 1920, 1080]],
         guide=[[469.0, 2.0], [482.0, 55.0], [495.5, 108.0]],
         what="X=1600 end white top board: front (Y=0, aisle) side boundary, white board on the left",
         model_line={"fixed": {"Y": 0}, "free": "X", "range_mm": [1435, 1600], "exact": False,
                     "note": "front face of the board visible; silhouette = lower edge of the front face"},
         note="clean; lower part (y>75) next to marker 4's card excluded"),
    # ---------------------------------------------------------------- divider flanges (straight, direction unknown)
    dict(id=R + "y_lip1", pol=+1, direction="unknown", note="crop checked: clean straight flange edge. " + LIP_NOTE,
         guide=[[525.7, 187.8], [430.0, 222.0], [341.9, 255.7]],
         what="first divider below the X=1600 board: top flange edge across the cart depth (light flange above, "
              "darker shelf below)", model_line=None),
    dict(id=R + "y_lip2", pol=+1, direction="unknown", note="crop checked: clean. " + LIP_NOTE,
         guide=[[601.9, 394.3], [510.0, 421.0], [413.5, 449.6]],
         what="second divider: top flange edge across the cart depth", model_line=None),
    dict(id=R + "y_lip4", pol=+1, direction="unknown", exclude=[[735, 0, 1920, 1080]], verified="partly",
         note="crop checked: thin, weak light flange line (strength ~11), lower boundary traced; end at the holder "
              "removed; +-0.6 px waviness. " + LIP_NOTE,
         guide=[[741.9, 759.0], [650.0, 790.0], [555.4, 819.9]],
         what="divider near the X=0 end: flange edge across the cart depth (lower boundary of the light flange "
              "line)", model_line=None),
    dict(id=R + "y_lip4b", pol=+1, direction="unknown",
         note="crop checked: thin light line, lower boundary traced. " + LIP_NOTE,
         guide=[[749.7, 695.4], [690.0, 715.0], [623.8, 736.4]],
         what="divider near the X=0 end: upper flange line across the cart depth", model_line=None),
    # ---------------------------------------------------------------- cart-Z (vertical) edges
    dict(id=R + "z_post_front_left_right", pol=+1, direction="cartZ", verified="partly",
         exclude=[[0, 762, 1920, 782], [0, 0, 1920, 596]],
         guide=[[829.2, 865.5], [846.0, 725.0], [862.6, 584.9]],
         what="front-left corner post (X=0, Y=0) of cart 310: right (aisle-side) silhouette of the dark post "
              "against the floor, from the top board down to the castor",
         model_line={"fixed": {"X": 0, "Y": 0}, "free": "Z", "range_mm": [-1800, 0], "exact": False,
                     "note": "outer corner line of the post; post section / inset relative to the drawing box unknown"},
         note="orange handle runs parallel next to the lower half, a reflective screw at y~668 and the castor "
              "bracket at the top end; residual ~0.35 px with ~1 px steps between segments"),
    dict(id=R + "z_post_front_right", pol=+1, direction="cartZ", verified="partly",
         guide=[[668.7, 26.3], [610.0, 19.0], [553.9, 12.2], [490.0, 4.5]],
         what="grey bar along the top image border from the X=1600 board corner to the far castor: lower "
              "boundary (grey bar -> dark gap above the wire mesh); identified as the front-right corner post "
              "(X=1600, Y=0) seen almost end-on towards the nadir",
         model_line={"fixed": {"X": 1600, "Y": 0}, "free": "Z", "range_mm": [-1800, 0], "exact": False,
                     "note": "identification inferred from its end points (board corner -> castor) and from its "
                             "direction pointing at the nadir (~(930,60)); post section unknown"},
         note="thin bar near the image border, wire mesh below, crossed by a thin rod; moderate contrast"),
    # ---------------------------------------------------------------- floor marking inside the region
    dict(id=R + "floor_tape_blue_upper", pol=+1, class_="floor_marking", direction="floor_plane",
         straight_3d=True, guide=[[429.5, 651.6], [300.0, 649.8], [171.5, 648.0]],
         what="blue floor tape left of cart 310 (runs under the cart): upper boundary (light floor above, blue "
              "tape below)", model_line=None,
         note="clean; small disturbances where the cart's back rails cross; tape straightness only ~mm "
              "(hand-laid)"),
    dict(id=R + "floor_tape_blue_lower", pol=+1, class_="floor_marking", direction="floor_plane",
         straight_3d=True, guide=[[175.6, 661.7], [300.0, 663.8], [434.7, 665.9]],
         what="blue floor tape left of cart 310: lower boundary (blue tape above, light floor below)",
         model_line=None, note="clean; tape straightness only ~mm (hand-laid)"),
]
