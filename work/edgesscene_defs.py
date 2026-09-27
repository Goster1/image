"""Hand-curated guide polylines of the straight structural edges in the scene (everything that is not
one of the two carts).  Guides were picked on zoomed crops (10_edges_scene_zoom.py /
10_edges_scene_lsdzoom.py) and from LSD segments; the sub-pixel position comes from the tracer.

fields: id, guide [[x,y],...], bright ('+x','-x','+y','-y' = side of the edge that is brighter in the
gray mean image; None = strongest gradient either way), kind ('edge' | 'dark_line' = centre of a thin
dark line), hw (search half-width px), cut (boxes [x0,y0,x1,y1] excluded: occluders / labels),
what, cls, straight_3d, dir0 (direction guess before the vanishing-point check), fit_deg.
"""

E = []


def add(id, guide, bright, what, cls="scene_structure", kind="edge", hw=4.0, cut=(), straight_3d=True,
        dir0="unknown", fit_deg=3, min_rel=0.4, sigma=1.0, min_strength=3.0, keep_color=None):
    E.append(dict(id="scene_" + id, guide=guide, bright=bright, what=what, cls=cls, kind=kind, hw=hw,
                  cut=list(cut), straight_3d=straight_3d, dir0=dir0, fit_deg=fit_deg, min_rel=min_rel,
                  sigma=sigma, min_strength=min_strength, keep_color=keep_color))


# ---------------------------------------------------------------------------------------- floor markings
add("tapeV_left", [[865.5, 690], [866, 716], [870.4, 838], [876.5, 960], [882.5, 1078]], "+x",
    "white floor tape running from the T-junction (865,670) down to the bottom border: LEFT edge "
    "(thin dark sharpening halo / floor on the left, white tape on the right)", cls="floor_marking",
    dir0="floor_plane")
add("tapeV_right", [[881, 686], [885, 760], [889.6, 843], [896.8, 1078]], "-x",
    "white floor tape running from the T-junction down to the bottom border: RIGHT edge "
    "(white tape on the left, floor on the right)", cls="floor_marking", dir0="floor_plane")
add("tapeH_top", [[872, 663.4], [915, 662.2], [962, 661.3], [1069, 658.2], [1150, 655.2]], "+y",
    "horizontal white floor tape between the carts: TOP edge (blue floor line above, white tape below)",
    cls="floor_marking", dir0="floor_plane")
add("tapeH_bottom", [[900, 678.3], [963, 677], [1080.7, 673.8], [1140, 671.9]], "-y",
    "horizontal white floor tape between the carts: BOTTOM edge (white tape above, floor below)",
    cls="floor_marking", dir0="floor_plane", cut=[[850, 660, 975, 690]])
add("blueH_top", [[872, 648.5], [1000, 645.3], [1150, 640.6]], "-y",
    "blue floor line directly above the horizontal white tape: TOP edge (floor above, blue below); low "
    "contrast in gray; the left part is worn/scratched and excluded", cls="floor_marking", dir0="floor_plane",
    hw=3.5, min_rel=0.45, cut=[[850, 630, 1045, 660]])
add("blueL_top", [[172, 648.6], [228, 650], [320, 650.5], [430, 651.6]], "-y",
    "blue floor line on the left (x 170-440): TOP edge (floor above, blue line below)",
    cls="floor_marking", dir0="floor_plane")
add("blueL_bottom", [[176, 661.8], [259, 664.2], [297, 665], [435, 665.6]], "+y",
    "blue floor line on the left (x 170-440): BOTTOM edge (blue line above, floor below)",
    cls="floor_marking", dir0="floor_plane")
add("joint450", [[826, 451.8], [905, 450.5], [950, 450.8], [1000, 452.5], [1095, 452.2], [1155, 453.2]], None,
    "floor joint (saw cut) crossing the aisle between the carts at y~452: centre of the thin dark line",
    cls="floor_marking", kind="dark_line", dir0="floor_plane", hw=4.0, sigma=1.2)
add("jointB_H", [[775, 971], [860, 971], [1000, 970], [1095, 969]], None,
    "floor joint near the bottom at y~970 (x 770-1095): centre of the dark band", cls="floor_marking",
    kind="dark_line", dir0="floor_plane", hw=6.0, sigma=1.5)
add("jointB_V_left", [[1085.5, 980], [1085.2, 1030], [1084.4, 1076]], "-x",
    "vertical floor joint near the bottom (x~1085-1100, y 975-1080): LEFT edge of the dark band",
    cls="floor_marking", dir0="floor_plane")
add("jointB_V_right", [[1100, 980], [1099.5, 1030], [1098.4, 1072]], "+x",
    "vertical floor joint near the bottom (x~1085-1100, y 975-1080): RIGHT edge of the dark band",
    cls="floor_marking", dir0="floor_plane")

# ---------------------------------------------------------------------------------------- blue box/column
add("blue_left_upper", [[467, 742], [440, 788], [412, 830]], "-x",
    "tall blue box-shaped object bottom-left (x 290-560, y 740-1080): LEFT outer edge, upper visible part "
    "y 742-830 (floor on the left, blue on the right; a red tape band crosses it at y~775)",
    cut=[[355, 832, 510, 930]])
add("blue_left_lower", [[318, 1019], [303, 1047], [287, 1077]], "-x",
    "tall blue box-shaped object bottom-left: LEFT outer edge, lower visible part y 1019-1077 below the white "
    "frame (floor on the left, blue on the right)", cut=[[300, 895, 368, 1016]])
add("blue_right", [[452, 978], [430, 1027], [407, 1078]], "+x",
    "tall blue box-shaped object bottom-left: RIGHT outer edge below the translucent sheet, y 993-1077 (blue "
    "on the left, floor on the right); the part next to / under the translucent sheet (y 900-990) is "
    "offset by ~1 px and excluded", cut=[[380, 850, 530, 993]])
add("blue_face", [[440, 930], [402, 1000], [363, 1072]], "+x",
    "long blue box-shaped object bottom-left: fold between the darker top face and the lighter side face",
    hw=4.0)

# ---------------------------------------------------------------------------------------- left rack / trolley
add("rack_tube_a", [[53, 10], [49, 35], [44, 85], [39, 135], [36, 160], [33, 210], [30, 285], [28, 360],
                    [27, 410], [27.5, 460], [28, 484], [31, 540], [36, 620]], "-x",
    "tall wire rack at the far-left border: light grey tube running along the whole rack (x 53->27->36, "
    "y 10-620), its right edge (tube on the left, darker gap on the right)", hw=3.0, dir0="horizontal_other")
add("rack_base", [[130, 555], [200, 510], [270, 462], [330, 421]], None,
    "tall wire rack far left: thin dark tube at the bottom of the rack side (x 130-330), centre line",
    kind="dark_line", hw=4.0, sigma=1.0)
add("trolley_rodA", [[110.6, 1008], [127, 1039], [143, 1069.5]], None,
    "grey wire trolley bottom-left corner: light tube of its frame (x 110-143, y 1008-1070), one edge",
    hw=3.0)
add("trolley_rodB", [[156.5, 969.5], [171, 999.5], [185.3, 1029.5]], None,
    "grey wire trolley bottom-left corner: second light tube of its frame (x 156-185, y 970-1030), one edge",
    hw=3.0)

# ---------------------------------------------------------------------------------------- right side
add("rpost_rod", [[1721, 340], [1716, 400], [1711, 460], [1704, 520], [1697, 580], [1691, 620], [1685, 660],
                  [1679, 700], [1672, 740], [1664, 780], [1657, 840]], "+x",
    "right side, left of the notice boards: long light grey pipe/rod (x 1721->1657, y 340-840), LEFT edge "
    "(dark wall on the left, pipe on the right)", hw=3.0, cut=[[1686, 628, 1708, 652]], dir0="horizontal_other")
add("rpost_rodR", [[1727, 340], [1723, 400], [1717, 460], [1711, 520], [1705, 560], [1700, 600],
                   [1694, 640], [1690, 660], [1685, 680], [1678, 720], [1671, 760], [1663, 800],
                   [1659, 840]], "-x",
    "right side, left of the notice boards: long light grey pipe/rod, RIGHT edge (pipe on the left, dark "
    "gap on the right); only y < 628: below, a light rim of the board runs alongside and the white lamp "
    "at (1697,640) interrupts it", hw=2.5, cut=[[1640, 628, 1712, 870]], dir0="horizontal_other")
add("beam_lower", [[1771, 322.5], [1800, 336], [1835, 353]], "-y",
    "right top: long light grey bar (column) in front of the roller conveyor: LOWER edge (bar above, dark "
    "gap above the notice board below), x 1771-1835 only (further left, against the dark opening, the edge "
    "is offset by ~0.8 px - different background; further right a lever/bracket touches it)",
    hw=3.5, cut=[[1836, 300, 1920, 420]])
add("beam_upper", [[1632, 259], [1680, 272], [1720, 285], [1762, 300]], "+y",
    "right top: long light grey bar (column): UPPER edge against the concrete wall (x 1630-1760 only; "
    "further right it crosses the conveyor plate and rollers); a bolt head at (1728,288) excluded", hw=3.5,
    cut=[[1721, 280, 1736, 297]])
add("leg_left", [[1744, 199], [1742, 245], [1740, 288]], "+x",
    "right top: dark conveyor leg under the yellow-black bollard: left edge (dark opening left, light rim "
    "of the leg right)", hw=4.0, dir0="horizontal_other")
add("leg_right", [[1773, 204], [1770.5, 255], [1768, 306]], "-x",
    "right top: dark conveyor leg under the yellow-black bollard: right edge (leg left, black gap and light "
    "plate right)", hw=4.0, dir0="horizontal_other")
add("roller_1", [[1794, 186], [1837, 194], [1879, 202]], None,
    "top right roller conveyor: silhouette edge of one roller (cylinder generator line)", hw=3.5)
add("roller_2", [[1798, 263], [1860, 277], [1919, 290]], None,
    "top right roller conveyor: silhouette edge of one roller (cylinder generator line)", hw=3.5)
add("roller_br1", [[1668, 1040], [1720, 1033.3], [1776, 1026.5]], "+y",
    "bottom right roller conveyor: roller (bright) / dark gap boundary on the upper side of one roller", hw=3.0)
add("roller_br2", [[1662, 1058.7], [1715, 1051.7], [1767, 1044.7]], "+y",
    "bottom right roller conveyor: roller (bright) / dark gap boundary on the upper side of the next roller",
    hw=3.0)
add("pole_top", [[1678, 77.5], [1700, 79], [1740, 84.5], [1760, 86], [1790, 90.5], [1800, 92]], "+y",
    "yellow-black striped bollard top right: upper silhouette edge; only the yellow bands (the black bands "
    "have no contrast against the dark background) and not the rounded end cap (x > 1800)", hw=4.0,
    keep_color=("+y", "yellow"), cut=[[1801, 0, 1920, 200]])
add("pole_bottom", [[1678, 104.5], [1700, 108], [1740, 115], [1750, 116.3], [1790, 123], [1800, 125]], "-y",
    "yellow-black striped bollard top right: lower silhouette edge; yellow bands only, no end cap", hw=4.0,
    keep_color=("-y", "yellow"), cut=[[1802, 0, 1920, 200]])
add("frame_br", [[1918, 785], [1902, 833], [1882, 900], [1863, 961], [1845, 1025], [1830, 1076]], None,
    "bottom right: light aluminium frame strip right of the poster board, its right edge against the dark "
    "machine", hw=4.0, dir0="horizontal_other")
add("wall_railA", [[1623, 300], [1618, 340], [1613, 380], [1607, 420], [1601, 460], [1595, 500], [1588, 540],
                   [1581, 580], [1573, 620]], "-x",
    "right side wall between cart 80 and the pipe: thin light rod (conduit) x~1623->1573, its RIGHT edge "
    "(a second rod runs 6-8 px to the left)", hw=2.5, dir0="horizontal_other")
add("wall_railB", [[1635, 300], [1630, 400], [1623.5, 500], [1610, 620], [1601, 690]], None,
    "right side wall: thin light rod x~1635->1600 (y 300-690), centre line", kind="bright_line", hw=3.0,
    sigma=0.8, dir0="horizontal_other")
