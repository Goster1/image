"""Curated guide table for region cart80 (used by 10_edges_cart80_trace.py).

Each entry: id (without the region prefix), guide = approximate polyline (full-image px, ordered;
obtained from LSD chains traced sub-pixel and grown along the edge, 10_edges_cart80_chains.py, printed
with 10_edges_cart80_mkguide.py), direction class, physical description, model line, options:
  ext   = [px, px] extension of the guide beyond its first / last point (edge followed further if it
          continues; the cleaning drops what is not the same edge),
  keep  = {"x": [lo, hi]} / {"y": [lo, hi]} window of the image kept after tracing,
  excl  = list of [x0, y0, x1, y1] boxes removed after tracing (marker cards, label holders, ...),
  hw    = half width of the search profile (px),
and the verification note written after looking at the zoomed crop.
Cart-80 frame: X runs from the image top (X=0 end) to the image bottom (X=1600 end), the front (Y=0,
aisle side) is the LEFT side of the cart in the image, lower levels appear shifted up-left (towards the
nadir).  The camera sits in front of / above the cart (initial calibration: X~145, Y~-645, Z~+1880 mm).
"""

LIPNOTE = ("front lip / front outline of the shelf at this level; the height of the visible outline "
           "above the nominal shelf surface and its exact Y are not in the drawing (bent-up lip with a "
           "rounded top, label holders)")
BACKNOTE = ("longitudinal member on the back (Y~450) side; its exact Y/Z (profile size, offset from the "
            "outline) is not in the drawing")
FLNOTE = ("top flange of a cross divider spanning the cart depth (label holder at its front end); position "
          "along X not in the drawing")


def xline(y, z, note, exact=False, rng=(0, 1600)):
    return {"fixed": {"Y": y, "Z": z}, "free": "X", "range_mm": list(rng), "exact": exact, "note": note}


EDGES = [
    # ============================================================== cart-X: shelf fronts (left side)
    dict(id="x_A_front_out", verified="yes",
         note="5 zoomed windows checked: points sit on the thin dark gap -> bright lip-top boundary over y 100..822; the stretches along the shelf-sticker cards (marker 3 at the X=0 end, markers 6 and 4 at the X=1600 end, card borders run within ~2-5 px of the lip) are excluded; label-holder interruptions removed (+-6 px)", direction="cartX",
         guide=[[1351.8, 23.6], [1343.4, 146.6], [1332.9, 269.3], [1320.4, 391.9], [1305.8, 514.1],
                [1289.2, 636.2], [1270.6, 758.0], [1250.1, 879.5]],
         keep={"y": [100, 826]},
         what="shelf A front (Y~0) along the whole cart: outer (aisle-side) outline of the front lip; thin "
              "dark gap on the left, bright rounded lip top on the right",
         model_line=xline(0, -195, LIPNOTE)),
    dict(id="x_A_front_in", verified="partly",
         note="weak inner boundary of the lip-top highlight (~7 px right of x_A_front_out); follows the line in most windows but wobbles by ~0.3 px in the upper part; not independent of x_A_front_out", direction="cartX",
         guide=[[1332.5, 333.1], [1326.9, 391.7], [1320.6, 450.2], [1313.6, 508.7], [1306.1, 567.1],
                [1298.0, 625.4], [1289.2, 683.7], [1280.0, 741.8]], ext=[30, 60],
         what="shelf A front lip: inner boundary of the lip's bright top strip (~7 px right of the outer "
              "outline)",
         model_line=xline(0, -195, LIPNOTE)),
    dict(id="x_B_front_top", verified="yes",
         note="checked: clean dark->bright lip outline for y 100..350; background on the left changes (plastic crates standing on level C) but the traced side is the lip; card zone (y<100) excluded; beyond y~350 the tracer drifts to another line, cut", direction="cartX",
         guide=[[1287.6, 26.0], [1285.6, 62.0], [1283.3, 97.9], [1281.0, 133.9], [1278.5, 169.8],
                [1275.8, 205.7], [1273.0, 241.7], [1270.1, 277.6]], ext=[0, 80], keep={"y": [0, 350]},
         what="shelf B front lip outline, X=0 part of the cart (dark gap / crates of level C on the left, "
              "bright lip on the right); beyond y~350 the line is lost behind label holders",
         model_line=xline(0, -595, LIPNOTE)),
    dict(id="x_B_front_bot", verified="yes",
         note="checked: lip outline y ~430..700, interrupted by label holders (removed); the part next to the printed label card and the marker-6 card (y > 700) excluded", direction="cartX",
         guide=[[1248.5, 493.2], [1244.2, 534.3], [1239.4, 575.5], [1234.3, 616.5], [1228.9, 657.6],
                [1223.5, 698.6], [1218.1, 739.6], [1212.9, 780.7]], ext=[150, 0], keep={"y": [0, 700]},
         what="shelf B front lip outline, X=1600 half of the cart (continues past the marker-6 card, card "
              "part excluded)",
         model_line=xline(0, -595, LIPNOTE)),
    dict(id="x_C_front", verified="yes",
         note="checked: one continuous lip outline y 100..637 (three LSD pieces verified to lie on one smooth curve, rms 0.22 px); cards of markers 7 and 8 excluded, label-holder gaps removed", direction="cartX",
         guide=[[1244.9, 100.5], [1240.1, 167.7], [1234.8, 234.8], [1229.0, 301.9], [1222.8, 369.0],
                [1216.1, 436.0], [1209.2, 503.0], [1201.9, 569.9], [1194.4, 636.9]], ext=[0, 20],
         what="shelf C front lip outline along the cart (dark gap on the left -> bright lip); card of "
              "marker 7 (X=0 end) and marker 8 (X=1600 end) excluded",
         model_line=xline(0, -995, LIPNOTE)),
    dict(id="x_D_front", verified="yes",
         note="checked: dark gap -> bright lip boundary y 36..~590, clean except at label holders (removed)", direction="cartX",
         guide=[[1221.5, 35.6], [1217.7, 100.0], [1213.4, 164.4], [1208.8, 228.8], [1203.7, 293.1],
                [1198.1, 357.5], [1192.2, 421.7], [1185.9, 486.0]], ext=[0, 100],
         what="shelf D front lip: dark gap -> bright lip boundary",
         model_line=xline(0, -1300, LIPNOTE)),
    dict(id="x_E_front", verified="yes",
         note="checked in 4 windows: sharp thin-dark-outline -> bright-lip edge of the lowest lip over y 27..600; the bracket at the X=0 corner rejected", direction="cartX",
         guide=[[1191.3, 71.0], [1186.4, 149.8], [1180.9, 228.5], [1174.7, 307.2], [1167.9, 385.8],
                [1160.6, 464.4], [1152.6, 542.9], [1144.2, 621.4]], ext=[60, 60],
         what="lowest front lip (shelf E): thin dark outline -> bright rounded lip top (1-2 px inside the "
              "silhouette)",
         model_line=xline(0, -1605, LIPNOTE)),
    dict(id="x_E_front_in", verified="yes",
         note="checked: other side of the same bright lip-top highlight (~3 px right of x_E_front), clean; NOT independent of x_E_front (same lip)", direction="cartX",
         guide=[[1196.3, 29.4], [1192.9, 89.9], [1189.2, 150.4], [1185.0, 210.8], [1180.5, 271.2],
                [1175.6, 331.6], [1170.3, 391.9], [1164.7, 452.2]], ext=[0, 150],
         what="lowest front lip (shelf E): inner boundary of the bright lip top (bright -> grey)",
         model_line=xline(0, -1605, LIPNOTE)),
    # ============================================================== cart-X: back side (right)
    dict(id="x_back_rail_out", verified="yes",
         note="checked in 5 windows: right silhouette of the bright back rail against the wall / background over y 146..802; very clean", direction="cartX",
         guide=[[1637.5, 145.6], [1628.9, 241.2], [1618.2, 336.4], [1605.5, 431.4], [1590.7, 525.9],
                [1573.6, 620.2], [1554.2, 714.1], [1532.5, 807.6]], ext=[0, 40],
         what="back (Y~450) top longitudinal rail: outer (right) silhouette of the bright rail against the "
              "wall / background behind the cart",
         model_line=xline(450, 0, BACKNOTE)),
    dict(id="x_back_rail_in", verified="partly",
         note="left boundary of the back rail face; clean in the lower half, weak and partly rejected in the upper part", direction="cartX",
         guide=[[1584.7, 481.4], [1577.4, 526.3], [1569.6, 571.1], [1561.2, 615.7], [1552.2, 660.3],
                [1542.9, 704.7], [1533.1, 749.1], [1522.8, 793.4]], ext=[300, 40],
         what="back top rail: inner (left) boundary of the bright rail face",
         model_line=xline(450, 0, BACKNOTE)),
    dict(id="x_back_low", verified="yes",
         note="checked: left boundary of a darker back-panel member y 147..~800, interrupted by cross members (removed)", direction="cartX",
         guide=[[1578.4, 147.3], [1572.9, 206.8], [1566.7, 266.2], [1559.9, 325.6], [1552.3, 384.8],
                [1543.9, 443.9], [1534.6, 502.9], [1524.4, 561.8]], ext=[0, 250],
         what="back side: lower longitudinal member (below the top rail), left boundary of its dark face",
         model_line=xline(450, -300, BACKNOTE)),
    dict(id="x_board0_front", verified="partly",
         note="white board front side; in the upper part the shelf-A sticker card (marker 3, lower level) lies directly next to the board edge, local noise 0.4 px", direction="cartX",
         guide=[[1401.9, 0.6], [1400.1, 16.9], [1398.3, 33.3], [1396.4, 49.6], [1394.6, 65.9],
                [1392.8, 82.3], [1390.9, 98.6], [1389.1, 115.0]], keep={"y": [4, 200]},
         what="X=0 end white top board: front (Y=0, aisle) side boundary; white board on the right, shelf "
              "sticker card / darker background on the left",
         model_line=xline(0, 0, "front side of the top plate at the X=0 end (drawing Y=0, Z=0), X ~0..175; check: homography "
                          "from markers 0/92 gives Y = -1..-3 mm",
                          exact=True, rng=(0, 170))),
    dict(id="x_board1600_front", verified="yes",
         note="checked: white board -> darker background boundary over the full board length", direction="cartX",
         guide=[[1289.5, 881.9], [1287.7, 894.6], [1285.9, 907.3], [1284.1, 920.0], [1282.3, 932.7],
                [1280.5, 945.4], [1278.7, 958.1], [1276.9, 970.8]],
         what="X=1600 end white top board: front (Y=0, aisle) side boundary; white board on the right",
         model_line=xline(0, 0, "front side of the top plate at the X=1600 end (drawing Y=0, Z=0), X ~1445..1600; check: "
                          "homography from markers 1/2 gives Y = -0.1..-0.7 mm",
                          exact=True, rng=(1435, 1600))),
    # ============================================================== cart-Y
    dict(id="y_board0_outer", verified="partly",
         note="clean white -> dark step, but within 3-27 px of the image top border (points < 3 px from the border dropped) and the curled back corner (x>1655) excluded", direction="cartY",
         guide=[[1501.1, 1.2], [1529.7, 5.5], [1558.1, 9.9], [1586.5, 14.7], [1614.9, 19.6], [1643.2, 24.9],
                [1671.5, 30.3], [1699.8, 36.0]], ext=[110, 0], keep={"x": [0, 1655], "y": [3, 1080]},
         what="X=0 end white top board: outer (X=0) edge of its top face (white board below, dark "
              "background above); the curled back corner (x > 1655) and the part at the image border "
              "excluded",
         model_line={"fixed": {"X": 0, "Z": 0}, "free": "Y", "range_mm": [0, 450], "exact": False,
                     "note": "nominally the outer end edge of the top plate at X=0 (drawing X=0, Z=0), but a plane "
                             "homography from the 8 corners of markers 0 and 92 on this board puts the traced edge at "
                             "X = -16..-21 mm -> the white board seems to overhang / be bent up at this end; "
                             "not used as exact"}),
    dict(id="y_board0_inner", verified="yes",
         note="checked: white board / grey interior boundary over the full width, clean", direction="cartY",
         guide=[[1388.9, 120.1], [1423.7, 121.8], [1458.5, 123.9], [1493.3, 126.6], [1528.0, 129.7],
                [1562.7, 133.4], [1597.3, 137.6], [1631.8, 142.5]],
         what="X=0 end white top board: inner edge (towards the cart centre); white board above, grey "
              "cart interior below",
         model_line={"fixed": {"X": 170, "Z": 0}, "free": "Y", "range_mm": [0, 450], "exact": False,
                     "note": "board width not in the drawing; homography from markers 0/92: X = 179-182 mm"}),
    dict(id="y_board1600_inner", verified="yes",
         note="checked: white top face / tan end-face boundary over the full width", direction="cartY",
         guide=[[1289.3, 893.6], [1322.9, 892.2], [1356.5, 891.0], [1390.1, 889.9], [1423.7, 888.7],
                [1457.2, 887.4], [1490.8, 885.7], [1524.3, 883.6]],
         what="X=1600 end white top board: inner edge of the white top face (white below, tan inner end "
              "face of the board above)",
         model_line={"fixed": {"X": 1444, "Z": 0}, "free": "Y", "range_mm": [0, 450], "exact": False,
                     "note": "board width not in the drawing; homography from markers 1/2: X = 1443-1445 mm"}),
    dict(id="y_board1600_inner_top", straight_3d=False, verified="yes",
         note="checked: tan end face / dark interior boundary over the full width. raw bow -0.09 px while the parallel edge 5 px away (y_board1600_inner) bows -0.72 px and the lens models predict -1.0..-2.1 px -> the lower inner board edge is not straight (warped board end); straight_3d set false", direction="cartY",
         guide=[[1292.6, 888.7], [1326.2, 887.4], [1359.8, 885.9], [1393.4, 884.2], [1427.0, 882.5],
                [1460.6, 880.9], [1494.2, 879.4], [1527.8, 878.1]],
         what="X=1600 end white top board: upper boundary of the tan inner end face (tan face below, "
              "darker cart interior above) = lower inner edge of the board",
         model_line={"fixed": {"X": 1444, "Z": -10}, "free": "Y", "range_mm": [0, 450], "exact": False,
                     "note": "board width and thickness not in the drawing (lower inner edge of the board, "
                             "homography in the Z=0 plane gives X = 1432-1434 mm)"}),
    dict(id="y_board1600_outer", verified="yes",
         note="checked: white board / dark background boundary; curled back corner excluded", direction="cartY",
         guide=[[1273.9, 972.4], [1312.9, 971.2], [1351.8, 969.7], [1390.8, 968.0], [1429.8, 966.0],
                [1468.7, 963.8], [1507.7, 961.3], [1546.6, 958.5]], keep={"x": [1279, 1520]},
         what="X=1600 end white top board: outer (X=1600) edge of the top face (white board above, darker "
              "background below); curled back corner excluded",
         model_line={"fixed": {"X": 1600, "Z": 0}, "free": "Y", "range_mm": [0, 450], "exact": True,
                     "note": "outer end edge of the top plate at X=1600 (drawing X=1600, Z=0); check: plane "
                             "homography from markers 1 and 2 on this board puts the edge at X = 1602-1603 mm"}),
    dict(id="y_div1_top", straight_3d=False, verified="partly",
         note="flange boundary with low contrast against the dusty shelf behind; ok in the front half, noisier (0.27 px) towards the back. raw bow +2.54 px over 212 px vs +0.9..+1.1 px predicted for a straight line by the initial / prototype lens models -> flange edge probably not straight; straight_3d set false", direction="cartY",
         guide=[[1355.9, 188.6], [1386.8, 189.2], [1417.6, 190.3], [1448.4, 191.9], [1479.2, 193.9],
                [1510.0, 196.3], [1540.8, 199.1], [1571.5, 202.1]], ext=[0, 30],
         what="first cross divider below the X=0 board: upper (towards X=0) edge of its bright top flange",
         model_line={"fixed": {"X": 280, "Z": 0}, "free": "Y", "range_mm": [0, 450], "exact": False,
                     "note": FLNOTE}),
    dict(id="y_div4_top", straight_3d=False, verified="partly",
         note="thin straight seam line on a flat plate near the X=1600 end; straight and clean but its physical identity (flange edge vs plate joint) is uncertain. raw bow -1.49 px over only 122 px vs -0.2..-0.3 px predicted for a straight line -> curved in 3D (or a seam, not an edge); straight_3d set false", direction="cartY",
         guide=[[1316.6, 755.0], [1332.9, 755.3], [1349.2, 755.6], [1365.5, 755.9], [1381.8, 756.2],
                [1398.0, 756.6], [1414.3, 756.9], [1430.6, 757.2], [1471.6, 757.6]], ext=[0, 20],
         what="cross divider near the X=1600 end: upper edge of its top flange",
         model_line={"fixed": {"X": 1190, "Z": 0}, "free": "Y", "range_mm": [0, 450], "exact": False,
                     "note": FLNOTE}),
    dict(id="y_div5_top", straight_3d=False, verified="yes",
         note="checked: clean plate edge just inside the X=1600 board. raw bow -1.49 px over only 92 px vs about -0.1..-0.2 px predicted for a straight line -> curved in 3D (sagging plate edge?); straight_3d set false", direction="cartY",
         guide=[[1339.7, 831.5], [1350.4, 831.9], [1361.1, 832.3], [1371.8, 832.7], [1382.5, 833.0],
                [1393.2, 833.4], [1403.9, 833.8], [1414.6, 834.2]], ext=[30, 60],
         what="cross member just inside the X=1600 board: upper edge",
         model_line={"fixed": {"X": 1330, "Z": 0}, "free": "Y", "range_mm": [0, 450], "exact": False,
                     "note": FLNOTE}),
    dict(id="u_div1_low", straight_3d=False, verified="yes",
         note="checked: clean flange lower edge from the label holder to the back post. BENT? raw bow -1.45 px over 236 px, while a straight line here would bow +1.1..+1.6 px under the initial / prototype lens models (opposite sign) -> flange edge probably sagging or curved; straight_3d set false", direction="unknown",
         guide=[[1386.6, 226.1], [1423.2, 232.9], [1459.8, 239.6], [1496.4, 246.2], [1533.0, 252.6],
                [1569.8, 258.5], [1606.6, 264.0], [1643.5, 268.8]],
         what="first cross divider: lower (towards X=1600) edge of the top flange; straight but ~5 deg off "
              "the cart-Y vanishing direction (flange not exactly horizontal / tapered)",
         model_line=None),
    dict(id="u_div2_low", verified="yes",
         note="checked: clean flange lower edge from the label holder to the back post", direction="unknown",
         guide=[[1368.5, 391.7], [1403.2, 394.6], [1438.0, 397.7], [1472.7, 401.1], [1507.3, 404.7],
                [1542.0, 408.5], [1576.6, 412.4], [1611.3, 416.6]],
         what="second cross divider: lower edge of the top flange; straight, ~3 deg off cart-Y",
         model_line=None),
    # ============================================================== cart-Z
    dict(id="z_post1600_sil", verified="partly",
         note="post silhouette above the handle; short (~100 px), several rungs/brackets interrupt it", direction="cartZ",
         guide=[[1136.3, 590.6], [1144.2, 611.2], [1152.0, 631.7], [1159.8, 652.3], [1167.6, 672.9],
                [1175.3, 693.5], [1183.0, 714.1], [1190.6, 734.7]], ext=[0, 10],
         what="front corner post at the X=1600 end (vertical, towards the nadir): left (aisle-side) "
              "silhouette against the floor, part above the push handle (y 590..740)",
         model_line={"fixed": {"X": 1600, "Y": 0}, "free": "Z", "range_mm": [-1800, 0], "exact": False,
                     "note": "silhouette = outer corner (X=1600, Y=0) of the post only if the post is flush with "
                             "the cart outline; post profile not in the drawing"}),
    dict(id="z_post1600_sil_low", verified="partly",
         note="post silhouette below the handle; short, interrupted; does not lie on one smooth curve with z_post1600_sil (0.6-0.9 px), so kept separate", direction="cartZ",
         guide=[[1202.3, 751.9], [1212.2, 779.4], [1222.1, 806.8], [1231.9, 834.2], [1241.8, 861.6],
                [1251.7, 889.0], [1261.6, 916.4], [1271.5, 943.9]], ext=[0, 10],
         what="front corner post at the X=1600 end: left silhouette of the post below the push handle "
              "(y 750..950), floor on the left",
         model_line={"fixed": {"X": 1600, "Y": 0}, "free": "Z", "range_mm": [-1800, 0], "exact": False,
                     "note": "post outline, profile not in the drawing"}),
    # ============================================================== scene structure right of the cart
    dict(id="s_rail_right", verified="partly",
         note="long straight dark steel rail behind the cart (scene, not cart); edge clean except next to the white reflector blob; physical straightness assumed (steel profile)", direction="horizontal_other", cls="scene_structure", cart=None,
         guide=[[1635.9, 318.8], [1631.8, 383.0], [1627.0, 447.2], [1621.2, 511.2], [1614.3, 575.1],
                [1606.2, 638.9], [1596.5, 702.6], [1585.1, 766.0]], ext=[0, 0],
         what="dark steel rail of the rack / ladder structure behind cart 80: the other long boundary",
         model_line=None),
    dict(id="s_frame", verified="partly",
         note="door/wall frame profile edge right of the cart (scene); clean; straightness assumed", direction="horizontal_other", cls="scene_structure", cart=None,
         guide=[[1718.2, 323.6], [1715.2, 371.6], [1711.5, 419.5], [1707.1, 467.3], [1702.1, 515.0],
                [1696.4, 562.7], [1689.9, 610.3], [1682.6, 657.7]], ext=[0, 0],
         what="door / wall frame right of cart 80: long straight boundary of the dark frame profile",
         model_line=None),
    dict(id="s_frame2", direction="horizontal_other", cls="scene_structure", cart=None, verified="partly",
         note="checked in 4 windows: clean boundary of the frame profile over y 300..800 (scene, not cart); physical "
              "straightness assumed (metal frame); not independent of s_frame / s_frame3 (same profile)",
         guide=[[1723.3, 301.1], [1718.4, 373.1], [1712.2, 444.9], [1704.8, 516.5], [1695.8, 588.0],
                [1685.4, 659.2], [1673.3, 730.3], [1659.6, 801.1]], ext=[0, 0],
         what="door / wall frame right of cart 80: second long boundary of the frame profile",
         model_line=None),
    dict(id="s_frame3", direction="horizontal_other", cls="scene_structure", cart=None, verified="partly",
         note="checked in 4 windows: clean, a few points next to a screw head rejected (scene, not cart); physical "
              "straightness assumed; same profile as s_frame2",
         guide=[[1729.0, 312.6], [1724.6, 375.4], [1719.3, 438.0], [1712.8, 500.4], [1705.0, 562.7],
                [1695.7, 624.9], [1684.6, 686.8], [1671.7, 748.4]], ext=[0, 0],
         what="door / wall frame right of cart 80: third long boundary of the frame profile",
         model_line=None),
]
