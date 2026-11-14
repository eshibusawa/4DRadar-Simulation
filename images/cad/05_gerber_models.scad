// The Gerber-based openEMS models of the RX array A of the MMWCAS-RF-EVM (Rev D), seen as OpenSCAD models (units: mm).
// The geometry is the one given to openEMS: patch_antenna_board.write_scad writes it as images/cad/data/05_patch_antenna_board*.scad
// (python Python/patch_antenna_board.py scad),
// and render_05_models.sh includes one of these files before this file.
//   gold:        the elements and their lines (rectangles, as openEMS gets them)       orange: the strip of the port (from the first mesh line, inside the PML)
//   wheat:       the copper pour (L1)          yellow-green: the dummy element (part of the pour)
//   black:       the L1-L2 micro vias (drawn 1.5 times larger than the 6 mil drill)
//   red / blue:  the start of the port of the excited element / of a matched load (the port starts inside the PML, far in front of the element)
//   green block: the substrate (RO3003, 5 mil), dark plate: the ground (L2)
// The vertical scale is exaggerated (zexag); the copper is a zero-thickness sheet in the simulation, drawn with a thickness only to be seen.
zexag = 6;
h = h_mm * zexag; t = 0.035 * zexag; via_k = 1.5;

function cum(rings, i) = i == 0 ? 0 : len(rings[i - 1]) + cum(rings, i - 1);
module poly_rings(rings) {
    pts = [for (r = rings) for (p = r) p];
    paths = [for (i = [0 : len(rings) - 1]) [for (j = [0 : len(rings[i]) - 1]) cum(rings, i) + j]];
    polygon(pts, paths);
}

module scene() {
    // inside the substrate: first (the substrate is translucent and is drawn last)
    color("black") for (v = vias) translate([v[0], v[1], 0]) cylinder(r = via_r * via_k, h = h + t + 0.03, $fn = 12);
    for (p = ports) color(p[3] == 1 ? "red" : "royalblue") translate([p[0] - strip_w / 2 - 0.02, p[1], 0]) cube([strip_w + 0.04, 0.05, h + t]);
    color("orange") for (p = ports) translate([p[0] - strip_w / 2, p[1], h]) cube([strip_w, p[2] - p[1], t]);
    color("goldenrod") for (b = boxes) translate([b[0], b[1], h]) cube([b[2] - b[0], b[3] - b[1], t]);
    color("wheat") for (q = pour) translate([0, 0, h]) linear_extrude(height = t) poly_rings(q);
    color("yellowgreen") for (q = dummy) translate([0, 0, h]) linear_extrude(height = t) poly_rings(q);
    color("dimgray") translate([sub_box[0], sub_box[1], -t]) cube([sub_box[2] - sub_box[0], sub_box[3] - sub_box[1], t]);
    color("seagreen", 0.3) translate([sub_box[0], sub_box[1], 0.004]) cube([sub_box[2] - sub_box[0], sub_box[3] - sub_box[1], h - 0.004]);   // (shifted by 0.004 from the ground plate: coplanar faces make the translucent block flicker)
}
scene();
