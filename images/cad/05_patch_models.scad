// Models of the microstrip patch antennas used in 05 (units: mm).
// Dimensions: TI SWRU553A Table 7 (mil) and RO3003 5 mil stack-up (Fig. 30).
// Render inside the container objective_hypatia with render_05.sh
// see render_05.sh
model = "series_element";   // "single" | "series_element" (patch_antenna4.series_element, A1 in red) | "series_element_plain" (A1 not marked) | "section" (single patch cut at y = 0 to show the feed port)
zexag = 6;              // exaggeration of the vertical scale (h = 0.127 mm is very thin)
mil = 0.0254;

h   = 0.127 * zexag;    // RO3003 5 mil
tcu = 0.04 * zexag;     // copper 1.6 mil (shown for clarity only; the simulation uses zero thickness)

// TI series-fed array (Table 7)
A1=27.568*mil; A2=11.802*mil; B1=47.24*mil; B2=49.21*mil; B3=45.27*mil; B4=49.21*mil;
C1=42.52*mil;  C2=40.95*mil;  E1=39.38*mil;  E2=55.12*mil;
F1=8.4*mil;    F2=10.62*mil;  F3=3.94*mil;

// textbook single patch (transmission-line model, 77 GHz)
Ws = 1.377; Ls = 1.064;

module cu(x0, x1, w) { translate([x0, -w/2, h]) cube([x1 - x0, w, tcu]); }

module board(x0, x1, gy) {
    color("seagreen", 0.45) translate([x0, -gy/2, 0]) cube([x1 - x0, gy, h]);      // RO3003
    color("dimgray")        translate([x0, -gy/2, -tcu]) cube([x1 - x0, gy, tcu]); // GND (L2)
}

module single() {
    board(-2, 2, 4);
    color("goldenrod") cu(-Ls/2, Ls/2, Ws);
    color("red") translate([0.2, 0, 0]) cylinder(d = 0.12, h = h);                  // lumped port (diameter exaggerated)
}

module series_element() {   // patch_antenna4.series_element
    a1w = (model == "a1_wrong") ? F1 : F2;   // A1 is red: the corrected section (series_element) or the misread one (a1_wrong)
    x = [0, A1, A1+A2, A1+A2+B1, A1+A2+B1+C1, A1+A2+B1+C1+B2,
         A1+A2+B1+C1+B2+C2, A1+A2+B1+C1+B2+C2+B3, A1+A2+B1+C1+B2+C2+B3+C2,
         A1+A2+B1+C1+B2+C2+B3+C2+B4, A1+A2+B1+C1+B2+C2+B3+C2+B4+C1];
    board(-1.3, x[10] + 2, 6);
    color("goldenrod") {
        cu(-1,   x[0], F1);  cu(x[1], x[2], F2);  cu(x[2], x[3], F3);
        cu(x[3], x[4], E1);  cu(x[4], x[5], F3);  cu(x[5], x[6], E2);  cu(x[6], x[7], F3);
        cu(x[7], x[8], E2);  cu(x[8], x[9], F3);  cu(x[9], x[10], E1);
    }
    color(model == "series_element_plain" ? "goldenrod" : "red") cu(x[0], x[1], a1w);          // A1 (red, except in series_element_plain)
    color("red") translate([-1, -F1/2, 0]) cube([0.02, F1, h]);                    // MSL port plane
}

module section() {   // half model (y >= 0), built directly so that the cut faces keep their colors
    color("red")            translate([0.2 - 0.06, 0.01, 0]) cube([0.12, 0.12, h]);   // lumped port (exaggerated)
    color("seagreen", 0.2) translate([-2, 0, 0]) cube([4, 2, h]);
    color("dimgray")        translate([-2, 0, -tcu]) cube([4, 2, tcu]);
    color("goldenrod")      translate([-Ls/2, 0, h]) cube([Ls, Ws/2, tcu]);
}

if (model == "single") single();
else if (model == "section") section();
else series_element();   // "series_element", "series_element_plain" and "a1_wrong"
