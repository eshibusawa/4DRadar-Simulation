# This file is part of 4DRadar-Simulation.
# Copyright (c) 2026, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""The element on the board (Section 3): a part of RX array A of the MMWCAS-RF-EVM (Rev D), built from the Gerber file of L1, with the
steps of patch_antenna_common.PatchAntenna.

The copper of L1 in a window (the active elements with their feed lines, the dummy element, the copper pour around them) is read by
gerber_geometry.py and given to openEMS as zero-thickness PEC on RO3003 (5 mil) over an ideal ground (L2): rectilinear polygons as
boxes, the other polygons without holes as polygons, the pour (with holes) as triangles, the L1-L2 micro vias of the drill file as
cylinders. Each active element gets a microstrip port; one port is excited and the others are matched loads.

    geometry = board_window(window, excite, isolate)    # the copper, the vias and the ports in model coordinates
    result = simulate_board(geometry, sim_path)          # the result as a dictionary (S parameters, far field)

Coordinates of the model are the board coordinates shifted so that the excited element starts at (0, 0): the long axis of the
elements is y, the E-plane is the yz-plane (phi = 90) and the H-plane the xz-plane (phi = 0).
The numbers without a basis are marked "basis: none".

print_patterns and figure_patterns compare the patterns at 76 GHz with Fig. 18 of the guide (patch_antenna4.load_ti_fig18).

write_scad writes the geometry as OpenSCAD data for images/cad/05_gerber_models.scad (images/cad/render_05_models.sh draws it).

usage: python patch_antenna_board.py scad      (the OpenSCAD data of the three models of Section 3:
                                                images/cad/data/05_patch_antenna_board_<model>.scad)
       python patch_antenna_board.py figures   (the figure of README.md: images/05_patch_antenna_board_eplane.png, the E-plane at
                                                76 GHz of the stored results against TI Fig. 18)
"""
import json, os, sys
import numpy as np
from shapely.geometry import box as sbox, LineString
from shapely.ops import unary_union

from patch_antenna import MIL, RO3003, beamwidth
from patch_antenna_common import PatchAntenna
from gerber_geometry import load_copper, load_vias, triangles, GTL, VIA_DIAMETER

# the board data (Rev D), read from the Gerber file by hand
ACTIVE_X = [137.62, 139.58, 141.53, 143.48, 145.43, 147.38, 149.33, 151.28]   # centres of RX1 ... RX8 [mm]
Y_START = 146.21          # the element (A1) starts here; the feed line comes from smaller y
Y_END = 156.30            # end of the last patch
F1 = 8.4 * MIL            # width of the feed line (Table 7)
DUMMY_X = 135.67          # the dummy element of RX array A (part of the pour); for the figures only
WINDOW = (134.6, 145.3, 138.6, Y_END + 0.2)    # x0, y0, x1, y1 [mm]: the dummy and RX1; 145.3 and + 0.2: basis: none


def rectangles(g, tol=2e-3):
    """a rectilinear polygon as a list of rectangles (x0, y0, x1, y1): slabs between the y values of the vertices, equal neighbours
    merged; None if not rectilinear (tol and the rounding to 3 decimals: basis: none)"""
    c = np.array(g.exterior.coords)
    d = np.abs(np.diff(c, axis=0))
    if not np.all((d[:, 0] < tol) | (d[:, 1] < tol)):
        return None
    ys = sorted(set(np.round(c[:, 1], 3)))
    out = []
    for y0, y1 in zip(ys[:-1], ys[1:]):
        if y1 - y0 < tol: continue
        inter = g.intersection(LineString([(g.bounds[0] - 1, (y0 + y1) / 2), (g.bounds[2] + 1, (y0 + y1) / 2)]))
        for s in ([inter] if inter.geom_type == 'LineString' else list(getattr(inter, 'geoms', []))):
            if s.length < tol: continue
            xa, xb = sorted([s.coords[0][0], s.coords[-1][0]])
            if out and abs(out[-1][0] - xa) < tol and abs(out[-1][2] - xb) < tol and abs(out[-1][3] - y0) < tol:
                out[-1] = (xa, out[-1][1], xb, y1)
            else:
                out.append((xa, y0, xb, y1))
    return out


def thin(v, dmin=0.03):
    """sorted mesh lines, no two closer than dmin (basis: none)"""
    v = np.sort(np.unique(np.round(v, 3))); out = [v[0]]
    for a in v[1:]:
        if a - out[-1] >= dmin: out.append(a)
    return out


def copper_primitives(simple, origin):
    """the polygons without holes as primitives of PatchAntenna.add_copper (model coordinates): rectilinear ones as boxes
    (the edge-refined mesh of openEMS works on boxes), the others as polygons"""
    ox, oy = origin
    out = []
    for g in simple:
        rects = rectangles(g)
        if rects is None:
            out.append(('polygon', np.array(g.exterior.coords)[:-1] - np.array([ox, oy])))
        else:
            out += [('box', xa - ox, ya - oy, xb - ox, yb - oy) for xa, ya, xb, yb in rects]
    return out


def board_window(window=WINDOW, excite=ACTIVE_X[0], isolate=False, margin=1.0, vias=True, copper=None):
    """the geometry of a window of the board in model coordinates (the excited element starts at (0, 0))
    window: x0, y0, x1, y1 of the copper [mm, board coordinates]; excite: x of the excited element; isolate: only the excited element
    and its feed line (no pour, no dummy, no vias); margin: the substrate and the ground outside the window [mm] (basis: none);
    copper: a shapely geometry instead of the copper of the Gerber file (board coordinates)"""
    X0, Y0, X1, Y1 = window
    xe = excite
    active = [x for x in ACTIVE_X if X0 + 0.7 < x < X1 - 0.7]          # 0.7: basis: none
    assert any(abs(x - xe) < 0.01 for x in active), 'the excited element must be inside the window'
    cu = load_copper(GTL, (X0, Y0, X1, Y1)) if copper is None else copper
    if isolate and copper is None:      # basis: none (board coordinates typed in)
        corridor = unary_union([sbox(xe - 0.7, 146.9, xe + 0.7, Y1 + 1), sbox(xe - 0.15, Y0 - 0.1, xe + 0.15, 147.0)])
        cu = cu.intersection(corridor)
    ox, oy = xe, Y_START
    # the copper below Y_START - 0.5 mm is replaced by the strip of the port (the round end of the Gerber line is not rectilinear);
    # the slivers are dropped (0.5 and the area 0.02 mm2: basis: none)
    cu = cu.intersection(sbox(X0 - 1, Y_START - 0.5, X1 + 1, Y1 + 1)).union(cu.intersection(sbox(X0 - 1, Y0 - 1, X1 + 1, Y_START - 0.5)).difference(unary_union([sbox(x_ - F1, Y0 - 1, x_ + F1, Y_START) for x_ in active])))
    polys = [g for g in getattr(cu, 'geoms', [cu]) if g.area > 0.02]
    simple = [g for g in polys if len(g.interiors) == 0]      # polygons without holes: boxes or polygons (edge-refined mesh)
    holed = [g for g in polys if len(g.interiors) > 0]       # the pour: triangles
    tri = [t - np.array([ox, oy]) for g in holed for t in triangles(g)]
    sx0, sx1, sy0, sy1 = X0 - ox - margin, X1 - ox + margin, Y0 - oy - 0.3, Y1 - oy + 2.0      # 0.3, 2.0: basis: none
    via_xy = load_vias(window=(X0, Y0, X1, Y1)) if (vias and not isolate) else np.empty((0, 2))
    rv = VIA_DIAMETER / 2
    # mesh lines at the edges of the pour and of the vias: the vertices, thinned out; only 0.2 mm inside the substrate (basis: none)
    vx, vy = [], []
    for g in holed:
        for ring in [g.exterior] + list(g.interiors):
            c = np.array(ring.coords) - np.array([ox, oy]); vx += list(c[:, 0]); vy += list(c[:, 1])
    vx, vy = np.array(vx), np.array(vy)
    if len(via_xy):
        vx = np.concatenate([vx, (via_xy[:, 0] - ox)[:, None].repeat(3, 1).ravel() + np.tile([-rv, 0, rv], len(via_xy))])
        vy = np.concatenate([vy, (via_xy[:, 1] - oy)[:, None].repeat(3, 1).ravel() + np.tile([-rv, 0, rv], len(via_xy))])
    keep = (vx > sx0 + 0.2) & (vx < sx1 - 0.2) & (vy > sy0 + 0.2) & (vy < sy1 - 0.2)
    lines = (thin(vx[keep]), thin(vy[keep])) if keep.any() else ([], [])
    return dict(window=tuple(window), excite=xe, active=active, origin=(ox, oy), simple=simple, holed=holed,
                copper=copper_primitives(simple, (ox, oy)), triangles=tri, vias=via_xy - np.array([ox, oy]) if len(via_xy) else via_xy,
                via_radius=rv, mesh_lines=lines, substrate=((sx0, sx1), (sy0, sy1)), feed_width=F1,
                port_end=Y_START - oy - 0.10,                                       # 0.10: basis: none
                center=((X0 + X1) / 2 - ox, (Y0 + Y1) / 2 - oy))


def simulate_board(geometry, sim_path, f0=78e9, fc=10e9, frequencies=(76e9, 77e9, 78e9, 79e9, 80e9, 81e9), substrate=RO3003,
                   cells_per_wavelength=20, metal_edge=0.04, substrate_cells=4, z_step=0.05, skip_face=2, run=True,
                   write_xml=False, scad=None):
    """the FDTD simulation of a window of the board (Section 3) and its result as a dictionary
    The domain is lambda0/2 of air + 9 cells (tutorial Horn_Antenna); the substrate and the ground run on the side of the feed lines
    to the first mesh line, where the microstrip ports start; the far field without the face y- that the feed lines cross
    (skip_face=2). metal_edge, z_step: basis: none (as in Section 2). run=False: post-processing of an earlier run;
    write_xml: write the model and stop; scad=(path, title): write the geometry as OpenSCAD data (write_scad) and stop."""
    g = geometry
    (sx0, sx1), (sy0, sy1) = g['substrate']
    pa = PatchAntenna(f0, fc, substrate, cells_per_wavelength)
    pa.setup_fdtd(nr_ts=300000)                                         # basis: none
    pa.set_domain((sx0, sx1), (sy0, sy1), (0, pa.h))
    pa.add_substrate_ground((sx0, sx1), (pa.y_first, sy1), substrate_cells)
    ox, oy = g['origin']
    strips = [('box', x - ox - g['feed_width'] / 2, pa.y_first, x - ox + g['feed_width'] / 2, g['port_end']) for x in g['active']]
    pa.add_copper(g['copper'] + strips, metal_edge)
    pa.add_triangles(g['triangles'])
    pa.add_vias(g['vias'], g['via_radius'])
    pa.add_mesh_lines(*g['mesh_lines'])
    pa.smooth_mesh(z_step)
    for k, x in enumerate(g['active']):
        pa.add_msl_port(g['feed_width'], g['port_end'],
                        feed_shift=lambda pml, length: pml + 0.1,                       # as in Section 2; basis: none
                        meas_shift=lambda pml, length: min(pml + 0.5, length - 0.3),
                        axis='y', center=x - ox, number=k + 1, excite=-1 if abs(x - g['excite']) < 0.01 else 0)
    cells = int(np.prod([len(pa.mesh.GetLines(a)) for a in 'xyz']))
    if scad is not None:
        write_scad(g, scad[0], scad[1], pa.y_first)
        return dict(cells=cells)
    pa.add_nf2ff(skip_face)
    if write_xml:
        pa.write_xml(sim_path + '.xml')
        return dict(cells=cells)
    wall = pa.run(sim_path) if run else 0.0
    f = np.linspace(68e9, 88e9, 401)
    for p in pa.ports:
        p.CalcPort(sim_path, f)
    S = {'S_%.2f' % x: p.uf_ref / pa.port.uf_inc for x, p in zip(g['active'], pa.ports)}   # S11 of the excited port, S_k1 the others
    fs = np.array(frequencies)
    theta = np.arange(-180, 180.1, 1.0)
    c = g['center']
    r = pa.nf2ff.CalcNF2FF(sim_path, fs, theta, [0., 90.], center=[c[0] * 1e-3, c[1] * 1e-3, pa.h * 1e-3])
    return dict(f=f, fs=fs, theta=theta, E=np.array(r.E_norm), D=np.array(r.Dmax), ports_x=np.array(g['active']),
                excited=g['excite'], **{k.replace('.', '_'): v for k, v in S.items()})


# ---------------------------------------------------------------- the patterns (notebook 05, Section 3)

COLORS = ('C0', 'C2', 'r')      # the colours of the three models in the figures
LABELS = {'isolated': 'isolated element', 'd_rx1': 'dummy + RX1', 'd_rx1_rx2': 'dummy + RX1 + RX2'}     # the names in the figures


def pattern_summary(result, f=76e9):
    """the numbers of the table of Section 3 at f: maximum directivity [dBi], direction and 3 dB width of the E-plane (phi = 90, yz),
    3 dB width of the H-plane (phi = 0, xz) [deg], and the level of the H-plane at +theta minus the level at -theta at 20, 30, 45 deg"""
    z = result; k = list(z['fs']).index(f); th = z['theta']
    Eb = 20 * np.log10(z['E'][k][:, 1]); Hb = 20 * np.log10(z['E'][k][:, 0])
    return dict(D=10 * np.log10(z['D'][k]), peak=th[np.argmax(Eb)], bw_E=beamwidth(th, Eb), bw_H=beamwidth(th, Hb),
                asym=[asymmetry(th, Hb, x) for x in (20, 30, 45)])


def asymmetry(theta, p, x):
    """the level at +x deg minus the level at -x deg"""
    return float(np.interp(x, theta, p) - np.interp(-x, theta, p))


def print_patterns(results, ti, f=76e9):
    """the table of Section 3: TI Fig. 18 (ti: patch_antenna4.load_ti_fig18; the widths of its legend, the asymmetry read from the
    curve) and the models (results: name -> result of simulate_board)"""
    th = ti['theta']
    print('%.0f GHz            | max [dBi] | E: peak, 3 dB width | H: 3 dB width | H: level(+theta) - level(-theta) at 20, 30, 45 deg' % (f / 1e9))
    print('TI Fig. 18        |   (10.2)  | {:4.0f} deg, {:5.1f} deg | {:6.1f} deg    | {}'.format(
        th[np.argmax(ti['E'])], 22.26, 62.89, ' '.join('{:+.1f}'.format(asymmetry(th, ti['H'], x)) for x in (20, 30, 45))))
    for name, z in results.items():
        s = pattern_summary(z, f)
        print('{:17s} |  {:6.2f}   | {:4.0f} deg, {:5.1f} deg | {:6.1f} deg    | {}'.format(
            name, s['D'], s['peak'], s['bw_E'], s['bw_H'], ' '.join('{:+.1f}'.format(v) for v in s['asym'])))


def figure_patterns(results, ti, f=76e9, planes=('E', 'H')):
    """the E-plane and the H-plane at f against TI Fig. 18, each relative to its maximum (the figure of TI is a gain, the simulation a
    directivity); results: name -> result of simulate_board, in the order of COLORS; planes: the panels, 'E' and / or 'H'"""
    import matplotlib.pyplot as plt
    plane = {'E': (1, 'E-plane at %.0f GHz' % (f / 1e9), -35),                                 # phi = 90: E-plane (yz)
             'H': (0, 'H-plane at %.0f GHz (+theta = +x, the side of RX2)' % (f / 1e9), -25)}  # phi = 0: H-plane (xz)
    fig, ax = plt.subplots(1, len(planes), figsize=(8 * len(planes), 5), squeeze=False)
    ax = ax[0]
    for a, p in zip(ax, planes):
        a.plot(ti['theta'], ti[p], 'k.', ms=2, label='TI Fig. 18')
    for (name, z), col in zip(results.items(), COLORS):
        k = list(z['fs']).index(f); th = z['theta']
        for a, p in zip(ax, planes):
            v = 20 * np.log10(z['E'][k][:, plane[p][0]])
            a.plot(th, v - v.max(), color=col, ls='-', lw=1.4, label=name)
    for a, p in zip(ax, planes):
        a.set_xlim(-90, 90); a.set_ylim(plane[p][2], 3); a.grid(); a.legend(loc='lower center', fontsize=9)
        a.set_xlabel('Theta [deg]'); a.set_ylabel('Relative to the main lobe [dB]'); a.set_title(plane[p][1])
    return fig


def write_scad(geometry, path, title, substrate_y0):
    """the geometry as given to openEMS, as OpenSCAD data for images/cad/05_gerber_models.scad (model coordinates);
    substrate_y0: the first mesh line on the side of the ports (PatchAntenna.y_first), where the substrate and the ports start"""
    g = geometry
    ox, oy = g['origin']
    shift = lambda c: [[round(float(a) - ox, 4), round(float(b) - oy, 4)] for a, b in c]
    dummy_region = sbox(DUMMY_X - 0.8, 146.9, DUMMY_X + 0.8, g['window'][3] + 5)
    rect_boxes, other, dummy, pour = [], [], [], []
    for s in g['simple']:
        rc = rectangles(s)
        if rc is None: other.append(s)
        else: rect_boxes += [[round(xa - ox, 4), round(ya - oy, 4), round(xb - ox, 4), round(yb - oy, 4)] for xa, ya, xb, yb in rc]
    other += g['holed']
    for s in other:
        for part, lst in ((s.difference(dummy_region), pour), (s.intersection(dummy_region), dummy)):
            for q in getattr(part, 'geoms', [part]):
                if q.geom_type == 'Polygon' and q.area > 1e-3:
                    lst.append([shift(q.exterior.coords)[:-1]] + [shift(i.coords)[:-1] for i in q.interiors])
    fmt = lambda v: json.dumps(v, default=float)
    (sx0, sx1), (sy0, sy1) = g['substrate']
    vias = [[round(float(a), 4), round(float(b), 4)] for a, b in g['vias']]
    with open(path, 'w') as fh:
        fh.write('// generated by patch_antenna_board.write_scad: the geometry given to openEMS, in mm, model coordinates (the excited element starts at y = 0)\n')
        fh.write('model_title = "%s";\n' % title)
        fh.write('h_mm = %g; via_r = %g; strip_w = %g;\n' % (RO3003['h'], g['via_radius'], g['feed_width']))
        fh.write('sub_box = %s;   // x0, y0, x1, y1 of the substrate and the ground; y0 is the first mesh line (inside the PML), where the ports start\n' % fmt([round(sx0, 4), round(substrate_y0, 4), round(sx1, 4), round(sy1, 4)]))
        fh.write('boxes = %s;\n' % fmt(rect_boxes))
        fh.write('ports = %s;   // x of the centre, y of the start, y of the end, 1 = excited, 0 = matched load\n' % fmt(
            [[round(x - ox, 4), round(substrate_y0, 4), round(g['port_end'], 4), 1 if abs(x - g['excite']) < 0.01 else 0] for x in g['active']]))
        fh.write('pour = %s;   // polygons with holes: the list of rings (the first is the outline)\n' % fmt(pour))
        fh.write('dummy = %s;\n' % fmt(dummy))
        fh.write('vias = %s;\n' % fmt(vias))


MODELS = {   # the three models of Section 3 (notebook 05): name: (arguments of board_window, title of the OpenSCAD figure)
    'isolated':  (dict(isolate=True), 'isolated element (element and feed line)'),
    'd_rx1':     (dict(), 'dummy + RX1 (copper pour, vias)'),
    'd_rx1_rx2': (dict(window=(WINDOW[0], WINDOW[1], 140.55, WINDOW[3])), 'dummy + RX1 + RX2 (copper pour, vias)'),
}


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'scad':
    cad = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images', 'cad', 'data')
    for name, (options, title) in MODELS.items():
        path = os.path.join(cad, '05_patch_antenna_board_%s.scad' % name)
        simulate_board(board_window(**options), path[:-5], scad=(path, title))
    print('saved', os.path.abspath(cad), ', '.join('05_patch_antenna_board_%s.scad' % n for n in MODELS))


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'figures':
    import matplotlib; matplotlib.use('Agg')
    from patch_antenna4 import load_ti_fig18
    here = os.path.dirname(os.path.abspath(__file__))
    results = {LABELS[n]: dict(np.load(os.path.join(here, 'openems', 'data', 'patch_antenna_board_%s.npz' % n))) for n in MODELS}
    out = os.path.join(here, '..', 'images', '05_patch_antenna_board_eplane.png')
    figure_patterns(results, load_ti_fig18(), planes=('E',)).savefig(out, dpi=85, bbox_inches='tight')
    print('saved', os.path.abspath(out))
