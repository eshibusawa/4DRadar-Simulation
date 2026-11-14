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

"""Read the copper of a Gerber layer (TIDM-464 Rev D) as shapely geometry, for the openEMS models.

Every graphic object of the layer (region, line, arc, flash) is converted to a polygon in mm; the polygons of a window are united,
so that copper connected to the large pour (for example the dummy elements at both ends of RX array A) is part of the same polygon,
as it is in the file.

usage (as a module):
    from gerber_geometry import load_copper, GTL
    cu = load_copper(GTL, (134, 144, 143, 158))     # shapely (Multi)Polygon in mm, board coordinates
"""
import os, re
import numpy as np
from gerbonara import GerberFile, utils
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

here = os.path.dirname(os.path.abspath(__file__))
GTL = os.path.join(here, '..', 'tmp', 'tidm464', 'Project Outputs', 'GerberNCdrills', 'PROC054D.GTL')   # L1 of Rev D

from patch_antenna import MIL      # the main line (one definition)
TX1 = os.path.join(os.path.dirname(GTL), 'PROC054D.TX1')   # NC drill file of the layer pair L1-L2 (the micro vias, 6 mil)
VIA_DIAMETER = 0.15                                           # mm (6 mil drill)


def _polygons(obj, max_error=0.005):
    """polygons (shapely) of one graphic object, in mm"""
    out = []
    for prim in obj.to_primitives(unit=utils.MM):
        ap = prim.to_arc_poly()
        ap = ap.approximate_arcs(max_error=max_error) if hasattr(ap, 'approximate_arcs') else ap
        pts = [(p[0], p[1]) for p in ap.outline]
        if len(pts) >= 3:
            g = Polygon(pts)
            if not g.is_valid:
                g = g.buffer(0)
            if not g.is_empty:
                out.append((g, getattr(prim, 'polarity_dark', True)))
    return out


def load_copper(path=GTL, window=None, max_error=0.005):
    """copper of the layer inside the window (x0, y0, x1, y1) in mm, as a shapely geometry clipped to the window.

    Objects with clear polarity are subtracted. Pads (flashes) are included with their outline."""
    f = GerberFile.open(path)
    dark, clear = [], []
    wbox = box(*window) if window else None
    for o in f.objects:
        (x0, y0), (x1, y1) = o.bounding_box(unit=utils.MM)
        if window and (x1 < window[0] or x0 > window[2] or y1 < window[1] or y0 > window[3]):
            continue
        for g, is_dark in _polygons(o, max_error):
            (dark if is_dark else clear).append(g)
    cu = unary_union(dark)
    if clear:
        cu = cu.difference(unary_union(clear))
    if wbox is not None:
        cu = cu.intersection(wbox)
    return cu


def elements(cu, length=10.09, width=55.12 * MIL, tol=0.02):
    """the antenna elements of Table 7 which are isolated polygons (the active elements): (bounding box, polygon)"""
    out = []
    for g in getattr(cu, 'geoms', [cu]):
        x0, y0, x1, y1 = g.bounds
        if abs(max(x1 - x0, y1 - y0) - length) < tol and abs(min(x1 - x0, y1 - y0) - width) < tol:
            out.append(g)
    return sorted(out, key=lambda g: g.bounds[0])


def triangles(geom):
    """triangulation of a (Multi)Polygon with holes (constrained Delaunay): list of 3x2 arrays"""
    import shapely
    tris = shapely.constrained_delaunay_triangles(geom)
    return [np.array(t.exterior.coords)[:3] for t in getattr(tris, 'geoms', [tris])]


def load_vias(path=TX1, window=None):
    """positions (x, y) in mm of the holes of an Excellon drill file (inch, 2:5 format; the coordinates may be modal: X or Y alone)"""
    pts, x, y, started = [], None, None, False
    for ln in open(path):
        t = ln.strip()
        if t == '%':
            started = True; continue
        if not started or (t[:1] in 'TMG;' and not t[:1] in 'XY'):
            continue
        mx, my = re.search(r'X(-?\d+)', t), re.search(r'Y(-?\d+)', t)
        if not (mx or my):
            continue
        if mx: x = int(mx.group(1)) / 1e5 * 25.4
        if my: y = int(my.group(1)) / 1e5 * 25.4
        pts.append((x, y))
    pts = np.array(pts)
    if window is not None:
        x0, y0, x1, y1 = window
        pts = pts[(pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)]
    return pts
