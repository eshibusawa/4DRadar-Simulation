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

"""The steps of the openEMS tutorial Simple_Patch_Antenna as the methods of a class, for the models of Section 2 and later.

Section 1 of notebook 05 follows the tutorial step by step in one function (patch_antenna.simulate_patch).
Section 2 changes the structure (a series-fed element, a microstrip port) and the conditions; for this the same steps are
written here as the methods of PatchAntenna, one method per step, and a model is a caller that calls them in order:

    General parameter setup     PatchAntenna(f0, fc, substrate, cells_per_wavelength)
    1. FDTD setup               setup_fdtd
    2. domain                   set_domain                        (sets x_first, used by add_element)
    2.1 substrate and ground    add_substrate_ground
    2.2 copper                  add_patch | add_element | add_copper, add_triangles, add_vias, add_mesh_lines (Section 3)
    2.3 / 2.4 port and mesh     add_lumped_port, then smooth_mesh (lumped port: before the mesh is smoothed, as in the tutorial)
                                smooth_mesh, then add_msl_port   (microstrip port: after the mesh is fixed, openEMS reference;
                                                                  along x (Section 2) or y (Section 3), one or several ports)
    2.5 near-field to far-field add_nf2ff
    3. run                      run
    4. post-processing          s11, far_field

single_patch(preset, sim_path) is the model of Section 1 with these steps; a preset is the dictionary of Section 1
(patch_antenna.settings completes it). compare_with_simulate_patch checks that it builds the same model as
patch_antenna.simulate_patch, for the presets stored in openems/data/patch_presets.
"""
import os
import time
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0, EPS0

from patch_antenna import RO3003, settings, simulate_patch, load_result


class PatchAntenna:
    """one openEMS model, built step by step as in the tutorial Simple_Patch_Antenna (lengths in mm)"""

    def __init__(self, f0, fc=10e9, substrate=RO3003, cells_per_wavelength=20):
        """General parameter setup: Gaussian excitation f0 +- fc; substrate: epsr, tand, h [mm];
        the largest cell C0/(f0+fc)/cells_per_wavelength (tutorial: mesh_res = C0/(f0+fc)/unit/20)"""
        self.f0, self.fc = f0, fc
        self.epsr, self.tand, self.h = substrate['epsr'], substrate['tand'], substrate['h']
        self.res = C0 / (f0 + fc) / 1e-3 / cells_per_wavelength

    def setup_fdtd(self, nr_ts, end_criteria=1e-4):
        """1. FDTD setup: Gaussian excitation, PML (8 cells) on all sides, end at -40 dB (tutorial: EndCriteria=1e-4)"""
        self.FDTD = openEMS(NrTS=nr_ts, EndCriteria=end_criteria)
        self.FDTD.SetGaussExcite(self.f0, self.fc)
        self.FDTD.SetBoundaryCond(['PML_8'] * 6)
        self.CSX = ContinuousStructure()
        self.FDTD.SetCSX(self.CSX)
        self.mesh = self.CSX.GetGrid()
        self.mesh.SetDeltaUnit(1e-3)

    def set_domain(self, x, y, z, air=None):
        """2. the domain: `air` outside the structure (x, y, z: its extent), then the NF2FF line and the PML (8 cells)
        air: default half a free-space wavelength at f0 (tutorial Horn_Antenna: margin = lam0/2 + bc_offset*max_res, bc_offset = 9)"""
        air = C0 / self.f0 / 1e-3 / 2 if air is None else air
        margin = air + 9 * self.res
        for axis, (a, b) in zip('xyz', (x, y, z)):
            self.mesh.AddLine(axis, [a - margin, b + margin])
        self.x_first = x[0] - margin               # the first mesh line on the -x side, inside the PML
        self.y_first = y[0] - margin               # the first mesh line on the -y side, inside the PML

    def add_substrate_ground(self, x, y, substrate_cells=4):
        """2.1 the substrate over x, y with substrate_cells in z (tutorial: 4), and the ground plane at z = 0"""
        sub = self.CSX.AddMaterial('substrate', epsilon=self.epsr, kappa=2 * np.pi * self.f0 * EPS0 * self.epsr * self.tand)
        sub.AddBox([x[0], y[0], 0], [x[1], y[1], self.h], priority=0)
        self.mesh.AddLine('z', np.linspace(0, self.h, substrate_cells + 1))
        gnd = self.CSX.AddMetal('gnd')
        gnd.AddBox([x[0], y[0], 0], [x[1], y[1], 0], priority=10)
        self.FDTD.AddEdges2Grid(dirs='xy', properties=gnd)

    def add_patch(self, L, W, metal_edge):
        """2.2 a rectangular patch (length L along x, width W along y) at z = h, zero-thickness PEC with the edge-refined mesh"""
        self.metal = self.CSX.AddMetal('patch')
        self.metal.AddBox([-L / 2, -W / 2, self.h], [L / 2, W / 2, self.h], priority=10)
        self.FDTD.AddEdges2Grid(dirs='xy', properties=self.metal, metal_edge_res=metal_edge)

    def add_element(self, sections, feed_width, metal_edge):
        """2.2 a series-fed element along x: the feed line from the first mesh line (inside the PML) to x = 0 and the sections
        (name, x0, x1, width), zero-thickness PEC boxes with the edge-refined mesh"""
        self.metal = self.CSX.AddMetal('cu')
        self.metal.AddBox([self.x_first, -feed_width / 2, self.h], [0, feed_width / 2, self.h], priority=10)
        for name, x0, x1, width in sections:
            self.metal.AddBox([x0, -width / 2, self.h], [x1, width / 2, self.h], priority=10)
        self.FDTD.AddEdges2Grid(dirs='xy', properties=self.metal, metal_edge_res=metal_edge)

    def add_copper(self, primitives, metal_edge):
        """2.2 copper of any shape at z = h (Section 3): primitives ('box', x0, y0, x1, y1) or ('polygon', points [n, 2]) in this
        order, zero-thickness PEC with the edge-refined mesh (it works on the boxes; a polygon gets no refinement)"""
        self.metal = self.CSX.AddMetal('cu')
        for p in primitives:
            if p[0] == 'box':
                self.metal.AddBox([p[1], p[2], self.h], [p[3], p[4], self.h], priority=10)
            else:
                self.metal.AddPolygon(np.asarray(p[1]).T, 'z', self.h, priority=10)
        self.FDTD.AddEdges2Grid(dirs='xy', properties=self.metal, metal_edge_res=metal_edge)

    def add_triangles(self, triangles):
        """2.2 copper with holes (the pour) as triangles [3, 2] at z = h, without the edge-refined mesh"""
        self.metal_tri = self.CSX.AddMetal('cu_tri')
        for t in triangles:
            self.metal_tri.AddPolygon(np.asarray(t).T, 'z', self.h, priority=10)

    def add_vias(self, positions, radius):
        """2.2 vias between the ground (z = 0) and the copper (z = h): PEC cylinders, part of the metal of add_copper"""
        for x, y in positions:
            self.metal.AddCylinder([x, y, 0], [x, y, self.h], radius, priority=15)

    def add_mesh_lines(self, x, y):
        """2.2 extra mesh lines in x and y (for example at the vertices of the triangles)"""
        if len(x):
            self.mesh.AddLine('x', x)
        if len(y):
            self.mesh.AddLine('y', y)

    def add_lumped_port(self, feed_x, R=50):
        """2.4 a lumped port (R ohm) from the ground to the patch at (feed_x, 0); it adds its own mesh lines, so it is created
        before the mesh is smoothed (tutorial)"""
        self.port = self.FDTD.AddLumpedPort(1, R, [feed_x, 0, 0], [feed_x, 0, self.h], 'z', 1.0, priority=5, edges2grid='xy')

    def smooth_mesh(self, z_step=None):
        """2.3 smooth the mesh: cells up to the largest cell, ratio 1.4 (tutorial); z_step: an extra limit of the cells in z"""
        self.mesh.SmoothMeshLines('all', self.res, 1.4)
        if z_step is not None:
            self.mesh.SmoothMeshLines('z', z_step, 1.4)

    def add_msl_port(self, width, stop, feed_shift, meas_shift, axis='x', center=0.0, number=1, excite=-1):
        """2.4 a microstrip port on a feed line along `axis` ('x' or 'y') centred at `center` across it, after the mesh is fixed;
        it starts at the first mesh line, inside the PML, and ends at `stop` (openEMS reference); feed_shift, meas_shift: functions
        of the PML thickness and the length of the port; excite = -1 for the excited port, 0 for a matched load.
        The ports are kept in self.ports, the excited one also in self.port."""
        lines = self.mesh.GetLines(axis)
        pml = float(np.sum(np.diff(lines[:9])))
        if axis == 'x':
            start, end = [lines[0], center - width / 2, self.h], [stop, center + width / 2, 0]
        else:
            start, end = [center - width / 2, lines[0], self.h], [center + width / 2, stop, 0]
        port = self.FDTD.AddMSLPort(number, self.metal, start, end, axis, 'z', excite=excite,
                                    FeedShift=feed_shift(pml, stop - lines[0]), MeasPlaneShift=meas_shift(pml, stop - lines[0]),
                                    priority=20)
        self.ports = getattr(self, 'ports', []) + [port]
        if excite:
            self.port = port
        return port

    def add_nf2ff(self, skip_face=None):
        """2.5 the near-field to far-field box; skip_face (0 ... 5 = x-, x+, y-, y+, z-, z+): a face left out"""
        if skip_face is None:
            self.nf2ff = self.FDTD.CreateNF2FFBox()
        else:
            self.nf2ff = self.FDTD.CreateNF2FFBox(directions=[k != skip_face for k in range(6)])

    def write_xml(self, path):
        """the model (CSXCAD), to compare it with another implementation"""
        self.CSX.Write2XML(path)

    def run(self, sim_path):
        """3. run the simulation; back to the working directory of the caller afterwards (openEMS.Run changes it to sim_path)"""
        cwd, t0 = os.getcwd(), time.time()
        try:
            self.FDTD.Run(sim_path, cleanup=True, verbose=0)
        finally:
            os.chdir(cwd)
        return time.time() - t0

    def s11(self, sim_path, n=401):
        """4. post-processing: the S11 over the band of the excitation"""
        f = np.linspace(self.f0 - self.fc, self.f0 + self.fc, n)
        self.port.CalcPort(sim_path, f)
        return f, self.port.uf_ref / self.port.uf_inc

    def far_field(self, sim_path, fs, center):
        """4. post-processing: the cuts phi = 0 and 90 deg at the frequencies fs; the pattern relative to the maximum [dB] and Dmax"""
        theta = np.arange(-180, 180.1, 1.0)
        r = self.nf2ff.CalcNF2FF(sim_path, fs, theta, [0., 90.], center=center)
        E = np.array(r.E_norm)
        return theta, 20 * np.log10(E / E.max(axis=(1, 2), keepdims=True)), np.array(r.Dmax)


# ---------------------------------------------------------------- the single patch of Section 1

def single_patch(preset, sim_path, fc=10e9, cells_per_wavelength=20, metal_edge=0.03, substrate_cells=4, write_xml=False):
    """the model of Section 1 (patch_antenna.simulate_patch) with the steps of PatchAntenna; same arguments and result
    write_xml=True: write the model (CSXCAD, sim_path + '.xml') and stop before the run"""
    s = settings(preset)
    g = s['gnd']
    pa = PatchAntenna(s['f0'], fc, s, cells_per_wavelength)
    pa.setup_fdtd(nr_ts=100000)
    pa.set_domain((-g / 2, g / 2), (-g / 2, g / 2), (0, pa.h), air=C0 / pa.f0 / 1e-3)     # one wavelength of air (Section 1)
    pa.add_substrate_ground((-g / 2, g / 2), (-g / 2, g / 2), substrate_cells)
    pa.add_patch(s['L'], s['W'], metal_edge)
    pa.add_lumped_port(s['feed_x'])
    pa.smooth_mesh()
    pa.add_nf2ff()
    if write_xml:
        pa.write_xml(sim_path + '.xml')
        return None
    wall = pa.run(sim_path)
    f, s11 = pa.s11(sim_path)
    zin = pa.port.uf_tot / pa.port.if_tot
    i = int(np.argmin(np.abs(s11)))
    theta, rel, D = pa.far_field(sim_path, [f[i]], center=[0, 0, pa.h * 1e-3])
    return dict(settings=s, f=f, s11=s11, zin=zin, f_res=f[i], theta=theta, pattern=rel[0] + 10 * np.log10(D[0]), Dmax=float(D[0]),
                wall_time=wall)


def compare_with_simulate_patch(npz_path, work_dir):
    """True if single_patch builds the same model as patch_antenna.simulate_patch for the preset stored in npz_path.

    The preset is restored from the settings stored with the result. Only the models are compared (the files written by
    CSXCAD), not the results of runs: openEMS evaluates the end criteria every few seconds of wall-clock time, so the time step
    at which a run stops, and with it the result, changes slightly from run to run, in the function and in the class alike
    (the option exact_endcriteria of FDTD.Run evaluates them on a fixed schedule of time steps instead).
    The display colours in the files are left out of the comparison (model_lines)."""
    preset = load_result(npz_path)['settings']
    name = os.path.splitext(os.path.basename(npz_path))[0]
    os.makedirs(work_dir, exist_ok=True)
    a, b = os.path.join(work_dir, name + '_function'), os.path.join(work_dir, name + '_class')
    simulate_patch(preset, a, write_xml=True)
    single_patch(preset, b, write_xml=True)
    return model_lines(a + '.xml') == model_lines(b + '.xml')


def model_lines(path):
    """the lines of a model file written by CSXCAD, without the display colours (CSXCAD gives each new property a random colour)"""
    with open(path) as fh:
        return [l for l in fh if 'FillColor' not in l and 'EdgeColor' not in l]


if __name__ == '__main__':
    import glob, sys
    here = os.path.dirname(os.path.abspath(__file__))
    work = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, '..', 'tmp', 'patch_antenna_common')
    for p in sorted(glob.glob(os.path.join(here, 'openems', 'data', 'patch_presets', '*.npz'))):
        print('{:20s} {}'.format(os.path.basename(p), 'same model' if compare_with_simulate_patch(p, work) else 'DIFFERENT'))
