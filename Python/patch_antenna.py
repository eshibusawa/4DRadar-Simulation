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

"""Microstrip patch antennas simulated with openEMS: a single rectangular patch designed by the transmission-line model
(design_patch, simulate_patch). The 4-patch series-fed element of the TI MMWCAS-RF-EVM is in patch_antenna4.py.

A preset is a dictionary of the settings of one simulation; the settings that are not given take the default values:

    f0       design frequency [Hz]                                  (77e9)
    epsr, tand, h   substrate: permittivity, loss tangent, thickness [mm]   (RO3003, 5 mil)
    W, L     patch width (y) and length (x) [mm]                    (designed for f0 by design_patch)
    gnd      side of the square ground plane and substrate [mm]     (4.0)
    feed_x   position of the feed from the patch centre along x [mm] (0.2)

    result = simulate_patch({'gnd': 8.0}, sim_path)
"""
import os
import time
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0, EPS0

MIL = 0.0254                                      # mm
RO3003 = dict(epsr=3.0, tand=0.001, h=0.127)
DEFAULTS = dict(f0=77e9, **RO3003, W=None, L=None, gnd=4.0, feed_x=0.2)


def run_fdtd(FDTD, sim_path):
    """FDTD.Run, then back to the working directory of the caller
    (openEMS.Run changes the working directory to sim_path and does not change it back: relative paths of the caller would break)"""
    cwd = os.getcwd()
    try:
        FDTD.Run(sim_path, cleanup=True, verbose=0)
    finally:
        os.chdir(cwd)


def design_patch(f0, epsr, h):
    """width W, length L, effective permittivity and length extension dL [mm] of a patch resonant at f0 (transmission-line model, Hammerstad)"""
    W = C0 / (2 * f0) * np.sqrt(2 / (epsr + 1)) * 1e3
    eeff = (epsr + 1) / 2 + (epsr - 1) / 2 / np.sqrt(1 + 12 * h / W)
    dL = 0.412 * h * (eeff + 0.3) * (W / h + 0.264) / ((eeff - 0.258) * (W / h + 0.8))
    L = C0 / (2 * f0 * np.sqrt(eeff)) * 1e3 - 2 * dL
    return W, L, eeff, dL


def settings(preset):
    """all the settings of a preset (the defaults completed, W and L designed if not given)"""
    s = dict(DEFAULTS, **preset)
    W, L, _, _ = design_patch(s['f0'], s['epsr'], s['h'])
    s['W'] = W if s['W'] is None else s['W']
    s['L'] = L if s['L'] is None else s['L']
    return s


def simulate_patch(preset, sim_path, fc=10e9, cells_per_wavelength=20, metal_edge=0.03, substrate_cells=4, write_xml=False):
    """run the FDTD simulation of a preset and return the result as a dictionary

    The patch is at z = h, the ground at z = 0, the patch length L along x (E-plane: xz, H-plane: yz).
    The domain is the ground plane plus one free-space wavelength at f0 of air on each side and the PML (8 cells) outside;
    the near-field to far-field box lies between the air and the PML.
    write_xml=True: write the model (CSXCAD, sim_path + '.xml') and stop before the run.
    """
    # General parameter setup
    s = settings(preset)
    f0, h, W, L, g = s['f0'], s['h'], s['W'], s['L'], s['gnd']
    res = C0 / (f0 + fc) * 1e3 / cells_per_wavelength       # largest cell [mm]
    air = C0 / f0 * 1e3                                      # one wavelength of air [mm]
    outer = air + 9 * res                                    # air, the NF2FF line and the PML (8 cells)


    # 1. FDTD setup
    # Limit the simulation to 100k timesteps
    # Define a reduced end criteria of -40dB
    FDTD = openEMS(NrTS=100000, EndCriteria=1e-4)
    FDTD.SetGaussExcite(f0, fc)
    FDTD.SetBoundaryCond(['PML_8'] * 6)

    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(1e-3)

    # 2. Generate properties, primitives and mesh-grid
    mesh.AddLine('x', [-g / 2 - outer, g / 2 + outer])
    mesh.AddLine('y', [-g / 2 - outer, g / 2 + outer])
    mesh.AddLine('z', [-outer, h + outer])
    # 2.1 create substrate
    sub = CSX.AddMaterial('substrate', epsilon=s['epsr'], kappa=2 * np.pi * f0 * EPS0 * s['epsr'] * s['tand'])
    sub.AddBox([-g / 2, -g / 2, 0], [g / 2, g / 2, h], priority=0)
    mesh.AddLine('z', np.linspace(0, h, substrate_cells + 1))
    # 2.2 create ground
    gnd = CSX.AddMetal('gnd'); gnd.AddBox([-g / 2, -g / 2, 0], [g / 2, g / 2, 0], priority=10)
    FDTD.AddEdges2Grid(dirs='xy', properties=gnd)
    # 2.3 create patch
    patch = CSX.AddMetal('patch'); patch.AddBox([-L / 2, -W / 2, h], [L / 2, W / 2, h], priority=10)
    FDTD.AddEdges2Grid(dirs='xy', properties=patch, metal_edge_res=metal_edge)
    # 2.4 apply the excitation & resist as a current source
    # (a lumped port adds its own mesh lines and is created before SmoothMeshLines; a microstrip port is placed on the mesh lines,
    #  so it has to be created after SmoothMeshLines, when the mesh is final)
    port = FDTD.AddLumpedPort(1, 50, [s['feed_x'], 0, 0], [s['feed_x'], 0, h], 'z', 1.0, priority=5, edges2grid='xy')
    mesh.SmoothMeshLines('all', res, 1.4)
    # 2.5 Add the nf2ff recording box
    nf2ff = FDTD.CreateNF2FFBox()
    # air between the structure and the NF2FF box (the box is on the 10th mesh line from the boundary)
    lines = [mesh.GetLines(a) for a in 'xyz']
    gap = min(-g / 2 - lines[0][9], lines[0][-10] - g / 2, -g / 2 - lines[1][9], lines[1][-10] - g / 2, -lines[2][9], lines[2][-10] - h)
    cells = int(np.prod([len(l) for l in lines]))
    if write_xml:                         # the model, to compare it with another implementation
        CSX.Write2XML(sim_path + '.xml')
        return dict(cells=cells)

    # 3. Run the simulation
    t0 = time.time()
    run_fdtd(FDTD, sim_path)
    wall = time.time() - t0

    f = np.linspace(f0 - fc, f0 + fc, 401)
    port.CalcPort(sim_path, f)
    s11 = port.uf_ref / port.uf_inc
    zin = port.uf_tot / port.if_tot
    i = int(np.argmin(np.abs(s11)))
    theta = np.arange(-180, 180.1, 1.0)
    r = nf2ff.CalcNF2FF(sim_path, f[i], theta, [0., 90.], center=[0, 0, h * 1e-3])
    Dmax = float(r.Dmax[0])
    E = r.E_norm[0]
    pattern = 20 * np.log10(E / E.max()) + 10 * np.log10(Dmax)          # directivity [dBi], [theta, phi = 0, 90]
    return dict(settings=s, f=f, s11=s11, zin=zin, f_res=f[i], theta=theta, pattern=pattern, Dmax=Dmax,
                cells=cells, gap=gap, wall_time=wall)


def save_result(result, path):
    s = result['settings']
    np.savez(path, **{k: v for k, v in result.items() if k != 'settings'},
             **{'setting_' + k: v for k, v in s.items()})


def load_result(path):
    d = np.load(path)
    r = {k: d[k] for k in d.files if not k.startswith('setting_')}
    r['settings'] = {k[8:]: d[k].item() if d[k].ndim == 0 else d[k] for k in d.files if k.startswith('setting_')}
    for k, v in r.items():
        if k != 'settings' and v.ndim == 0:
            r[k] = v.item()
    return r


def to_db(x):
    """the magnitude of the complex amplitude x in dB (20 log10 |x|)"""
    return 20 * np.log10(np.abs(x))


def beamwidth(theta, p):
    """3 dB beamwidth [deg] of the main lobe around theta = 0 (|theta| < 90 deg)"""
    idx = np.where((p > p.max() - 3) & (np.abs(theta) < 90))[0]
    return theta[idx[-1]] - theta[idx[0]]


def summary(result):
    """resonant frequency [GHz], minimum S11 [dB], maximum directivity [dBi], 3 dB beamwidths [deg] (E, H), front-to-back ratio [dB]"""
    th, p = result['theta'], result['pattern']
    back = p[np.argmin(np.abs(np.abs(th) - 180)), 0]
    return dict(f_res=result['f_res'] / 1e9, s11_min=to_db(result['s11']).min(), Dmax=10 * np.log10(result['Dmax']),
                bw_E=beamwidth(th, p[:, 0]), bw_H=beamwidth(th, p[:, 1]), front_to_back=p[np.argmin(np.abs(th)), 0] - back)


# ---------------------------------------------------------------- figures

LABELS = dict(f0=('design frequency', 1e-9, 'GHz'), epsr=('permittivity', 1, ''), h=('substrate thickness', 1, 'mm'),
              W=('patch width W', 1, 'mm'), L=('patch length L', 1, 'mm'), gnd=('ground plane (square)', 1, 'mm'),
              feed_x=('feed position', 1, 'mm'))


def draw_top_view(ax, s, extent=9.0):
    """the ground plane, the patch and the feed seen from above, to scale"""
    import matplotlib.patches as mp
    g = s['gnd']
    ax.add_patch(mp.Rectangle((-g / 2, -g / 2), g, g, fc='#b8d8b0', ec='#2e7d32', label='substrate and ground'))
    ax.add_patch(mp.Rectangle((-s['L'] / 2, -s['W'] / 2), s['L'], s['W'], fc='#d4a017', ec='k', lw=0.5, label='patch'))
    ax.plot(s['feed_x'], 0, 'r.', ms=6, label='feed')
    lam = C0 / s['f0'] * 1e3
    ax.plot([-extent + 0.5, -extent + 0.5 + lam], [-extent + 0.7] * 2, 'k-', lw=2)
    ax.text(-extent + 0.5 + lam / 2, -extent + 1.0, 'one wavelength', ha='center', fontsize=8)
    ax.set_xlim(-extent, extent); ax.set_ylim(-extent, extent); ax.set_aspect('equal')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]'); ax.set_title('model (top view)')
    ax.legend(loc='upper right', fontsize=7)


def draw_settings(ax, s, ref=None):
    """the settings as a table; the values that differ from `ref` in red"""
    ax.axis('off')
    for k, (key, (name, scale, unit)) in enumerate(LABELS.items()):
        v = s[key] * scale
        changed = ref is not None and abs(ref[key] * scale - v) > 1e-9
        ax.text(0.0, 0.92 - 0.12 * k, name, fontsize=10, transform=ax.transAxes)
        ax.text(1.0, 0.92 - 0.12 * k, ('%.3f %s' % (v, unit)) if key not in ('f0', 'epsr') else ('%g %s' % (v, unit)),
                fontsize=10, ha='right', transform=ax.transAxes, color='r' if changed else 'k', fontweight='bold' if changed else 'normal')
    ax.set_title('settings (red: changed)')


def draw_s11(ax, result, ref=None):
    if ref is not None:
        ax.plot(ref['f'] / 1e9, to_db(ref['s11']), color='0.7', label='design')
    ax.plot(result['f'] / 1e9, to_db(result['s11']), 'b', label='this model')
    ax.axvline(result['settings']['f0'] / 1e9, color='k', ls=':', lw=1)
    ax.set_ylim(-30, 0); ax.grid(); ax.legend(fontsize=8)
    ax.set_xlabel('Frequency [GHz]'); ax.set_ylabel('$S_{11}$ [dB]'); ax.set_title('reflection coefficient')


def polar_axes(ax, rmin, rmax, rticks, rlabel_position):
    """a polar axis as in Fig. 18 of the guide: 0 deg at the top, positive angles to the right, angles -180 ... 180;
    the radial axis in dB from rmin (centre) to rmax"""
    ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
    ax.set_rlim(rmin, rmax); ax.set_rticks(rticks); ax.set_rlabel_position(rlabel_position)
    ax.set_thetagrids(np.arange(0, 360, 30), labels=[str(a) if a <= 180 else str(a - 360) for a in np.arange(0, 360, 30)])


def draw_polar(ax, result, col, title, ref=None, rmin=-30, rmax=10):
    th = np.deg2rad(result['theta'])
    if ref is not None:
        ax.plot(th, np.maximum(ref['pattern'][:, col], rmin), color='0.7', label='design')
    ax.plot(th, np.maximum(result['pattern'][:, col], rmin), 'b', label='this model')
    polar_axes(ax, rmin, rmax, np.arange(rmin, rmax + 1, 10), 150)
    ax.set_title(title + ' [dBi]', fontsize=10)


def show_result(result, ref=None):
    """the settings and the results of one model side by side; ref (the design) is drawn in gray"""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    draw_top_view(fig.add_subplot(gs[0, 0]), result['settings'])
    draw_settings(fig.add_subplot(gs[0, 1]), result['settings'], None if ref is None else ref['settings'])
    ax = fig.add_subplot(gs[0, 2]); ax.axis('off')
    sm = summary(result)
    rows = [('resonant frequency', '%.1f GHz' % sm['f_res']), ('minimum $S_{11}$', '%.1f dB' % sm['s11_min']),
            ('maximum directivity', '%.1f dBi' % sm['Dmax']), ('3 dB beamwidth, E-plane', '%.0f deg' % sm['bw_E']),
            ('3 dB beamwidth, H-plane', '%.0f deg' % sm['bw_H']), ('front-to-back ratio', '%.1f dB' % sm['front_to_back'])]
    for k, (a, b) in enumerate(rows):
        ax.text(0.0, 0.92 - 0.14 * k, a, fontsize=10, transform=ax.transAxes)
        ax.text(1.0, 0.92 - 0.14 * k, b, fontsize=10, ha='right', transform=ax.transAxes)
    ax.set_title('results')
    draw_s11(fig.add_subplot(gs[1, 0]), result, ref)
    draw_polar(fig.add_subplot(gs[1, 1], projection='polar'), result, 0, 'E-plane (xz)', ref)
    draw_polar(fig.add_subplot(gs[1, 2], projection='polar'), result, 1, 'H-plane (yz)', ref)
    fig.tight_layout()
    return fig


def file_name(name):
    """file name of the result of the preset `name`"""
    import re
    return re.sub('[^0-9A-Za-z]+', '_', name).strip('_') + '.npz'


def compare(results, names, title=''):
    """S11 and the E-plane and H-plane patterns of several models in one figure"""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 4.6))
    ax0 = fig.add_subplot(1, 3, 1)
    axE = fig.add_subplot(1, 3, 2, projection='polar'); axH = fig.add_subplot(1, 3, 3, projection='polar')
    for k, n in enumerate(names):
        r = results[n]; c = 'k' if k == 0 else 'C%d' % (k - 1)
        ax0.plot(r['f'] / 1e9, to_db(r['s11']), color=c, label=n)
        th = np.deg2rad(r['theta'])
        axE.plot(th, np.maximum(r['pattern'][:, 0], -30), color=c); axH.plot(th, np.maximum(r['pattern'][:, 1], -30), color=c)
    ax0.set_ylim(-30, 0); ax0.grid(); ax0.legend(fontsize=8)
    ax0.set_xlabel('Frequency [GHz]'); ax0.set_ylabel('$S_{11}$ [dB]'); ax0.set_title('reflection coefficient')
    for ax, t in ((axE, 'E-plane (xz)'), (axH, 'H-plane (yz)')):
        ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
        ax.set_rlim(-30, 10); ax.set_rticks([-30, -20, -10, 0, 10]); ax.set_rlabel_position(150)
        ax.set_thetagrids(np.arange(0, 360, 30), labels=[str(a) if a <= 180 else str(a - 360) for a in np.arange(0, 360, 30)])
        ax.set_title(t + ' [dBi]', fontsize=10)
    fig.suptitle(title); fig.tight_layout()
    return fig
