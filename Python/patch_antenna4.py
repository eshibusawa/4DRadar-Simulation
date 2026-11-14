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

"""The 4-patch series-fed element of Section 2 with the steps of patch_antenna_common.PatchAntenna.

The geometry is declared by a free function (series_element) and given to the steps; simulate_element is the model of Section 2.
The numbers without a basis are marked "basis: none".

    geometry = series_element()                      # Table 7
    result = simulate_element(geometry, sim_path)    # the result as a dictionary (save_result, load_result)

series_summary and show_series_result show a result, polar_panels the patterns side by side (pattern_curves: the curves of a result at
some frequencies; notebook 05, Section 2);
load_ti_fig18 and plot_ti_fig18 compare the E-plane with Fig. 18 of the guide, read from the figure (notebook 05, Section 2.4).

usage: python patch_antenna4.py figures      (the figure of README.md: images/05_patch_antenna4_polar.png, the patterns of the stored
                                              result at 76.8 GHz)
"""
import os, sys
import numpy as np

from patch_antenna import MIL, RO3003, to_db, beamwidth, draw_polar, polar_axes
from patch_antenna_common import PatchAntenna

TABLE7 = dict(A1=27.568, A2=11.802, B1=47.24, B2=49.21, B3=45.27, B4=49.21, C1=42.52, C2=40.95,
              E1=39.38, E2=55.12, F1=8.4, F2=10.62, F3=3.94)            # SWRU553A Table 7 [mil]


def series_element(dims=TABLE7, gnd_width=6.0):
    """the geometry of the element [mm] from the dimensions of Table 7 [mil] (SWRU553A Fig. 16, Table 7)
    sections: (name, x0, x1, width) along x from A1 at x = 0, in the order A1, A2, B1, C1, B2, C2, B3, C2, B4, C1;
    A1 and A2 are one section of the width F2, F1 is the width of the feed line in front of it (checked against the Gerber file)"""
    d = {k: v * MIL for k, v in dims.items()}
    seq = [('A1', d['A1'], d['F2']), ('A2', d['A2'], d['F2']), ('B1', d['B1'], d['F3']), ('C1', d['C1'], d['E1']),
           ('B2', d['B2'], d['F3']), ('C2', d['C2'], d['E2']), ('B3', d['B3'], d['F3']), ('C2', d['C2'], d['E2']),
           ('B4', d['B4'], d['F3']), ('C1', d['C1'], d['E1'])]
    x, sections = 0.0, []
    for name, length, width in seq:
        sections.append((name, x, x + length, width)); x += length
    return dict(dims=dict(dims), sections=sections, feed_width=d['F1'], x_end=x,
                ground_x=(-1.3, x + 2.0),     # basis: none (the structure on the feed side; 2 mm of ground behind the last patch)
                ground_width=gnd_width)       # basis: none


def simulate_element(geometry, sim_path, f0=78e9, fc=10e9, frequencies=(76e9, 76.8e9, 77e9, 78e9, 79e9), substrate=RO3003,
                     cells_per_wavelength=20, metal_edge=0.04, substrate_cells=4, z_step=0.05, run=True, write_xml=False):
    """the FDTD simulation of the element (Section 2) and its result as a dictionary
    The substrate and the ground run on the feed side, with the feed line, to the first mesh line, where the microstrip port starts;
    the domain is lambda0/2 of air + 9 cells (tutorial Horn_Antenna); the far field without the face x- that the feed line crosses.
    metal_edge, z_step: basis: none. run=False: post-processing of an earlier run; write_xml: write the model and stop."""
    g = geometry
    gx0, gx1 = g['ground_x']
    gy = g['ground_width'] / 2
    pa = PatchAntenna(f0, fc, substrate, cells_per_wavelength)
    pa.setup_fdtd(nr_ts=150000)                                         # basis: none
    pa.set_domain((gx0, gx1), (-gy, gy), (0, pa.h))
    pa.add_substrate_ground((pa.x_first, gx1), (-gy, gy), substrate_cells)
    pa.add_element(g['sections'], g['feed_width'], metal_edge)
    pa.smooth_mesh(z_step)
    pa.add_msl_port(g['feed_width'], 0.0,
                    feed_shift=lambda pml, length: pml + 0.1,                       # just behind the PML; 0.1: basis: none
                    meas_shift=lambda pml, length: min(pml + 0.5, length - 0.3))    # 0.5, 0.3: basis: none
    pa.add_nf2ff(skip_face=0)
    cells = int(np.prod([len(pa.mesh.GetLines(a)) for a in 'xyz']))
    if write_xml:
        pa.write_xml(sim_path + '.xml')
        return dict(cells=cells)
    wall = pa.run(sim_path) if run else 0.0
    f, s11 = pa.s11(sim_path)
    fs = np.array(frequencies)
    theta, pattern, D = pa.far_field(sim_path, fs, center=[g['x_end'] / 2 * 1e-3, 0, pa.h * 1e-3])
    settings = dict(f0=f0, **substrate, gnd_width=g['ground_width'], frequencies=list(frequencies))
    settings.update({'dim_' + k: v for k, v in g['dims'].items()})
    return dict(settings=settings, f=f, s11=s11, fs=fs, theta=theta, pattern=pattern, Dmax=D, x_end=g['x_end'], cells=cells,
                wall_time=wall)


# ---------------------------------------------------------------- results

def series_summary(result):
    """per frequency of the far field: S11 [dB], direction of the E-plane maximum [deg], 3 dB beamwidths [deg] (E, H)"""
    th, P = result['theta'], result['pattern']
    return [dict(f=fk / 1e9, s11=float(np.interp(fk, result['f'], to_db(result['s11']))),
                 peak=th[np.argmax(P[k, :, 0])], bw_E=beamwidth(th, P[k, :, 0]), bw_H=beamwidth(th, P[k, :, 1]))
            for k, fk in enumerate(result['fs'])]


def draw_series_top_view(ax, result):
    """the element and the ground plane seen from above, to scale"""
    import matplotlib.patches as mp
    s = result['settings']
    g = series_element({k[4:]: v for k, v in s.items() if k.startswith('dim_')}, s['gnd_width'])
    F1 = g['feed_width']
    gy, gx1, x0 = s['gnd_width'] / 2, result['x_end'] + 2.0, -2.4
    ax.add_patch(mp.Rectangle((x0, -gy), gx1 - x0, 2 * gy, fc='#b8d8b0', ec='none'))
    ax.plot([x0, gx1, gx1, x0], [gy, gy, -gy, -gy], color='#2e7d32')
    ax.add_patch(mp.Rectangle((x0, -F1 / 2), -x0, F1, fc='#d4a017', ec='none'))
    for name, a, b, w in g['sections']:
        ax.add_patch(mp.Rectangle((a, -w / 2), b - a, w, fc='#d4a017', ec='none'))
    ax.annotate('feed line (port)', (x0 + 0.2, 0), xytext=(x0 + 0.2, -2.4), fontsize=8, arrowprops=dict(arrowstyle='->'))
    ax.set_xlim(x0 - 0.3, gx1 + 0.5); ax.set_ylim(-gy - 0.5, gy + 0.5); ax.set_aspect('equal')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]'); ax.set_title('model (top view)')


def show_series_result(result, f):
    """the model, S11 and the patterns (relative to the maximum) at the frequency f [Hz] of the far field"""
    import matplotlib.pyplot as plt
    k = int(np.argmin(np.abs(result['fs'] - f)))
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3)
    draw_series_top_view(fig.add_subplot(gs[0, :2]), result)
    ax = fig.add_subplot(gs[0, 2]); ax.axis('off')
    row = series_summary(result)[k]
    rows = [('frequency', '%.1f GHz' % row['f']), ('$S_{11}$', '%.1f dB' % row['s11']),
            ('direction of the maximum (E-plane)', '%.0f deg' % row['peak']), ('3 dB beamwidth, E-plane', '%.0f deg' % row['bw_E']),
            ('3 dB beamwidth, H-plane', '%.0f deg' % row['bw_H'])]
    for i, (a, b) in enumerate(rows):
        ax.text(0.0, 0.92 - 0.14 * i, a, fontsize=10, transform=ax.transAxes)
        ax.text(1.0, 0.92 - 0.14 * i, b, fontsize=10, ha='right', transform=ax.transAxes)
    ax.set_title('results')
    a0 = fig.add_subplot(gs[1, 0])
    a0.plot(result['f'] / 1e9, to_db(result['s11']), 'b')
    a0.axvline(result['fs'][k] / 1e9, color='r', lw=1)
    a0.set_ylim(-30, 0); a0.grid(); a0.set_xlabel('Frequency [GHz]'); a0.set_ylabel('$S_{11}$ [dB]'); a0.set_title('reflection coefficient')
    one = dict(theta=result['theta'], pattern=result['pattern'][k])
    for j, (col, name) in enumerate(((0, 'E-plane (xz)'), (1, 'H-plane (yz)'))):
        a = fig.add_subplot(gs[1, 1 + j], projection='polar')
        draw_polar(a, one, col, name, rmin=-30, rmax=0)
        a.set_title(name + ' [dB, relative to the maximum]', fontsize=10)
    fig.tight_layout()
    return fig


def pattern_curves(result, f_list, styles=('-', '--')):
    """the curves of polar_panels for a result of simulate_element: the E-plane and the H-plane at each frequency of f_list [GHz]"""
    out = []
    for f_show, ls in zip(f_list, styles):
        k = list(result['fs'] / 1e9).index(f_show)
        out += [(result['pattern'][k, :, 0], 'E-plane {:g} GHz'.format(f_show), '#6b2d6b', ls),
                (result['pattern'][k, :, 1], 'H-plane {:g} GHz'.format(f_show), 'red', ls)]
    return out


def polar_panels(theta_deg, panels, rmin=-30, rmax=0):
    """several polar plots side by side in the style of Fig. 18 of the guide (0 deg at the top, positive angles to the right);
    panels: list of (title, curves), curves: list of (pattern [dB] on theta_deg, label, colour, line style);
    the 3 dB beamwidth of each curve is added to its label"""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5.6 * len(panels), 6.4))
    th = np.deg2rad(theta_deg)
    for i, (title, curves) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1, projection='polar')
        for p, label, col, ls in curves:
            ax.plot(th, np.maximum(p, rmin), color=col, ls=ls, lw=1.8, label='%s (3 dB: %.0f deg)' % (label, beamwidth(theta_deg, p)))
        polar_axes(ax, rmin, rmax, np.arange(rmin, rmax + 1, 10), 15)
        ax.grid(color='lightgray', lw=0.7)
        ax.set_title(title, fontsize=11, pad=14)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- Fig. 18 of the guide (SWRU553A)

TI_FIG18_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'openems', 'data', 'ti_fig18_curves.csv')   # read from the figure of the guide


def load_ti_fig18(path=TI_FIG18_CSV):
    """the curves of Fig. 18 read from the figure [dB]: angle; the upper / lower envelope of the E-plane relative to the main lobe
    (high, low; Section 2.4); the E-plane and the H-plane as the mean of their two envelopes, each relative to its maximum (E, H;
    Section 3)"""
    c = np.genfromtxt(path, delimiter=',', names=True)
    th = c['theta_deg']
    def fill(v):
        ok = ~np.isnan(v); return np.interp(th, th[ok], v[ok])
    hi, lo = fill(c['E_high_dB']), fill(c['E_low_dB'])
    E, H = (lo + hi) / 2, (fill(c['H_low_dB']) + fill(c['H_high_dB'])) / 2
    return {'theta': th, 'high': hi - hi.max(), 'low': lo - hi.max(), 'peak_db': hi.max(), 'E': E - E.max(), 'H': H - H.max()}


def plot_ti_fig18(ax, ti, th_sim, sim, label_sim='simulation, 76 GHz (corrected geometry, base mesh)'):
    """the E-plane of Fig. 18 (ti: load_ti_fig18) and of the simulation (sim [dB] on th_sim), relative to the main lobe"""
    ax.plot(ti['theta'], ti['high'], 'k.', ms=2, label='TI Fig. 18, E-plane (read from the figure)')
    ax.plot(ti['theta'], ti['low'], 'k.', ms=2)
    ax.plot(th_sim, sim, 'r-', lw=1.2, label=label_sim)
    ax.set_xlim(-90, 90); ax.set_ylim(-35, 3); ax.grid(); ax.legend(loc='lower center')
    ax.set_xlabel('Theta [deg]'); ax.set_ylabel('Relative to the main lobe [dB]'); ax.set_title('E-plane at 76 GHz: TI Fig. 18 and the simulation')


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'figures':
    import matplotlib; matplotlib.use('Agg')
    from patch_antenna import load_result
    here = os.path.dirname(os.path.abspath(__file__))
    a4 = load_result(os.path.join(here, 'openems', 'data', 'patch_antenna4.npz'))
    out = os.path.join(here, '..', 'images', '05_patch_antenna4_polar.png')
    polar_panels(a4['theta'], [('76.8 GHz', pattern_curves(a4, [76.8]))]).savefig(out, dpi=85, bbox_inches='tight')
    print('saved', os.path.abspath(out))
