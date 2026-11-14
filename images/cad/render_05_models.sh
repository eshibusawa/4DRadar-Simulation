#!/bin/bash
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
#
# Render the figures (PNG) of the Gerber-based models of Section 3 with images/cad/05_gerber_models.scad. Run in the container objective_hypatia (OpenSCAD 2021.01):
#   docker exec objective_hypatia bash /home/ej-sib/Camp/4DRadar-Simulation/images/cad/render_05_models.sh [parent directory of the output]
# The data (images/cad/data/*.scad) are written by patch_antenna_board.write_scad:
#   python Python/patch_antenna_board.py scad -> data/05_patch_antenna_board_<model>.scad
# The figures are named 05_patch_antenna_board_<model>_model.png.
cd "$(dirname "$0")"
cad=$(pwd)
root=${1:-../..}
main=$root/images; mkdir -p $main
r() { data=$1; out=$2; view=$3; cx=$4; cy=$5; dist=$6
	w=$(mktemp /tmp/w_XXXXXX.scad)
	printf 'include <%s/data/%s.scad>\ninclude <%s/05_gerber_models.scad>\n' "$cad" "$data" "$cad" > $w
	if [ $view = oblique ]; then
		xvfb-run -a openscad -o $out --imgsize=1300,800 --camera=$cx,$cy,0,52,0,28,$dist --projection=p --colorscheme=Tomorrow $w
	else
		xvfb-run -a openscad -o $out --imgsize=900,1300 --camera=$cx,$cy,0,0,0,0,$dist --projection=o --colorscheme=Tomorrow $w
	fi
	rm -f $w; }
# Section 3 of 05 (the domain is lambda0/2 + 9 cells; the substrate and the ports start at the first mesh line)
r 05_patch_antenna_board_isolated   $main/05_patch_antenna_board_isolated_model.png  oblique -1 3.5 31
r 05_patch_antenna_board_d_rx1      $main/05_patch_antenna_board_d_rx1_model.png     oblique -1 3.5 31
r 05_patch_antenna_board_d_rx1_rx2  $main/05_patch_antenna_board_d_rx1_rx2_model.png oblique  0 3.5 35
