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
# Render the figures (PNG) of images/cad/05_patch_models.scad. Run in the container objective_hypatia (OpenSCAD 2021.01):
#   docker exec objective_hypatia bash /home/ej-sib/Camp/4DRadar-Simulation/images/cad/render_05.sh [parent directory of the output]
# The figures of Section 2 are named 05_<module>_<figure>.png (patch_antenna4.series_element).
cd "$(dirname "$0")"
root=${1:-../..}
main=$root/images; mkdir -p $main
r() { m=$1; out=$2; size=$3; camera=$4; proj=$5
	xvfb-run -a openscad -D "model=\"$m\"" -o $out --imgsize=$size --camera=$camera --projection=$proj --colorscheme=Tomorrow 05_patch_models.scad; }
# Section 1
r single   $main/05_model_single.png  1200,700 0,0,0,50,0,30,15   p
r section  $main/05_model_section.png 1200,700 0,0.5,0.1,72,0,20,6 p
# Section 2: the whole element
r series_element_plain $main/05_patch_antenna4_model_plain.png          1200,700 4,0,0,50,0,30,30 p
