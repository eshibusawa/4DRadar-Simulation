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
# Create the conda environment 4dradar-openems and build openEMS (the C++ core and the Python bindings) into it.
# openEMS is on neither conda-forge nor PyPI, so it is built from the source. Steps already done are skipped: the script can be run repeatedly.
#
#   tools/setup_openems.sh
#   ENV_NAME=openems-test JOBS=4 tools/setup_openems.sh
#
# Environment variables
#   ENV_NAME  name of the conda environment (default: 4dradar-openems)
#   SRC_DIR   where the source is placed (default: tmp/openEMS-Project under the repository)
#   JOBS      number of parallel build jobs (default: 8)
#
# Prerequisite: conda (or mamba). The compilers, cmake and the libraries come with the environment file.
# The GUI (AppCSXCAD) is not built: the geometry is checked with matplotlib or with a CAD program.
set -euo pipefail

ENV_NAME=${ENV_NAME:-4dradar-openems}
REPO=$(cd "$(dirname "$0")/.." && pwd)
SRC_DIR=${SRC_DIR:-$REPO/tmp/openEMS-Project}
JOBS=${JOBS:-8}
# the commit of openEMS-Project that was tested
REV=34770fd060814b92b49ef1518109cf564f6fd579

command -v conda >/dev/null 2>&1 || { echo "error: conda not found" >&2; exit 1; }

echo "[1/4] conda environment $ENV_NAME"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
	echo "  already there, skipped"
else
	conda env create -q -n "$ENV_NAME" -f "$REPO/Python/environment_openems.yaml"
fi
PREFIX=$(conda run -n "$ENV_NAME" printenv CONDA_PREFIX)
echo "  prefix: $PREFIX"

echo "[2/4] source ($SRC_DIR)"
if [ -d "$SRC_DIR/.git" ]; then
	echo "  already there, skipped"
else
	git clone -q https://github.com/thliebig/openEMS-Project.git "$SRC_DIR"
	git -C "$SRC_DIR" checkout -q "$REV"
	git -C "$SRC_DIR" submodule update --init --depth 1 fparser CSXCAD openEMS
fi

echo "[3/4] C++ core (fparser, CSXCAD, openEMS) into $PREFIX"
if [ -f "$PREFIX/lib/libopenEMS.so" ] && [ -f "$PREFIX/lib/libCSXCAD.so" ]; then
	echo "  already there, skipped"
else
	conda run -n "$ENV_NAME" --no-capture-output cmake -S "$SRC_DIR" -B "$SRC_DIR/build" \
		-DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_PREFIX_PATH="$PREFIX" \
		-DCMAKE_BUILD_TYPE=Release -DBUILD_APPCSXCAD=NO
	conda run -n "$ENV_NAME" --no-capture-output cmake --build "$SRC_DIR/build" -j "$JOBS"
fi

echo "[4/4] Python bindings"
if (cd /tmp && "$PREFIX/bin/python" -c "import CSXCAD, openEMS" 2>/dev/null); then
	echo "  already there, skipped"
else
	for d in CSXCAD openEMS; do
		(
			cd "$SRC_DIR/$d/python"
			CSXCAD_INSTALL_PATH="$PREFIX" OPENEMS_INSTALL_PATH="$PREFIX" \
			CSXCAD_NOSCM=1 OPENEMS_NOSCM=1 \
			CPATH="$PREFIX/include" LIBRARY_PATH="$PREFIX/lib" \
				"$PREFIX/bin/pip" install . --no-build-isolation
		)
	done
fi

echo "check: import CSXCAD, openEMS in $ENV_NAME"
(cd /tmp && "$PREFIX/bin/python" -c "import CSXCAD, openEMS; print('ok')")
