# This file is part of 4DRadar-Simulation.
# Copyright (c) 2024, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
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

import numpy as np

class ULA:
    def __init__(self, nr: int, d: float) -> None:
        self.nr = nr
        self.d = d

    def get_steering_vector(self, theta: float = 0) -> np.array:
        ret = np.exp(2j * np.pi * self.d * np.arange(self.nr) * np.sin(theta))
        return ret

    def get_steering_vector_matrix(self, theta: np.array) -> np.array:
        ret = np.exp(2j * np.pi * self.d * np.arange(self.nr)[:,None] * np.sin(theta)[None,:])
        return ret

    def get_beam_pattern(self, theta: float = 0, n_theta: int = 256) -> tuple[np.array, np.array]:
        angle_bins = self.get_angle_bins(n_theta)
        pdss = np.pi * self.d * (np.sin(angle_bins) - np.sin(theta))
        p = np.sin(self.nr * pdss) / np.sin(pdss) / self.nr

        p_dB = 20*np.log10(np.abs(p))
        p_dB -= np.max(p_dB)
        return p_dB, angle_bins

    @staticmethod
    def get_angle_bins(d: float, fft_size: int = 256):
        radrange = np.arange(-np.pi, np.pi, 2 * np.pi / fft_size)
        ret = np.arcsin(radrange / 2 / np.pi / d)
        return ret

    @staticmethod
    def get_beam_pattern_fft(coefficient: np.array = 0, fft_size: int = 256) -> tuple[np.array, np.array]:
        C = np.fft.fftshift(np.fft.fft(coefficient, fft_size))
        C_dB = 20*np.log10(np.abs(C))
        C_dB -= np.max(C_dB)

        angle_bins = ULA.get_angle_bins(fft_size)
        return C_dB, angle_bins
