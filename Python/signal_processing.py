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

from typing import Callable

import numpy as np

from radar_simulation import ChirpConfiguration
from radar_simulation import MIMOConfiguration
from sensor import ULA

class RangeDopplerFFT:
    def __init__(self):
        self.use_2D_FFT = False

    @staticmethod
    def get_fft_size(sz:int) -> int:
        ret = (2 ** int(np.ceil(np.log(sz) / np.log(2))))
        return ret

    @staticmethod
    def get_fft_sizes(szl:list[int]) -> tuple[int]:
        ret = list()
        for sz in szl:
            ret.append(RangeDopplerFFT.get_fft_size(sz))
        return tuple(ret)

    @staticmethod
    def get_range_bins(cc:ChirpConfiguration) -> np.array:
        sz = RangeDopplerFFT.get_fft_size(cc.num_adc)
        if cc.num_adc == sz:
            ret = np.arange(0, cc.range_max, cc.range_resolution)
        else:
            ret = np.arange(0, cc.range_max, cc.range_max / sz)
        return ret

    @staticmethod
    def get_velocity_bins(cc:ChirpConfiguration) -> np.array:
        return np.linspace(-cc.doppler_max, cc.doppler_max, RangeDopplerFFT.get_fft_size(cc.num_chirps))

    def get_2d_fft(self, data_cube:np.array) -> np.array:
        szl = RangeDopplerFFT.get_fft_sizes(data_cube.shape[-2:])
        if self.use_2D_FFT:
            dfft = np.fft.fftshift(np.fft.fft2(data_cube, s=szl[::-1], axes=(-1, -2)), axes=-2)
        else:
            # use this like process for windowing
            rfft = np.fft.fft(data_cube, n=szl[-1], axis=-1)
            dfft = np.fft.fft(rfft, n=szl[-2], axis=-2)
            dfft = np.fft.fftshift(dfft, axes=-2)

        return dfft

    def get_1d_fft(self, data_cube:np.array) -> np.array:
        sz = RangeDopplerFFT.get_fft_size(data_cube.shape[-1])
        rfft = np.fft.fft(data_cube, n=sz, axis=-1)

        return rfft

class BeamFormingDOA:
    def __init__(self, mc:MIMOConfiguration):
        self.is_supported_array = False
        self.array_dimension = 0
        self.angle_fft_size = 256

        va_arr = np.array(mc.va)
        has_axis = np.any(va_arr[0], axis=0)
        if (has_axis[0]) and (not has_axis[1]) and (not has_axis[2]):
            print('the array seems ULA')
            self.is_supported_array = True
            self.array_dimension = 1
        elif (has_axis[0]) and (not has_axis[1]) and (has_axis[2]):
            print('the array seems xz-2D array')
            self.is_supported_array = False
            self.array_dimension = 2
        else:
            print('the array is not supported')
            self.is_supported_array = False
            self.array_dimension = 0

    def get_angle_bins(self, d: float) -> np.array:
        if not self.is_supported_array:
            return None
        if self.array_dimension == 1:
            ret = ULA.get_angle_bins(d, self.angle_fft_size)
        elif self.array_dimension == 2:
            ret = None # not implemented yet
        else:
            ret = None
        return ret

    def get_angle_fft(self, X: np.array) -> np.array:
        if not self.is_supported_array:
            return None
        if self.array_dimension == 1:
            ret = np.fft.fftshift(np.fft.fft(X, n=self.angle_fft_size, axis=-1), axes=-1)
        elif self.array_dimension == 2:
            ret = np.fft.fftshift(np.fft.fft2(X, n=(self.angle_fft_size, self.angle_fft_size), axis=(-1, -2)), axes=(-1, -2))
        else:
            ret = None
        return ret

class MUSIC:
    def __init__(self, rank_estimator: Callable):
        self.rank_estimator = rank_estimator

    def get_music_spetrum(self, Rxx: np.array, Atheta: np.array) -> np.array:
        ret = np.empty(Atheta.shape[1], dtype=Rxx.dtype)
        w, V = np.linalg.eig(Rxx)
        w = np.real(w)
        rank = self.rank_estimator(w)
        En = V[:,rank:]

        for s in range(Atheta.shape[1]):
            EnVs = np.dot(En.conj().T, Atheta[:,s])
            ret[s] = np.inner(Atheta[:,s].conj(), Atheta[:,s]) / np.inner(EnVs.conj(), EnVs)

        return ret