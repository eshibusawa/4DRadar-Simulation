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
from radar_simulation import MIMOConfiguration

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
        angle_bins = self.get_angle_bins_full(n_theta)
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
    def get_angle_bins_full(fft_size: int = 256):
        ret = np.arange(-np.pi/2, np.pi/2, np.pi / fft_size)
        return ret

    def get_beam_pattern_fft(self, coefficient: np.array = 0, fft_size: int = 256) -> tuple[np.array, np.array]:
        C = np.fft.fftshift(np.fft.fft(coefficient, fft_size))
        C_dB = 20*np.log10(np.abs(C))
        C_dB -= np.max(C_dB)

        angle_bins = self.get_angle_bins(self.d, fft_size)
        return C_dB, angle_bins

class MIMOVirtualArray:
    def __init__(self, mc: MIMOConfiguration):
        # (0) setup
        self.use_conjugate = False # apply complex conjugate for obtaining signals

        # (1) check array structure
        va_arr = np.array(mc.va, dtype=np.int32)
        # (1.1) check element coordinate
        has_neg = np.min(va_arr)
        has_z = np.any(va_arr[:,:,2])
        is_supported = (not has_neg) and (not has_z)
        if not is_supported:
            raise ValueError ('only x-linear array or xz-planer array')

        # (1.2) compute mask
        nx_max = np.max(va_arr[:,:,0])
        ny_max = np.max(va_arr[:,:,1])
        self.va_mask = np.zeros((ny_max + 1, nx_max + 1), dtype=np.int8)
        has_y_axis = False
        for k in range(va_arr.shape[0]):
            for l in range(va_arr.shape[1]):
                x = va_arr[k][l][0]
                y = va_arr[k][l][1]
                self.va_mask[y, x] = 1
                if y != 0:
                    has_y_axis = True

        # (1.3) compute mask indices
        self.is_uniform = self.va_mask.size == np.count_nonzero(self.va_mask)
        index_full = self.get_full_index(self.va_mask.shape)
        self.index_vec = index_full[self.va_mask == 1]
        self.index_mat = np.unravel_index(self.index_vec, self.va_mask.shape)

        if not has_y_axis:
            type_str = 'uniform' if self.is_uniform else 'general'
            print('the array seems {} linear array'.format(type_str))
            self.array_dimension = 1
            self.la = ULA(self.va_mask.shape[1], mc.d)
            self.angle_bins_func = lambda angle_fft_size: self.la.get_angle_bins(mc.d, angle_fft_size)
            self.steering_vector_matrix_func = lambda theta: self.la.get_steering_vector_matrix(theta)
        else:
            print('the array seems xz-planer array')
            self.array_dimension = 2
            raise ValueError ('is not implemented yet')

    @staticmethod
    def get_full_index(sz):
        iy = np.arange(sz[0], dtype=np.int32)
        ix = np.arange(sz[1], dtype=np.int32)
        ret = np.zeros(sz, dtype=np.int32)
        ret[:,:] = iy[:,None]
        ret[:,:] = ix[None,:]
        return ret

    def get_array_dimension(self) -> int:
        return self.array_dimension

    def get_fft_angle_bins(self, angle_fft_size) -> np.array:
        return self.angle_bins_func(angle_fft_size)

    def get_steering_vector_matrix(self, theta) -> np.array:
        ret = self.steering_vector_matrix_func(theta)
        if not self.is_uniform:
            ret = ret[self.index_vec, :]
        return ret

    def get_signal(self, X: np.array) -> np.array:
        if len(X.shape) == 2:
            ret = np.reshape(X, -1)
        else:
            ret = np.reshape(X, (X.shape[0] * X.shape[1], *X.shape[2:]))
        ret = np.squeeze(ret)
        if self.use_conjugate:
            ret = np.conj(ret)
        return ret

    def get_signal_padded(self, X: np.array) -> np.array:
        if len(X.shape) == 2:
            ret = np.zeros(self.va_mask.shape, dtype=X.dtype)
            ret[self.index_mat[0], self.index_mat[1]] = np.reshape(X, -1)
        else:
            ret = np.zeros((*self.va_mask.shape, *X.shape[2:]), dtype=X.dtype)
            ret[self.index_mat[0], self.index_mat[1]] = np.reshape(X, (X.shape[0] * X.shape[1], *X.shape[2:]))
        ret = np.squeeze(ret)
        if self.use_conjugate:
            ret = np.conj(ret)
        return ret
