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
    def get_angle_bins(d: float, fft_size: int = 256) -> np.array:
        radrange = np.arange(-np.pi, np.pi, 2 * np.pi / fft_size)
        ret = np.arcsin(radrange / 2 / np.pi / d)
        return ret

    @staticmethod
    def get_angle_bins_full(fft_size: int = 256) -> np.array:
        ret = np.arange(-np.pi/2, np.pi/2, np.pi / fft_size)
        return ret

    def get_beam_pattern_fft(self, coefficient: np.array = 0, fft_size: int = 256) -> tuple[np.array, np.array]:
        C = np.fft.fftshift(np.fft.fft(coefficient, fft_size))
        C_dB = 20*np.log10(np.abs(C))
        C_dB -= np.max(C_dB)

        angle_bins = self.get_angle_bins(self.d, fft_size)
        return C_dB, angle_bins

class URA:
    def __init__(self, nr: int, d: float) -> None:
        self.nr = nr
        self.d = d

    def get_steering_vector(self, theta_phi: tuple[float, float] = (0, 0)) -> np.array:
        theta, phi = theta_phi[0], theta_phi[1]
        ret_el = np.exp(2j * np.pi * self.d * np.arange(self.nr[0]) * np.sin(theta))
        ret_az = np.exp(2j * np.pi * self.d * np.arange(self.nr[1]) * np.sin(phi) * np.cos(theta))
        ret = np.kron(ret_el, ret_az) # equal to ret_el[:,None] * ret_az[None, :]
        return ret

    def get_steering_vector_matrix(self, theta_phi: tuple[np.array, np.array]) -> np.array:
        complex_type = np.complex64 if theta_phi[0].dtype == np.float32 else np.complex128
        ret = np.zeros((self.nr[0] * self.nr[1], theta_phi[0].shape[0] * theta_phi[1].shape[0]), dtype=complex_type)
        k = 0
        for theta in theta_phi[0]:
            for phi in theta_phi[1]:
                ret[:,k] = self.get_steering_vector((theta, phi))
                k += 1
        return ret

    def get_beam_pattern_fft(self, coefficient: np.array = 0, fft_size: int = 256) -> tuple[np.array, np.array]:
        C = np.fft.fftshift(np.fft.fft2(coefficient, fft_size, axes=(-1, -2)), axes=(-1, -2))
        C_dB = 20*np.log10(np.abs(C))
        C_dB -= np.max(C_dB)

        angle_bins = self.get_angle_bins(self.d, fft_size)
        return C_dB, angle_bins

    @staticmethod
    def get_angle_bins(d: float, fft_size: tuple[int, int] = (256, 256)) -> np.array:
        radrange_el = np.arange(-np.pi, np.pi, 2 * np.pi / fft_size[0])
        ret_el = np.arcsin(radrange_el / 2 / np.pi / d)
        radrange_az = radrange_el
        if (fft_size[0] != fft_size[1]):
            radrange_az = np.arange(-np.pi, np.pi, 2 * np.pi / fft_size[1])
        val_az = radrange_az[None,:] / 2 / np.pi / d / np.cos(ret_el[:,None])
        mask = np.abs(val_az) <= 1
        ret_az = np.full((fft_size[0], fft_size[1]), np.pi, dtype=radrange_el.dtype) # np.pi is invalid value
        ret_az[mask] = np.arcsin(val_az[mask])
        return ret_el, ret_az

    @staticmethod
    def get_angle_bins_full(fft_size: int = 256) -> np.array:
        return URA.get_angle_bins(d=0.5, fft_size=fft_size)

    @staticmethod
    def get_valid_angle_mask(angle_bins) -> np.array:
        return np.abs(angle_bins) <= (np.pi / 2)

    @staticmethod
    def get_valid_angle_mask_theta_phi(theta_phi: tuple[np.array, np.array]) -> np.array:
        cos_theta = np.cos(theta_phi[0])
        sin_phi = np.sin(theta_phi[1])
        angle_bins = sin_phi[None, :] / cos_theta[:,None]
        return np.abs(angle_bins) <= 1

class MIMOVirtualArray:
    def __init__(self, mc: MIMOConfiguration):
        # (0) setup
        self.valid_angle_mask_func = None

        # (1) check array structure
        va_arr = np.array(mc.va, dtype=np.int32)
        # (1.1) check element coordinate
        has_neg = np.min(va_arr)
        has_y = np.any(va_arr[:,:,1])
        is_supported = (not has_neg) and (not has_y)
        if not is_supported:
            raise ValueError ('only x-linear array or xz-planer array')

        # (1.2) compute mask
        nx_max = np.max(va_arr[:,:,0])
        nz_max = np.max(va_arr[:,:,2])
        self.va_mask = np.zeros((nz_max + 1, nx_max + 1), dtype=np.int8)
        has_z_axis = False
        index_vec = list()
        index_mat = list((list(), list()))
        for k in range(va_arr.shape[0]):
            for l in range(va_arr.shape[1]):
                x = va_arr[k][l][0]
                z = va_arr[k][l][2]
                if self.va_mask[z, x] == 0:
                    self.va_mask[z, x] = 1
                    index_vec.append(z * self.va_mask.shape[1] + x)
                    index_mat[0].append(z)
                    index_mat[1].append(x)
                if z != 0:
                    has_z_axis = True

        # (1.3) compute mask indices
        self.is_uniform = self.va_mask.size == np.count_nonzero(self.va_mask)
        self.index_vec = index_vec
        self.index_mat = index_mat

        if not has_z_axis:
            type_str = 'uniform' if self.is_uniform else 'general'
            print('the array seems {} linear array'.format(type_str))
            self.array_dimension = 1
            self.la = ULA(self.va_mask.shape[1], mc.d)
            self.angle_bins_func = lambda angle_fft_size: self.la.get_angle_bins(mc.d, angle_fft_size)
            self.steering_vector_func = lambda theta: self.la.get_steering_vector(theta)
            self.steering_vector_matrix_func = lambda theta: self.la.get_steering_vector_matrix(theta)
            self.steering_vector_matrix_mask_theta_phi_func = None
        else:
            type_str = 'xz uniform rectangular' if self.is_uniform else 'general xz planaer'
            print('the array seems {} array'.format(type_str))
            self.array_dimension = 2
            self.pa = URA(self.va_mask.shape, mc.d)
            self.angle_bins_func = lambda angle_fft_size: self.pa.get_angle_bins(mc.d, angle_fft_size)
            self.steering_vector_func = lambda theta: self.pa.get_steering_vector(theta)
            self.steering_vector_matrix_func = lambda theta: self.pa.get_steering_vector_matrix(theta)
            self.valid_angle_mask_func = lambda theta: self.pa.get_valid_angle_mask(theta)
            self.steering_vector_matrix_mask_theta_phi_func = lambda theta: self.pa.get_valid_angle_mask_theta_phi(theta)

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

    def get_valid_angle_mask(self, theta) -> np.array:
        return self.valid_angle_mask_func(theta)

    def get_steering_vector(self, theta) -> np.array:
        ret = self.steering_vector_func(theta)
        if not self.is_uniform:
            ret = ret[self.index_vec]
        return ret

    def get_steering_vector_matrix(self, theta) -> np.array:
        ret = self.steering_vector_matrix_func(theta)
        if not self.is_uniform:
            ret = ret[self.index_vec]
        if self.steering_vector_matrix_mask_theta_phi_func is not None:
            mask = self.steering_vector_matrix_mask_theta_phi_func(theta)
            ret = ret[:,np.reshape(mask, -1)], mask
        return ret

    def get_signal(self, X: np.array) -> np.array:
        if len(X.shape) == 2:
            vec_X = np.reshape(X, -1)
        else:
            vec_X = np.reshape(X, (-1, *X.shape[2:]))
        ret = np.squeeze(vec_X)
        return ret

    def get_signal_padded(self, X: np.array) -> np.array:
        if len(X.shape) == 2:
            ret = np.zeros(self.va_mask.shape, dtype=X.dtype)
            vec_X = np.reshape(X, -1)
        else:
            ret = np.zeros((*self.va_mask.shape, *X.shape[2:]), dtype=X.dtype)
            vec_X = np.reshape(X, (-1, *X.shape[2:]))
        ret[self.index_mat[0], self.index_mat[1]] = vec_X
        ret = np.squeeze(ret)
        return ret
