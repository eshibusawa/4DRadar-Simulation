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

class ChirpConfiguration:
    def __init__(self):
        # USRR settings (inter-chirp duration is not considered)
        # https://www.ti.com/lit/an/swra553a/swra553a.pdf
        self.speed_of_light = 299792458.0 # [m/s]
        self.start_freq = 76.8 # [GHz]
        self.chirp_duration = 50 # [us]
        self.num_adc = 250
        self.num_chirps = 128
        self.chirp_slope = 30 # [MHz/us]
        self.update()

    def update(self):
        self.start_lambda = self.speed_of_light / self.start_freq / 1E9 # [m]
        self.sampling_rate = self.num_adc / self.chirp_duration # [MHz]
        self.sampling_delta = 1 / self.sampling_rate # [us]
        self.bandwidth = self.chirp_duration * self.chirp_slope # [MHz]
        self.chirp_timesteps = np.arange(0, self.num_adc * self.sampling_delta, self.sampling_delta) # [us]

        self.range_max = self.sampling_rate * self.speed_of_light / (self.chirp_slope * 1E6) / 2
        self.range_resolution = self.speed_of_light / (self.bandwidth * 1E6) / 2
        self.doppler_max = self.start_lambda / (self.chirp_duration * 1E-6) / 4

class MIMOConfiguration:
    def __init__(self, txl:np.array, rxl:np.array, lambda_m:float, d:float = 0.5):
        va = list()
        for k in range(txl.shape[0]):
            va.append([])
            for l in range(rxl.shape[0]):
                va[k].append(txl[k] + rxl[l])

        self.txl = txl
        self.rxl = rxl
        self.d = d
        self.lambda_m = lambda_m
        self.d_m = d * lambda_m
        self.va = va

class TargetObject:
    def __init__(self, r: float, rvel: float, angle: float):
        ar = np.deg2rad(angle)
        loc = np.empty((3, 1), dtype=np.float64)
        loc[0] = r * np.sin(ar)
        loc[1] = r * np.cos(ar)
        loc[2] = 0
        vel = np.empty_like(loc)
        vel[0] = rvel * np.sin(ar)
        vel[1] = rvel * np.cos(ar)
        vel[2] = 0

        self.r = r
        self.rvel = rvel
        self.angle = angle
        self.loc = loc
        self.vel = vel

    def get_trajectries(self, ts:np.array) -> np.array:
        ret = self.loc + self.vel * (ts[None, :])
        return ret

class FMCWMIMORadar:
    def __init__(self, cc: ChirpConfiguration, mc: MIMOConfiguration):
        self.cc = cc
        self.mc = mc

    def get_timesteps_(self, num_frames: int) -> np.array:
        num_steps = self.cc.num_adc * self.cc.num_chirps * num_frames
        end_time = self.cc.chirp_duration * self.cc.num_chirps * num_frames * 1E-6 # [us] -> [s]
        ret = np.linspace(0, end_time, num_steps)
        ret = np.reshape(ret, (num_frames, self.cc.num_chirps, self.cc.num_adc))
        return ret

    def get_delay_(self, trajectory: np.array) -> np.array:
        ret = np.empty((self.mc.txl.shape[0], self.mc.rxl.shape[0], trajectory.shape[1]), dtype=trajectory.dtype)
        for k in range(ret.shape[0]):
            tx_k = self.mc.d_m * self.mc.txl[k][:,None]
            for l in range(ret.shape[1]):
                rx_l = self.mc.d_m * self.mc.rxl[l][:,None] + tx_k
                delta1 = np.linalg.norm(trajectory - tx_k, axis=0) # TX -> Target
                delta2 = np.linalg.norm(rx_l - trajectory, axis=0) # RX -> Target
                ret[k, l, :] = (delta1 + delta2) / self.cc.speed_of_light; # TOF
        return ret

    def get_data_cube(self, num_frames: int, targets: list[TargetObject]) -> np.array:
        num_adc = self.cc.num_adc
        num_chirp = self.cc.num_chirps
        num_tx = self.mc.txl.shape[0]
        num_rx = self.mc.rxl.shape[0]

        slope = self.cc.chirp_slope * 1E12 # [MHz/us] -> [Hz/s]
        fs = self.cc.start_freq * 1E9 # [GHz] -> [Hz]
        complexPhase = lambda tv: 2 * np.pi * (fs * tv + slope/2 * tv * tv)

        ct = self.cc.chirp_timesteps / 1E6 # [us] -> [s]
        phase0 = complexPhase(ct)
        phase0 = phase0[None, None, None, None, :]

        ts = self.get_timesteps_(num_frames)
        tsf = ts.flatten()
        t_complex = (np.exp(1j * ct[0])).dtype
        ret = np.zeros((num_tx, num_rx, num_frames, num_chirp, num_adc), dtype=t_complex)
        ct = ct[None, None, None, None, :]
        for tg in targets:
            # trajectory
            traj = tg.get_trajectries(tsf)
            # delay
            d = self.get_delay_(traj)
            d = np.reshape(d, (num_tx, num_rx, num_frames, num_chirp, num_adc))
            # phase
            phase = complexPhase(ct - d)
            signal = np.exp(1j * (phase0 - phase))
            # add to ret
            ret += signal
        ret = np.transpose(ret, [2, 0, 1, 3, 4])
        return ret
