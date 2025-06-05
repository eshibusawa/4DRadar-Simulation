# Theoretical Background of Direction of Arrival (DoA) Methods for ULA

This document provides the fundamental theoretical derivations for common Direction of Arrival (DoA) methods used with Uniform Linear Arrays (ULA).
Understanding these concepts will deepen your comprehension of how radar systems estimate the angles of incoming signals.
We'll cover Delay-and-Sum Beamforming (DSBF), its relationship to FFT Beamforming (FFT-BF), and the high-resolution MUSIC (MUltiple SIgnal Classification) algorithm.

First, we define a right-handed coordinate system as follows: The $x$-axis points to the right, the $y$-axis points upward, and the positive direction of the azimuth angle is counterclockwise around the $z$-axis.
Next, we consider a uniform linear array (ULA) with $N$ elements placed along the x-axis, with a uniform inter-element spacing of $d$.
The element numbers increase in the positive $x$-axis direction. Here, $d$ is given as a fraction of the wavelength; for example, $d = 0.5$.
The coordinate system of this ULA is illustrated in Fig. 1.

<figure style="background-color: white; padding: 15px; display: inline-block;">
  <img src="./images/01_coordinate_system_of_ULA.svg" width="600" style="background-color: white;">
  <figcaption style="text-align: center; margin-top: 10px; font-size: 0.9em; color: #555;">
    Fig. 1: Illustration of the ULA (Uniform Linear Array) Coordinate System
  </figcaption>
</figure>

## Delay-and-Sum Beamforming (DSBF) for ULA

Delay-and-Sum Beamforming (DSBF) is a fundamental technique for spatially filtering signals received by an array antenna.
It works by applying a specific phase shift (delay) to the signal received at each antenna element before summing them.
This coherent summation effectively "steers" a beam in a desired direction, maximizing sensitivity to signals arriving from that angle while suppressing signals from other directions.

Assume a planar wave arrives at this ULA from a direction $\theta_\mathrm{incident}$.
If the incident signal is a complex sinusoid $x$, the received signal $y_k$ at each element $k$ (where $k=0, 1, \dots, N-1$) will also be a complex sinusoid, with a phase difference proportional to its position:

$$y_k = e^{j 2 \pi k d \sin(\theta_\mathrm{incident})}x$$

These received signals form the received signal vector $\boldsymbol{y}$:

$$
\boldsymbol{y} = \begin{bmatrix}
y_0 \\
y_1 \\
\vdots \\
y_{N-1}
\end{bmatrix}
=
\begin{bmatrix}
e^{j 2 \pi \cdot 0 \cdot d \sin(\theta_\mathrm{incident})}x \\
e^{j 2 \pi \cdot 1 \cdot d \sin(\theta_\mathrm{incident})}x \\
\vdots \\
e^{j 2 \pi (N-1) d \sin(\theta_\mathrm{incident})}x
\end{bmatrix}
$$

A steering vector $\boldsymbol{s}(\theta)$ represents the phase progression across the array elements for a hypothetical signal arriving from a direction $\theta$:

$$\boldsymbol{s}(\theta) = \begin{bmatrix}
e^{j 2 \pi \cdot 0 \cdot d \sin(\theta)} \\
e^{j 2 \pi \cdot 1 \cdot d \sin(\theta)} \\
\vdots \\
e^{j 2 \pi (N-1) d \sin(\theta)}
\end{bmatrix}$$

In DSBF, a weight vector $\boldsymbol{w}(\theta_\mathrm{steer})$ is used to steer the beam towards a specific direction $\theta_\mathrm{steer}$.
This weight vector is defined as the conjugate of the steering vector for the desired direction, normalized by the number of elements $N$:

$$\boldsymbol{w}(\theta_\mathrm{steer}) = \frac{1}{N} \boldsymbol{s}^H(\theta_\mathrm{steer}) = \frac{1}{N} \begin{bmatrix}
e^{-j 2 \pi \cdot 0 \cdot d \sin(\theta_\mathrm{steer})} \\
e^{-j 2 \pi \cdot 1 \cdot d \sin(\theta_\mathrm{steer})} \\
\vdots \\
e^{-j 2 \pi (N-1) d \sin(\theta_\mathrm{steer})}
\end{bmatrix}$$

The output of the DSBF, $y_{DSBF}$, is computed as the inner product of the received signal vector $\boldsymbol{y}$ and the weight vector $\boldsymbol{w}(\theta_\mathrm{steer})$:

$$y_{DSBF}(\theta_\mathrm{incident}, \theta_\mathrm{steer}) = \sum_{k=0}^{N-1} y_k \cdot w_k(\theta_\mathrm{steer})$$

Substituting $y_k = e^{j 2 \pi k d \sin(\theta_\mathrm{incident})}x$ and $w_k(\theta_\mathrm{steer}) = \frac{1}{N} e^{-j 2 \pi k d \sin(\theta_\mathrm{steer})}$:

$$y_{DSBF}(\theta_\mathrm{incident}, \theta_\mathrm{steer}) = \sum_{k=0}^{N-1} \left( e^{j 2 \pi k d \sin(\theta_\mathrm{incident})}x \right) \cdot \left( \frac{1}{N} e^{-j 2 \pi k d \sin(\theta_\mathrm{steer})} \right)$$

$$y_{DSBF}(\theta_\mathrm{incident}, \theta_\mathrm{steer}) = \frac{x}{N} \sum_{k=0}^{N-1} e^{j 2 \pi k d (\sin(\theta_\mathrm{incident}) - \sin(\theta_\mathrm{steer}))}$$

Let $C = 2 \pi d (\sin(\theta_\mathrm{incident}) - \sin(\theta_\mathrm{steer}))$. The summation becomes a geometric series:

$$y_{DSBF}(\theta_\mathrm{incident}, \theta_\mathrm{steer}) = \frac{x}{N} \sum_{k=0}^{N-1} (e^{jC})^k = \frac{x}{N} \frac{1 - e^{jCN}}{1 - e^{jC}}$$

When $e^{jC} = 1$ (i.e., $\sin(\theta_\mathrm{incident}) = \sin(\theta_\mathrm{steer})$), the formula simplifies to $y_{DSBF}(\theta_\mathrm{incident}, \theta_\mathrm{steer}) = \frac{x}{N} \cdot N = x$. This means that when the incident signal's direction perfectly aligns with the steered direction, the output (normalized by $N$) is the original signal $x$.

For the evaluation of the array response or beam pattern, we set $x = 1$ without loss of generality.
This allows us to focus solely on the angular characteristics, independent of the incident signal amplitude.

The magnitude of this output, $|y_{DSBF}(\theta_\mathrm{incident}, \theta_\mathrm{steer})|$, represents the array's response (or gain) in the direction $\theta_\mathrm{incident}$ when steered to $\theta_\mathrm{steer}$. It can be shown that the magnitude of this geometric series sum is given by:

$$|y_{DSBF}(\theta_\mathrm{incident}, \theta_\mathrm{steer})| = \left| \frac{\sin\left[N\pi d\left( \sin\theta_\mathrm{incident} - \sin\theta_\mathrm{steer}\right)\right]}{N\sin\left[\pi d\left( \sin\theta_\mathrm{incident} - \sin\theta_\mathrm{steer}\right)\right]} \right|$$

This expression is commonly known as the array factor (or beam pattern) for a uniform linear array.
It quantifies how the array's sensitivity varies with the angle of an incoming signal relative to the steered direction.
The maximum response occurs when $\sin\theta_\mathrm{incident} = \sin\theta_\mathrm{steer}$, corresponding to the main lobe of the beam pattern.

## FFT Beamforming (FFT-BF) for ULA

FFT Beamforming (FFT-BF) is a computationally efficient method to perform beamforming, particularly for equally spaced arrays.
It leverages the Discrete Fourier Transform (DFT) to simultaneously calculate the beam response for a set of predefined, discrete steering angles.
This is often used for fast angle spectrum estimation.

Consider a ULA with $N$ elements and normalized inter-element spacing $d$. The received signal at element $k$ from an incident direction $\theta$ is:

$$y_k = e^{j 2 \pi k d \sin(\theta)}x$$

These received signals form the received signal vector $\boldsymbol{y}$:

$$
\boldsymbol{y} = \begin{bmatrix}
y_0 \\
y_1 \\
\vdots \\
y_{N-1}
\end{bmatrix}
=
\begin{bmatrix}
e^{j 2 \pi \cdot 0 \cdot d \sin(\theta)}x \\
e^{j 2 \pi \cdot 1 \cdot d \sin(\theta)}x \\
\vdots \\
e^{j 2 \pi (N-1) d \sin(\theta)}x
\end{bmatrix}
$$

FFT Beamforming is achieved by applying the Discrete Fourier Transform (DFT) to the received signal vector $\boldsymbol{y}$.
We define the DFT output $Y[m]$ using a normalized angular frequency $\omega_m$.
To align with typical angle ranges (e.g., $[-90^\circ, 90^\circ)$), the DFT index $m$ is typically chosen symmetrically around zero, from $-\lfloor N/2 \rfloor$ to $\lceil N/2 \rceil - 1$.

$$\omega_m = \frac{2\pi m}{N}, \quad m = -\lfloor N/2 \rfloor, \dots, -1, 0, 1, \dots, \lceil N/2 \rceil - 1$$

The DFT formula, when normalized by $N$ to match the DSBF output, is:

$$Y[m] = \frac{1}{N} \sum_{k=0}^{N-1} y_k \cdot e^{-j \omega_m k}$$

By substituting $y_k$ into the DFT equation:

$$
Y[m] = \frac{1}{N} \sum_{k=0}^{N-1} \left( e^{j 2 \pi k d \sin(\theta)}x \right) \cdot e^{-j \omega_m k} \\
Y[m] = \frac{x}{N} \sum_{k=0}^{N-1} e^{j k (2 \pi d \sin(\theta) - \omega_m)}
$$

Each DFT output bin $Y[m]$ can be interpreted as the output of a DSBF steered towards a "virtual steering direction" $\theta_m$.
This correspondence arises by equating the phase compensation term in the DFT ($-\omega_m k$) with that of the DSBF weight vector ($ -2 \pi k d \sin(\theta_m)$):

$$2 \pi k d \sin(\theta_{m}) = \omega_m k$$

This leads to the relationship between the angle $\theta_m$ and the normalized angular frequency $\omega_m$:

$$\sin(\theta_{m}) = \frac{\omega_m}{2 \pi d}$$

Thus, the virtual steering direction $\theta_{m}$ is given by:

$$\theta_{m} = \sin^{-1}\left(\frac{\omega_m}{2 \pi d}\right)$$

This definition ensures that $\theta_m$ typically corresponds to angles within the physical range of $[-90^\circ, 90^\circ)$.

Each DFT output $Y[m]$ is equivalent to the inner product of the received signal $\boldsymbol{y}$ and a corresponding weight vector $\boldsymbol{w}(\theta_{m})$. This weight vector has elements $w_k(\theta_{m}) = \frac{1}{N} e^{-j \omega_m k}$, which formally matches the DSBF weight vector definition.

$$Y[m] = \sum_{k=0}^{N-1} y_k \cdot w_k(\theta_{m})$$

This equation formally demonstrates that FFT-BF efficiently computes DSBF outputs for a discrete set of angles.
Expressing it as a geometric series:

$$Y[m] = \frac{x}{N} \sum_{k=0}^{N-1} (e^{jC'})^k = \frac{x}{N} \frac{1 - e^{jC'N}}{1 - e^{jC'}}$$

where $C' = 2 \pi d \sin(\theta) - \omega_m$. When $e^{jC'} = 1$ (i.e., $2 \pi d \sin(\theta) = \omega_m$), then $Y[m] = \frac{x}{N} \cdot N = x$.
This indicates that each DFT bin enhances the signal component arriving from the angle corresponding to that bin, with a normalized output of $x$. FFT-BF simultaneously forms $N$ beams in different directions through a single DFT operation.

Similar to DSBF, we set $x=1$ when evaluating the array response so that only the angular characteristics are considered; the magnitude of the FFT-BF output for each bin $m$, $|Y[m]|$, then follows the array factor pattern:

$$|Y[m]| = \left| \frac{\sin\left[N\pi d\left( \sin\theta - \sin\theta_m\right)\right]}{N\sin\left[\pi d\left( \sin\theta - \sin\theta_m\right)\right]} \right|$$

Here, $\theta$ is the true angle of arrival, and $\theta_m$ is the virtual steering angle corresponding to the $m$-th DFT bin.
This explicitly shows that FFT-BF produces an array factor, effectively calculating the beam pattern at discrete angular points determined by the DFT bins.
The highest response for a given incident angle $\theta$ occurs at the DFT bin $m$ where $\theta_m$ is closest to $\theta$.

## MUSIC (MUltiple SIgnal Classification) for ULA

The MUSIC (MUltiple SIgnal Classification) algorithm is a high-resolution Direction of Arrival (DoA) estimation technique.
Unlike traditional beamforming methods, which often suffer from wide main lobes and high sidelobes, MUSIC provides sharper peaks at true DoA values by exploiting the subspace properties of the received signal.
It achieves this by separating the received signal space into a signal subspace and a noise subspace.

Consider a ULA with $N$ elements.
Assume there are $M$ incoming signals, each arriving from angles $\theta_1, \theta_2, \dots, \theta_M$.
The inter-element spacing is given as a normalized spacing $d$.

The received signal vector $\boldsymbol{y}(l)$ for $L$ snapshots (time samples) can be expressed as:

$$\boldsymbol{y}(l) = \boldsymbol{S} \boldsymbol{x}(l) + \boldsymbol{n}(l)$$

Where:

* $\boldsymbol{y}(l)$ is the $N \times 1$ received signal vector for snapshot $l$.
* $\boldsymbol{S} = [\boldsymbol{s}(\theta_1), \boldsymbol{s}(\theta_2), \dots, \boldsymbol{s}(\theta_M)]$ is the $N \times M$ steering matrix, where each column is a steering vector for an incoming signal.
* $\boldsymbol{s}(\theta_i) = \begin{bmatrix} e^{j 2 \pi \cdot 0 \cdot d \sin(\theta_i)} \\ e^{j 2 \pi \cdot 1 \cdot d \sin(\theta_i)} \\ \vdots \\ e^{j 2 \pi (N-1) d \sin(\theta_i)} \end{bmatrix}$ is the steering vector for the $i$-th signal.
* $\boldsymbol{x}(l)$ is the $M \times 1$ signal vector, containing the complex amplitudes of each signal source.
* $\boldsymbol{n}(l)$ is the $N \times 1$ noise vector.

The covariance matrix $\boldsymbol{R}_y$ of the received signals is estimated using $L$ snapshots:

$$\boldsymbol{R}_y = E[\boldsymbol{y}(l) \boldsymbol{y}(l)^H] \approx \frac{1}{L} \sum_{l=1}^{L} \boldsymbol{y}(l) \boldsymbol{y}(l)^H$$

Here, $(\cdot)^H$ denotes the Hermitian (conjugate) transpose.

Perform eigenvalue decomposition on the covariance matrix $\boldsymbol{R}_y$:

$$\boldsymbol{R}_y = \boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{U}^H$$

Where:

* $\boldsymbol{U} = [\boldsymbol{u}_1, \boldsymbol{u}_2, \dots, \boldsymbol{u}_N]$ is an $N \times N$ unitary matrix whose columns are the eigenvectors.
* $\boldsymbol{\Sigma} = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_N)$ is an $N \times N$ diagonal matrix containing the corresponding eigenvalues.
* Eigenvalues are assumed to be sorted in descending order: $\lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_N$.

From the eigenvalue decomposition, we separate the signal subspace and the noise subspace.
The dimension of the signal subspace, $M$, is equal to the number of incoming signals and is typically determined by counting the number of significantly large eigenvalues.

* Signal Subspace: This subspace is spanned by the eigenvectors corresponding to the $M$ largest eigenvalues: $\boldsymbol{U}_S = [\boldsymbol{u}_1, \boldsymbol{u}_2, \dots, \boldsymbol{u}_M]$.
* Noise Subspace: This subspace is spanned by the eigenvectors corresponding to the remaining $N-M$ smaller eigenvalues: $\boldsymbol{U}_N = [\boldsymbol{u}_{M+1}, \boldsymbol{u}_{M+2}, \dots, \boldsymbol{u}_N]$.

The key principle of MUSIC is that the steering vectors corresponding to the true directions of arrival are theoretically orthogonal to the noise subspace.
That is:

$$\boldsymbol{s}(\theta_i)^H \boldsymbol{U}_N = \boldsymbol{0}$$

The MUSIC pseudo-spectrum $P_\mathrm{MUSIC}(\theta)$ is a measure of how orthogonal a test steering vector $\boldsymbol{s}(\theta)$ is to the noise subspace $\boldsymbol{U}_N$.
When $\boldsymbol{s}(\theta)$ is orthogonal to the noise subspace, the denominator approaches zero, leading to a peak in the spectrum.

$$P_\mathrm{MUSIC}(\theta) = \frac{1}{\boldsymbol{s}(\theta)^H \boldsymbol{U}_N \boldsymbol{U}_N^H \boldsymbol{s}(\theta)}$$

Here, $\boldsymbol{s}(\theta)$ is a test steering vector for the angle $\theta$ being searched.

The DoAs are estimated by finding the peaks in the MUSIC pseudo-spectrum $P_\mathrm{MUSIC}(\theta)$.
The number and positions of these peaks correspond to the number and angles of the incoming signals.
The estimated angle $\hat{\theta}$ maximizes $P_\mathrm{MUSIC}(\theta)$:

$$\hat{\theta} = \arg \max_{\theta} P_\mathrm{MUSIC}(\theta)$$
