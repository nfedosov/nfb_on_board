# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 18:32:24 2022

@author: Fedosov
"""

from __future__ import annotations

from cmath import exp
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from mne.io.brainvision.brainvision import read_raw_brainvision



"""Complex numbers manipulation utilities"""

import random



def complex_randn() -> complex:
    """Generate random complex number with Re and Im sampled from N(0, 1)"""
    return random.gauss(0, 1) + 1j * random.gauss(0, 1)


def complex2vec(z: complex):
    """Convert complex number to 2d vector"""
    return np.array([[z.real], [z.imag]])


def vec2complex(v) -> complex:
    """Convert 2d vector to a complex number"""
    return v[0, 0] + 1j * v[1, 0]


def complex2mat(z: complex):
    """Convert complex number to 2x2 antisymmetrical matrix"""
    return np.array([[z.real, -z.imag], [z.imag, z.real]])


def mat2complex(M) -> complex:
    """Convert complex number to 2x2 antisymmetrical matrix"""
    return M[0, 0] + 1j * M[1, 0]






from typing import Any

import numpy as np
import numpy.typing as npt







class SignalGenerator(Protocol):
    def step(self) -> complex:
        """Generate single noise sample"""
        ...


@dataclass
class MatsudaParams:
    A: float
    freq: float
    sr: float

    def __post_init__(self):
        self.Phi = self.A * exp(2 * np.pi * self.freq / self.sr * 1j)


@dataclass
class SingleRhythmModel:
    mp: MatsudaParams
    sigma: float
    x: complex = 0

    def step(self) -> complex:
        """Update model state and generate measurement"""
        self.x = self.mp.Phi * self.x + complex_randn() * self.sigma
        return self.x


def gen_ar_noise_coefficients(alpha: float, order: int):
    """
    Parameters
    ----------
    order : int
        Order of the AR model
    alpha : float in the [-2, 2] range
        Alpha as in '1/f^alpha' PSD profile

    References
    ----------
    .. [1] Kasdin, N.J. “Discrete Simulation of Colored Noise and Stochastic
    Processes and 1/f/Sup /Spl Alpha// Power Law Noise Generation.” Proceedings
    of the IEEE 83, no. 5 (May 1995): 802–27. https://doi.org/10.1109/5.381848.

    """
    a: list[float] = [1]
    for k in range(1, order + 1):
        a.append((k - 1 - alpha / 2) * a[-1] / k)  # AR coefficients as in [1]
    return -np.array(a[1:])


class ArNoise:
    """
    Generate 1/f^alpha noise with truncated autoregressive process, as described in [1]

    Parameters
    ----------
    x0 : np.ndarray of shape(order,)
        Initial conditions vector for the AR model
    order : int
        Order of the AR model
    alpha : float in range [-2, 2]
        Alpha as in '1/f^alpha'
    s : float, >= 0
        White noise standard deviation (see [1])

    References
    ----------
    .. [1] Kasdin, N.J. “Discrete Simulation of Colored Noise and Stochastic
    Processes and 1/f/Sup /Spl Alpha// Power Law Noise Generation.” Proceedings
    of the IEEE 83, no. 5 (May 1995): 802–27. https://doi.org/10.1109/5.381848.

    """

    def __init__(self, x0: np.ndarray, order: int = 1, alpha: float = 1, s: float = 1):
        assert (len(x0) == order), f"x0 length must match AR order; got {len(x0)=}, {order=}"
        self.a = gen_ar_noise_coefficients(alpha, order)
        self.x = x0
        self.s = s

    def step(self) -> float:
        """Make one step of the AR process"""
        y_next = self.a @ self.x + np.random.randn() * self.s
        self.x = np.concatenate([[y_next], self.x[:-1]])  # type: ignore
        return float(y_next)


class RealNoise:
    def __init__(self, single_channel_eeg, s: float):
        self.single_channel_eeg = single_channel_eeg
        self.ind = 0
        self.s = s

    def step(self) -> float:
        n_samp = len(self.single_channel_eeg)
        if self.ind >= len(self.single_channel_eeg):
            raise IndexError(f"Index {self.ind} is out of bounds for data of length {n_samp}")
        self.ind += 1
        return self.single_channel_eeg[self.ind] * self.s


def prepare_real_noise(
    raw_path: str, s: float = 1, minsamp: int = 0, maxsamp: int | None = None
) -> tuple[RealNoise, float]:
    raw = read_raw_brainvision(raw_path, preload=True, verbose="ERROR")
    raw.pick_channels(["FC2"])
    raw.crop(tmax=244)
    raw.filter(l_freq=0.1, h_freq=None, verbose="ERROR")

    data = np.squeeze(raw.get_data())
    data /= data.std()
    data -= data.mean()
    crop = slice(minsamp, maxsamp)
    return RealNoise(data[crop], s), raw.info["sfreq"]


def collect(signal_generator: SignalGenerator, n_samp: int):
    return np.array([signal_generator.step() for _ in range(n_samp)])




import matplotlib.pyplot as plt
from scipy.signal import hilbert, welch


def plot_generated_signal(noise, meas, sr, alpha, legend, tmin=0, tmax=2):
    freqs, psd_noise = welch(noise, fs=sr, nperseg=1024)
    freqs, psd_signal = welch(meas, fs=sr, nperseg=1024)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
    freq_lim = 1000
    one_over_f = np.array([1 / f**alpha for f in freqs[1:freq_lim]])

    # bring 1/f line closer to data
    one_over_f *= psd_noise[min(len(psd_noise), freq_lim) - 1] / one_over_f[-1]

    ax1.loglog(freqs[1:freq_lim], psd_signal[1:freq_lim])
    ax1.loglog(freqs[1:freq_lim], one_over_f)
    ax1.loglog(freqs[1:freq_lim], psd_noise[1:freq_lim], alpha=0.5)

    ax1.legend(legend)
    ax1.set_xlabel("Frequencies, Hz")
    ax1.grid()
    t = np.linspace(tmin, tmax, (tmax - tmin) * sr, endpoint=False)
    ax2.plot(t, meas[int(tmin * sr) : int(tmax * sr)])
    ax2.grid()
    ax2.set_xlabel("Time, sec")

    return fig, ax1, ax2


def plot_timeseries(ax, times, timeseries):
    for ts in timeseries:
       ax.plot(times, ts[0],**ts[1])
    ax.legend()
    ax.grid()



def plot_kalman_vs_cfir(
    meas, gt_states, kf_states, cfir_states, plv_win_kf, plv_win_cfir, n_samp, sr, delay
):
    times = np.arange(n_samp) / sr
    meas = meas[2 * n_samp : 3 * n_samp]
    gt_states = gt_states[2 * n_samp : 3 * n_samp]
    kf_states = np.roll(kf_states, shift=-delay)[2 * n_samp : 3 * n_samp]
    cfir_states = np.roll(cfir_states, shift=-delay)[2 * n_samp : 3 * n_samp]

    plv_win_kf = plv_win_kf[2 * n_samp : 3 * n_samp]
    plv_win_cfir = plv_win_cfir[2 * n_samp : 3 * n_samp]

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(9, 10))
    plot_timeseries(
        ax1,
        times,
        [
            [np.real(kf_states),dict(alpha=0.9, label="kalman state (Re)")],
            [np.real(cfir_states), dict(alpha=0.9, label="cfir state (Re)")],
            [meas, dict(alpha=0.3, linewidth=1, label="measurements")],
            [np.real(gt_states),dict(alpha=0.3, linewidth=4, label="ground truth state (Re)")],
        ],
    )

    plot_timeseries(
        ax2,
        times,
        [
            [np.abs(plv_win_kf), dict(linewidth=2, label="plv(gt, kf)")],
            [np.abs(plv_win_cfir),dict(linewidth=2, label="plv(gt, cfir)")],
        ],
    )

    plot_timeseries(
        ax3,
        times,
        [
            [np.abs(kf_states),dict(alpha=0.9, label="kalman envelope")],
            [np.abs(cfir_states),dict(alpha=0.7, label="cfir envelope")],
            [np.abs(gt_states),dict(alpha=0.7, label="gt envelope")],
            [np.abs(hilbert(meas)),dict(alpha=0.3, label="meas envelope")],  # pyright: ignore
        ]
    )
    plt.xlabel("Time, sec")
    return f, ax1, ax2, ax3


def plot_crosscorrelations(t_ms, corrs_cfir, corrs_kf):
    fig = plt.figure(figsize=(9, 5))
    # fig = plt.figure()
    ax = plt.subplot()
    ind_cfir = np.argmax(corrs_cfir)
    ind_kf = np.argmax(corrs_kf)

    C1 = "#d1a683"
    C2 = "#005960"
    ax.plot(t_ms, corrs_cfir, color=C1, label="CFIR")
    ax.axvline(t_ms[ind_cfir], color=C1)
    ax.axhline(corrs_cfir[ind_cfir], color=C1)
    ax.plot(t_ms, corrs_kf, C2, label="Kalman")
    ax.axvline(t_ms[ind_kf], color=C2)
    ax.axhline(corrs_kf[ind_kf], color=C2)
    ax.set_xlabel("delay, ms", fontsize=14)
    ax.set_ylabel("correlation", fontsize=14)
    ax.set_ylim([0, 1])
    ax.legend(fontsize=14)
    ax.grid()

    ax.annotate(f"{t_ms[ind_kf]} ms", (t_ms[ind_kf] + 1, 0.02), color=C2, fontsize=16)
    ax.annotate(f"{t_ms[ind_cfir]} ms", (t_ms[ind_cfir] + 1, 0.02), color=C1, fontsize=16)
    ax.annotate(f"{corrs_kf[ind_kf]:.2f}", (-100, corrs_kf[ind_kf] + 0.01), color=C2, fontsize=16)
    ax.annotate(
        f"{corrs_cfir[ind_cfir]:.2f}",
        (-100, corrs_cfir[ind_cfir] + 0.01),
        color=C1,
        fontsize=16,
    )

    # plt.subplots_adjust(wspace=0.5, hspace=0.5)
    return fig, ax




"""

References
----------
.. [1] Matsuda, Takeru, and Fumiyasu Komaki. “Time Series Decomposition into
Oscillation Components and Phase Estimation.” Neural Computation 29, no. 2
(February 2017): 332–67. https://doi.org/10.1162/NECO_a_00916.

.. [2] Chang, G. "On kalman filter for linear system with colored measurement
noise". J Geod 88, 1163–1170, 2014 https://doi.org/10.1007/s00190-014-0751-7

.. [3] Wang, Kedong, Yong Li, and Chris Rizos. “Practical Approaches to Kalman
Filtering with Time-Correlated Measurement Errors.” IEEE Transactions on
Aerospace and Electronic Systems 48, no. 2 (2012): 1669–81.
https://doi.org/10.1109/TAES.2012.6178086.

.. [4] Kasdin, N.J. “Discrete Simulation of Colored Noise and Stochastic
Processes and 1/f/Sup /Spl Alpha// Power Law Noise Generation.” Proceedings of
the IEEE 83, no. 5 (May 1995): 802–27. https://doi.org/10.1109/5.381848.

"""
from abc import ABC
from cmath import exp
from typing import Any, NamedTuple

import numpy as np



class DifferenceColoredKF:
    """
    'Alternative approach' implementation for KF with colored noise from [2]

    Parameters
    ----------
    Phi : np.ndarray of shape(n_states, n_states)
        State transfer matrix
    Q : np.ndarray of shape(n_states, n_states)
        Process noise covariance matrix (see eq.(1) in [2])
    H : np.ndarray of shape(n_meas, n_states)
        Matrix of the measurements model (see eq.(2) in [2]); maps state to
        measurements
    Psi : np.ndarray of shape(n_meas, n_meas)
        Measurement noise transfer matrix (see eq. (3) in [2])
    R : np.ndarray of shape(n_meas, n_meas)
        Driving noise covariance matrix for the noise AR model (cov for e_{k-1}
        in eq. (3) in [2])

    """

    def __init__(self, Phi, Q, H, Psi, R):
        n_states = Phi.shape[0]
        n_meas = H.shape[0]

        self.Phi = Phi
        self.Q = Q
        self.H = H
        self.Psi = Psi
        self.R = R

        self.x = np.zeros((n_states, 1))  # posterior state (after update)
        self.P = np.zeros((n_states, n_states))  # posterior state covariance (after update)

        self.y_prev = np.zeros((n_meas, 1))

    def predict(self, x, P):
        x_ = self.Phi @ x  # eq. (26) from [1]
        P_ = self.Phi @ P @ self.Phi.T + self.Q  # eq. (27) from [1]
        return x_, P_

    def update(self, y: Vec, x_: Vec, P_: Cov) -> tuple[Vec, Cov]:
        A = self.Psi @ self.H
        B = self.H @ self.Phi
        P, H, R = self.P, self.H, self.R

        z = y - self.Psi @ self.y_prev  # eq. (35) from [1]
        n = z - self.H @ x_ + A @ self.x  # eq. (37) from [1]
        Sigma = H @ P_ @ H.T + A @ P @ A.T + R - B @ P @ A.T - A @ P @ B.T  # eq. (38) from [1]
        Pxn = P_ @ self.H.T - self.Phi @ P @ A.T  # eq. (39) from [1]

        K = Pxn / Sigma  # eq. (40) from [1]
        self.x = x_ + K * n  # eq. (41) from [1]
        self.P = P_ - K * Sigma @ K.T  # eq. (42) from [1]
        self.y_prev = y
        return self.x, self.P

    def update_no_meas(self, x_: Vec, P_: Cov):
        """Update step when the measurement is missing"""
        self.x = x_
        self.P = P_
        self.y_prev = self.H @ x_
        return x_, P_

    def step(self, y: Vec | None) -> tuple[Vec, Cov]:
        x_, P_ = self.predict(self.x, self.P)
        return self.update(y, x_, P_) if y is not None else self.update_no_meas(x_, P_)


class SimpleKF:
    """
    Standard Kalman filter implementation

    Implementation follows eq. (2, 3) from [3]

    Parameters
    ----------
    Phi : np.ndarray of shape(n_states, n_states)
        State transfer matrix
    Q : np.ndarray of shape(n_states, n_states)
        Process noise covariance matrix
    H : np.ndarray of shape(n_meas, n_states)
        Matrix of the measurements model; maps state to measurements
    R : np.ndarray of shape(n_meas, n_meas)
        Driving noise covariance matrix for the noise AR model

    """

    def __init__(self, Phi: Mat, Q: Cov, H: Mat, R: Cov):
        self.Phi = Phi
        self.Q = Q
        self.H = H
        self.R = R

        n_states = Phi.shape[0]
        self.x = np.zeros((n_states, 1), dtype = 'int32')  # posterior state (after update)
        self.P = np.zeros((n_states, n_states), dtype = 'int32')  # posterior state covariance (after update)

    def predict(self, x: Vec, P: Cov) -> tuple[Vec, Cov]:
        x_ = ((self.Phi @ x).astype('int32') / (0x07FF)).astype('int32') # 
        #print(P[:10,:10])
        P_ = ((((self.Phi @ P).astype('int32') / (0x07FF)).astype('int32') @ self.Phi.T).astype('int32')/(0x0004)).astype('int32') + (self.Q/0x0004).astype('int32')
        
        #print(x_[:8])
        #print((((self.Phi @ P).astype('int32') / (0x07FF))).astype('int32')[:10,:10])
        #print(P_[:3,:3])
        
        return x_, P_

    def update(self, y: Vec, x_: Vec, P_: Cov) -> tuple[Vec, Cov]:
        coef = int((0x07FF)/0x0004)
        Sigma = ((self.H @ P_ @ self.H.T + self.R)/coef).astype('int32')
        if Sigma ==0:
            Sigma = 1
            
        #print(Sigma)
        Pxn = (P_ @ self.H.T).astype('int32')
        
        #print(Pxn[:10])

        K = (Pxn / Sigma ).astype('int32')
        #print(K[:8])
        n = (y - self.H @ x_).astype('int32')
        #print(n[:8])
        self.x = x_ + ((K @ n)/coef).astype('int16')
        #print(self.x[:8])
        
        # P is very low resolutional (improve it)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #print(P_[:4,:4])
        self.P = ((((np.eye(len(self.P))*coef-K@self.H)@P_)/coef/coef).astype('int32'))
        #print(self.P[:4,:4])
        #self.P = P_ - ((((K @ Sigma)).astype('int32') @ K.T)/coef/coef).astype('int32')
        return self.x, self.P

    def update_no_meas(self, x_: Vec, P_: Cov):
        """Update step when the measurement is missing"""
        self.x = x_
        self.P = P_
        return x_, P_

    def step(self, y: Vec | None) -> tuple[Vec, Cov]:
        x_, P_ = self.predict(self.x, self.P)
    
        return self.update(y, x_, P_) if y is not None else self.update_no_meas(x_, P_)


class PerturbedPKF(SimpleKF):
    """
    Perturbed P implementation from [3] for KF with augmented state space

    Parameters
    ----------
    Phi : np.ndarray of shape(n_aug_states, n_aug_states)
        Augmented state transfer matrix (see eq. (9) in [3])
    Q : np.ndarray of shape(n_aug_states, n_aug_states)
        Augmented process noise covariance matrix (see eq.(9) in [3])
    H : np.ndarray of shape(n_meas, n_aug_states)
        Augmented matrix of the measurements model (see eq.(9) in [3]); maps
        augmented state to measurements
    R : np.ndarray of shape(n_meas, n_meas)
        Measurements covariance matrix, usually of zeroes, see notes
    lambda_ : float, default=1e-6
        Perturbation factor for P, see eq. (19) in [3].

    Notes
    -----
    R is added for possible regularization and normally must be a zero matrix,
    since the measurement errors are incorporated into the augmented state
    vector

    """

    def __init__(self, Phi: Mat, Q: Cov, H: Mat, R: Cov, lambda_: float = 1e-6):
        super().__init__(Phi, Q, H, R)
        self.lambda_ = lambda_

    def update(self, y: Vec, x_: Vec, P_: Cov) -> tuple[Vec, Cov]:
        super().update(y, x_, P_)
        self.P += 0#np.eye(len(self.P)) * self.lambda_
        return self.x, self.P


class Gaussian(NamedTuple):
    mu: Vec
    Sigma: Cov


class OneDimKF(ABC):
    """Single oscillation - single measurement Kalman filter abstraction"""
    KF: Any

    def predict(self, X: Gaussian) -> Gaussian:
        return Gaussian(*self.KF.predict(X.mu, X.Sigma))

    def update(self, y: float, X_: Gaussian) -> Gaussian:
        y_arr = np.array([[y]])
        return Gaussian(*self.KF.update(y=y_arr, x_=X_.mu, P_=X_.Sigma))

    def update_no_meas(self, X_: Gaussian) -> Gaussian:
        """Update step when the measurement is missing"""
        return Gaussian(*self.KF.update_no_meas(x_=X_.mu, P_=X_.Sigma))

    def step(self, y: float | None) -> Gaussian:
        X_ = self.predict(Gaussian(self.KF.x, self.KF.P))
        return self.update_no_meas(X_) if y is None else self.update(y, X_)


class Difference1DMatsudaKF(OneDimKF):
    """
    Single oscillation - single measurement Kalman filter with AR(1) colored noise

    Using Matsuda's model for oscillation prediction, see [1], and a difference
    scheme to incorporate AR(1) 1/f^a measurement noise, see [2]. Wraps
    DifferenceColoredKF to avoid trouble with properly arranging matrix and
    vector shapes.

    Parameters
    ----------
    A : float
        A in Matsuda's step equation: x_next = A * exp(2 * pi * i * f / sr) * x + n
    f : float
        Oscillation frequency; f in Matsuda's step equation:
        x_next = A * exp(2 * pi * i * f / sr) * x + n
    sr : float
        Sampling rate
    q_s : float
        Standard deviation of model's driving noise (std(n) in the formula above),
        see eq. (1) in [2] and the explanation below
    psi : float
        Coefficient of the AR(1) process modelling 1/f^a colored noise;
        see eq. (3) in [2]; 0.5 corresponds to 1/f noise, 0 -- to white noise,
        1 -- to Brownian motion, see [4]. In between values are also allowed.
    r_s : float
        Driving white noise standard deviation for the noise AR model
        (see cov for e_{k-1} in eq. (3) in [2])

    """

    def __init__(self, A: float, f: float, sr: float, q_s: float, psi: float, r_s: float):
        Phi = complex2mat(A * exp(2 * np.pi * f / sr * 1j))
        Q = np.eye(2) * q_s**2
        H = np.array([[1, 0]])
        Psi = np.array([[psi]])
        R = np.array([[r_s**2]])
        self.KF = DifferenceColoredKF(Phi=Phi, Q=Q, H=H, Psi=Psi, R=R)


class PerturbedP1DMatsudaKF(OneDimKF):
    """
    Single oscillation - single measurement Kalman filter with AR(n_ar) colored noise

    Using Matsuda's model for oscillation prediction, see [1], and AR(n) to
    make account for 1/f^a measurement noise. Previous states for
    AR(n_ar) are included via state-space augmentation with the Perturbed P
    stabilization technique, see [3]. Wraps PerturbedPKF to avoid trouble with
    properly arranging matrix and vector shapes.

    Parameters
    ----------
    A : float
        A in Matsuda's step equation: x_next = A * exp(2 * pi * i * f / sr) * x + n
    f : float
        Oscillation frequency; f in Matsuda's step equation:
        x_next = A * exp(2 * pi * i * f / sr) * x + n
    sr : float
        Sampling rate
    q_s : float
        Standard deviation of model's driving noise (std(n) in the formula above),
        see eq. (1) in [2] and the explanation below
    psi : np.ndarray of shape(n_ar,)
        Coefficients of the AR(n_ar) process modelling 1/f^a colored noise;
        used to set up Psi as in eq. (3) in [2];
        coefficients correspond to $-a_i$ in eq. (115) in [4]
    r_s : float
        Driving white noise standard deviation for the noise AR model
        (see cov for e_{k-1} in eq. (3) in [2])
    lambda_ : float, default=1e-6
        Perturbation factor for P, see eq. (19) in [3]

    """

    def __init__(
        self,
        A: float,
        f: float,
        sr: float,
        q_s: float,
        psi: np.ndarray,
        r_s: float,
        lambda_: float = 1e-6,
    ):
        ns = len(psi)  # number of noise states

        Phi = (np.block(
            [  # pyright: ignore
                [complex2mat(A * exp(2 * np.pi * f / sr * 1j)), np.zeros([2, ns])],
                [np.zeros([1, 2]), psi[np.newaxis, :]],
                [np.zeros([ns - 1, 2]), np.eye(ns - 1), np.zeros([ns - 1, 1])],
            ]
        )*0x07FF).astype('int16')
        print(repr(np.concatenate([Phi[:2,:2].ravel(),Phi[2,2:].ravel()])))
        print(Phi[3,2])
        #### Phi - 16 bit
        
        Q_noise = np.zeros([ns, ns])
        Q_noise[0, 0] = r_s**2
        Q = (np.block(
            [  # pyright: ignore
                [np.eye(2) * q_s**2, np.zeros([2, ns])],
                [np.zeros([ns, 2]), Q_noise],
            ]
        )*0x07FF).astype('int16')
        print(repr(Q.ravel()[Q.ravel()!=0]))
        #print(Q)
        #### Q - 16 bit

        H = np.array([[1, 0, 1] + [0] * (ns - 1)])
        R = np.array([[0]])
        self.KF = PerturbedPKF(Phi=Phi, Q=Q, H=H, R=R, lambda_=lambda_)


def apply_kf(kf: OneDimKF, signal, delay: int) :
    """Convenience function to filter all signal samples at once with KF"""
    if delay > 0:
        raise NotImplementedError("Kalman smoothing is not implemented")
    res = []
    for y in signal:
        state = kf.step(y)
        for _ in range(abs(delay)):
            state = kf.predict(state)
        res.append(vec2complex(state.mu))
    return np.array(res)





from dataclasses import asdict, dataclass

import numpy as np
import scipy.signal as sg



class CFIRBandDetector:
    """
    Complex-valued FIR envelope detector based on analytic signal reconstruction

    Parameters
    ----------
    band : tuple[float, float]
        Frequency range to apply band-pass filtering
    sr : float
        Sampling frequency
    delay : int
        Delay of ideal filter in samples
    n_taps : positive int
        Length of FIR filter
    n_fft : positive int
        Length of frequency grid to estimate ideal freq. response
    weights : array of shape(n_weights,) or None
        Least squares weights. If None match WHilbertFilter

    """

    def __init__(
        self,
        band: tuple[float, float],
        sr: float,
        delay: int,
        n_taps: int = 500,
        n_fft: int = 2000,
        weights: None = None,
    ):
        w = np.arange(n_fft)
        H = 2 * np.exp(-2j * np.pi * w / n_fft * delay)
        H[(w / n_fft * sr < band[0]) | (w / n_fft * sr > band[1])] = 0
        F = np.array(
            [np.exp(-2j * np.pi / n_fft * k * np.arange(n_taps)) for k in np.arange(n_fft)]
        )
        if weights is None:
            self.b = F.T.conj().dot(H) / n_fft
        else:
            W = np.diag(weights)
            self.b = np.linalg.solve(F.T.dot(W.dot(F.conj())), (F.T.conj()).dot(W.dot(H)))
        self.a = np.array([1.0])
        self.zi = np.zeros(len(self.b) - 1)

    def apply(self, signal):
        y, self.zi = sg.lfilter(self.b, self.a, signal, zi=self.zi)
        return y


@dataclass
class CFIRParams:
    band: tuple[float, float]
    sr: float
    n_taps: int = 500
    n_fft: int = 2000
    weights: None = None


def apply_cfir(cfir_params: CFIRParams, signal, delay: int) :
    cfir_params = asdict(cfir_params)
    cfir_params["delay"] = delay
    cfir_params_dict = cfir_params#asdict(cfir_params) | {"delay": delay}
    cfir = CFIRBandDetector(**cfir_params_dict)
    return cfir.apply(signal)



import numpy as np


def plv(x1, x2, ma_window_samp: int):
    x1, x2 = x1.copy(), x2.copy()
    x1 /= np.abs(x1)
    x2 /= np.abs(x2)
    prod = np.conj(x1) * x2

    ma_kernel = np.ones(ma_window_samp) / ma_window_samp
    assert ma_window_samp > 0
    return np.convolve(prod, ma_kernel, mode="same"), prod[ma_window_samp:-ma_window_samp].mean()


def env_cor(x1, x2):
    return np.corrcoef(np.abs(x1), np.abs(x2))[0, 1]


def crosscorr(
    c1: np.ndarray, c2: np.ndarray, sr: float, shift_nsamp: int = 500
) -> tuple[np.ndarray, np.ndarray]:
    c1, c2 = c1.copy(), c2[shift_nsamp:-shift_nsamp].copy()
    c1 -= c1.mean()
    c2 -= c2.mean()
    cc1, cc2 = np.correlate(c1, c1), np.correlate(c2, c2)
    times = np.arange(-shift_nsamp, shift_nsamp + 1) / sr
    return times, np.correlate(c2, c1) / np.sqrt(cc1 * cc2)




















np.random.seed(0)

random.seed(0)


















SRATE = 250
N_SAMP = 100_000




# Setup oscillatioins model and generate oscillatory signal
FREQ_GT = 10
A_GT = 0.99          # as in x_next = A*exp(2*pi*OSCILLATION_FREQ / sr)
SIGNAL_SIGMA_GT = 4     # std of the model-driving white noise in the Matsuda model

MP = MatsudaParams(A = A_GT, freq = FREQ_GT, sr = SRATE)
oscillation_model = SingleRhythmModel(mp = MP,  sigma=SIGNAL_SIGMA_GT)
gt_states = collect(oscillation_model, N_SAMP)


# Setup simulated noise and measurements
NOISE_AR_ORDER = 15#100
ALPHA = 1
NOISE_SIGMA_GT = 1  # std of white noise driving the ar model for the colored noise

noise_model = ArNoise(x0=np.random.rand(NOISE_AR_ORDER), alpha=ALPHA, order=NOISE_AR_ORDER, s=NOISE_SIGMA_GT)
noise = collect(noise_model, N_SAMP)
meas = np.real(gt_states) + noise


# Plot generated signal
legend = ["Generated signal", f"$1/f^{ {ALPHA} }$", f"AR({NOISE_AR_ORDER})" f" for $1/f^{ {ALPHA} }$ noise"]
plot_generated_signal(noise, meas, sr=SRATE, alpha=ALPHA, legend=legend, tmin=0, tmax=2)
plt.show()











# Setup filters

A_KF = A_GT
FREQ_KF = FREQ_GT
SIGNAL_SIGMA_KF = SIGNAL_SIGMA_GT
# PSI = 0
PSI = 0.5
# PSI = -0.5
NOISE_SIGMA_KF = NOISE_SIGMA_GT
print(NOISE_SIGMA_KF)
DELAY = 0


kf = PerturbedP1DMatsudaKF(A=A_KF, f=FREQ_KF, sr=SRATE, q_s=SIGNAL_SIGMA_KF, psi=noise_model.a, r_s=NOISE_SIGMA_KF, lambda_=0)
cfir = CFIRParams([8, 12], SRATE)




# Filter measurements with simulated noise
meas = (meas/np.max(np.abs(meas))*0x7FFF).astype('int16')

cfir_states = apply_cfir(cfir, meas, delay=DELAY)
kf_states = apply_kf(kf, meas, delay=DELAY)



print(repr(meas[:20]))
print(np.real(kf_states[:20]))



# Plot results for simulated noise

plv_win_kf, plv_tot_kf = plv(gt_states, kf_states, int(0.5 * SRATE))
plv_win_cfir, plv_tot_cfir = plv(gt_states, cfir_states, int(0.5 * SRATE))
envcor_kf = env_cor(gt_states.copy(), np.roll(kf_states.copy(), shift=-DELAY))
envcor_cfir = env_cor(gt_states.copy(), np.roll(cfir_states.copy(), shift=-DELAY))
print("KF total PLV = ", round(np.abs(plv_tot_kf), 2), "CFIR total PLV = ", round(np.abs(plv_tot_cfir), 2), end=" ")
print("KF envcor = ", round(envcor_kf, 2), "CFIR envcor = ", round(envcor_cfir, 2))

plot_kalman_vs_cfir(meas, gt_states, kf_states, cfir_states, plv_win_kf, plv_win_cfir, 1000, SRATE, DELAY)
plt.show()

# Plot envelope cross-correlations and delays

t, corrs_cfir = crosscorr(np.abs(gt_states), np.abs(cfir_states), SRATE, 50)
t, corrs_kf = crosscorr(np.abs(gt_states), np.abs(kf_states), SRATE, 50)
t_ms = t * 1000
res = plot_crosscorrelations(t_ms, corrs_cfir, corrs_kf)
plt.show()




















