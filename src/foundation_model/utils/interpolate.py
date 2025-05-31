# Copyright 2025 TsumiNa.
# SPDX-License-Identifier: Apache-2.0


from typing import Literal, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


class InterpSmooth:
    """
    A class for 1D linear interpolation of (x, y) → x_new, plus optional smoothing.
    Supported smoothing methods: "savgol", "gaussian", "moving average", or None.
    Default smooth_method is "savgol".

    Parameters
    ----------
    x : np.ndarray
        1D array of original x-values (e.g., energies for DOS). Must be sorted ascending
        and already restricted to the interval of interest.
    y : np.ndarray
        1D array of original y-values corresponding to x. Length must match len(x).
    smooth_method : Literal["savgol", "gaussian", "moving average"] or None, default="savgol"
        - "savgol":         apply a Savitzky–Golay filter after interpolation.
        - "gaussian":       apply a Gaussian‐kernel convolution after interpolation.
        - "moving average": apply a fixed-window moving-average filter after interpolation.
        - None:             do not apply any smoothing (return raw interpolation).
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        smooth_method: Optional[Literal["savgol", "gaussian", "moving average"]] = "savgol",
    ):
        # Convert inputs to numpy arrays
        self.x = np.asarray(x)
        self.y = np.asarray(y)

        # Basic sanity checks
        if self.x.ndim != 1 or self.y.ndim != 1:
            raise ValueError("Input x and y must be 1D arrays.")
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("x and y must have the same length.")

        # Ensure x is strictly increasing; if not, sort
        if not np.all(np.diff(self.x) > 0):
            idx_sort = np.argsort(self.x)
            self.x = self.x[idx_sort]
            self.y = self.y[idx_sort]

        # Validate smoothing method
        valid_methods = {"savgol", "gaussian", "moving average", None}
        if smooth_method not in valid_methods:
            raise ValueError(f"smooth_method must be one of {valid_methods}, got '{smooth_method}'")
        self.smooth_method = smooth_method

    def __call__(self, x_new, s_factor: float = 0.0, normalize: bool = False) -> np.ndarray:
        """
        Interpolate (self.x, self.y) onto x_new via linear interpolation,
        then optionally apply smoothing as specified by smooth_method and s_factor.
        Finally, if normalize=True, scale the output so that its maximum absolute value is 1.

        Parameters
        ----------
        x_new : array_like
            1D array of x-coordinates where we want interpolated (and possibly smoothed) y-values.
            Must be strictly increasing and uniformly spaced.
        s_factor : float, default=0.0
            - If s_factor <= 0.0: no smoothing → return raw interpolated y.
            - If smooth_method == "savgol": s_factor is treated as window length in x-units.
              Internally, window_points = int(s_factor / dx), forced to be an odd integer ≥ 3.
            - If smooth_method == "gaussian": s_factor is interpreted as Gaussian sigma in x-units.
              Internally, sigma_pts = s_factor / dx (points). If sigma_pts ≤ 0, no smoothing.
            - If smooth_method == "moving average": s_factor is treated as window length in x-units.
              Internally, window_points = int(s_factor / dx). If window_points ≤ 1, no smoothing.
        normalize : bool, default=False
            If True, after interpolation (and smoothing), divide the output array by its maximum
            absolute value. If the maximum is zero, returns the array unchanged to avoid division by zero.

        Returns
        -------
        y_out : np.ndarray
            1D array of length len(x_new). Either raw interpolated values
            (if no smoothing), or smoothed values after applying the chosen method.
            If normalize=True, the result is scaled so that max(abs(y_out)) == 1.
        """
        # Convert x_new to numpy and basic checks
        x_new_arr = np.asarray(x_new)
        if x_new_arr.ndim != 1 or x_new_arr.shape[0] < 2:
            raise ValueError("x_new must be a 1D array with at least two points.")
        if not np.all(np.diff(x_new_arr) > 0):
            raise ValueError("x_new must be strictly increasing and uniformly spaced.")

        # STEP 1: Linear interpolation
        y_interp = np.interp(x_new_arr, self.x, self.y)

        # STEP 2: If no smoothing requested, or s_factor <= 0, assign raw interpolation
        if self.smooth_method is None or s_factor <= 0.0:
            y_out = y_interp
        else:
            # Compute dx from x_new (assumes uniform spacing)
            dx = x_new_arr[1] - x_new_arr[0]
            if dx <= 0:
                raise ValueError("x_new must be strictly increasing with positive spacing.")

            # ------------------------------------------------------------------
            #  Savitzky–Golay smoothing
            # ------------------------------------------------------------------
            if self.smooth_method == "savgol":
                # Interpret s_factor as window length in x-units → convert to number of points
                win_pts = int(s_factor / dx)

                # Enforce minimum odd window length (≥ 3)
                if win_pts < 3:
                    # Too small: skip smoothing
                    y_out = y_interp
                else:
                    if win_pts % 2 == 0:
                        win_pts += 1  # make it odd

                    # Choose polynomial order (must be < win_pts). Here polyorder=2 by default.
                    polyorder = 2
                    if win_pts <= polyorder:
                        # Window too small to support polyorder
                        y_out = y_interp
                    else:
                        # Apply Savitzky–Golay filter
                        y_out = savgol_filter(y_interp, window_length=win_pts, polyorder=polyorder, mode="interp")

            # ------------------------------------------------------------------
            #  Gaussian Filter smoothing
            # ------------------------------------------------------------------
            elif self.smooth_method == "gaussian":
                # Interpret s_factor as sigma in x-units
                sigma_pts = s_factor / dx
                if sigma_pts <= 0:
                    # No smoothing
                    y_out = y_interp
                else:
                    # Apply 1D Gaussian filter (mode='nearest' to handle boundaries)
                    y_out = gaussian_filter1d(y_interp, sigma=sigma_pts, mode="nearest")

            # ------------------------------------------------------------------
            #  Moving-average smoothing
            # ------------------------------------------------------------------
            elif self.smooth_method == "moving average":
                # Interpret s_factor as window length in x-units → convert to number of points
                n_conv = int(s_factor / dx)
                if n_conv <= 1:
                    # Too small: skip smoothing
                    y_out = y_interp
                else:
                    # Build uniform boxcar kernel and convolve
                    kernel = np.ones(n_conv, dtype=np.float64) / n_conv
                    y_out = np.convolve(y_interp, kernel, mode="same")

            # ------------------------------------------------------------------
            #  Should not reach here due to validation in __init__
            # ------------------------------------------------------------------
            else:
                raise RuntimeError(f"Unsupported smoothing method '{self.smooth_method}'")

        # STEP 3: Normalize if requested
        if normalize:
            max_val = np.max(np.abs(y_out))
            if max_val > 0:
                y_out = y_out / max_val

        return y_out
