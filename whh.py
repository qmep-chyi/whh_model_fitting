"""Werthamer-Helfand-Hohenberg (WHH) upper-critical-field model.

Implements the dirty-limit WHH equation for the upper critical field
``H_c2(T)`` of a type-II superconductor, including the Maki orbital/paramagnetic
pair-breaking parameter ``alpha`` and spin-orbit scattering ``l_so``
(``lambda_SO``), in the form given by Kim et al. (2025), Eq. (6) [1]_, which
follows Werthamer, Helfand & Hohenberg (1966) [2]_.

The model is an *implicit* relation between the reduced temperature
``t = T / T_c`` and a reduced field ``bar_h``::

    ln(1/t) = sum_{nu=-inf}^{inf} {  1/|2*nu+1|
                                     - [ |2*nu+1| + bar_h/t
                                         + (alpha*bar_h/t)**2
                                           / (|2*nu+1| + (bar_h + l_so)/t) ]**-1 }

The reduced field uses the Werthamer convention (Ref. [2]_)::

    bar_h = (4/pi**2) * H_c2 / |dH_c2/dt|_{t=1}
          = (4/pi**2) * H_c2 / (T_c * |dH_c2/dT|_{T_c})

Because the equation is implicit in ``H_c2``, both *fitting* (recover
``alpha``, ``l_so`` from measured points) and *plotting* (draw ``H_c2(T)`` for
given ``alpha``, ``l_so``) reduce to the same core: form the residual
``ln(1/t) - sum{...}`` and either minimise it over the measured points or
drive it to zero at a single point.

References
----------
.. [1] S. Kim et al., "Spin-orbit coupling induced enhancement of upper
   critical field in superconducting A15 single crystals", J. Alloys Compd.
   1037, 182350 (2025).
.. [2] N. R. Werthamer, E. Helfand & P. C. Hohenberg, "Temperature and Purity
   Dependence of the Superconducting Critical Field, H_c2. III. Electron Spin
   and Spin-Orbit Effects", Phys. Rev. 147, 295 (1966).
"""

from __future__ import annotations
from typing import Literal

import numpy as np
from mpmath import nsum, inf
from scipy.optimize import brentq
from scipy.linalg import sqrtm
from scipy.special import digamma, polygamma
import warnings


class WHHModel:
    """WHH upper-critical-field model evaluated in a fixed unit scale.

    An instance stores the *scaling* constants (``t_c``, ``slope``) and
    default *shape* parameters (``alpha``, ``l_so``).

    ``alpha``, ``l_so`` and ``slope`` can be held fixed (its stored value is
    used) or promoted to a fit variable (see :meth:`make_residual`). ``alpha``
    and ``l_so`` are the usual fit targets;

    Parameters
    ----------
    alpha : float, optional
        Maki parameter, the ratio of orbital to paramagnetic pair breaking,
        Reduces Hc2(T). Default 0.0 (purely orbital limit).
    l_so : float, optional
        Spin-orbit scattering parameter ``lambda_SO``. Put Hc2(T) back to the
        orbital limits, against the pauli effect. Default 0.0.
    slope : float or ndarray, optional
        ``|dH_c2/dT|`` at ``T_c``,  equivalently ``Tc*|dH_c2/dt|`` at ``t=1``
        where t=T/Tc, in field-per-temperature units (e.g. T/K). Default 1.0.
    t_c : float or ndarray, optional
        Critical temperature ``T_c``, in the same temperature unit as the data
        passed to the methods. Default 1.0.

    Notes
    -----
    The infinite series of the right-hand side is derived as
        (1) polygammam functions (Default)
        (2):func:`mpmath.nsum` (``'r+s+e'`` method: Richardson extrapolation +
        Shanks transformation + Euler-Maclaurin). The summand decays as
        ``~1/nu**2``, so plain truncation converges slowly; the extrapolation
        handles the tail to full precision with few evaluations. (Old default)

    """

    #: parameters that may be either fixed or fitted
    PARAM_NAMES = ("alpha", "l_so", "slope", "t_c")

    def __init__(
        self,
        *,
        alpha=0.0,
        l_so=0.0,
        slope=1.0,
        t_c=1.0,
        summation: Literal["mpmath_nsum", "polygamma"] = "polygamma",
    ):
        self.params = dict(
            alpha=alpha, l_so=l_so, slope=np.abs(slope), t_c=t_c, summation=summation
        )

    def __repr__(self):
        inner = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"WHHModel({inner})"

    # ------------------------------------------------------------------ #
    # Core equation, broken up to match the brackets in Kim et al. Eq. 6
    # ------------------------------------------------------------------ #
    @staticmethod
    def reduced_field(field, slope, t_c):
        """Reduced field ``bar_h`` from a (normalized) measured field.

        Computes ``bar_h = (4/pi**2) * H_c2 / (T_c * slope)``

        Parameters
        ----------
        field : float or ndarray
            Measured field, raw ``H_c2``
        slope, t_c : float or ndarray
            Scaling constants; see :class:`WHHModel`.

        Returns
        -------
        float or ndarray
            The dimensionless reduced field ``bar_h``.
              = (4/pi**2)*(Hc2(t)/(Tc*slope))
            where slope = |dHc2/dT| at T=Tc or t=T/Tc=1.
        """
        return (4.0 / np.pi**2) * field / (t_c * slope)

    @staticmethod
    def _bracket_term(nu, bar_h, t, alpha, l_so):
        """The square bracket ``[ ... ]`` that is inverted inside the summand.

        Corresponds to ``|2*nu+1| + bar_h/t + (alpha*bar_h/t)**2
        / (|2*nu+1| + (bar_h + l_so)/t)`` in Eq. (6). Scalar ``nu`` and ``t``.
        """
        odd = abs(2 * nu + 1)
        numerator = (alpha * bar_h / t) ** 2
        denominator = odd + (bar_h + l_so) / t
        return odd + bar_h / t + numerator / denominator

    @classmethod
    def _brace_term(cls, nu, bar_h, t, alpha, l_so):
        """The curly brace ``{ ... }`` summand for a single index ``nu``.

        Corresponds to ``1/|2*nu+1| - 1/[bracket]`` in Eq. (6).
        """
        return 1.0 / abs(2 * nu + 1) - 1.0 / cls._bracket_term(
            nu, bar_h, t, alpha, l_so
        )

    @classmethod
    def _on_seam(cls, l_so, alpha, bar_h, disc_tol=1e-6):
        return np.abs(l_so - 2 * alpha * bar_h) < disc_tol

    @classmethod
    def _brace_term_polygamma(
        cls, bar_h, t, alpha, l_so, on_seam, imag_tol=1e-9
    ) -> float:
        """summation of the WHH model, depends on the degeneracy

        satisfies
        abs(l_so - 2 * alpha * bar_h)>tol

        then summation becomes.. #TODO: validate equation

        Parameters
        ----------
        bar_h : float
            reduced field, bar_h, see WHHModel.reduced_field
        t : float
            reduced temperature, t=T/Tc
        alpha : float
            Maki parameter
        l_so : float
            Spin orbit scattering parameter.
        on_seam : bool
            if True, alpha and l_so have high degeneracy on the bar_h.

        Returns
        -------
        float
            summation value, real number.
        """
        root = l_so**2 / 4 - (alpha * bar_h) ** 2
        q = np.sqrt(root) / t if root >= 0 else np.sqrt(complex(root)) / t
        w = (bar_h + l_so / 2) / t

        trigamma = 0
        if on_seam:
            c1 = 1 / 2
            c2 = 1 / 2
            trigamma = l_so / (4 * t) * polygamma(1, (1 + w) / 2)  # trigamma
        else:
            c1 = 1 / 2 - l_so / (4 * t * q)
            c2 = 1 / 2 + (l_so / (4 * t * q))

        simple_pole1 = c1 * digamma((1 + q + w) / 2)
        simple_pole2 = c2 * digamma((1 - q + w) / 2)
        out = simple_pole1 + simple_pole2 - digamma(1 / 2) - trigamma

        if isinstance(out, complex):
            if abs(out.imag) > imag_tol:
                raise ArithmeticError(f"should be real but {out=}")
            else:
                out = out.real
        return out

    @classmethod
    def _series(cls, bar_h, t, alpha, l_so, summation, disc_tol=1e-6):
        """Right-hand-side series ``sum_nu { ... }`` for scalar inputs."""
        if summation == "mpmath_nsum":

            def summand(nu):
                return cls._brace_term(nu, bar_h, t, alpha, l_so)

            return float(nsum(summand, [-inf, inf], method="r+s+e"))
        elif summation == "polygamma":
            on_seam = cls._on_seam(l_so, alpha, bar_h, disc_tol=disc_tol)
            return cls._brace_term_polygamma(bar_h, t, alpha, l_so, on_seam)
        else:
            raise ValueError(summation)

    @classmethod
    def equation_residual(cls, bar_h, t, alpha, l_so, summation):
        """WHH equation residual ``ln(1/t) - sum_nu{...}`` (scalar).

        Equals zero exactly on the physical solution. Used both as the per-
        point residual for fitting and as the function whose root gives the
        forward curve. Note ``t`` must lie in ``(0, 1)``.
        """
        return np.log(1.0 / t) - cls._series(bar_h, t, alpha, l_so, summation)

    # ------------------------------------------------------------------ #
    # Residual builder for scipy.optimize.least_squares
    # ------------------------------------------------------------------ #
    def _residual_vector(self, t, field, alpha, l_so, slope, t_c, summation):
        """Residuals at every measured point, looping point-by-point.

        Vectorisation is done with a plain Python loop because the series is
        summed per scalar point; ``slope``/``t_c`` may be scalars
        or ``(n,)`` arrays and are broadcast to the shape of ``t``.
        """
        t = np.asarray(t, dtype=float)
        field = np.asarray(field, dtype=float)
        slope = np.broadcast_to(np.asarray(np.abs(slope), float), t.shape)
        t_c = np.broadcast_to(np.asarray(t_c, float), t.shape)
        reduced_t = t / t_c
        out = np.empty_like(t)
        for i in range(t.size):
            bar_h = self.reduced_field(field[i], slope[i], t_c[i])
            out[i] = self.equation_residual(bar_h, reduced_t[i], alpha, l_so, summation)
        return out

    def make_residual(self, t, field, fit=("alpha", "l_so"), x0=None, x0_eps=1e-3):
        """Build a residual function and initial guess for ``least_squares``.

        Bakes the *fixed* parameters (everything not named in ``fit``) into a
        closure, so the returned callable is a function of the fit vector only.

        Parameters
        ----------
        t : array_like, shape (n,)
            Measured  temperature ``T``. Held fixed.
            ***Do not use reduced t=T/T_c***
        t_c : float. Superconducting critical temperature.
        field : array_like, shape (n,)
            Measured field. Held fixed.
        fit : sequence of str, optional
            Which of ``{'alpha', 'l_so', 'slope', 't_c'}`` are
            *fitted*. All others are held at the instance's stored value.
            Default ``('alpha', 'l_so')``.
        x0 : array_like, optional
            Initial guess, one entry per name in ``fit`` and in the same order.
            If omitted, the instance's stored values are used.
        x0_eps : float, optional
            Lower nudge for the start point. Any component of ``x0`` smaller
            than ``x0_eps`` is increased by ``x0_eps``. This lifts near-zero
            starts off the flat-gradient region: ``alpha`` enters the equation
            only as ``alpha**2``, so the cost gradient vanishes at
            ``alpha = 0`` and an optimizer started there never moves. Safe to
            apply to every component because all five parameters are
            non-negative. Set ``x0_eps=0`` to disable. Default 1e-3.

        Returns
        -------
        residual : callable
            ``residual(x) -> ndarray`` of shape ``(n,)``, suitable as the first
            argument of :func:`scipy.optimize.least_squares`. The order of ``x``
            matches ``fit``.
        x0 : ndarray
            Initial guess aligned with ``fit``, after the ``x0_eps`` nudge.

        Examples
        --------
        >>> model = WHHModel(t_c=5.28, slope=1.4)
        >>> resid, x0 = model.make_residual(t, h_norm, fit=("alpha", "l_so"))
        >>> res = least_squares(resid, x0, bounds=((0, 0), (5, 5)))
        >>> alpha_fit, l_so_fit = res.x
        """
        fit = tuple(fit)
        unknown = set(fit) - set(self.PARAM_NAMES)
        if unknown:
            raise ValueError(
                f"unknown fit parameter(s) {unknown}; choose from {self.PARAM_NAMES}"
            )

        if x0 is None:
            x0 = np.array([self.params[name] for name in fit], dtype=float)
        else:
            x0 = np.asarray(x0, dtype=float)
            if x0.shape != (len(fit),):
                raise ValueError(
                    f"x0 has shape {x0.shape}, expected ({len(fit)},) "
                    f"to match fit={fit}"
                )

        # lift near-zero starts off the flat-gradient region (see x0_eps doc)
        x0 = np.where(x0 < x0_eps, x0 + x0_eps, x0)

        def residual(x):
            values = dict(self.params)  # start from the fixed defaults
            values.update(zip(fit, x))  # overwrite the fitted ones
            return self._residual_vector(t, field, **values)

        return residual, x0

    # ------------------------------------------------------------------ #
    # Forward curve for plotting
    # ------------------------------------------------------------------ #
    def predict(
        self, t_grid, alpha=None, l_so=None, slope=None, t_c=None, bar_h_max=100.0
    ):
        """Solve the implicit equation for the field at each temperature.

        For each ``t`` in ``t_grid`` the WHH equation is monotonic in
        ``bar_h``, so the field is found by a 1-D bracketed root search
        (:func:`scipy.optimize.brentq`) rather than a least-squares solve.

        Parameters
        ----------
        t_grid : array_like, shape (m,)
            Reduced temperatures in ``(0, 1)``.
        alpha, l_so : float, optional
            Shape parameters; default to the instance's stored values.
        bar_h_max : float, optional
            Upper bracket for the root search. The default comfortably exceeds
            the physical ``bar_h`` (which is below ~0.3 in reduced units).

        Returns
        -------
        ndarray, shape (m,)
            Predicted field ``H_c2`` at each ``t``.
        """
        alpha = self.params["alpha"] if alpha is None else alpha
        l_so = self.params["l_so"] if l_so is None else l_so
        slope = self.params["slope"] if slope is None else np.abs(slope)
        t_c = self.params["t_c"] if t_c is None else t_c

        t_grid = np.asarray(t_grid, dtype=float)
        bar_h = np.empty_like(t_grid)

        if max(t_grid) > 1.0:
            if len(t_grid[t_grid > 1.0]) == 1 and len(t_grid) > 7:
                warnings.warn(
                    f"t_grid should be a reduced critical temperature, t/t_c but {max(t_grid)=}",
                    UserWarning,
                )
                mask_t = t_grid < 1.0
                t_grid = t_grid[mask_t]
                bar_h[mask_t] = None
            else:
                raise ValueError(f"two many {t_grid=} values over 1.0")

        for i, t in enumerate(t_grid):
            if t == 1:
                # exceptional case t=1 (T=Tc)
                bar_h[i] = 0.0
            else:

                def f_hb(hb):
                    return self.equation_residual(
                        hb, t, alpha, l_so, summation=self.params["summation"]
                    )

                bar_h[i] = brentq(
                    f_hb,
                    0.0,
                    bar_h_max,
                )
        h_c2 = bar_h * (np.pi**2 / 4.0) * (t_c * slope)
        return h_c2

    def curve(
        self,
        t_grid,
        t_c=None,
        h_orb=None,
        alpha=None,
        l_so=None,
        slope=None,
        return_reduced=True,
    ):
        """Forward ``H_c2(T)`` curve, ready to hand to a plot call.

        Thin wrapper over :meth:`predict` that returns the temperature and
        field arrays together.

        Parameters
        ----------
        t_grid : array_like, shape (m,)
            reduced temperatures in ``(0, 1.0)``.
        alpha, l_so : float, optional
            Shape parameters; default to the instance's stored values.
        return_reduced : bool, default True.
            If True (default) return ``(t, H_c2/H_orb)`` — matching a
            ``T/T_c`` vs ``H/H_orb`` plot. If False, return physical
            ``(T, H_c2) = (t*t_c, (H_c2))``.

        Returns
        -------
        temperature : ndarray, shape (m,)
        field : ndarray, shape (m,)
        """
        if self.params.get("t_c") is None and t_c is None:
            if not return_reduced:
                raise ValueError("t_c is not given")
        if self.params.get("t_c") is not None and t_c is not None:
            raise ValueError(
                f"t_c is repeatedly given by the __init__ {self.params["t_c"]=} and the method argument {t_c=}"
            )
        t_c = self.params["t_c"] if t_c is None else t_c

        field = self.predict(t_grid, alpha=alpha, l_so=l_so, slope=slope, t_c=t_c)
        t_grid = np.asarray(t_grid, dtype=float)
        if return_reduced:
            if h_orb is None:
                raise ValueError(f"h_orb required if {return_reduced=} but {h_orb=}")
            return t_grid, field / h_orb
        else:
            return t_grid * t_c, field

    # ------------------------------------------------------------------ #
    # Goodness of fit
    # ------------------------------------------------------------------ #
    def score(self, t_obs, field_obs, alpha=None, l_so=None):
        """Goodness-of-fit metrics comparing the curve to observed points.

        Parameters
        ----------
        t_obs : array_like, shape (n,)
            Reduced temperatures of the measurements.
        field_obs : array_like, shape (n,)
            Measured field at ``t_obs``.
        alpha, l_so : float, optional
            Shape parameters; default to the instance's stored values.

        Returns
        -------
        dict
            ``{'r2', 'rmse', 'residuals', 'predicted'}`` where ``r2`` is the
            coefficient of determination, ``rmse`` the root-mean-square error,
            and the arrays are the field-space residuals and predictions.
        """
        field_obs = np.asarray(field_obs, dtype=float)

        mask_t = t_obs <= 1
        t_obs = t_obs[mask_t]
        field_obs = field_obs[mask_t]

        predicted = self.predict(t_obs, alpha=alpha, l_so=l_so)

        residuals = field_obs - predicted
        return {
            "r2": self.r2(field_obs, predicted),
            "rmse": float(np.sqrt(np.mean(residuals**2))),
            "residuals": residuals,
            "predicted": predicted,
        }

    @staticmethod
    def r2(observed, predicted):
        """Coefficient of determination R**2, observed-vs-predicted.

        Defined as ``1 - SS_res / SS_tot`` with ``SS_tot`` taken about the mean
        of ``observed``. R**2 is **not** symmetric in its arguments, so the
        order matters: ``observed`` first, ``predicted`` second (same
        convention as :func:`sklearn.metrics.r2_score`, which is
        ``r2_score(y_true, y_pred)``).
        """
        observed = np.asarray(observed, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        ss_res = np.sum((observed - predicted) ** 2)
        ss_tot = np.sum((observed - observed.mean()) ** 2)
        return 1.0 - ss_res / ss_tot

    @staticmethod
    def param_uncertainty(result):
        """One-sigma parameter errors from a ``least_squares`` result.

        Approximates the covariance as ``sigma**2 * (J^T J)^-1`` with the
        residual variance ``sigma**2 = 2*cost / (n_data - n_param)`` and the
        Jacobian ``J`` at the solution. Returns the square root of its
        diagonal, aligned with ``result.x``.

        Parameters
        ----------
        result : OptimizeResult
            The object returned by :func:`scipy.optimize.least_squares`.

        Returns
        -------
        ndarray
            Estimated standard errors, same length as ``result.x``. Entries are
            NaN if the Jacobian is rank-deficient (non-identifiable parameter).
        """
        jac = result.jac
        n_data, n_param = jac.shape
        dof = max(n_data - n_param, 1)
        sigma2 = 2.0 * result.cost / dof
        try:
            cov = sigma2 * np.linalg.inv(jac.T @ jac)
            return np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            return np.full(n_param, np.nan)
