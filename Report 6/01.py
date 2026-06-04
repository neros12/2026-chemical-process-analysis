# ============================================================
# Linear stability analysis of isothermal Newtonian spinning
#
# Dimensionless equations:
#   a_t + (a v)_z = 0
#   d/dz ( a v_z ) = 0
#
# Steady solution:
#   v_s(z) = r^z
#   a_s(z) = 1 / v_s(z)
#
# Perturbation:
#   a(z,t) = a_s(z) + alpha(z) exp(lambda t)
#   v(z,t) = v_s(z) + beta(z) exp(lambda t)
#
# Boundary conditions:
#   alpha(0) = 0
#   beta(0)  = 0
#   beta(1)  = 0
# ============================================================

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

NumpyArray = NDArray[np.float64]
FloatArray = NDArray[np.float64]
SteadyValues = tuple[float, float, float]
ResponsePair = tuple[float, float]
ResponseParts = tuple[ResponsePair, ResponsePair, ResponsePair]


def Vs(z: NumpyArray, *, r=25.0) -> NumpyArray:
    return r**z


def Vs_dz(z: NumpyArray, *, r=25.0) -> NumpyArray:
    return np.log(r) * r**z


def As(z: NumpyArray, *, r=25.0) -> NumpyArray:
    return 1 / r**z


def secant_method(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    *,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    f0 = f(x0)
    f1 = f(x1)

    for _ in range(max_iter):
        if abs(f1) < tol:
            return x1

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        f2 = f(x2)

        x0 = x1
        f0 = f1
        x1 = x2
        f1 = f2

    raise RuntimeError(f"ERROR:: 최대 iteration({max_iter}) 안에 수렴하지 못했습니다!")


def steady_shooting_constant(draw_ratio: float, reynolds: float) -> float:
    """
    Return q for V' = q V + Re V^2 with V(0)=1, V(1)=draw_ratio.
    """

    if abs(reynolds) < 1e-14:
        return float(np.log(draw_ratio))

    def residual(q: float) -> float:
        return (1.0 + reynolds / q) * np.exp(-q) - reynolds / q - 1.0 / draw_ratio

    return secant_method(residual, np.log(draw_ratio), 1.2 * np.log(draw_ratio))


def steady_velocity(
    x: float, draw_ratio: float, reynolds: float, q: float
) -> SteadyValues:
    if abs(reynolds) < 1e-14:
        velocity = draw_ratio**x
        velocity_x = np.log(draw_ratio) * velocity
    else:
        inverse_velocity = (1.0 + reynolds / q) * np.exp(-q * x) - reynolds / q
        velocity = 1.0 / inverse_velocity
        velocity_x = q * velocity + reynolds * velocity**2

    log_velocity_x = velocity_x / velocity
    return float(velocity), float(velocity_x), float(log_velocity_x)


def transit_time(
    draw_ratio: float, reynolds: float, q: float, *, n_grid: int = 5001
) -> float:
    xs = np.linspace(0.0, 1.0, n_grid)
    inverse_velocity = np.empty_like(xs)
    for i, x in enumerate(xs):
        velocity, _, _ = steady_velocity(float(x), draw_ratio, reynolds, q)
        inverse_velocity[i] = 1.0 / velocity

    return float(np.trapezoid(inverse_velocity, xs))


def system_matrix(
    x: float,
    omega: float,
    draw_ratio: float,
    reynolds: float,
    q: float,
) -> FloatArray:
    velocity, velocity_x, log_velocity_x = steady_velocity(x, draw_ratio, reynolds, q)
    omega_over_velocity = omega / velocity
    log_velocity_omega_over_velocity = log_velocity_x * omega_over_velocity

    return np.array(
        [
            [0.0, omega_over_velocity, 0.0, 0.0, -1.0, 0.0],
            [-omega_over_velocity, 0.0, 0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [
                0.0,
                -log_velocity_omega_over_velocity,
                reynolds * velocity_x,
                -reynolds * omega,
                reynolds * velocity,
                0.0,
            ],
            [
                log_velocity_omega_over_velocity,
                0.0,
                reynolds * omega,
                reynolds * velocity_x,
                0.0,
                reynolds * velocity,
            ],
        ],
        dtype=float,
    )


def integrate_basis_solutions(
    omega: float,
    draw_ratio: float,
    reynolds: float,
    q: float,
    total_transit_time: float,
    *,
    points_per_cycle: int = 40,
    min_steps: int = 700,
) -> FloatArray:
    phase = abs(omega) * total_transit_time
    n_steps = max(min_steps, int(np.ceil(points_per_cycle * phase / (2.0 * np.pi))))
    h = 1.0 / n_steps

    y = np.eye(6, dtype=float)
    x = 0.0
    for _ in range(n_steps):
        k1 = system_matrix(x, omega, draw_ratio, reynolds, q) @ y
        k2 = system_matrix(x + 0.5 * h, omega, draw_ratio, reynolds, q) @ (
            y + 0.5 * h * k1
        )
        k3 = system_matrix(x + 0.5 * h, omega, draw_ratio, reynolds, q) @ (
            y + 0.5 * h * k2
        )
        k4 = system_matrix(x + h, omega, draw_ratio, reynolds, q) @ (y + h * k3)

        y += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        x += h

    return y


def frequency_response_at(
    omega: float,
    draw_ratio: float,
    reynolds: float,
    q: float,
    total_transit_time: float,
) -> ResponseParts:
    basis = integrate_basis_solutions(
        omega, draw_ratio, reynolds, q, total_transit_time
    )

    def solve_for_response(
        known_initial: FloatArray,
        target_beta_end: ResponsePair,
    ) -> ResponsePair:
        known_end = basis @ known_initial
        beta_x_sensitivity = basis[[2, 3], 4:6]
        residual = np.array(target_beta_end, dtype=float) - known_end[[2, 3]]
        beta_x_initial = np.linalg.solve(beta_x_sensitivity, residual)

        initial = known_initial.copy()
        initial[4] = beta_x_initial[0]
        initial[5] = beta_x_initial[1]
        end = basis @ initial

        return float(end[0]), float(end[1])

    spinneret_area_initial = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    initial_velocity_initial = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    takeup_velocity_initial = np.zeros(6, dtype=float)

    spinneret_area = solve_for_response(spinneret_area_initial, (0.0, 0.0))
    initial_velocity = solve_for_response(initial_velocity_initial, (0.0, 0.0))
    takeup_velocity = solve_for_response(takeup_velocity_initial, (1.0, 0.0))

    return spinneret_area, initial_velocity, takeup_velocity


def frequency_response(
    *,
    draw_ratio: float = 25.0,
    reynolds: float = 0.05,
    n_points: int = 140,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    frequencies = np.logspace(0.0, 4.0, n_points)
    q = steady_shooting_constant(draw_ratio, reynolds)
    total_transit_time = transit_time(draw_ratio, reynolds, q)

    spinneret_area = np.empty_like(frequencies)
    initial_velocity = np.empty_like(frequencies)
    takeup_velocity = np.empty_like(frequencies)

    for i, omega in enumerate(frequencies):
        area, initial, takeup = frequency_response_at(
            float(omega), draw_ratio, reynolds, q, total_transit_time
        )
        spinneret_area[i] = np.hypot(*area)
        initial_velocity[i] = np.hypot(*initial)
        takeup_velocity[i] = np.hypot(*takeup)

    return frequencies, spinneret_area, initial_velocity, takeup_velocity


if __name__ == "__main__":
    draw_ratio = 25.0
    reynolds = 0.05

    frequencies, spinneret_area, initial_velocity, takeup_velocity = frequency_response(
        draw_ratio=draw_ratio,
        reynolds=reynolds,
    )

    # ============================================================
    # Plot
    # ============================================================
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    ax.loglog(frequencies, spinneret_area, "k-.", lw=1.4, label="Spinneret area")
    ax.loglog(frequencies, initial_velocity, "k--", lw=1.4, label="Initial velocity")
    ax.loglog(frequencies, takeup_velocity, "k-", lw=1.4, label="Take-up velocity")

    ax.set_xlim(1.0, 1.0e4)
    ax.set_ylim(1.0e-1, 1.0e2)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Response of take-up area")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.legend()
    fig.tight_layout()
    plt.title(f"Re={reynolds:g}, r={draw_ratio:g}")
    plt.show()
