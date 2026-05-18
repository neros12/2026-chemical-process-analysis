from typing import TypedDict

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class History(TypedDict):
    time: list[float]
    A: list[FloatArray]
    V: list[FloatArray]


class SpinningResult(TypedDict):
    x: FloatArray
    A: FloatArray
    V: FloatArray
    time: float
    history: History


# ============================================================
# Tridiagonal solver: Thomas algorithm
# ============================================================
def thomas_solver(
    lower: FloatArray,
    diag: FloatArray,
    upper: FloatArray,
    rhs: FloatArray,
) -> FloatArray:
    """
    Solve tridiagonal matrix system.

    lower: length n-1
    diag : length n
    upper: length n-1
    rhs  : length n
    """
    n = len(diag)

    a = lower.astype(float).copy()
    b = diag.astype(float).copy()
    c = upper.astype(float).copy()
    d = rhs.astype(float).copy()

    # Forward elimination
    for i in range(1, n):
        m = a[i - 1] / b[i - 1]
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]

    # Back substitution
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]

    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x


# ============================================================
# Solve EOC for A
# ============================================================
def solve_A(A_old: FloatArray, V: FloatArray, dx: float, dt: float) -> FloatArray:
    """
    Solve continuity equation:

        dA/dt + d(AV)/dx = 0

    using backward Euler in time and central difference in space.

    Boundary:
        A(0) = 1
    """

    N = len(A_old) - 1

    # Unknowns: A_1, A_2, ..., A_N
    n = N

    lower = np.zeros(n - 1)
    diag = np.zeros(n)
    upper = np.zeros(n - 1)
    rhs = np.zeros(n)

    # Interior nodes: i = 1, ..., N-1
    for i in range(1, N):
        row = i - 1

        # Equation:
        # (A_i^{n+1} - A_i^n)/dt
        # + ((AV)_{i+1}^{n+1} - (AV)_{i-1}^{n+1}) / (2 dx) = 0
        #
        # Multiply by 2dx:
        # -V_{i-1} A_{i-1}^{n+1}
        # + (2dx/dt) A_i^{n+1}
        # + V_{i+1} A_{i+1}^{n+1}
        # = (2dx/dt) A_i^n

        diag[row] = 2.0 * dx / dt
        rhs[row] = 2.0 * dx / dt * A_old[i]

        # A_{i-1}
        if i - 1 == 0:
            # A_0 = 1 is known
            rhs[row] += V[i - 1] * 1.0
        else:
            lower[row - 1] = -V[i - 1]

        # A_{i+1}
        upper[row] = V[i + 1]

    # Last node: i = N
    # Use backward difference for flux:
    #
    # (A_N^{n+1} - A_N^n)/dt
    # + ((AV)_N^{n+1} - (AV)_{N-1}^{n+1}) / dx = 0
    #
    # Multiply by dx:
    # - V_{N-1} A_{N-1}^{n+1}
    # + (dx/dt + V_N) A_N^{n+1}
    # = dx/dt A_N^n

    row = N - 1
    diag[row] = dx / dt + V[N]
    rhs[row] = dx / dt * A_old[N]

    if N - 1 == 0:
        rhs[row] += V[N - 1] * 1.0
    else:
        lower[row - 1] = -V[N - 1]

    A_unknown = thomas_solver(lower, diag, upper, rhs)

    A_new = np.zeros(N + 1)
    A_new[0] = 1.0
    A_new[1:] = A_unknown

    return A_new


# ============================================================
# Solve EOM for V
# ============================================================
def solve_V(A: FloatArray, r: float, dx: float) -> FloatArray:
    """
    Solve momentum equation:

        d/dx ( A dV/dx ) = 0

    Expanded form:

        dA/dx dV/dx + A d2V/dx2 = 0

    Boundary:
        V(0) = 1
        V(1) = r
    """

    N = len(A) - 1

    # Unknowns: V_1, V_2, ..., V_{N-1}
    n = N - 1

    lower = np.zeros(n - 1)
    diag = np.zeros(n)
    upper = np.zeros(n - 1)
    rhs = np.zeros(n)

    for i in range(1, N):
        row = i - 1

        # dA/dx * dV/dx + A * d2V/dx2 = 0
        #
        # dA/dx ≈ (A_{i+1} - A_{i-1}) / (2dx)
        # dV/dx ≈ (V_{i+1} - V_{i-1}) / (2dx)
        # d2V/dx2 ≈ (V_{i+1} - 2V_i + V_{i-1}) / dx^2

        coeff_grad = (A[i + 1] - A[i - 1]) / (4.0 * dx**2)

        c_left = A[i] / dx**2 - coeff_grad
        c_mid = -2.0 * A[i] / dx**2
        c_right = A[i] / dx**2 + coeff_grad

        diag[row] = c_mid

        # V_{i-1}
        if i - 1 == 0:
            # V_0 = 1
            rhs[row] -= c_left * 1.0
        else:
            lower[row - 1] = c_left

        # V_{i+1}
        if i + 1 == N:
            # V_N = r
            rhs[row] -= c_right * r
        else:
            upper[row] = c_right

    V_inner = thomas_solver(lower, diag, upper, rhs)

    V = np.zeros(N + 1)
    V[0] = 1.0
    V[N] = r
    V[1:N] = V_inner

    return V


# ============================================================
# Full transient solver
# ============================================================
def spinning_solver(
    r_initial: float = 15.0,
    r_final: float = 15.1,
    N: int = 100,
    dt: float = 1e-3,
    t_max: float = 20.0,
    steady_tol: float = 1e-8,
    inner_tol: float = 1e-10,
    max_inner_iter: int = 50,
    save_interval: int = 100,
) -> SpinningResult:
    """
    Transient solver for spinning process.

    Initial condition:
        A(x,0) = r_initial^(-x)
        V(x,0) = r_initial^(x)

    Then boundary condition V(1,t) is changed to r_final.
    """

    dx = 1.0 / N
    x = np.linspace(0.0, 1.0, N + 1)

    # Initial steady state at r_initial
    A = r_initial ** (-x)
    V = r_initial**x

    # New boundary condition
    V[0] = 1.0
    V[-1] = r_final

    history: History = {
        "time": [],
        "A": [],
        "V": [],
    }

    n_steps = int(t_max / dt)
    t = 0.0

    for step in range(n_steps):
        t = (step + 1) * dt

        A_prev_time = A.copy()
        V_prev_time = V.copy()

        # Fixed-point iteration inside one time step
        A_guess = A.copy()
        V_guess = V.copy()

        for inner in range(max_inner_iter):
            A_new = solve_A(A_prev_time, V_guess, dx, dt)
            V_new = solve_V(A_new, r_final, dx)

            err_A = np.max(np.abs(A_new - A_guess))
            err_V = np.max(np.abs(V_new - V_guess))
            err = max(err_A, err_V)

            A_guess = A_new
            V_guess = V_new

            if err < inner_tol:
                break

        A = A_guess
        V = V_guess

        # Save history
        if step % save_interval == 0:
            history["time"].append(t)
            history["A"].append(A.copy())
            history["V"].append(V.copy())

        # Check steady state
        err_time_A = np.max(np.abs(A - A_prev_time))
        err_time_V = np.max(np.abs(V - V_prev_time))
        err_time = max(err_time_A, err_time_V)

        if err_time < steady_tol:
            print(f"Converged at t = {t:.6f}, step = {step + 1}")
            break

    history["time"].append(t)
    history["A"].append(A.copy())
    history["V"].append(V.copy())

    result: SpinningResult = {
        "x": x,
        "A": A,
        "V": V,
        "time": t,
        "history": history,
    }

    return result


# ============================================================
# Run examples
# ============================================================
if __name__ == "__main__":

    r_initial = 15.0

    # 문제에서 요구하는 r 값으로 바꿔서 사용하면 됨
    r_cases = [15.1, 35.0]

    for r_final in r_cases:
        print("=" * 60)
        print(f"Solving transient from r = {r_initial} to r = {r_final}")

        result = spinning_solver(
            r_initial=r_initial,
            r_final=r_final,
            N=120,
            dt=1e-3,
            t_max=20.0,
            steady_tol=1e-8,
            inner_tol=1e-10,
            max_inner_iter=50,
            save_interval=500,
        )

        x = result["x"]
        A = result["A"]
        V = result["V"]

        # Analytic steady solution for comparison
        A_exact = r_final ** (-x)
        V_exact = r_final**x

        err_A = np.max(np.abs(A - A_exact))
        err_V = np.max(np.abs(V - V_exact))

        print(f"Final time = {result['time']:.6f}")
        print(f"max |A - A_exact| = {err_A:.6e}")
        print(f"max |V - V_exact| = {err_V:.6e}")

        # Plot A
        plt.figure()
        plt.plot(x, A, label="Numerical A")
        plt.plot(x, A_exact, "--", label="Exact steady A")
        plt.xlabel("x")
        plt.ylabel("A")
        plt.title(f"A(x), r = {r_final}")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot V
        plt.figure()
        plt.plot(x, V, label="Numerical V")
        plt.plot(x, V_exact, "--", label="Exact steady V")
        plt.xlabel("x")
        plt.ylabel("V")
        plt.title(f"V(x), r = {r_final}")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot transient profiles of A
        plt.figure()
        for t, A_hist in zip(result["history"]["time"], result["history"]["A"]):
            plt.plot(x, A_hist, label=f"t={t:.3f}")
        plt.xlabel("x")
        plt.ylabel("A")
        plt.title(f"Transient A profiles, r = {r_final}")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot transient profiles of V
        plt.figure()
        for t, V_hist in zip(result["history"]["time"], result["history"]["V"]):
            plt.plot(x, V_hist, label=f"t={t:.3f}")
        plt.xlabel("x")
        plt.ylabel("V")
        plt.title(f"Transient V profiles, r = {r_final}")
        plt.legend()
        plt.grid(True)
        plt.show()
