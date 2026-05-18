import numpy as np

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


def solve_problem(r: float, N=300) -> tuple[float, float]:
    """
    Compute eigenvalues for a given drawdown ratio r and mesh number N.

    Parameters
    ----------
    r : float
        Drawdown ratio.
    N : int
        Number of mesh intervals.

    Returns
    -------
    eigvals : ndarray
        Eigenvalues lambda.
    """

    dz = 1.0 / N
    z = np.linspace(0.0, 1.0, N + 1)

    vs = r**z
    vs_z = np.log(r) * vs
    vs_zz = np.log(r) ** 2 * vs

    # Unknowns:
    # alpha_1, alpha_2, ..., alpha_N
    # beta_1, beta_2, ..., beta_{N-1}
    #
    # alpha_0 = 0
    # beta_0 = 0
    # beta_N = 0

    n_alpha = N
    n_beta = N - 1
    n_total = n_alpha + n_beta

    A = np.zeros((n_total, n_total), dtype=float)

    # ------------------------------------------------------------
    # Linearized EOC:
    #
    # lambda alpha =
    #   - v_s alpha_z
    #   - v_s_z alpha
    #   + v_s_z / v_s^2 beta
    #   - 1 / v_s beta_z
    # ------------------------------------------------------------
    for i in range(1, N + 1):
        row = i - 1

        # - v_s alpha_z
        if i == N:
            # alpha_N index = N - 1
            # alpha_{N-1} index = N - 2
            A[row, N - 1] += -vs[i] / dz
            A[row, N - 2] += vs[i] / dz
        else:
            # alpha_{i+1}
            if i + 1 <= N:
                A[row, (i + 1) - 1] += -vs[i] / (2.0 * dz)

            # alpha_{i-1}
            if i - 1 >= 1:
                A[row, (i - 1) - 1] += vs[i] / (2.0 * dz)

        # - v_s_z alpha_i
        A[row, i - 1] += -vs_z[i]

        # - 1 / v_s beta_z
        if i == N:
            # beta_N = 0
            # beta_{N-1} index = n_alpha + (N-1) - 1
            A[row, n_alpha + N - 2] += 1.0 / (vs[i] * dz)
        else:
            # beta_{i+1}
            if i + 1 <= N - 1:
                A[row, n_alpha + (i + 1) - 1] += -1.0 / (vs[i] * 2.0 * dz)

            # beta_{i-1}
            if i - 1 >= 1:
                A[row, n_alpha + (i - 1) - 1] += 1.0 / (vs[i] * 2.0 * dz)

        # + v_s_z / v_s^2 beta_i
        if i <= N - 1:
            A[row, n_alpha + i - 1] += vs_z[i] / vs[i] ** 2

    # ------------------------------------------------------------
    # Linearized EOM:
    #
    # 0 =
    #   v_s v_s_zz alpha
    #   + v_s v_s_z alpha_z
    #   - v_s_z / v_s beta_z
    #   + beta_zz
    # ------------------------------------------------------------
    for i in range(1, N):
        row = n_alpha + i - 1

        # v_s v_s_zz alpha_i
        A[row, i - 1] += vs[i] * vs_zz[i]

        # v_s v_s_z alpha_z
        # alpha_{i+1}
        if i + 1 <= N:
            A[row, (i + 1) - 1] += vs[i] * vs_z[i] / (2.0 * dz)

        # alpha_{i-1}
        if i - 1 >= 1:
            A[row, (i - 1) - 1] += -vs[i] * vs_z[i] / (2.0 * dz)

        # beta_zz
        # beta_i
        A[row, n_alpha + i - 1] += -2.0 / dz**2

        # beta_{i+1}
        if i + 1 <= N - 1:
            A[row, n_alpha + (i + 1) - 1] += 1.0 / dz**2

        # beta_{i-1}
        if i - 1 >= 1:
            A[row, n_alpha + (i - 1) - 1] += 1.0 / dz**2

        # - v_s_z / v_s beta_z
        coeff = -vs_z[i] / vs[i]

        # beta_{i+1}
        if i + 1 <= N - 1:
            A[row, n_alpha + (i + 1) - 1] += coeff / (2.0 * dz)

        # beta_{i-1}
        if i - 1 >= 1:
            A[row, n_alpha + (i - 1) - 1] += -coeff / (2.0 * dz)

    # Generalized eigenvalue problem:
    # A q = lambda B q
    Aaa = A[:n_alpha, :n_alpha]
    Aab = A[:n_alpha, n_alpha:]
    Aba = A[n_alpha:, :n_alpha]
    Abb = A[n_alpha:, n_alpha:]

    Aeff = Aaa - Aab @ np.linalg.inv(Abb) @ Aba
    eigvals = np.linalg.eigvals(Aeff)
    lam = eigvals[np.argmax(np.real(eigvals))]

    return lam.real, np.abs(lam.imag)


def critical_draw_ratio(N=300, *, max_iter=100, tol=1e-6):
    r0 = 15
    r1 = 16
    for _ in range(max_iter):
        f0, _ = solve_problem(r0, N)
        f1, _ = solve_problem(r1, N)

        if abs(f1) < tol:
            return r1

        r2 = r1 - f1 * (r1 - r0) / (f1 - f0)
        r0 = r1
        r1 = r2
    else:
        raise Exception("ERROR::Iteration Failed!")


if __name__ == "__main__":
    rs = [15.0, 20.0, 20.218, 22.0, 25.0]
    N = 300
    for r in rs:
        real, imag = solve_problem(r, N)
        print(f"{r:5.3f} {real:15.4f} {imag:18.3f}")

    print()
    print("---------------------------------------")
    print()

    Ns = [100, 200, 300, 400, 500, 700, 1000, 1100, 1200, 2000]
    for N in Ns:
        r_crit = critical_draw_ratio(N)
        print(f"{N:4.0f} {r_crit:10.4f}")
