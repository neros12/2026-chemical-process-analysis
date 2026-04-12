import numpy as np
import matplotlib.pyplot as plt


def f(t, y, z):
    # y' = z
    return z


def g(t, y, z):
    # z' = 8 - y/4
    return 8.0 - y / 4.0


def rk4_ivp(y0, z0, a, b, h):
    n = int((b - a) / h)
    t = np.linspace(a, b, n + 1)

    y = np.zeros(n + 1)
    z = np.zeros(n + 1)

    y[0] = y0
    z[0] = z0

    for i in range(n):
        k1y = f(t[i], y[i], z[i])
        k1z = g(t[i], y[i], z[i])

        k2y = f(t[i] + h / 2, y[i] + h * k1y / 2, z[i] + h * k1z / 2)
        k2z = g(t[i] + h / 2, y[i] + h * k1y / 2, z[i] + h * k1z / 2)

        k3y = f(t[i] + h / 2, y[i] + h * k2y / 2, z[i] + h * k2z / 2)
        k3z = g(t[i] + h / 2, y[i] + h * k2y / 2, z[i] + h * k2z / 2)

        k4y = f(t[i] + h, y[i] + h * k3y, z[i] + h * k3z)
        k4z = g(t[i] + h, y[i] + h * k3y, z[i] + h * k3z)

        y[i + 1] = y[i] + (h / 6) * (k1y + 2 * k2y + 2 * k3y + k4y)
        z[i + 1] = z[i] + (h / 6) * (k1z + 2 * k2z + 2 * k3z + k4z)

    return t, y, z


def shoot_bvp(y0, yb, a, b, h, s0, s1, tol=1e-8, max_iter=50):
    # 첫 번째 추정
    t, y, z = rk4_ivp(y0, s0, a, b, h)
    F0 = y[-1] - yb

    # 두 번째 추정
    t, y, z = rk4_ivp(y0, s1, a, b, h)
    F1 = y[-1] - yb

    for _ in range(max_iter):
        if abs(F1) < tol:
            return s1, t, y, z

        # secant update
        s2 = s1 - F1 * (s1 - s0) / (F1 - F0)

        s0, F0 = s1, F1
        s1 = s2

        t, y, z = rk4_ivp(y0, s1, a, b, h)
        F1 = y[-1] - yb

    raise RuntimeError("수렴하지 않았습니다.")


# 문제 설정
a = 0.0
b = 10.0
h = 0.1

y0 = 0.0
yb = 0.0

# 초기기울기 추정값 2개
s0 = 5.0
s1 = 10.0

s, t, y, z = shoot_bvp(y0, yb, a, b, h, s0, s1)

print(f"찾은 초기기울기 y'(0) = {s:.10f}")
print(f"y(10) = {y[-1]:.10e}")

# # 몇 개 점 출력
# for i in range(0, len(t), 20):
#     print(f"t = {t[i]:6.2f}, y = {y[i]:12.6f}")

# 그래프
plt.plot(t, y, label="y(t)")
plt.plot(t, z, "--", label="z(t)=y'(t)")
plt.xlabel("t")
plt.ylabel("solution")
plt.title(r"Solution of $y'' + \frac14 y = 8,\; y(0)=0,\; y(10)=0$")
plt.grid(True)
plt.legend()
plt.show()
