import numpy as np
import matplotlib.pyplot as plt


def f(t, y):
    return -y / t + 3 * t


def euler_method(f, t0, y0, T, h):
    N = int((T - t0) / h)
    t = np.linspace(t0, T, N + 1)
    y = np.zeros(N + 1)
    y[0] = y0

    for i in range(N):
        y[i + 1] = y[i] + h * f(t[i], y[i])

    return t, y


t0 = 1
T = 2
y0 = 1
h = 0.1

t_euler, y_euler = euler_method(f, t0, y0, T, h)


def runge_kutta_4(f, t0, y0, T, h):
    N = int((T - t0) / h)
    t = np.linspace(t0, T, N + 1)
    y = np.zeros(N + 1)
    y[0] = y0

    for i in range(N):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i] + h / 2, y[i] + k1 / 2)
        k3 = h * f(t[i] + h / 2, y[i] + k2 / 2)
        k4 = h * f(t[i] + h, y[i] + k3)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return t, y


t_rk, y_rk = runge_kutta_4(f, t0, y0, T, h)


def analytical_solution(t):
    return t * t

t_analytical = np.linspace(t0, T, 100)
y_analytical = analytical_solution(t_analytical)

plt.plot(t_euler, y_euler, 'o-', label='Euler method')
plt.plot(t_rk, y_rk, 's-', label='Runge-Kutta 4th order')
plt.plot(t_analytical, y_analytical, '^-', label='Analytical solution')
plt.legend()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Comparison of Euler, Runge-Kutta and Analytical solutions')
plt.grid(True)
plt.show()

def max_error(y_approx, y_exact):
    return np.max(np.abs(y_approx - y_exact))

y_exact_euler = analytical_solution(t_euler)
y_exact_rk = analytical_solution(t_rk)

error_euler = max_error(y_euler, y_exact_euler)
error_rk = max_error(y_rk, y_exact_rk)

print(f'Max error (Euler method): {error_euler}')
print(f'Max error (Runge-Kutta method): {error_rk}')


h_euler = 0.1
error_target = error_rk
while True:
    t_euler, y_euler = euler_method(f, t0, y0, T, h_euler)
    y_exact_euler = analytical_solution(t_euler)
    error_euler = max_error(y_euler, y_exact_euler)
    if error_euler <= error_target:
        break
    h_euler /= 2

print(f'Optimal step for Euler method: {h_euler}')
