# First task
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def f(t, y):
#     return -y / t + 3 * t
#
#
# def euler_method(f, t0, y0, T, h):
#     N = int((T - t0) / h)
#     t = np.linspace(t0, T, N + 1)
#     y = np.zeros(N + 1)
#     y[0] = y0
#
#     for i in range(N):
#         y[i + 1] = y[i] + h * f(t[i], y[i])
#
#     return t, y
#
#
# t0 = 1
# T = 2
# y0 = 1
# h = 0.1
#
# t_euler, y_euler = euler_method(f, t0, y0, T, h)
#
#
# def runge_kutta_4(f, t0, y0, T, h):
#     N = int((T - t0) / h)
#     t = np.linspace(t0, T, N + 1)
#     y = np.zeros(N + 1)
#     y[0] = y0
#
#     for i in range(N):
#         k1 = h * f(t[i], y[i])
#         k2 = h * f(t[i] + h / 2, y[i] + k1 / 2)
#         k3 = h * f(t[i] + h / 2, y[i] + k2 / 2)
#         k4 = h * f(t[i] + h, y[i] + k3)
#         y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
#
#     return t, y
#
#
# t_rk, y_rk = runge_kutta_4(f, t0, y0, T, h)
#
#
# def analytical_solution(t):
#     return t * t
#
# t_analytical = np.linspace(t0, T, 100)
# y_analytical = analytical_solution(t_analytical)
#
# plt.plot(t_euler, y_euler, 'o-', label='Euler method')
# plt.plot(t_rk, y_rk, 's-', label='Runge-Kutta 4th order')
# plt.plot(t_analytical, y_analytical, '^-', label='Analytical solution')
# plt.legend()
# plt.xlabel('t')
# plt.ylabel('y(t)')
# plt.title('Comparison of Euler, Runge-Kutta and Analytical solutions')
# plt.grid(True)
# plt.show()
#
# def max_error(y_approx, y_exact):
#     return np.max(np.abs(y_approx - y_exact))
#
# y_exact_euler = analytical_solution(t_euler)
# y_exact_rk = analytical_solution(t_rk)
#
# error_euler = max_error(y_euler, y_exact_euler)
# error_rk = max_error(y_rk, y_exact_rk)
#
# print(f'Max error (Euler method): {error_euler}')
# print(f'Max error (Runge-Kutta method): {error_rk}')
#
#
# h_euler = 0.1
# error_target = error_rk
# while True:
#     t_euler, y_euler = euler_method(f, t0, y0, T, h_euler)
#     y_exact_euler = analytical_solution(t_euler)
#     error_euler = max_error(y_euler, y_exact_euler)
#     if error_euler <= error_target:
#         break
#     h_euler /= 2
#
# print(f'Optimal step for Euler method: {h_euler}')
#
#

# Second task
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def f(t, y):
#     return -t * y + (t - 1) * np.exp(t) * y ** 2
#
#
# def runge_kutta_4(f, t0, y0, T, h):
#     N = int((T - t0) / h)
#     t = np.linspace(t0, T, N + 1)
#     y = np.zeros(N + 1)
#     y[0] = y0
#
#     for i in range(N):
#         k1 = h * f(t[i], y[i])
#         k2 = h * f(t[i] + h / 2, y[i] + k1 / 2)
#         k3 = h * f(t[i] + h / 2, y[i] + k2 / 2)
#         k4 = h * f(t[i] + h, y[i] + k3)
#         y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
#
#     return t, y
#
#
# t0 = 0
# T = 1
# y0 = 1
# h = 0.1
#
# t_rk, y_rk = runge_kutta_4(f, t0, y0, T, h)
# t_rk_half, y_rk_half = runge_kutta_4(f, t0, y0, T, h / 2)
#
#
# def adams_bashforth_3(f, t0, y0, T, h):
#     # Начальные значения получаем с помощью метода Рунге-Кутты 4-го порядка
#     t_rk, y_rk = runge_kutta_4(f, t0, y0, t0 + 2 * h, h)
#
#     N = int((T - t0) / h)
#     t = np.linspace(t0, T, N + 1)
#     y = np.zeros(N + 1)
#     y[:3] = y_rk[:3]
#
#     for i in range(2, N):
#         y[i + 1] = y[i] + h / 12 * (23 * f(t[i], y[i]) - 16 * f(t[i - 1], y[i - 1]) + 5 * f(t[i - 2], y[i - 2]))
#
#     return t, y
#
#
# t_ab, y_ab = adams_bashforth_3(f, t0, y0, T, h)
# t_ab_half, y_ab_half = adams_bashforth_3(f, t0, y0, T, h / 2)
#
# def runge_rule(y_h, y_h2):
#     y_h2_interp = y_h2[::2]
#     return np.max(np.abs(y_h - y_h2_interp) / (2**3 - 1))
#
# def refined_solution(y_h, y_h2):
#     y_h2_interp = y_h2[::2]
#     return y_h + (y_h2_interp - y_h) / (2**3 - 1)
#
# error_rk = runge_rule(y_rk, y_rk_half)
# error_ab = runge_rule(y_ab, y_ab_half)
#
# y_rk_refined = refined_solution(y_rk, y_rk_half)
# y_ab_refined = refined_solution(y_ab, y_ab_half)
#
# print(f'Max Runge error (Runge-Kutta method): {error_rk}')
# print(f'Max Runge error (Adams-Bashforth method): {error_ab}')
#
# plt.plot(t_rk, y_rk, 'o-', label='Runge-Kutta 4th order (h)')
# plt.plot(t_rk_half, y_rk_half, 's-', label='Runge-Kutta 4th order (h/2)')
# plt.plot(t_ab, y_ab, '^-', label='Adams-Bashforth 3rd order (h)')
# plt.plot(t_ab_half, y_ab_half, 'd-', label='Adams-Bashforth 3rd order (h/2)')
# plt.plot(t_rk, y_rk_refined, 'x-', label='Refined Runge-Kutta')
# plt.plot(t_ab, y_ab_refined, 'v-', label='Refined Adams-Bashforth')
# plt.legend()
# plt.xlabel('t')
# plt.ylabel('y(t)')
# plt.title('Comparison of Numerical Methods and Refined Solutions')
# plt.grid(True)
# plt.show()
