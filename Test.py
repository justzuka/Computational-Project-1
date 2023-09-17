# as we increase points error gets smaller, float32 and float64 has similar results not much different in this case

import numpy as np
import matplotlib.pyplot as plt

# Define the function to plot
def f(x):
    return np.sin(x)

# Define the derivative of the function
def df(x):
    return np.cos(x)

# Define the number of points to use for the derivative approximation
n = 10

# Generate n points to use for the derivative approximation with float32 data type
x_float32 = np.linspace(0, 2*np.pi, n, dtype=np.float32)
y_float32 = f(x_float32)
dx_float32 = x_float32[1] - x_float32[0]
dy_float32 = np.zeros_like(y_float32, dtype=np.float32)
dy_float32[0] = (y_float32[1] - y_float32[0]) / dx_float32
dy_float32[1:-1] = (y_float32[2:] - y_float32[:-2]) / (2*dx_float32)
dy_float32[-1] = (y_float32[-1] - y_float32[-2]) / dx_float32

# Generate n points to use for the derivative approximation with float64 data type
x_float64 = np.linspace(0, 2*np.pi, n, dtype=np.float64)
y_float64 = f(x_float64)
dx_float64 = x_float64[1] - x_float64[0]
dy_float64 = np.zeros_like(y_float64, dtype=np.float64)
dy_float64[0] = (y_float64[1] - y_float64[0]) / dx_float64
dy_float64[1:-1] = (y_float64[2:] - y_float64[:-2]) / (2*dx_float64)
dy_float64[-1] = (y_float64[-1] - y_float64[-2]) / dx_float64

# Plot the function and its derivative for float32 data type
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
ax1.plot(x_float32, f(x_float32), label="f(x)")
ax1.plot(x_float32, df(x_float32), label="f'(x)")
ax1.set_title("Function and its derivative (float32)")
ax1.legend()

# Plot the derivative approximation and the absolute error for float32 data type
ax2.plot(x_float32, df(x_float32), label="f'(x)")
ax2.plot(x_float32, dy_float32, label="Approximation")
ax2.plot(x_float32, np.abs(df(x_float32) - dy_float32), label="Absolute error")
ax2.set_title(f"Derivative approximation with {n} points (float32)")
ax2.legend()



# Plot the function and its derivative for float64 data type
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
ax1.plot(x_float64, f(x_float64), label="f(x)")
ax1.plot(x_float64, df(x_float64), label="f'(x)")
ax1.set_title("Function and its derivative (float64)")
ax1.legend()

# Plot the derivative approximation and the absolute error for float64 data type
ax2.plot(x_float64, df(x_float64), label="f'(x)")
ax2.plot(x_float64, dy_float64, label="Approximation")
ax2.plot(x_float64, np.abs(df(x_float64) - dy_float64), label="Absolute error")
ax2.set_title(f"Derivative approximation with {n} points (float64)")
ax2.legend()

plt.show()
