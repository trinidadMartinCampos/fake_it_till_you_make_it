# Project: python test numpy
# Author: Trinidad Martín Campos
# Created: November 6, 2023
# Description: This script performs a simple test for python pandas

# !/usr/bin/env python
# coding: utf-8

# ## Importing NumPy and Creating NumPy Arrays

import numpy as np
import matplotlib.pyplot as plt

# create a 1D NumPy array
a1 = np.array([1, 2, 3, 4, 5])

# create a 2D NumPy array
a2 = np.array([[1, 2, 3], [4, 5, 6]])

# create a NumPy array with all elements set to 0
a4 = np.zeros(shape=(3, 3))

# ## Manipulating and Indexing NumPy Arrays

arr = np.random.randint(0, 11, size=(4, 4))

second_row = arr[1, :]

second_col = arr[:, 1]

arr[:, 1] = 1

# ## Basic Statistical Operations on NumPy Arrays

# Create a NumPy array with random values
arr = np.random.rand(10)

# # Calculate the mean of the array
mean = np.mean(arr)

# # Calculate the median of the array
median = np.median(arr)

# # Calculate the standard deviation of the array
std = np.std(arr)

# # Calculate the variance of the array
var = np.var(arr)

# ## Linear Algebra Operations with NumPy


# Create two NumPy arrays

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# # Calculate the dot product of the arrays
dot_product = np.dot(a, b)

# # Transpose the first array
a_transposed = np.transpose(a)

# # Calculate the inverse of the second array
b_inverse = np.linalg.inv(b)

# ## Reshaping and Stacking NumPy Arrays


# Create a NumPy array
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr_1 = np.array([[7, 8, 9], [10, 11, 12]])
arr_2 = np.array([[13, 14, 15], [16, 17, 18]])

# # Reshape the array to a new shape
reshaped_arr = arr.reshape(3, 2)

# Resize the array to a new shape, potentially adding or removing elements
resized_arr = np.resize(arr, (4, 3))

# Stack the arrays vertically (row-wise) to create a single array
vertical_stack = np.vstack((arr, arr_1, arr_2))

# Stack the arrays horizontally (column-wise) to create a single array
horizontal_stack = np.hstack((arr, arr_1, arr_2))

# ## Saving and Loading NumPy Arrays to/from Files

# Create a NumPy array
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Save the array to a binary file in NumPy's .npy format
np.save('array.npy', arr)

# Save the array to a text file
np.savetxt('array.txt', arr)

# Load the array from the binary file
loaded_arr = np.load('array.npy')

# ---------------MATRICES-----------------------
# Define the matrices
A = np.array([[1, 2], [3, 4]])  # Matrix A with shape (2, 2)
B = np.array([[5, 6, -1, 0], [7, 8, 3, 1]])  # Matrix B with shape (2, 4)

# Calculate the matrix multiplication using dot product
rows = A.shape[0]  # Number of rows in resulting matrix
cols = B.shape[1]  # Number of cols in resulting matrix
C = np.zeros((rows, cols))  # Initialize the result matrix C

for i in range(rows):
    for j in range(cols):
        C[i, j] = np.dot(A[i, :], B[:, j])  # Dot product of row A[i] and column B[j]

# Print the result
print(C)
print(f'Shape of resulting matrix is: {C.shape}')

# EASIER
C = A.dot(B)
print(C)
print(f'Shape of resulting matrix is: {C.shape}')


# ------------------- Matplot ----------------------------


# Function to construct a rotation matrix
def rotation_matrix(angle_param):
    cos_theta = np.cos(angle_param)
    sin_theta = np.sin(angle_param)
    return np.array([[cos_theta, -sin_theta],
                     [sin_theta, cos_theta]])


# Function to construct a scaling matrix
def scaling_matrix(scale_x_param, scale_y_param):
    return np.array([[scale_x_param, 0],
                     [0, scale_y_param]])


# Function to plot a square
def plot_square(square_param, color):
    square_param = np.vstack((square_param, square_param[0]))  # Close the square by repeating the first point
    plt.plot(square[:, 0], square[:, 1], color=color)


# Define a square using its corners as a matrix
square = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
    [0, 0]  # Close the square
]).T  # Transpose to have x and y as separate rows

# Plot the original square
plot_square(square.T, 'blue')

# Define the rotation angle and scaling factors
angle = np.pi / 3
scale_x, scale_y = 2, 0.5

# Construct the rotation and scaling matrices
rotation = rotation_matrix(angle)
scaling = scaling_matrix(scale_x, scale_y)

# Apply the transformations to the square
rotated_square = rotation @ square
transformed_square = scaling @ rotated_square

# Plot the transformed square
plot_square(transformed_square.T, 'red')

# Set the aspect ratio to equal and adjust the axis limits
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(-3, 3)
plt.ylim(-3, 3)

# Display the plot
plt.show()

# ------------------ SLE ---------------------

# Define the coefficient matrix A and the constant vector B
A = np.array([[2, 3, -1], [4, -1, 5], [10, 2, 6]])
y = np.array([10, 4, 1])

# Solve the SLE using np.linalg.solve()
x = np.linalg.solve(A, y)
print(f'Solution using np.linalg.solve(): {x}')

# Calculate the inverse of matrix A
A_inv = np.linalg.inv(A)

# Solve the SLE using the inverse matrix method
x_inv = A_inv @ y
print(f'Solution using inverse matrix method: {x_inv}')

# ------------- Derivatives -----------------------
import sympy as sp

# Define the time variable
t = sp.symbols('t')

# Define the position function
s = 5 * t ** 3 - 2 * t ** 2 + 3 * t + 1

# Calculate the velocity and acceleration
v = sp.diff(s, t)  # Velocity
a = sp.diff(s, t, 2)  # Acceleration

# Evaluate the velocity at t = 5
point = {t: 5}
v_5 = v.subs(point)

# Print the velocity and acceleration
print(f'Velocity: {v}')
print(f'Acceleration: {a}')

# Print the velocity and acceleration at t = 5
print(f'Velocity at t = 5: {v_5}')

# Example 2
# Define the symbols
t = sp.symbols('t')

# Define the height function h(t)
v = 15.0  # Initial velocity in m/s
g = 9.81  # Acceleration due to gravity in m/s^2
h = v * t - (1 / 2) * g * t ** 2

# Calculate the first and second derivatives of h(t)
h_prime = sp.diff(h, t)
h_double_prime = sp.diff(h_prime, t)

# Find the critical points by solving h'(t) = 0
critical_points = sp.solve(h_prime, t)

# Identify the maximum height by evaluating h''(t) at each critical point
"""Let's consider a physics-related optimization problem where we need to find the maximum height reached by an 
object thrown vertically upward with a given initial velocity. We have the following equation: h = v * t - 0.5 * g * 
t**2 that describes the motion of an object. Our task is to find the time t when the object reaches its maximum 
height and then find the maximum height h_max"""
maximum_height = -1
for t_critical in critical_points:
    h_double_prime_value = h_double_prime.subs({t: t_critical})
    if h_double_prime_value < 0:
        maximum_height = h.subs({t: t_critical})

    # Print the result
    if maximum_height != -1:
        print(f'The object reaches its maximum height of {maximum_height:.2f} meters at time t = '
              f'{t_critical:.2f} seconds.')
    else:
        print('There is no maximum height (object doesnt\'t reach the ground).')

# ------------------------ ML - Gradient Descent ----------------------
"""The most commonly used loss function in linear regression is the Mean Squared Error (MSE) loss function. This 
function is the squared Euclidean distance between the variable's real value and the value we obtained using linear 
regression approximation. Since this is a function of several variables, we can optimize it using gradient descent.
Your task is to use the optimization method to find the best parameters of the linear regression function"""

from scipy.optimize import minimize

# Generate some sample data
x = np.linspace(0, 10, 100)
y_true = 2 * x + 3 + np.random.normal(0, 1, 100)


# Define the MSE loss function
def mse_loss(params):
    m, b = params
    y_pred = m * x + b
    return np.mean((y_true - y_pred) ** 2)


# Initial guess for the parameters m and b
initial_params = [1, 1]

# Optimize the loss function using scipy.optimize.minimize
result = minimize(mse_loss, initial_params)

# Extract the optimized parameters
m_optimized, b_optimized = result.x

print(f'Optimized slope (m): {m_optimized:.4f}')
print(f'Optimized y-intercept (b): {b_optimized:.4f}')

# Plot the original data points
plt.scatter(x, y_true, label='Data Points')

# Plot the optimized linear regression line
plt.plot(x, m_optimized * x + b_optimized, color='red', label='Linear Regression Line')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Linear Regression with MSE Loss')
plt.grid(True)
plt.show()
