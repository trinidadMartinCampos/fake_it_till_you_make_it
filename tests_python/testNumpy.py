# Project: python test 3
# Author: Trinidad Martín Campos
# Created: November 6, 2023
# Description: This script performs a simple test for python pandas

# !/usr/bin/env python
# coding: utf-8

# ## Importing NumPy and Creating NumPy Arrays

import numpy as np

# create a 1D NumPy array
a1 = np.array([1, 2, 3, 4, 5])

# create a 2D NumPy array
a2 = np.array([[1,2,3],[4,5,6]])

# create a NumPy array with all elements set to 0
a4 = np.zeros(shape=(3, 3))


# ## Manipulating and Indexing NumPy Arrays

arr = np.random.randint(0, 11, size=(4,4))

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





