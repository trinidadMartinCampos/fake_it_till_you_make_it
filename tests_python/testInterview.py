# Project: python test interview
# Author: Trinidad Martín Campos
# Created: November 6, 2023
# Description: This script performs a simple test for python interview

def square_even(nums):
    """Given a list of numbers, write a Python function to square all even numbers in the list
        Params:
            nums: List of int,
                The input list
        Returns:
            A list with the square of even numbers.
        """
    return [number ** 2 for number in nums if number % 2 == 0]


def reverse_words(string):
    words = string.split(' ')  # 1. Split the string
    reversed_words = [word[::-1] for word in words]  # 2. Reverse every word
    return ' '.join(reversed_words)  # 3. Join the list of words


def filter_items_by_price(items, threshold):
    """Given a dictionary of items and their prices, write a function to return items with prices greater than a
    given value. Params: nums: List of int, The input list Returns: A list with the square of even numbers.
    Params:
            items: Dictionary,
            threshold: condition int
        Returns:
            A dictionary filtered by threshold.
    """
    new_dict = {k: v for k, v in items.items() if v > threshold}
    return new_dict


def print_matrix(rows, columns):
    """The goal is to generate a matrix (a list of lists) where the outer list contains n lists and each inner list
    contains m integers. These integers should be in ascending order starting from 1
    Params:
            row,columns: int,
                matrix dimensions
        Returns:
            A matrix"""
    count = 0
    matrix = []
    for i in range(rows):  # 1. Set up for loop
        row = []
        j = 0
        while j < m:  # 2. Set up while loop
            row.append(count)  # 3. Append inner row
            count += 1  # 4. Increase counter
            j += 1  # 5. Increase row index
        matrix.append(row)  # 6. Append matrix
    print(matrix)


print(square_even([1, 2, 3, 4, 5]))  # list comprehension
print(reverse_words("olleH dlroW"))  # String manipulation
items_dict = {
    'apple': 0.5,
    'banana': 0.25,
    'cherry': 1.2,
    'date': 2.5
}
print(filter_items_by_price(items_dict, 1))  # Dictionaries manipulation
n, m = 3, 4
print(print_matrix(n, m))  # nested loops

# --------------- OOP ------------------------------
import math


class Shape:
    def area(self):
        pass


class Rectangle(Shape):  # 1. Inherit class Rectangle from class Shape
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height  # 2. Define area method for Rectangle class


class Circle(Shape):  # 3. Inherit class Circle from class Shape
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return math.pi * self.radius ** 2  # 4. Define area method for Rectangle class


rect = Rectangle(10, 5)
print(rect.area())

circle = Circle(7)
print(circle.area())

# ------------- Numpy ---------------------------
import numpy as np

# 1. Use numpy to create an array of 10 zeros.
array_zeros = np.zeros(10, dtype=int)
print(array_zeros)

# 2. Now, create an array of 10 fives.
array_fives = np.zeros(10, dtype=int) + 5
print(array_fives)

# 3. Generate an array with numbers from 10 to 20.
range_array = np.arange(10, 21, dtype=int)
print(range_array)
# ---------------------------------
# Given the following array
arr = np.arange(0, 25).reshape(5, 5)

# 1. Extract the diagonal elements.
diagonal_elements = np.diag(arr)
print(diagonal_elements)

# 2. Retrieve the elements in the second row.
second_row = arr[1,]
print(second_row)

# 3. Extract the 2nd and 3rd rows and change their shape to (5, 2).
reshaped_rows = arr[[1, 2],].reshape(5, 2)
print(reshaped_rows)
# ---------------------------------
# Generate random array of 20 values
np.random.seed(1)
data = np.random.randn(20)  # randn for negative values
print(data)

# 1. Compute the mean of the data.
mean_value = np.mean(data)
print(mean_value)

# 2. Get the standard deviation of the data.
std_dev = np.std(data)
print(std_dev)

# 3. Find the value closest to the mean.
closest_to_mean = data[np.abs(data - mean_value).argmin()]  # argmin()
print(closest_to_mean)
# ---------------------------------
# Generate array with NaN
arr_with_nan = np.array([1, 2, np.nan, 4, 5])

# 1. Check for the presence of NaN values.
has_nan = np.isnan(arr_with_nan).any()
print(has_nan)

# 2. Replace NaN values with 0.
arr_without_nan = arr_with_nan
arr_without_nan[np.isnan(arr_with_nan)] = 0
# Convert all elements to integers
arr_without_nan = arr_without_nan.astype(int)  # convert to int
print(arr_without_nan)
# ---------------------------------
# Generate random matrix 5x5
np.random.seed(1)
arr = np.random.randint(1, 100, (5, 5))
print(arr, '\n')

# 1. Extract the central subarray
central_subarray = arr[1:4, 1:4]
print(central_subarray)

# 2. Flatten the extracted array
flat_subarray = central_subarray.flatten()

# 3. Sort the subarray
sorted_subarray = np.sort(flat_subarray)  # np.sort()!!

# 4. Integrate it back into the original array
arr[1:4, 1:4] = sorted_subarray.reshape(3, 3)
print(arr)

# ------------- Pandas ---------------------------
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar'],
    'B': [1, 2, 1, 4, 5, 2],
    'C': [2.5, 3.5, 4.5, 2.5, 3.5, 4.5]
})

# 1. Group data by a single column A.
grouped_A = df.groupby('A')

# 2. Sum all data grouped for column `A` using the built-in function.
sum_grouped_A = grouped_A.sum()
print(sum_grouped_A)
print('-' * 20)

# 3. Apply multiple aggregation functions simultaneously.
multi_aggregate = grouped_A.agg({'B': 'sum', 'C': 'mean'})
print(multi_aggregate)
print('-' * 20)

# 4. Group by multiple columns and sum.
grouped_A_B = df.groupby(['A', 'B']).sum()
print(grouped_A_B)
# --------------------------------- indexing
# Sample DataFrame
df = pd.DataFrame({
    'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'Temperature': [22, 24, 23],
    'Humidity': [56, 58, 57]
})

# 1. Set a column as the index of a DataFrame.
indexed_df = df.set_index('Date')
print(indexed_df)
print('-' * 40)

# 2. Reset the index of a DataFrame.
reset_df = indexed_df.reset_index()
print(reset_df)
print('-' * 40)

# 3. Create a DataFrame with a MultiIndex.
arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
multi_indexed_df = pd.DataFrame({
    'Value': [10, 20, 30, 40]
}, index=pd.MultiIndex.from_arrays(arrays, names=('Letter', 'Number')))
print(multi_indexed_df)
print('-' * 20)

# 4. Access data from a MultiIndexed DataFrame.
retrieved_data = multi_indexed_df.loc['A', 1]
print(retrieved_data)

# --------------------------------- altering DataFrame
# Sample DataFrame
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'San Francisco', 'Los Angeles']
})

# 1. Add a new column to a DataFrame.
df['Occupation'] = ['Engineer', 'Doctor', 'Artist']
print(df)
print('-' * 40)

# 2. Rename columns in a DataFrame.
renamed_df = df.rename(columns={'Name': 'Full Name', 'Age': 'Age (years)'})
print(renamed_df)
print('-' * 40)

# 3. Drop a column from a DataFrame.
reduced_df = renamed_df.drop('City', axis=1)
print(reduced_df)
print('-' * 40)

# 4. Sort a DataFrame based on a specific column.
sorted_df = reduced_df.sort_values(by='Age (years)', ascending=True)
print(sorted_df)

# ---------------------------------  Iterating Over Data
# Sample DataFrame
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'San Francisco', 'Los Angeles']
})

# 1. Iterate over rows of a DataFrame.
print('--- Task 1 ---')
for index, row in df.iterrows():  # iterrrows() !!
    print(row)
    print('-' * 20)

# 2. Iterate over columns of a DataFrame.
print('\n--- Task 2 ---')
for column in df:  # iteration by columns !!
    print(column)

# 3. Apply a custom function to each cell in a DataFrame column.
print('\n--- Task 3 ---')


def add_suffix(cell):
    return str(cell) + "_suffix"


df['Name'] = df['Name'].apply(add_suffix)  # apply()
print(df)

# 4. Replace values in the City column according replacement_dict.
print('\n--- Task 4 ---')
replacement_dict = {
    'New York': 'NY',
    'San Francisco': 'SF',
    'Los Angeles': 'LA'
}

df['City'] = df['City'].map(replacement_dict)  # map()
print(df)

# --------------------------------- Matplot
import matplotlib.pyplot as plt

# Fix seed
np.random.seed(1)

# 1. Plot a simple line graph.
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.show()

# 2. Create a scatter plot.
x_scatter = np.random.rand(50)
y_scatter = np.random.rand(50)
plt.scatter(x_scatter, y_scatter)
plt.show()

# 3. Generate a histogram.
data = np.random.randn(1000)
plt.hist(data, bins=30)
plt.show()
# ---------------------------------
# 1. Plot a bar chart with error bars.
labels = ['A', 'B', 'C', 'D', 'E']
means = [20, 35, 30, 35, 27]
errors = [2, 3, 4, 1, 2]
plt.bar(labels, means, yerr=errors, align='center', alpha=0.7, ecolor='black', capsize=10)
plt.show()

# 2. Generate a stacked bar plot.
labels = ['A', 'B', 'C', 'D']
y1 = [20, 34, 30, 35]
y2 = [25, 32, 34, 20]
plt.bar(labels, y1, label='y1')
plt.bar(labels, y2, bottom=y1, label='y2')
plt.legend()
plt.show()

# 3. Construct a pie chart.
labels = 'Frogs', 'Hogs', 'Dogs', 'Ducks'
sizes = [15, 30, 45, 10]
explode = (0, 0.1, 0, 0)
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%')
plt.axis('equal')
plt.show()
# ---------------------------------
# 1. Create a 2x2 grid of subplots.
fig, axs = plt.subplots(2, 2)

# 2. Plot a line graph on the top-left subplot.
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
axs[0,0].plot(x, y1)
axs[0,0].set_title('Line Graph')

# 3. Plot a scatter plot on the bottom-right subplot.
y2 = np.random.rand(100)
axs[1,1].scatter(x, y2)
axs[1,1].set_title('Scatter Plot')

plt.tight_layout()
plt.show()

# --------------------------------- grid
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 1. Set the title, x-label, and y-label with custom font sizes.
plt.title('Customized Sinusoidal Curve', fontsize=16)
plt.xlabel('X-axis', fontsize=14)
plt.ylabel('Y-axis', fontsize=14)

# 2. Change the line style, width, and color.
plt.plot(x, y, linestyle='--', linewidth=2, color='purple')

# 3. Customize the x and y axis ticks and their labels.
plt.xticks(np.arange(0, 11, 2), fontsize=12)  # X axis
plt.yticks(np.linspace(-1, 1, 5), fontsize=12)  # Y axis

# 4. Add a grid with a specific style.
plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
plt.show()

# --------------------------------- annotation
x = [1, 2, 3, 4, 5]
y = [2, 5, 1, 3, 4]

fig, ax = plt.subplots()

# Plot a simple line.
ax.plot(x, y)

# Display a title for the plot.
ax.set_title('Simple Line Plot with Annotations')

# 1. Label both the x and y axes.
ax.set_xlabel('X-Axis')
ax.set_ylabel('Y-Axis')

# 2. Annotate the maximum and minimum points.
max_val = max(y)
min_val = min(y)
max_idx = y.index(max_val)
min_idx = y.index(min_val)

ax.annotate('Max: {}'.format(max_val), xy=(x[max_idx], max_val), xytext=(x[max_idx]+0.5, max_val), arrowprops=dict(facecolor='black', arrowstyle='->'))
ax.annotate('Min: {}'.format(min_val), xy=(x[min_idx], min_val), xytext=(x[min_idx]-0.75, min_val), arrowprops=dict(facecolor='black', arrowstyle='->'))

# 3. Place a textual note.
ax.text(2.45, 2, 'The greatest decline', rotation=-72)
plt.show()


# --------------------------------- Seaborn!!!
import seaborn as sns

# Sample data
x = [2, 5, 1, 3, 4, 5, 6, 7, 8, 2, 3, 5, 4, 6]
y = [5, 3, 6, 5, 4, 6, 3, 2, 1, 7, 8, 3, 2, 1]

# 1. Plotting univariate distribution
sns.displot(x, kde=True)
plt.title('Univariate Distribution')
plt.show()

# 2. Plotting bivariate distribution
sns.jointplot(x=x, y=y, kind='kde')
plt.show()

# ---------------------------------
# Sample data
data = sns.load_dataset("tips")
days = data['day']
total_bill = data['total_bill']
time = data['time']

# 1. Creating a box plot
sns.boxplot(x=days, y=total_bill)
plt.title('Box plot of Total Bill across Days')
plt.show()

# 2. Displaying distribution using a swarm plot
sns.swarmplot(x=days, y=total_bill)
plt.title('Swarm plot of Total Bill across Days')
plt.show()

# 3. Count of observations using a bar plot
sns.countplot(x=days)
plt.title('Bar plot of Days')
plt.show()
# ---------------------------------

# Sample data
data = sns.load_dataset('flights')
months = data['month']
passengers = data['passengers']
year = data['year']

# 1. Tracking changes using a line plot
sns.lineplot(x=months, y=passengers)
plt.title('Monthly Passenger Counts Over Time')
plt.show()

# 2. Scatter plot with color semantics
sns.scatterplot(x=year, y=passengers, hue=months)
plt.title('Yearly Passenger Count Differentiated by Month')
plt.show()

# ---------------------------------
# Sample data
data = sns.load_dataset('flights')
months = data['month']
passengers = data['passengers']
year = data['year']

# 1. Regression line for two variables
sns.regplot(x=year, y=passengers)
plt.title('Regression Line Showing Relationship Between Year and Passenger Count')
plt.show()

# 2. Differentiating regression line with hue
sns.lmplot(x='year', y='passengers', hue='month', data=data)
plt.title('Regression Line Differentiated by Month')
plt.show()
# --------------------------------- heatmap
# Sample data
data = sns.load_dataset('flights').pivot(index='month', columns='year', values='passengers')

# 1. Heatmap of a correlation matrix
sns.heatmap(data.corr())
plt.title('Correlation Matrix Heatmap')
plt.show()

# 2. Annotated heatmap
sns.heatmap(data.corr(), annot=True)
plt.title('Annotated Correlation Matrix Heatmap')
plt.show()
# --------------------------------- STATS!!

from scipy.stats import shapiro, probplot, norm

# Load the dataset
data = sns.load_dataset('tips')

# 1. Compute descriptive statistics for the 'total_bill' column
print(data['total_bill'].describe())

# 2. Assess the normality of the data using a Q-Q plot
probplot(data['total_bill'], plot=plt)
plt.title('Q-Q Plot of Total Bill')
plt.show()

# 3. Assess the normality of the data using the Shapiro-Wilk test
stat, p = shapiro(data['total_bill'])
alpha = 0.05
if p > alpha:
    print(f"The data appears to be normally distributed (p={p:.2f}).")
else:
    print(f"The data does not appear to be normally distributed (p={p:.2f}).")

# 4. Compute the probability that a randomly chosen bill is more than $20
prob = len(data[data['total_bill'] > 20 ]) / len(data)
print(f"The probability that a randomly chosen bill is more than $20 is {prob:.2%}.")

# --------------------------------- Bayes' Theorem
# Given probabilities
p_disease = 0.01
p_positive_given_disease = 0.99
p_negative_given_no_disease = 0.98

# Calculate P(Positive|No Disease)
p_positive_given_no_disease = 1 - p_negative_given_no_disease

# Compute P(Positive)
p_positive = p_positive_given_disease * p_disease + p_positive_given_no_disease * (1 - p_disease)

# Use Bayes' theorem to compute P(Disease|Positive)
p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive

print(f"The probability that a person who tests positive actually has the disease is: {p_disease_given_positive:.2f}")

# --------------------------------- Mann Whitneyu
from scipy.stats import mannwhitneyu, chi2_contingency
# Load the dataset
data = sns.load_dataset('tips')

# 1. Test whether there is a significant difference in 'total_bill' between smokers and non-smokers using Mann–Whitney test.
smokers = data[data['smoker'] == 'Yes']['total_bill']
non_smokers = data[data['smoker'] == 'No']['total_bill']
u_val, p_val = mannwhitneyu(smokers, non_smokers)
alpha = 0.1
if p_val < alpha:
    print(f"There is a significant difference in 'total_bill' between smokers and non-smokers (p={p_val:.2f}).")
else:
    print(f"There is no significant difference in 'total_bill' between smokers and non-smokers (p={p_val:.2f}).")

# 2.  Test whether there is a relationship between 'sex' and 'smoker' using a chi-squared test.
contingency_table = pd.crosstab(data['sex'], data['smoker'])
chi2, p_val2, _, _ = chi2_contingency(contingency_table)
alpha = 0.1
if p_val2 < alpha:
    print(f"There is a significant relationship between 'sex' and 'smoker' (p={p_val2:.2f}).")
else:
    print(f"There is no significant relationship between 'sex' and 'smoker' (p={p_val2:.2f}).")

from scipy.stats import sem, t

# --------------------------------- Confidence intervals
# Generate a random sample data
np.random.seed(42)
sample_data = pd.Series(np.random.randn(100) * 5 + 50)  # Normally distributed data with mean 50 and standard deviation 5

# 1. Compute the sample mean and standard error
sample_mean = sample_data.mean()
standard_error = sem(sample_data)

# 2. Set the confidence level
confidence_level = 0.95
degrees_freedom = len(sample_data) - 1
confidence_value = t.ppf((1 + confidence_level) / 2., degrees_freedom)

# Compute the confidence interval
ci_lower = sample_mean - confidence_value * standard_error
ci_upper = sample_mean + confidence_value * standard_error

print(f"The {confidence_level*100}% confidence interval for the sample mean is: ({ci_lower:.2f}, {ci_upper:.2f})")

# --------------------------------- Correlation
# Load the dataset
data = sns.load_dataset('tips')

# 1. Compute the Pearson correlation coefficient between 'total_bill' and 'tip'
correlation_coefficient = data['total_bill'].corr(data['tip'])
print(f"The Pearson correlation coefficient between 'total_bill' and 'tip' is: {correlation_coefficient:.2f}")

# 2. Visualize the relationship between 'total_bill' and 'tip' using a linear regression plot
sns.lmplot(x='total_bill', y='tip', data=data)
plt.title("Relationship between 'total_bill' and 'tip'")
plt.show()

# Calculates Cramér's V measure between categorical variables
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

# 3. Generate a matrix of correlations (Cramér's V) between categorical variables
categorical_cols = ['sex', 'smoker', 'day', 'time']
correlation_matrix = pd.DataFrame(index=categorical_cols, columns=categorical_cols)  # Label the X and Y axes with the names of the selected columns
for col1 in categorical_cols:
    for col2 in categorical_cols:
        correlation_matrix.loc[col1,col2] = cramers_v(data[col1], data[col2])

sns.heatmap(correlation_matrix.astype(float), annot=True)
plt.show()