import numpy as np
import matplotlib.pyplot as plt
import time
import random
from objective import objective_function_softImpute, objective_function_ALS, objective_function_softImputeALS
from algorithm import ALS, softImpute, softImputeALS, softImputeALS_second, softImputeALS

def generate_test_matrix(m, n, mask, rank):
    """
    Parameters:
    mask (float): Fraction of elements to be masked as missing (NaN). Range [0, 1].
    rank (int): The highest rank of the generated matrix.
    """
    # Generate a low rank matrix
    low_rank_matrix = np.dot(np.random.randn(m, rank), np.random.randn(rank, n))

    # Create a mask for missing values
    missing_mask = np.random.rand(m, n) < mask

    # Apply the mask to the matrix
    matrix_with_missing = low_rank_matrix.copy()
    matrix_with_missing[missing_mask] = np.nan

    return matrix_with_missing

# Parameters for the test matrix
# m, n = 300, 200  # Dimensions of the matrix
# mask = 0.7       # Portion of the matrix that is missing
# rank = 25        # Highest rank of the matrix
# lambda_ = 120    # Regularization parameter

# m, n = 10, 9  # Dimensions of the matrix
# mask = 0.2       # Portion of the matrix that is missing
# rank = 5         # Highest rank of the matrix
# lambda_ = 0.1    # Regularization parameter

m, n = 50, 45  # Dimensions of the matrix
mask = 0.6       # Portion of the matrix that is missing
rank = 10         # Highest rank of the matrix
lambda_ = 10    # Regularization parameter

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)


# Generate test matrix
test_matrix = generate_test_matrix(m, n, mask, rank)
place = np.isnan(test_matrix)  # Boolean mask for missing entries

# Initialize matrices for methods
M_initial = np.nan_to_num(test_matrix)  # Initial impute
A0, B0 = np.random.rand(m, rank), np.random.rand(n, rank)  # Initial guesses for ALS

# Time and performance tracking


U, V, D, timeSIALS, objSIALS = softImputeALS_second(test_matrix, lambda_, rank)
M_initial, timeSI, objSI = softImpute(test_matrix, M_initial, lambda_)
A, B, timeALS, objALS = ALS(test_matrix, A0, B0, lambda_)

# Creating the plot
plt.figure(figsize=(10, 6))

# Plotting each dataset
plt.plot(timeSIALS[15:], objSIALS[15:], label='softImputeALS', marker='o')
plt.plot(timeSI[15:], objSI[15:], label='softImpute', marker='s')
plt.plot(timeALS[15:], objALS[15:], label='ALS', marker='^')

# Log scale for y-axis
# plt.yscale('log')

# Adding labels and title
plt.xlabel('Time')
plt.ylabel('Obj')
plt.title('Obj vs Time')
plt.yscale('log')

# Adding a legend
plt.legend()

# Showing the plot
plt.show()

plt.savefig('./plot4.pdf')
