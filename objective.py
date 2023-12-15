import numpy as np

def objective_function_softImpute(X, M, lambda_):
    """
    Parameters:
    place (np.array): Boolean mask array where True indicates missing entries.
    lambda_ (float): Regularization parameter.

    """
    # Perform Singular Value Decomposition
    U, Sigma, Vt = np.linalg.svd(M)
    # Sum of the singular values
    sum_singular_values = np.sum(Sigma)
    squared_error = np.nansum((X - M)**2)  # Sum of squared differences only over known values
    return 0.5 * squared_error + lambda_ * sum_singular_values

def objective_function_softImputeALS(X, U, V, D, lambda_):
    """
    Parameters:
    place (np.array): Boolean mask array where True indicates missing entries.
    lambda_ (float): Regularization parameter.

    """
    # Perform Singular Value Decomposition
    M = U @ D @ (V.T)
    # Sum of the singular values
    # U, Sigma, Vt = np.linalg.svd(M)
    sum_singular_values = np.sum(D)
    squared_error = np.nansum((X - M)**2)  # Sum of squared differences only over known values
    return 0.5 * squared_error + lambda_ * sum_singular_values

def objective_function_ALS(X, A, B, lambda_):
    """
    place (np.array): Boolean mask array where True indicates missing entries.
    lambda_ (float): Regularization parameter.

    Returns:
    float: The value of the objective function.
    """
    M = A @ B.T
    U, Sigma, Vt = np.linalg.svd(M)
    sum_singular_values = np.sum(Sigma)
    squared_error = np.nansum((X - M)**2)  # Sum of squared differences only over known values
    return 0.5 * squared_error + lambda_ * sum_singular_values