import numpy as np
import time
from objective import objective_function_softImpute, objective_function_ALS, objective_function_softImputeALS


def ALS(X_original, A0, B0, lambda_, max_iter=500, tol=1e-5):
    '''
    Solves a ridge regression problem using a closed form solution:

        w_i = (X'X + lambda * I)^-1 X'y
    
    for all i in the target matrix.
    '''
    start_time = time.time()
    localtime = []
    objective = []
    A = A0.copy().T
    B = B0.copy().T
    for iteration in range(max_iter):
        X_A = X_original.T
        # Update A
        for i in range(A.shape[1]):
            # Handling nan values instead of nonzeros
            valid_entries = ~np.isnan(X_A[:, i])
            y = X_A[valid_entries, i]
            X = B[:, valid_entries].T

            A[:, i] = np.squeeze(np.linalg.inv(X.T.dot(X) + lambda_ * np.eye(X.shape[1])).dot(X.T.dot(y)))

        for j in range(B.shape[1]):
            # Handling nan values instead of nonzeros
            valid_entries = ~np.isnan(X_original[:, j])
            y = X_original[valid_entries, j]
            X = A[:, valid_entries].T

            B[:, j] = np.squeeze(np.linalg.inv(X.T.dot(X) + lambda_ * np.eye(X.shape[1])).dot(X.T.dot(y)))

        if iteration % 10 == 9:
            localtime.append(time.time() - start_time)
            objective.append(objective_function_ALS(X_original, A.T, B.T, lambda_))

        if np.linalg.norm(X_original - A.T @ B, 'fro') / np.linalg.norm(X_original, 'fro') < tol:
            break

    return A.T, B.T, localtime, objective

def softImpute(X, M, lambda_, max_iter=500, tol=1e-5):
    start_time = time.time()
    localtime = []
    objective = []

    for iteration in range(max_iter):
        # Step 1: Replace missing entries in X with entries from M
        omega = ~np.isnan(X)  # Indicator matrix for observed values
        X_imputed = np.where(omega, X, M)

        # Step 2: Update M by computing the soft-thresholded SVD of X
        U, D, Vt = np.linalg.svd(X_imputed, full_matrices=False)
        D_soft_thresholded = np.maximum(D - lambda_, 0)  # Soft-thresholding
        M_updated = np.dot(U, np.dot(np.diag(D_soft_thresholded), Vt))

        M = M_updated

        if iteration % 10 == 9:
            localtime.append(time.time() - start_time)
            objective.append(objective_function_softImpute(X, M_updated, lambda_))

        if np.linalg.norm(X_imputed - M_updated, 'fro') / np.linalg.norm(X_imputed, 'fro') < tol:
            print(np.linalg.norm(X_imputed - M_updated, 'fro') / np.linalg.norm(X_imputed, 'fro'))
            break

    return M_updated, localtime, objective

import numpy as np

def softImputeALS(X, lambda_, r, max_iter=500, tol=1e-5):
    start_time = time.time()
    localtime = []
    objective = []
    m, n = X.shape
    U = np.random.randn(m, r)
    U, _ = np.linalg.qr(U)  # Orthonormalize U
    D = np.eye(r)
    V = np.zeros((n, r))
    A = U @ D
    B = V @ D
    omega = ~np.isnan(X)  # Indicator matrix for observed values

    for iteration in range(max_iter):
        # Update B
        X_star = np.where(omega, X, A @ B.T)
        B_new = (np.linalg.inv(D**2 + lambda_ * np.eye(r)) @ D @ U.T @ X_star).T
        V, D_diag, U_tilde = np.linalg.svd(B_new @ D, full_matrices=False)
        D = np.diag(np.sqrt(D_diag))
        B = V @ D

        # Update A
        X_star = np.where(omega, X, A @ B.T)
        A_new = (np.linalg.inv(D**2 + lambda_ * np.eye(r)) @ D @ V.T @ X_star.T).T
        U, D_diag, V_tilde = np.linalg.svd(A_new @ D, full_matrices=False)
        D = np.diag(np.sqrt(D_diag))
        A = U @ D

        # Check for convergence
        if np.linalg.norm(X_star - A @ B.T, 'fro') / np.linalg.norm(X_star, 'fro') < tol:
            break

        if iteration % 10 == 9:
            localtime.append(time.time() - start_time)
            M_test = X_star @ V
            U_test, D_sigma_test, R_test = np.linalg.svd(M_test, full_matrices=False)
            D_sigma_lambda = np.diag(np.maximum(D_sigma_test - lambda_, 0))
            objective.append(objective_function_softImputeALS(X, U_test, V@(R_test.T), D_sigma_lambda, lambda_))

    M = X_star @ V
    U, D_sigma, R = np.linalg.svd(M, full_matrices=False)
    D_sigma_lambda = np.diag(np.maximum(D_sigma - lambda_, 0))
    return U, V@(R.T), D_sigma_lambda, localtime, objective


def softImputeALS_second(X, lambda_, r, max_iter=500, tol=1e-5):
    m, n = X.shape
    omega = ~np.isnan(X)  # Indicator matrix for observed values
    X_star = np.where(omega, X, 0)
    start_time = time.time()
    localtime = []
    objective = []

    # Initialization
    V = np.zeros((n, r))
    U = np.random.normal(0.0, 1.0, (m, r))
    U, _ = np.linalg.qr(U)
    D = np.ones((r, 1))

    ratio = 1.0
    iteration = 0
    for iteration in range(max_iter):

        # Update B (V in your code)
        B = U.T@X_star
        B *= D / (D + lambda_)
        V, D_diag, Uh = np.linalg.svd(B.T, full_matrices=False)
        D = D_diag.reshape((r, 1))
        U = U @ Uh
        M = U @ np.diag(D_diag) @ V.T

        X_star = np.where(omega, X_star, M)
        # Update A (U in your code)
        A = X_star.dot(V).T
        A *= D / (D + lambda_)
        U, D_diag, Vh = np.linalg.svd(A.T, full_matrices=False)
        D = D_diag.reshape((r, 1))
        V = V @ Vh
        M = U @ np.diag(D_diag) @ V.T

        X_star = np.where(omega, X_star, M)

        if iteration % 10 == 9:
            localtime.append(time.time() - start_time)
            objective.append(objective_function_softImpute(X, M, lambda_))

    return U[:, :r], V[:, :r], np.diag(D_diag[:r]), localtime, objective


# Example usage
# X is a matrix with missing values (NaN)
# clf = SoftImpute(J=2, lambda_=0.1)
# fit = clf.fit(X)
# X_imp = clf.predict(X)


# Usage example
# X is the input data matrix with missing values filled with zeros or mean
# lambda_ is the regularization parameter
# r is the rank restriction
# Replace X, lambda_, and r with actual values
# U, D_sigma_lambda = softImpute_ALS(X, lambda_, r)
