# utils/linear_algebra_utils.py

import numpy as np
import pandas as pd
from numpy.linalg import (eig, 
                          svd, 
                          lstsq, 
                          inv, 
                          det, 
                          matrix_rank)

def compute_eigen(matrix):
    """Compute eigenvalues and eigenvectors of a matrix."""
    values, vectors = eig(matrix)
    return {"eigenvalues": values, "eigenvectors": vectors}

def compute_svd(matrix):
    """Compute singular value decomposition."""
    U, S, Vt = svd(matrix)
    return {"U": U, "S": S, "Vt": Vt}

def solve_least_squares(A, b):
    """Solve Ax = b using least squares."""
    x, residuals, rank, s = lstsq(A, b, rcond=None)
    return {"solution": x, "residuals": residuals, "rank": rank, "singular_values": s}

def compute_determinant(matrix):
    """Compute determinant of a square matrix."""
    return det(matrix)

def compute_inverse(matrix):
    """Compute matrix inverse if invertible."""
    return inv(matrix)

def matrix_summary_df(matrix):
    """
    Return a DataFrame summary of matrix properties:
    Shape, Rank, Determinant, and Condition Number.
    """
    data = {
        "Shape": [matrix.shape],
        "Rank": [matrix_rank(matrix)],
        "Determinant": [det(matrix) if matrix.shape[0] == matrix.shape[1] else None],
        "Condition Number": [np.linalg.cond(matrix)]
    }
    return pd.DataFrame(data)


def generate_matrix(matrix_type: str = "Random Symmetric", dim: int = 4) -> pd.DataFrame:
    """
    Generates a matrix based on the selected type.

    Parameters
    ----------
    matrix_type : str
        Type of matrix ("Random Symmetric" or "Tall Matrix")
    dim : int
        Matrix dimension (rows = dim, cols = dim or smaller)

    Returns
    -------
    pd.DataFrame
        Generated matrix as a DataFrame for display.
    """
    if matrix_type == "Random Symmetric":
        # Create a random symmetric matrix
        A = np.random.randn(dim, dim)
        A = (A + A.T) / 2  # make symmetric
    elif matrix_type == "Tall Matrix":
        # Create a rectangular tall matrix (more rows than cols)
        rows = max(dim + 2, dim * 2)  # ensure it's tall
        A = np.random.randn(rows, dim)
    else:
        raise ValueError("Unsupported matrix type selected.")

    return pd.DataFrame(A, columns=[f"x{i+1}" for i in range(A.shape[1])])


def least_squares_solution(A, b):
    """Solve Ax = b using least squares."""
    x, residuals, rank, s = lstsq(A, b, rcond=None)
    return x, residuals


def compute_eigendecomposition(matrix):
    """Compute eigenvalues and eigenvectors of a matrix, ensuring float type."""
    values, vectors = eig(matrix)
    values = np.array(values, dtype=float)
    vectors = np.array(vectors, dtype=float)
    return values, vectors


def compute_svd_adv(matrix):
    """Compute singular value decomposition with enforced numeric output."""
    # Ensure matrix is numeric
    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.to_numpy(dtype=float)
    else:
        matrix = np.asarray(matrix, dtype=float)
    
    # Perform SVD
    U, S, Vt = svd(matrix)
    
    return U, S, Vt
