# utils/linear_algebra_utils.py

import numpy as np
import pandas as pd
from numpy.linalg import (eig, 
                          svd, 
                          lstsq, 
                          inv, 
                          det, 
                          matrix_rank)

def compute_eigen(
    matrix: "np.ndarray",
) -> dict[str, "np.ndarray"]:
    """
    Compute eigenvalues and eigenvectors of a square matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Square numeric matrix.

    Returns
    -------
    dict
        {'eigenvalues': np.ndarray, 'eigenvectors': np.ndarray}
        Eigenvectors are column vectors in the returned matrix.

    Notes
    -----
    Uses numpy.linalg.eig which may return complex values for asymmetric matrices.
    For symmetric matrices, prefer numpy.linalg.eigh for real eigenvalues.
    """
    values, vectors = eig(matrix)
    return {"eigenvalues": values, "eigenvectors": vectors}

def compute_svd(
    matrix: "np.ndarray",
) -> dict[str, "np.ndarray"]:
    """
    Compute the full Singular Value Decomposition of a matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix of shape (m, n).

    Returns
    -------
    dict
        {'U': np.ndarray (m x m), 'S': np.ndarray (min(m,n),), 'Vt': np.ndarray (n x n)}
        Satisfies matrix = U @ np.diag(S) @ Vt.
    """
    U, S, Vt = svd(matrix)
    return {"U": U, "S": S, "Vt": Vt}

def solve_least_squares(
    A: "np.ndarray", b: "np.ndarray"
) -> dict[str, "np.ndarray"]:
    """
    Solve the least-squares problem: minimize ||Ax - b||^2.

    Parameters
    ----------
    A : np.ndarray
        Coefficient matrix of shape (m, n).
    b : np.ndarray
        Target vector of shape (m,).

    Returns
    -------
    dict
        {'solution': np.ndarray, 'residuals': np.ndarray, 'rank': int, 'singular_values': np.ndarray}
    """
    x, residuals, rank, s = lstsq(A, b, rcond=None)
    return {"solution": x, "residuals": residuals, "rank": rank, "singular_values": s}

def compute_determinant(matrix: "np.ndarray") -> float:
    """
    Compute the determinant of a square matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix.

    Returns
    -------
    float
        Determinant value. Zero indicates a singular (non-invertible) matrix.
    """
    return det(matrix)

def compute_inverse(matrix: "np.ndarray") -> "np.ndarray":
    """
    Compute the inverse of a square matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Square, non-singular matrix.

    Returns
    -------
    np.ndarray
        The inverse matrix such that matrix @ inverse ≈ I.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the matrix is singular (determinant = 0).
    """
    return inv(matrix)

def matrix_summary_df(matrix: "np.ndarray") -> "pd.DataFrame":
    """
    Compute a summary of key matrix properties.

    Parameters
    ----------
    matrix : np.ndarray
        Input numeric matrix.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns: Shape, Rank, Determinant, Condition Number.
        Determinant is None for non-square matrices.
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


def least_squares_solution(
    A: "np.ndarray", b: "np.ndarray"
) -> tuple["np.ndarray", "np.ndarray"]:
    """
    Solve the least-squares problem and return solution and residuals.

    Parameters
    ----------
    A : np.ndarray
        Coefficient matrix of shape (m, n).
    b : np.ndarray
        Target vector of shape (m,).

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (x, residuals) where x is the solution vector.
    """
    x, residuals, rank, s = lstsq(A, b, rcond=None)
    return x, residuals


def compute_eigendecomposition(
    matrix: "np.ndarray",
) -> tuple["np.ndarray", "np.ndarray"]:
    """
    Compute eigenvalues and eigenvectors, ensuring float output.

    Parameters
    ----------
    matrix : np.ndarray
        Square numeric matrix.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (eigenvalues, eigenvectors) both as float64 arrays.
        Complex parts are discarded — use only for real-eigenvalue matrices (e.g., symmetric).
    """
    values, vectors = eig(matrix)
    values = np.array(values, dtype=float)
    vectors = np.array(vectors, dtype=float)
    return values, vectors


def compute_svd_adv(
    matrix: "np.ndarray | pd.DataFrame",
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """
    Compute SVD with enforced float numeric output.

    Parameters
    ----------
    matrix : np.ndarray or pd.DataFrame
        Input matrix. DataFrames are converted to float numpy arrays.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        (U, S, Vt) matrices from the SVD decomposition.
    """
    # Ensure matrix is numeric
    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.to_numpy(dtype=float)
    else:
        matrix = np.asarray(matrix, dtype=float)
    
    # Perform SVD
    U, S, Vt = svd(matrix)
    
    return U, S, Vt
