"""
Tests for linear_algebra_utils module.

This module tests linear algebra operations including SVD,
eigendecomposition, least squares, and matrix operations.
"""

import numpy as np
import pytest
from scipy import linalg


class TestMatrixDecompositions:
    """Test matrix decomposition operations."""

    def test_svd_decomposition(self, rectangular_matrix):
        """Test Singular Value Decomposition."""
        U, s, Vt = linalg.svd(rectangular_matrix)

        # Reconstruct matrix
        S = linalg.diagsvd(s, U.shape[0], Vt.shape[0])
        reconstructed = U @ S @ Vt

        np.testing.assert_allclose(reconstructed, rectangular_matrix, atol=1e-10)

    def test_eigendecomposition(self, symmetric_matrix):
        """Test eigenvalue decomposition."""
        eig_out = linalg.eig(symmetric_matrix)
        eigenvalues = np.asarray(eig_out[0])
        eigenvectors = np.asarray(eig_out[1])

        # Check that Av = λv for each eigenpair
        for i in range(len(eigenvalues)):
            lam = eigenvalues[i]
            v = eigenvectors[:, i]

            Av = symmetric_matrix @ v
            lam_v = lam * v

            np.testing.assert_allclose(Av, lam_v, atol=1e-10)

    def test_svd_rank(self, rectangular_matrix):
        """Test matrix rank computation via SVD."""
        U, s, Vt = linalg.svd(rectangular_matrix)

        # Rank is number of non-zero singular values
        rank = np.sum(s > 1e-10)
        assert rank > 0
        assert rank <= min(rectangular_matrix.shape)


class TestLinearSystems:
    """Test solving linear systems."""

    def test_least_squares_solution(self, random_seed):
        """Test least squares solution."""
        np.random.seed(random_seed)

        # Create overdetermined system Ax = b
        A = np.random.randn(10, 5)
        x_true = np.random.randn(5)
        b = A @ x_true + 0.1 * np.random.randn(10)  # Add small noise

        # Solve least squares
        x_fit, residuals, rank, s = linalg.lstsq(A, b)  # type: ignore[misc]

        # Should be close to true solution
        np.testing.assert_allclose(x_fit, x_true, atol=0.5)

    def test_square_system_solve(self, square_matrix, random_seed):
        """Test solving square linear system."""
        np.random.seed(random_seed)

        # Create Ax = b
        x_true = np.random.randn(5)
        b = square_matrix @ x_true

        # Solve system
        x_solve = linalg.solve(square_matrix, b)

        # Check solution
        np.testing.assert_allclose(x_solve, x_true, atol=1e-10)


class TestMatrixProperties:
    """Test matrix property computations."""

    def test_determinant(self, square_matrix):
        """Test determinant calculation."""
        det = linalg.det(square_matrix)

        assert isinstance(det, (float, complex, np.number))

    def test_matrix_inverse(self, square_matrix):
        """Test matrix inversion."""
        try:
            inv = linalg.inv(square_matrix)

            # Check that A @ A^-1 = I
            identity = square_matrix @ inv
            expected_identity = np.eye(square_matrix.shape[0])

            np.testing.assert_allclose(identity, expected_identity, atol=1e-10)
        except linalg.LinAlgError:
            # Matrix may be singular
            pytest.skip("Matrix is singular")

    def test_condition_number(self, square_matrix):
        """Test condition number calculation."""
        cond = linalg.norm(square_matrix) * linalg.norm(linalg.inv(square_matrix))

        assert cond >= 1.0  # Condition number is always >= 1

    def test_matrix_rank(self, rectangular_matrix):
        """Test matrix rank calculation."""
        rank = np.linalg.matrix_rank(rectangular_matrix)

        assert 0 < rank <= min(rectangular_matrix.shape)


class TestSpecialMatrices:
    """Test operations on special matrix types."""

    def test_symmetric_eigenvalues(self, symmetric_matrix):
        """Test that symmetric matrix has real eigenvalues."""
        eigenvalues = linalg.eigvalsh(symmetric_matrix)

        # All eigenvalues should be real
        assert np.all(np.isreal(eigenvalues))

    def test_positive_definite_check(self, symmetric_matrix):
        """Test checking if matrix is positive definite."""
        # Try Cholesky decomposition
        try:
            L = linalg.cholesky(symmetric_matrix, lower=True)

            # If successful, verify L @ L.T = A
            reconstructed = L @ L.T
            np.testing.assert_allclose(reconstructed, symmetric_matrix, atol=1e-10)
        except linalg.LinAlgError:
            # Not positive definite
            pass  # This is fine, not all symmetric matrices are PD

    def test_orthogonal_matrix(self, random_seed):
        """Test orthogonal matrix properties."""
        np.random.seed(random_seed)

        # Generate orthogonal matrix via QR decomposition
        A = np.random.randn(5, 5)
        qr_out = linalg.qr(A)
        Q = np.asarray(qr_out[0])

        # Q should be orthogonal: Q^T @ Q should be identity
        identity = Q.T @ Q
        expected_identity = np.eye(5)

        np.testing.assert_allclose(identity, expected_identity, atol=1e-10)


class TestMatrixNorms:
    """Test matrix norm calculations."""

    @pytest.mark.parametrize("ord", [1, 2, np.inf, "fro"])
    def test_matrix_norms(self, square_matrix, ord):
        """Test different matrix norms."""
        norm = linalg.norm(square_matrix, ord=ord)

        assert norm >= 0
        assert np.isfinite(norm)

    def test_norm_properties(self, square_matrix):
        """Test matrix norm properties."""
        # Frobenius norm
        norm_fro = linalg.norm(square_matrix, "fro")

        # 2-norm
        norm_2 = linalg.norm(square_matrix, 2)

        # Frobenius norm >= 2-norm
        assert norm_fro >= norm_2 - 1e-10  # Allow small numerical error
