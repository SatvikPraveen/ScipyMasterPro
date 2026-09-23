"""
Smoke tests for :mod:`utils.viz_utils`.

Plots are rendered on the non-interactive Agg backend and the returned figure
objects are checked for basic structure. The goal is to catch API breakage
(renamed seaborn/matplotlib arguments, pandas changes) rather than pixels.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy import stats  # noqa: E402

from utils import viz_utils as viz  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _assert_figure(fig) -> None:
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1


class TestDescriptivePlots:
    def test_histograms(self, simple_dataframe):
        _assert_figure(viz.plot_histograms(simple_dataframe, ["normal", "skewed"], bins=10))

    def test_boxplots(self, simple_dataframe):
        _assert_figure(viz.plot_boxplots(simple_dataframe, ["normal", "skewed"]))

    def test_correlation_heatmap(self, multivariate_dataframe):
        _assert_figure(viz.plot_correlation_heatmap(multivariate_dataframe))

    def test_covariance_heatmap(self, multivariate_dataframe):
        _assert_figure(viz.plot_covariance_heatmap(multivariate_dataframe))

    def test_pairplot(self, multivariate_dataframe):
        grid = viz.plot_pairplot(multivariate_dataframe.iloc[:30])
        assert grid.figure is not None

    def test_ecdf(self, simple_dataframe):
        _assert_figure(viz.plot_ecdf(simple_dataframe, "normal"))

    def test_qq(self, normal_data):
        _assert_figure(viz.plot_qq(normal_data))

    def test_violin_swarm(self, categorical_dataframe):
        _assert_figure(viz.plot_violin_swarm(categorical_dataframe, "category", "value"))


class TestDistributionPlots:
    def test_pdf_overlay(self, normal_data):
        params = stats.norm.fit(normal_data)
        _assert_figure(viz.plot_pdf_overlay(normal_data, stats.norm, params))

    def test_cdf_overlay(self, normal_data):
        params = stats.norm.fit(normal_data)
        _assert_figure(viz.plot_cdf_overlay(normal_data, stats.norm, params))

    def test_multi_distribution_overlay(self, skewed_data):
        _assert_figure(
            viz.plot_multi_distribution_overlay(skewed_data, [stats.gamma, stats.lognorm])
        )

    def test_ecdf_comparison_multi(self, normal_data, skewed_data):
        def ecdf(a):
            x = np.sort(a)
            return x, np.arange(1, x.size + 1) / x.size

        _assert_figure(
            viz.plot_ecdf_comparison_multi(
                {"normal": ecdf(normal_data), "skewed": ecdf(skewed_data)}
            )
        )


class TestSamplingAndBootstrapPlots:
    def test_ranked_plots(self, normal_data, skewed_data):
        xr, yr = stats.rankdata(normal_data), stats.rankdata(skewed_data)
        _assert_figure(viz.plot_ranked_barplots(xr, yr))
        _assert_figure(viz.plot_ranked_boxplot(xr, yr))

    def test_sampling_distributions(self, normal_data, uniform_data):
        _assert_figure(
            viz.plot_sampling_distributions({"normal": normal_data, "uniform": uniform_data})
        )

    def test_bootstrap_distribution(self, normal_data):
        rng = np.random.default_rng(0)
        boots = np.array(
            [rng.choice(normal_data, size=normal_data.size).mean() for _ in range(200)]
        )
        lo, hi = np.percentile(boots, [2.5, 97.5])
        _assert_figure(viz.plot_bootstrap_distribution(boots, ci_bounds=(lo, hi), true_stat=0.0))


class TestMultivariatePlots:
    @pytest.fixture
    def outlier_df(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame(rng.normal(size=(60, 3)), columns=["X1", "X2", "X3"])
        df["mahalanobis"] = np.sqrt((df[["X1", "X2", "X3"]] ** 2).sum(axis=1))
        df["is_outlier"] = df["mahalanobis"] > df["mahalanobis"].quantile(0.9)
        return df

    def test_mahalanobis_scatter(self, outlier_df):
        _assert_figure(viz.plot_mahalanobis_outliers(outlier_df))

    def test_mahalanobis_distribution(self, outlier_df):
        _assert_figure(viz.plot_mahalanobis_distance_distribution(outlier_df, threshold=2.0))

    def test_mahalanobis_3d_returns_plotly_figure(self, outlier_df):
        fig = viz.plot_mahalanobis_outliers_3d(outlier_df)
        assert hasattr(fig, "to_dict")


class TestOptimizationAndLinearAlgebraPlots:
    @pytest.fixture
    def surface(self):
        x = np.linspace(-2, 2, 25)
        X, Y = np.meshgrid(x, x)
        return X, Y, X**2 + Y**2

    def test_loss_surface(self, surface):
        X, Y, Z = surface
        _assert_figure(viz.plot_3d_loss_surface(X, Y, Z, optimum=(0, 0, 0)))
        _assert_figure(viz.plot_contour_loss_surface(X, Y, Z, optimum=(0, 0)))

    def test_scalar_function(self):
        _assert_figure(
            viz.plot_scalar_function(lambda x: (x - 1) ** 2, x_range=(-3, 5), optimum=(1.0, 0.0))
        )

    def test_eigen_and_singular(self, symmetric_matrix):
        vals, vecs = np.linalg.eigh(symmetric_matrix)
        _assert_figure(viz.plot_eigenvectors(symmetric_matrix[:2, :2], vals[:2], vecs[:2, :2]))
        _assert_figure(viz.plot_eigenvectors_safe(symmetric_matrix[:2, :2], vals[:2], vecs[:2, :2]))
        s = np.linalg.svd(symmetric_matrix, compute_uv=False)
        _assert_figure(viz.plot_singular_values(s))
        _assert_figure(viz.plot_singular_values_safe(s))

    def test_least_squares(self, rectangular_matrix):
        x = np.arange(rectangular_matrix.shape[0])
        observed = rectangular_matrix[:, 0]
        predicted = observed + 0.1
        _assert_figure(viz.plot_least_squares_fit(x, observed, predicted))
        _assert_figure(viz.plot_least_squares_residuals(x, observed - predicted))
        _assert_figure(viz.plot_residuals(predicted, observed))


class TestInterpolationPlots:
    @pytest.fixture
    def curves(self, interpolation_data):
        x, y = interpolation_data
        x_new = np.linspace(x.min(), x.max(), 50)
        y_new = np.interp(x_new, x, y)
        return x, y, x_new, y_new

    def test_interpolation_comparison(self, curves):
        x, y, x_new, y_new = curves
        _assert_figure(viz.plot_interpolation_comparison(x, y, x_new, y_new, y_new, y_new))

    def test_curve_fit_plots(self, curves):
        x, y, x_new, y_new = curves
        _assert_figure(
            viz.plot_curve_fits_with_bands(x, y, x_new, y_new, y_new - 0.1, y_new + 0.1, y_new)
        )
        _assert_figure(
            viz.plot_gaussian_fit_with_band(x, y, x_new, y_new, y_new - 0.1, y_new + 0.1)
        )
        _assert_figure(viz.plot_polynomial_fit(x, y, x_new, y_new, degree=3))
        _assert_figure(
            viz.plot_all_fits_comparison(x, y, x_new, y_new, y_new, y_new, y_new, y_new, y_new)
        )
        _assert_figure(viz.plot_weighted_vs_unweighted_fit(x, y, x_new, y_new, y_new))

    def test_residuals_comparison(self, curves):
        x, y, _, _ = curves
        r = y - y.mean()
        _assert_figure(viz.plot_residuals_comparison(x, r, r, r, r))

    def test_2d_grids(self):
        g = np.linspace(0, 1, 10)
        gx, gy = np.meshgrid(g, g)
        z = np.sin(gx) * np.cos(gy)
        _assert_figure(viz.plot_multivariate_griddata(gx, gy, z))
        _assert_figure(viz.plot_rbf_interpolation(gx, gy, z))
        _assert_figure(viz.plot_multivariate_error_heatmap(gx, gy, np.abs(z)))


class TestInferencePlots:
    def test_confidence_interval(self):
        _assert_figure(viz.plot_confidence_interval(10.0, (9.0, 11.0), pop_mean=10.5))
        _assert_figure(
            viz.plot_multiple_confidence_intervals(10.0, {0.90: (9.2, 10.8), 0.95: (9.0, 11.0)})
        )
        _assert_figure(viz.plot_residuals_vs_population(10.0, 10.5, -0.5))

    def test_power_curve(self):
        n = np.arange(10, 100, 10)
        _assert_figure(viz.plot_power_curve(n, 1 - np.exp(-n / 40)))

    def test_ecdf_manual_vs_stats(self, normal_data):
        x = np.sort(normal_data)
        y = np.arange(1, x.size + 1) / x.size
        _assert_figure(viz.plot_ecdf_comparison_manual_vs_stats(x, y, x, y))


def test_save_and_show_plot_writes_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    viz.save_and_show_plot(fig, "unit_test_plot.png")
    written = list(tmp_path.rglob("unit_test_plot.png"))
    assert written, "save_and_show_plot did not write a file"
