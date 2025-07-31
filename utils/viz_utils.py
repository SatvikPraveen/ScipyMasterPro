import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px


def apply_theme(style="whitegrid", palette="Set2", context="notebook", font_scale=1.1):
    sns.set_theme(style=style, palette=palette, context=context, font_scale=font_scale)

def plot_histograms(df, columns, bins=30, title=None, figsize=(12, 6), palette="Set2"):
    fig, axs = plt.subplots(1, len(columns), figsize=figsize)
    for i, col in enumerate(columns):
        sns.histplot(df[col], bins=bins, kde=True, ax=axs[i], color=sns.color_palette(palette)[i % len(columns)])
        axs[i].set_title(f"Histogram: {col}")
        axs[i].grid(True)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_boxplots(df, columns, title=None, figsize=(8, 5), palette="Set2"):
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df[columns], palette=palette, orient="v", ax=ax)
    ax.set_title(title or "Boxplots")
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_pairplot(df, hue=None, diag_kind="kde", palette="Spectral"):
    return sns.pairplot(df, hue=hue, diag_kind=diag_kind, palette=palette)


def plot_correlation_heatmap(df, annot=True, cmap="coolwarm"):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=annot, cmap=cmap, ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig


def save_and_show_plot(fig, filename):
    fig.savefig(filename, bbox_inches="tight")
    plt.show()


# ECDF Plotter
def plot_ecdf(df, column, color="darkorange"):
    data = df[column]
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, marker='.', linestyle='none', color=color)
    ax.set_title(f"Empirical CDF: {column}")
    ax.set_xlabel("Value")
    ax.set_ylabel("ECDF")
    ax.grid(True)
    return fig


# PDF Overlay Plot
def plot_pdf_overlay(data, dist_obj, params, title="PDF Overlay"):

    x = np.linspace(min(data), max(data), 200)
    fig, ax = plt.subplots(figsize=(7, 4))

    # Use a modern palette color for histogram
    sns.histplot(
        data,
        bins=40,
        kde=False,
        stat="density",
        label="Data",
        color=sns.color_palette("crest")[2],  # Gradient blue-green
        alpha=0.5,
        edgecolor="black"
    )

    # Choose a contrasting color for PDF line
    line_color = sns.color_palette("flare")[3]
    ax.plot(
        x,
        dist_obj.pdf(x, *params),
        color=line_color,
        lw=2.5,
        linestyle="-",
        label=f'{dist_obj.name} fit'
    )

    # Beautify plot
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    return fig


# CDF vs ECDF Overlay Plot
def plot_cdf_overlay(data, dist_obj, params, title="CDF vs ECDF"):

    sorted_data = np.sort(data)
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    cdf_vals = dist_obj.cdf(sorted_data, *params)

    fig, ax = plt.subplots(figsize=(7, 4))

    # Empirical CDF: dotted blue line with markers
    ax.plot(
        sorted_data,
        ecdf,
        marker='o',
        linestyle='-',
        color=sns.color_palette("crest")[4],
        label='Empirical CDF',
        alpha=0.8,
        markersize=4
    )

    # Fitted CDF: smooth contrasting line
    ax.plot(
        sorted_data,
        cdf_vals,
        color=sns.color_palette("flare")[2],
        linestyle='--',
        linewidth=2.2,
        label=f'{dist_obj.name} CDF'
    )

    # Beautify plot
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Value")
    ax.set_ylabel("Cumulative Probability")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()
    plt.tight_layout()

    return fig



# Covariance Heatmap Plot
def plot_covariance_heatmap(df, title="Covariance Matrix"):
    """
    Plots a fancy covariance heatmap with improved styling and color palette.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing numerical columns.
    title : str
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Heatmap figure object for saving or further customization.
    """
    
    # Compute covariance matrix
    cov_matrix = df.cov()

    # Setup figure
    fig, ax = plt.subplots(figsize=(7.5, 6))
    
    # Create heatmap with a better diverging palette
    sns.heatmap(
        cov_matrix,
        annot=True,
        fmt=".2f",
        cmap=sns.color_palette("coolwarm", as_cmap=True),
        linewidths=0.6,
        linecolor='white',
        cbar=True,
        cbar_kws={'shrink': 0.8, 'label': 'Covariance'},
        square=True,
        ax=ax
    )

    # Beautify plot
    ax.set_title(title, fontsize=14, fontweight='bold', color="#333333", pad=12)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', rotation=0, labelsize=10)
    plt.xticks(ha='right')
    plt.yticks(va='center')
    plt.tight_layout()
    
    return fig




# -------------------------------------------
# 📈 Q–Q PLOT (Normal)
# -------------------------------------------
from scipy import stats

def plot_qq(data, title="Q–Q Plot (Normal)", figsize=(6, 5), marker_color="#1f77b4"):
    fig = plt.subplots(figsize=figsize)[0]
    ax = fig.gca()
    stats.probplot(data, dist="norm", plot=ax)
    for line in ax.get_lines():
        line.set_color(marker_color)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig

# -------------------------------------------
# 🟣 VIOLIN + SWARM OVERLAY
# -------------------------------------------
def plot_violin_swarm(df, x_col, y_col, title="Violin + Swarm", figsize=(7, 5), palette="Set2"):
    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(data=df, x=x_col, y=y_col, inner=None, palette=palette, ax=ax)
    sns.swarmplot(data=df, x=x_col, y=y_col, color="k", size=2, alpha=0.6, ax=ax)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# -------------------------------------------
# 🔹 RANKED BARPLOT DISTRIBUTIONS
# -------------------------------------------
def plot_ranked_barplots(x_ranked, y_ranked, title1="x_ranked", title2="y_ranked", figsize=(12, 5)):
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    sns.set(style="whitegrid")

    # Prepare aggregated data
    df_x = pd.Series(x_ranked).value_counts().sort_index().reset_index()
    df_x.columns = ["score", "count"]
    sns.barplot(x="score", y="count", data=df_x, ax=ax[0], palette="Blues_d")
    ax[0].set_title(f"Distribution of {title1}")
    ax[0].set_xlabel("Likert Score")
    ax[0].set_ylabel("Frequency")

    df_y = pd.Series(y_ranked).value_counts().sort_index().reset_index()
    df_y.columns = ["score", "count"]
    sns.barplot(x="score", y="count", data=df_y, ax=ax[1], palette="Greens_d")
    ax[1].set_title(f"Distribution of {title2}")
    ax[1].set_xlabel("Likert Score")
    ax[1].set_ylabel("Frequency")

    plt.tight_layout()
    return fig


# -------------------------------------------
# 🔹 BOX PLOT FOR RANKED DISTRIBUTIONS
# -------------------------------------------
def plot_ranked_boxplot(x_ranked, y_ranked, figsize=(7, 5), title="Boxplot of Ranked Distributions"):
    df_ranked = pd.DataFrame({
        "x_ranked": x_ranked,
        "y_ranked": y_ranked
    })
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df_ranked, palette="Set2", ax=ax)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_multi_distribution_overlay(data, distribution_list, title="Multi-Distribution Fit", bins=40):
    """
    Plot histogram of data and overlay PDFs of multiple fitted distributions.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data, bins=bins, stat="density", color="gray", alpha=0.4, label="Data")

    x = np.linspace(min(data), max(data), 300)
    for dist in distribution_list:
        params = dist.fit(data)
        y = dist.pdf(x, *params)
        ax.plot(x, y, lw=2, label=f"{dist.name}")
    
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


# -------------------------------------------
# 📈 ECDF COMPARISON PLOT
# -------------------------------------------
def plot_ecdf_comparison_multi(ecdf_data, title="ECDF Comparison", figsize=None, palette=None):
    """
    Plot ECDF curves for multiple datasets on a single figure.

    Parameters
    ----------
    ecdf_data : dict
        Dictionary of the form {label: (x_values, y_values)}
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size (width, height). Defaults to adaptive size based on dataset count.
    palette : list, optional
        List of colors to use for each dataset
    """
    n_curves = len(ecdf_data)
    if figsize is None:
        figsize = (8, 4.5 + 0.3 * n_curves)  # Adjust height for more curves

    # Use Seaborn palette if not specified
    if palette is None:
        palette = sns.color_palette("husl", n_curves)

    fig, ax = plt.subplots(figsize=figsize)

    # Define different line styles for variety
    linestyles = ['-', '--', '-.', ':']
    
    for (label, (x, y)), color, ls in zip(ecdf_data.items(), palette, linestyles * (n_curves // 4 + 1)):
        ax.step(x, y, where="post", label=label, color=color, linestyle=ls, linewidth=1.8)

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel("Value", fontsize=11)
    ax.set_ylabel("ECDF", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title="Samples", fontsize=9)
    
    plt.tight_layout()
    return fig


# -------------------------------------------
# 🎨 SAMPLING DISTRIBUTIONS COMPARISON PLOT
# -------------------------------------------
def plot_sampling_distributions(samples_dict, 
                                title="Sampling Distributions Comparison", 
                                bins=20, 
                                figsize=(12, 6), 
                                colors=None):
    """
    Plot multiple sampling distributions side by side for comparison.

    Parameters:
    - samples_dict: dict where { "label": sample_array }
    - title: figure title
    - bins: number of histogram bins
    - figsize: size of the figure
    - colors: list of colors for histograms
    """
    if colors is None:
        colors = sns.color_palette("Set2", len(samples_dict))

    fig, ax = plt.subplots(figsize=figsize)
    for (label, sample), color in zip(samples_dict.items(), colors):
        sample = np.array(sample) # ensure it's always a NumPy array
        sns.histplot(sample, bins=bins, stat="density", label=label, 
                     alpha=0.5, color=color, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Values")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# -----------------------------------------------------
# 🎨 BOOTSTRAP PLOT WITH CI AND TRUE STATISTIC
# -----------------------------------------------------
def plot_bootstrap_distribution(estimates, ci_bounds=None, true_stat=None, 
                                title="Bootstrap Distribution", bins=30):
    """
    Plot bootstrap distribution with colorful, informative highlights:
    - Histogram with gradient-like color
    - CI bounds in distinct contrasting colors
    - True statistic highlighted with a bold, distinct line
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Use a more appealing palette for histogram
    sns.histplot(
        estimates,
        bins=bins,
        stat="frequency",
        edgecolor="black",
        color=sns.color_palette("crest", 8)[4],
        alpha=0.6,
        ax=ax
    )
    
    # Mean line (optional visual aid)
    mean_val = np.mean(estimates)
    ax.axvline(mean_val, color=sns.color_palette("Blues", 8)[6], linestyle="-", linewidth=2, label="Bootstrap Mean")
    
    # Add true statistic line (if provided)
    if true_stat is not None:
        ax.axvline(true_stat, color="red", linestyle="--", linewidth=2.5, label="Sample Statistic")
    
    # Add CI bounds (green lower, orange upper for contrast)
    if ci_bounds:
        ax.axvline(ci_bounds[0], color="#2ca02c", linestyle="dashed", linewidth=2.2, label="Lower CI")
        ax.axvline(ci_bounds[1], color="#ff7f0e", linestyle="dashed", linewidth=2.2, label="Upper CI")
        
        # Shade CI region for visibility
        ax.axvspan(ci_bounds[0], ci_bounds[1], color="yellow", alpha=0.15, label="CI Range")
    
    # Beautify plot
    ax.set_title(title, fontsize=13, fontweight='bold', color="#333333")
    ax.set_xlabel("Bootstrap Estimates", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    
    return fig


def plot_mahalanobis_outliers(df, x_col="X1", y_col="X2", outlier_col="is_outlier", 
                              title="Mahalanobis Outliers Highlighted", 
                              figsize=(7, 5), cmap=None, alpha=0.7):
    """
    Scatter plot highlighting Mahalanobis outliers with improved color styling.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the variables to plot and the outlier flag column.
    x_col : str
        Column name for the x-axis.
    y_col : str
        Column name for the y-axis.
    outlier_col : str
        Column name with boolean or 0/1 values marking outliers.
    title : str
        Plot title.
    figsize : tuple
        Size of the figure.
    cmap : str or matplotlib colormap, optional
        Color map for differentiating outliers vs non-outliers.
    alpha : float
        Transparency level for scatter points.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object for saving or further customization.
    """
    
    if cmap is None:
        cmap = sns.color_palette("coolwarm", as_cmap=True)

    fig, ax = plt.subplots(figsize=figsize)

    # ✅ Separate inliers and outliers for custom styling
    inliers = df[~df[outlier_col]]
    outliers = df[df[outlier_col]]

    # Plot inliers
    ax.scatter(
        inliers[x_col], inliers[y_col],
        c="steelblue", alpha=alpha, edgecolor='k',
        label="Inliers", s=60
    )

    # Plot outliers with distinct color & marker
    ax.scatter(
        outliers[x_col], outliers[y_col],
        c="crimson", alpha=0.9, edgecolor='k',
        label="Outliers", s=100, marker="X", linewidths=1.2
    )

    # Titles and labels
    ax.set_title(title, fontsize=13, fontweight='bold', color="#333333")
    ax.set_xlabel(x_col, fontsize=11)
    ax.set_ylabel(y_col, fontsize=11)

    # Colorbar (optional: binary mapping for clarity)
    cbar = plt.colorbar(scatter := ax.scatter(df[x_col], df[y_col], c=df[outlier_col],
                                              cmap=cmap, alpha=0))
    cbar.set_label("Outlier (1=True)", fontsize=10)

    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    return fig



def plot_mahalanobis_outliers_3d(df, x_col="X1", y_col="X2", z_col="X3", outlier_col="is_outlier",
                                 title="3D Outlier Visualization using Mahalanobis Distance",
                                 opacity=0.85, colors=None):
    """
    Interactive 3D scatter plot highlighting Mahalanobis outliers with custom colors.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the 3D coordinates and outlier flag column.
    x_col, y_col, z_col : str
        Column names for 3D axes.
    outlier_col : str
        Column name indicating outliers (boolean or 0/1 values).
    title : str
        Plot title.
    opacity : float
        Transparency level of points.
    colors : dict
        Custom color mapping for outliers vs non-outliers.

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        Interactive Plotly figure for 3D visualization.
    """
    
    if colors is None:
        colors = {False: '#1f77b4', True: '#d62728'}  # Blue for inliers, Red for outliers

    fig = px.scatter_3d(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=outlier_col,
        color_discrete_map=colors,
        title=title,
        opacity=opacity,
        symbol=outlier_col,  # Different symbol for outliers
        symbol_map={False: "circle", True: "x"}
    )

    fig.update_traces(
        marker=dict(
            size=6,
            line=dict(width=1, color='DarkSlateGrey')
        ),
        selector=dict(mode='markers')
    )

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(title="Outlier", font=dict(size=10)),
        scene=dict(
            xaxis=dict(title=x_col),
            yaxis=dict(title=y_col),
            zaxis=dict(title=z_col)
        )
    )

    return fig


def plot_mahalanobis_distance_distribution(df, distance_col="mahalanobis", threshold=None, 
                                           title="Mahalanobis Distance Distribution"):
    """
    Plots the distribution of Mahalanobis distances with enhanced colors and threshold annotation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing Mahalanobis distances.
    distance_col : str
        Column name with computed Mahalanobis distances.
    threshold : float, optional
        Chi-square threshold for marking potential outliers.
    title : str
        Plot title.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object for saving or further customization.
    """
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Main histogram
    sns.histplot(
        df[distance_col],
        bins=30,
        kde=True,
        color=sns.color_palette("crest")[3],
        alpha=0.7,
        edgecolor="black",
        ax=ax
    )

    # Threshold line and shaded region
    if threshold is not None:
        ax.axvline(
            threshold,
            color=sns.color_palette("flare")[2],
            linestyle="--",
            linewidth=2.2,
            label=f"Threshold = {threshold:.2f}"
        )
        ax.axvspan(
            threshold,
            df[distance_col].max(),
            color=sns.color_palette("flare")[1],
            alpha=0.15,
            label="Outlier Region"
        )

    # Titles and labels
    ax.set_title(title, fontsize=13, fontweight='bold', color="#333333")
    ax.set_xlabel("Mahalanobis Distance", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()
    plt.tight_layout()
    
    return fig


def plot_3d_loss_surface(X, Y, Z, optimum=None, title="Loss Surface 3D"):
    """
    Plots 3D loss surface with optional optimal solution marker.

    Parameters
    ----------
    X, Y, Z : np.ndarray
        Grid coordinates and cost values
    optimum : tuple (x, y, z)
        Coordinates of the optimum point to mark on the plot
    title : str
        Plot title
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor='k', alpha=0.7)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=8, label='Cost')

    if optimum:
        ax.scatter(optimum[0], optimum[1], optimum[2],
                   color='red', s=80, marker='o', label='Optimum')
        ax.legend()

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Cost")
    plt.tight_layout()
    return fig


def plot_contour_loss_surface(X, Y, Z, optimum=None, title="Loss Surface Contour"):
    """
    Plots 2D contour map of a loss surface with optimum point highlighted.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    cp = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
    fig.colorbar(cp, ax=ax, label='Cost')
    
    if optimum:
        ax.scatter(optimum[0], optimum[1], color='red', s=80, marker='x', label='Optimum')
        ax.legend()

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    return fig

# --- New visualization utilities ---

def plot_eigenvectors(matrix, eigenvalues, eigenvectors, title="Eigenvectors Visualization"):
    """
    Plot eigenvectors on a 2D grid with scaling according to eigenvalues.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    origin = [0], [0]
    colors = ['r', 'b', 'g', 'm', 'y']

    for i in range(eigenvectors.shape[1]):
        ax.quiver(*origin, 
                  eigenvectors[0, i], eigenvectors[1, i], 
                  angles='xy', scale_units='xy', scale=1,
                  color=colors[i % len(colors)],
                  label=f"Eigenvector {i+1} (λ={eigenvalues[i]:.2f})",
                  width=0.005)
    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_singular_values(singular_values, title="Singular Values"):
    """
    Plot singular values from SVD as a bar chart.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=list(range(1, len(singular_values)+1)), 
                y=singular_values, palette="crest", ax=ax)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Component")
    ax.set_ylabel("Singular Value")
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    return fig


def plot_least_squares_fit(x_values, observed, predicted, 
                           title="Least Squares Fit", 
                           figsize=(5, 4), 
                           observed_color='blue', 
                           predicted_color='red'):
    """
    Plot observed data points and the fitted line from least squares regression.

    Parameters
    ----------
    x_values : array-like
        Independent variable values (x-axis).
    observed : array-like
        Observed dependent variable values.
    predicted : array-like
        Predicted dependent variable values from least squares solution.
    title : str, default="Least Squares Fit"
        Plot title.
    figsize : tuple, default=(5, 4)
        Size of the figure.
    observed_color : str, default='blue'
        Color for observed points.
    predicted_color : str, default='red'
        Color for fitted line.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated plot figure.
    """
    x_values = np.array(x_values)
    observed = np.array(observed)
    predicted = np.array(predicted)

    if len(x_values) != len(observed) or len(x_values) != len(predicted):
        raise ValueError("x_values, observed, and predicted must have the same length.")

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x_values, observed, color=observed_color, label='Observed', s=70, alpha=0.85, edgecolor='k', linewidth=0.8)
    ax.plot(x_values, predicted, color=predicted_color, label='Fitted Line', linewidth=2)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("X values")
    ax.set_ylabel("Y values")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    return fig


def plot_least_squares_residuals(x_values, residuals, 
                                 title="Residuals of Least Squares Fit"):
    """
    Plot residuals for a least squares fit using a stem plot.

    Parameters
    ----------
    x_values : array-like
        The x-values corresponding to residuals.
    residuals : array-like
        The residuals (observed - predicted) to plot.
    title : str, optional
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object for further manipulation or display.
    """
    fig, ax = plt.subplots(figsize=(5, 3))

    ax.stem(x_values, residuals, basefmt=" ", linefmt='C0-', markerfmt='C0o')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("x values")
    ax.set_ylabel("Residuals")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    return fig


def plot_interpolation_comparison(x, y, x_new, y_lin, y_cub, y_spl,
                                  title="Interpolation Comparison",
                                  figsize=(10, 6)):
    """
    Plot comparison of Linear, Cubic, and Spline interpolation.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, color='gray', label="Original Points")
    ax.plot(x_new, y_lin, label="Linear", color='blue')
    ax.plot(x_new, y_cub, label="Cubic", color='green')
    ax.plot(x_new, y_spl, label="Spline", linestyle='--', color='purple')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig


def plot_curve_fits_with_bands(x, y, x_new, y_exp_pred, y_exp_lower, y_exp_upper,
                               y_gauss, title="Curve Fitting with Confidence Bands",
                               figsize=(10, 6)):
    """
    Plot Exponential and Gaussian curve fits with confidence bands for Exponential.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, color='gray', label="Data")
    ax.plot(x_new, y_exp_pred, label="Exponential Fit", color="orange")
    ax.fill_between(x_new, y_exp_lower, y_exp_upper, color="orange", alpha=0.2, label="Exp ±2σ Band")
    ax.plot(x_new, y_gauss, label="Gaussian Fit", color="blue")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig


def plot_multivariate_griddata(grid_x, grid_y, grid_z,
                               title="2D Multivariate Interpolation using griddata",
                               cmap='viridis', figsize=(8, 6)):
    """
    Plot 2D multivariate interpolation results from griddata.
    """
    fig, ax = plt.subplots(figsize=figsize)
    c = ax.contourf(grid_x, grid_y, grid_z, cmap=cmap)
    plt.colorbar(c, ax=ax)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.3)
    return fig


def plot_rbf_interpolation(grid_x, grid_y, z_rbf,
                           title="2D RBF Interpolation",
                           cmap='plasma', figsize=(8, 6)):
    """
    Plot 2D Radial Basis Function interpolation results.
    """
    fig, ax = plt.subplots(figsize=figsize)
    c = ax.contourf(grid_x, grid_y, z_rbf, cmap=cmap)
    plt.colorbar(c, ax=ax)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.3)
    return fig


def plot_gaussian_fit_with_band(x, y, x_new, y_pred, y_lower, y_upper,
                                title="Gaussian Curve Fit with Confidence Band",
                                figsize=(10, 6)):
    """
    Plot Gaussian fit with ±2σ confidence band.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_new, y_pred, label="Gaussian Fit", color="blue")
    ax.fill_between(x_new, y_lower, y_upper, color="blue", alpha=0.2, label="±2σ Band")
    ax.scatter(x, y, color='gray', alpha=0.6, label="Data")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig


def plot_polynomial_fit(x, y, x_new, y_poly, degree=2, 
                        title="Polynomial Curve Fit", figsize=(10, 6)):
    """
    Plot polynomial fit alongside original data.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, color='gray', label="Data")
    ax.plot(x_new, y_poly, label=f"Polynomial Fit (deg={degree})", color='green', linewidth=2)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    return fig


def plot_all_fits_comparison(x, y, x_new, y_lin, y_cub, y_spl, y_exp, y_gauss, y_poly,
                             figsize=(10, 6), title="All Fits Overlay Comparison"):
    """
    Plot all interpolation and curve fitting results on one figure for visual comparison.

    Parameters
    ----------
    x, y : array-like
        Original data points.
    x_new : array-like
        Range of new x-values for prediction.
    y_lin, y_cub, y_spl, y_exp, y_gauss, y_poly : array-like
        Predicted values from each fit.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, color='black', label="Original Data", alpha=0.7)
    ax.plot(x_new, y_lin, label="Linear", linestyle='-', linewidth=2)
    ax.plot(x_new, y_cub, label="Cubic", linestyle='--', linewidth=2)
    ax.plot(x_new, y_spl, label="Spline", linestyle='-.', linewidth=2)
    ax.plot(x_new, y_exp, label="Exponential", linewidth=2)
    ax.plot(x_new, y_gauss, label="Gaussian", linewidth=2)
    ax.plot(x_new, y_poly, label="Polynomial", linewidth=2)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    return fig


def plot_residuals_comparison(x, residual_exp, residual_gauss, residual_poly, residual_weighted,
                              figsize=(10, 6), title="Residuals Comparison"):
    """
    Plot residuals for multiple fitted models to compare error distributions.

    Parameters
    ----------
    x : array-like
        Original x-values.
    residual_exp, residual_gauss, residual_poly, residual_weighted : array-like
        Residuals for exponential, gaussian, polynomial, and weighted exponential fits.
    figsize : tuple, default=(10, 6)
        Figure size for the plot.
    title : str, default="Residuals Comparison"
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated residual comparison plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.stem(x, residual_exp, markerfmt='o', linefmt='r-', basefmt=" ", label="Exponential")
    ax.stem(x, residual_gauss, markerfmt='s', linefmt='g-', basefmt=" ", label="Gaussian")
    ax.stem(x, residual_poly, markerfmt='^', linefmt='b-', basefmt=" ", label="Polynomial")
    ax.stem(x, residual_weighted, markerfmt='x', linefmt='m-', basefmt=" ", label="Weighted Exp")
    
    ax.set_title(title)
    ax.set_xlabel("X values")
    ax.set_ylabel("Residuals")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig


def plot_weighted_vs_unweighted_fit(x, y, x_new, y_unweighted, y_weighted,
                                    figsize=(10, 6), title="Weighted vs Unweighted Exponential Fit"):
    """
    Compare weighted and unweighted exponential curve fitting visually.

    Parameters
    ----------
    x, y : array-like
        Original data points.
    x_new : array-like
        Range for prediction.
    y_unweighted, y_weighted : array-like
        Predicted values from unweighted and weighted fits.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, color='gray', alpha=0.7, label="Data")
    ax.plot(x_new, y_unweighted, label="Unweighted Fit", color='orange', linewidth=2)
    ax.plot(x_new, y_weighted, label="Weighted Fit", color='blue', linewidth=2, linestyle='--')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    return fig


def plot_multivariate_error_heatmap(grid_x, grid_y, error_grid,
                                    figsize=(8, 6), title="Interpolation Error Heatmap"):
    """
    Plot error distribution for 2D interpolation as a heatmap.

    Parameters
    ----------
    grid_x, grid_y : 2D arrays
        Grid coordinates.
    error_grid : 2D array
        Absolute errors between true and interpolated values.
    """
    fig, ax = plt.subplots(figsize=figsize)
    c = ax.contourf(grid_x, grid_y, error_grid, cmap='coolwarm')
    plt.colorbar(c, ax=ax, label="Absolute Error")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.3)
    return fig


def plot_confidence_interval(mean, ci, pop_mean=None, figsize=(6, 4), title="Confidence Interval for Sample Mean"):
    """
    Plot a confidence interval for a sample mean with optional population mean reference line.

    Parameters:
    -----------
    mean : float
        Sample mean
    ci : tuple
        Confidence interval (low, high)
    pop_mean : float, optional
        Population mean to mark with a reference line
    figsize : tuple
        Size of the plot
    title : str
        Title of the plot

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(1, mean,
                yerr=[[mean - ci[0]], [ci[1] - mean]],
                fmt='o', capsize=5, label="95% t-CI")
    
    if pop_mean is not None:
        ax.axhline(y=pop_mean, color='r', linestyle='--', label="Population Mean")
    
    ax.set_xlim(0.5, 1.5)
    ax.set_xticks([1])
    ax.set_xticklabels(["Sample Mean"])
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residuals_vs_population(sample_mean, pop_mean, residual, figsize=(6, 4)):
    """
    Plot residual difference between sample mean and population mean.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(["Residual"], [residual], color='skyblue')
    ax.axhline(0, color='red', linestyle='--', label='No Difference')
    ax.set_ylabel("Difference (Sample - Population)")
    ax.set_title("Residual Difference from Population Mean")
    ax.legend()
    return fig


def plot_multiple_confidence_intervals(mean, ci_dict, figsize=(7, 5)):
    """
    Plot multiple confidence intervals for various confidence levels.
    """
    fig, ax = plt.subplots(figsize=figsize)
    levels = []
    for i, (conf, (low, high)) in enumerate(ci_dict.items(), start=1):
        ax.errorbar(i, mean,
                    yerr=[[mean - low], [high - mean]],
                    fmt='o', capsize=5, label=f"{int(conf*100)}% CI")
        levels.append(i)
    ax.set_xticks(levels)
    ax.set_xticklabels([f"{int(l*100)}%" for l in ci_dict.keys()])
    ax.set_title("Confidence Intervals at Multiple Levels")
    ax.legend()
    return fig


def plot_ecdf_comparison_manual_vs_stats(x_manual, y_manual, x_sm, y_sm, title="ECDF Comparison"):
    fig, ax = plt.subplots()
    ax.step(x_manual, y_manual, where='post', label="Manual ECDF", color='green')
    ax.step(x_sm, y_sm, where='post', linestyle='--', label="Statsmodels ECDF", color='orange')
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("Cumulative Probability")
    ax.legend()
    ax.grid()
    return fig


def plot_power_curve(sample_sizes, power_values, alpha=0.05):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sample_sizes, power_values, marker='o', label="Power")
    ax.axhline(y=0.8, color='r', linestyle='--', label="Target Power (0.8)")
    ax.set_title("Power Curve vs Sample Size")
    ax.set_xlabel("Sample Size (n)")
    ax.set_ylabel("Power")
    ax.legend()
    ax.grid(True)
    return fig


def plot_scalar_function(func, x_range=(-10, 10), optimum=None, title="Scalar Function"):
    """
    Plot a scalar cost function over a given range.

    Parameters
    ----------
    func : callable
        The scalar function to plot.
    x_range : tuple
        Range of x values to plot.
    optimum : tuple
        (x_opt, y_opt) location of optimum, optional.
    title : str
        Title of the plot.
    """
    x = np.linspace(x_range[0], x_range[1], 400)
    y = [func(val) for val in x]
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, label="Cost Function", color='steelblue')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    
    if optimum is not None:
        ax.scatter(optimum[0], optimum[1], color='red', s=80, label="Optimal Point", zorder=5)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    plt.tight_layout()
    
    return fig


def plot_residuals(predicted, actual, figsize=(6, 4)):
    """
    Simple wrapper to plot residuals (Predicted - Actual).
    """
    residuals = predicted - actual
    fig, ax = plt.subplots(figsize=figsize)
    ax.stem(range(len(residuals)), residuals, markerfmt='o', linefmt='r-', basefmt=" ")
    ax.axhline(0, color='blue', linestyle='--')
    ax.set_title("Residuals Plot")
    ax.set_xlabel("Index")
    ax.set_ylabel("Residual")
    return fig


def plot_singular_values_safe(singular_values, title="Singular Values (Safe Plot)"):
    """
    A safe version of plot_singular_values that:
    - Converts input to numeric floats
    - Handles lists, Series, or arrays with mixed types
    - Plots without breaking due to dtype issues
    """
    # Force numeric conversion and drop NaNs
    try:
        singular_values = pd.to_numeric(np.ravel(singular_values), errors='coerce')
        singular_values = singular_values[~np.isnan(singular_values)]
    except Exception:
        singular_values = np.array([], dtype=float)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(singular_values) > 0:
        components = list(range(1, len(singular_values) + 1))
        sns.barplot(
            x=components,
            y=singular_values,
            hue=components,      # ✅ Add hue
            palette="crest",
            dodge=False,
            legend=False,
            ax=ax
        )

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Component")
        ax.set_ylabel("Singular Value")
    else:
        ax.text(0.5, 0.5, "No valid singular values", 
                ha='center', va='center', fontsize=12, color='red')
        ax.set_axis_off()
    
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    return fig


def plot_eigenvectors_safe(matrix, eigvals, eigvecs, title="Eigenvectors Visualization"):
    """
    Plots eigenvectors safely in 2D space.
    If matrix dimension > 2, only the first 2 components are plotted.

    Parameters
    ----------
    matrix : np.ndarray
        Original matrix (for scale reference)
    eigvals : array-like
        Eigenvalues
    eigvecs : array-like
        Eigenvectors

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    eigvals = np.array(eigvals, dtype=float)
    eigvecs = np.array(eigvecs, dtype=float)

    # Keep only first 2 dimensions
    if eigvecs.shape[0] > 2:
        eigvecs = eigvecs[:2, :]
    if eigvecs.shape[1] > 5:
        eigvecs = eigvecs[:, :5]  # avoid overcrowded plot

    fig, ax = plt.subplots(figsize=(6, 6))
    origin = np.zeros(2)

    # Plot each eigenvector
    for i in range(eigvecs.shape[1]):
        vec = eigvecs[:, i]
        ax.quiver(*origin, *vec, angles='xy', scale_units='xy', scale=1,
                  label=f"Eigenvector {i+1} (λ={eigvals[i]:.2f})")

    ax.axhline(0, color='grey', lw=1)
    ax.axvline(0, color='grey', lw=1)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    return fig

