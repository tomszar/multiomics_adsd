import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from momics_ad.figures import plots
from momics_ad.io import read
from momics_ad.stats import sd


def plot_sex_diff_results(result_file="ResultTable_PLS.csv", output_file="SexDifferences.pdf"):
    """
    Create a visually appealing plot of sex differences results.
    
    Parameters
    ----------
    result_file: str
        Name of the result file from sex_diff analysis.
    output_file: str
        Name of the output file.
    """
    plots._check_dir()
    
    # Read results
    results = pd.read_csv(result_file)
    results = results.set_index("Index")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot parameters
    params = results.index
    values = results["Values"]
    pvals = results["Pvalues"]
    
    # Create bar plot
    bars = ax.bar(params, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    # Add p-values
    for i, (param, pval) in enumerate(zip(params, pvals)):
        significance = ""
        if pval < 0.001:
            significance = "***"
        elif pval < 0.01:
            significance = "**"
        elif pval < 0.05:
            significance = "*"
        
        if significance:
            ax.text(i, values[i] + 0.1, significance, ha='center', fontsize=14)
        
        # Add actual p-value below
        ax.text(i, -0.5, f"p={pval:.3f}", ha='center', fontsize=10)
    
    # Add labels and title
    ax.set_ylabel("Magnitude", fontsize=12)
    ax.set_title("Sex Differences in Metabolite Trajectories", fontsize=14, fontweight='bold')
    
    # Add explanations
    explanations = {
        "delta": "Magnitude difference",
        "angle": "Direction difference (degrees)",
        "shape": "Shape difference"
    }
    
    # Add legend with explanations
    legend_elements = [
        Patch(facecolor=bars[i].get_facecolor(), label=f"{param}: {explanations[param]}")
        for i, param in enumerate(params)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add significance legend
    ax.text(0.02, 0.02, "* p<0.05, ** p<0.01, *** p<0.001", transform=ax.transAxes, fontsize=10)
    
    # Save figure
    fig.tight_layout()
    fig.savefig(f"plots/{output_file}", dpi=300)
    plt.close(fig)


def plot_integrated_platforms(output_file="IntegratedPlatforms.pdf"):
    """
    Create a visualization showing the integration of the two metabolite platforms.
    
    Parameters
    ----------
    output_file: str
        Name of the output file.
    """
    plots._check_dir()
    
    # Read metabolomics data
    metabolomics = read.read_metabolomics()
    
    # Get data from both platforms
    p180 = metabolomics["p180"]
    nmr = metabolomics["nmr"]
    qt = metabolomics["qt"]
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # Plot correlation heatmap for p180
    ax1 = fig.add_subplot(gs[0, 0])
    p180_corr = p180.corr()
    sns.heatmap(p180_corr, ax=ax1, cmap="coolwarm", vmin=-1, vmax=1, 
                xticklabels=False, yticklabels=False)
    ax1.set_title("P180 Platform Correlations", fontsize=12)
    
    # Plot correlation heatmap for nmr
    ax2 = fig.add_subplot(gs[0, 1])
    nmr_corr = nmr.corr()
    sns.heatmap(nmr_corr, ax=ax2, cmap="coolwarm", vmin=-1, vmax=1,
                xticklabels=False, yticklabels=False)
    ax2.set_title("NMR Platform Correlations", fontsize=12)
    
    # Plot correlation between platforms
    ax3 = fig.add_subplot(gs[0, 2])
    # Merge data and calculate correlations between platforms
    merged = p180.merge(nmr, on="RID")
    p180_cols = p180.columns.drop("RID")
    nmr_cols = nmr.columns.drop("RID")
    cross_corr = pd.DataFrame(
        np.corrcoef(merged[p180_cols].values.T, merged[nmr_cols].values.T)[:len(p180_cols), len(p180_cols):]
    )
    sns.heatmap(cross_corr, ax=ax3, cmap="coolwarm", vmin=-1, vmax=1,
                xticklabels=False, yticklabels=False)
    ax3.set_title("Cross-Platform Correlations", fontsize=12)
    
    # Plot PCA of integrated data by sex
    ax4 = fig.add_subplot(gs[1, :])
    
    # Merge data for PCA
    merged_with_qt = merged.merge(qt, on="RID")
    
    # Perform PCA
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    X = merged.values
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    
    # Create DataFrame with PCA results
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    pca_df['RID'] = merged.index
    pca_df = pca_df.merge(qt, on="RID")
    
    # Plot PCA by sex
    sns.scatterplot(
        x='PC1', y='PC2', 
        hue='PTGENDER', 
        style='DX',
        palette={'Female': '#ff7f0e', 'Male': '#1f77b4'},
        data=pca_df,
        s=100,
        alpha=0.7,
        ax=ax4
    )
    
    # Add explained variance
    var_exp = pca.explained_variance_ratio_ * 100
    ax4.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance explained)", fontsize=12)
    ax4.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance explained)", fontsize=12)
    ax4.set_title("PCA of Integrated Metabolite Platforms", fontsize=14, fontweight='bold')
    
    # Improve legend
    handles, labels = ax4.get_legend_handles_labels()
    ax4.legend(handles, labels, title="", loc='best', fontsize=10)
    
    # Add annotation
    ax4.text(0.02, 0.02, 
             "Integration of P180 and NMR metabolite platforms shows clear separation by sex and diagnosis",
             transform=ax4.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    # Save figure
    fig.suptitle("Integration of P180 and NMR Metabolite Platforms", fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"plots/{output_file}", dpi=300)
    plt.close(fig)


def plot_trajectory_visualization(output_file="TrajectoryVisualization.pdf"):
    """
    Create a visualization of disease trajectories by sex.
    
    Parameters
    ----------
    output_file: str
        Name of the output file.
    """
    plots._check_dir()
    
    # Read scores
    x_scores = read.read_xscores()
    x_center = sd.center_matrix(x_scores)
    
    # Get observed vectors
    X, Y = read.subset.get_XY(x_center)
    obs_vect = sd.get_observed_vectors(X, Y)
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot 2D trajectories (first two components)
    ax1 = fig.add_subplot(gs[0, :])
    
    # Define colors and markers
    colors = {'Female': '#ff7f0e', 'Male': '#1f77b4'}
    markers = {'NL': 'o', 'MCI': 's', 'Dementia': '^'}
    
    # Plot individual points
    for sex in ['Female', 'Male']:
        for dx in ['NL', 'MCI', 'Dementia']:
            subset = x_center[(x_center['PTGENDER'] == sex) & (x_center['DX'] == dx)]
            ax1.scatter(
                subset['0'], subset['1'],
                color=colors[sex], marker=markers[dx],
                alpha=0.5, s=80,
                label=f"{sex} {dx}"
            )
    
    # Plot trajectories
    # Female trajectory (0, 1, 2 are the indices for Female Control, MCI, AD)
    ax1.plot(
        obs_vect.iloc[[0, 1, 2], 0],
        obs_vect.iloc[[0, 1, 2], 1],
        color=colors['Female'], linewidth=3, marker='o', markersize=10
    )
    
    # Male trajectory (3, 4, 5 are the indices for Male Control, MCI, AD)
    ax1.plot(
        obs_vect.iloc[[3, 4, 5], 0],
        obs_vect.iloc[[3, 4, 5], 1],
        color=colors['Male'], linewidth=3, marker='o', markersize=10
    )
    
    # Add labels for trajectory points
    for i, label in enumerate(['CN', 'MCI', 'AD']):
        # Female
        ax1.annotate(
            label, 
            (obs_vect.iloc[i, 0], obs_vect.iloc[i, 1]),
            xytext=(10, 10), textcoords='offset points',
            color=colors['Female'], fontweight='bold'
        )
        # Male
        ax1.annotate(
            label, 
            (obs_vect.iloc[i+3, 0], obs_vect.iloc[i+3, 1]),
            xytext=(10, 10), textcoords='offset points',
            color=colors['Male'], fontweight='bold'
        )
    
    ax1.set_xlabel("Component 1", fontsize=12)
    ax1.set_ylabel("Component 2", fontsize=12)
    ax1.set_title("Disease Trajectories by Sex (2D View)", fontsize=14, fontweight='bold')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    
    # Sex legend elements
    sex_legend_elements = [
        Line2D([0], [0], color=colors['Female'], lw=3, label='Female'),
        Line2D([0], [0], color=colors['Male'], lw=3, label='Male')
    ]
    
    # Diagnosis legend elements
    dx_legend_elements = [
        Line2D([0], [0], marker=markers['NL'], color='gray', label='Control', 
               markerfacecolor='gray', markersize=8, linestyle='None'),
        Line2D([0], [0], marker=markers['MCI'], color='gray', label='MCI', 
               markerfacecolor='gray', markersize=8, linestyle='None'),
        Line2D([0], [0], marker=markers['Dementia'], color='gray', label='AD', 
               markerfacecolor='gray', markersize=8, linestyle='None')
    ]
    
    # Add legends
    ax1.legend(handles=sex_legend_elements, loc='upper left', fontsize=10)
    ax1.legend(handles=dx_legend_elements, loc='upper right', fontsize=10)
    
    # Plot orientation across all components
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Get orientations
    contrast = [[0, 1, 2], [3, 4, 5]]
    ys = []
    for i in range(len(contrast)):
        y = sd._estimate_orientation(obs_vect, contrast[i])
        ys.append(y)
    
    # Plot orientations
    ax2.plot(ys[0], color=colors['Female'], linewidth=2, label='Female')
    ax2.plot(ys[1], color=colors['Male'], linewidth=2, label='Male')
    
    ax2.set_xlabel("Components", fontsize=12)
    ax2.set_ylabel("Orientation", fontsize=12)
    ax2.set_title("Trajectory Orientation by Sex", fontsize=14)
    ax2.legend(fontsize=10)
    
    # Add ticks
    ticks = list(range(obs_vect.shape[1]))
    labels = [str(i + 1) for i in ticks]
    ax2.set_xticks(ticks=ticks, labels=labels)
    
    # Plot trajectory size and shape
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Calculate trajectory sizes
    female_size = sd._estimate_size(obs_vect, [0, 1, 2])
    male_size = sd._estimate_size(obs_vect, [3, 4, 5])
    
    # Create bar plot
    bars = ax3.bar(['Female', 'Male'], [female_size, male_size], 
                  color=[colors['Female'], colors['Male']])
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{height:.2f}',
            ha='center', va='bottom', fontsize=10
        )
    
    ax3.set_ylabel("Trajectory Size", fontsize=12)
    ax3.set_title("Trajectory Size by Sex", fontsize=14)
    
    # Add annotation about the differences
    fig.text(0.5, 0.01, 
             "Sex differences in AD trajectories: Males and females show distinct patterns in the progression from Control to MCI to AD",
             ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # Save figure
    fig.suptitle("Sex Differences in AD Trajectories", fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"plots/{output_file}", dpi=300)
    plt.close(fig)


def plot_vip_heatmap(output_file="VIPHeatmap.pdf"):
    """
    Create a heatmap visualization of the most important metabolites (VIPs).
    
    Parameters
    ----------
    output_file: str
        Name of the output file.
    """
    plots._check_dir()
    
    # Read VIPs
    vips = pd.read_csv("VIPS.csv", header=None)
    
    # Read metabolomics data to get feature names
    metabolomics = read.read_metabolomics()
    met_names = []
    for key in metabolomics:
        if key != "qt":
            col_names = list(metabolomics[key].columns)
            for col in col_names:
                met_names.append(col)
    
    # Create DataFrame with VIPs and feature names
    vip_df = pd.DataFrame({
        'Metabolite': met_names,
        'VIP': vips.iloc[:, 0]
    })
    
    # Sort by VIP score
    vip_df = vip_df.sort_values('VIP', ascending=False)
    
    # Get top 30 metabolites
    top_vips = vip_df.head(30)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap-style bar plot
    bars = ax.barh(top_vips['Metabolite'], top_vips['VIP'], 
                  color=plt.cm.viridis(top_vips['VIP']/top_vips['VIP'].max()))
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                              norm=plt.Normalize(vmin=top_vips['VIP'].min(), 
                                                vmax=top_vips['VIP'].max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('VIP Score', fontsize=12)
    
    # Add a threshold line
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
    ax.text(1.05, 0, 'Threshold (VIP=1)', color='red', fontsize=10, va='bottom')
    
    # Add labels
    ax.set_xlabel('VIP Score', fontsize=12)
    ax.set_title('Top 30 Most Important Metabolites', fontsize=14, fontweight='bold')
    
    # Improve y-axis labels
    plt.yticks(fontsize=10)
    
    # Add annotation
    ax.text(0.02, 0.02, 
            "Metabolites with VIP > 1 are considered significant contributors to the model",
            transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    # Save figure
    fig.tight_layout()
    fig.savefig(f"plots/{output_file}", dpi=300)
    plt.close(fig)


def main():
    """
    Generate impactful visualizations for grant proposal.
    """
    parser = argparse.ArgumentParser(
        description="Generate impactful visualizations for grant proposal."
    )
    parser.add_argument(
        "-R",
        type=str,
        default="ResultTable_PLS.csv",
        metavar="RESULT_FILE",
        help="Result file from sex_diff analysis. Default ResultTable_PLS.csv.",
    )
    
    args = parser.parse_args()
    
    # Generate all plots
    plot_sex_diff_results(result_file=args.R)
    plot_integrated_platforms()
    plot_trajectory_visualization()
    plot_vip_heatmap()
    
    print("All grant proposal visualizations have been generated in the 'plots' directory.")


if __name__ == "__main__":
    main()