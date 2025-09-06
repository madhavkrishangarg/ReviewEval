import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols
import statsmodels.api as sm
from itertools import combinations

# Increase font size for all plots
plt.rcParams.update({
    'font.size': 14,           # Base font size
    'axes.titlesize': 24,      # Title font size
    'axes.labelsize': 20,      # Axis label font size 
    'xtick.labelsize': 16,     # X-axis tick labels
    'ytick.labelsize': 16,     # Y-axis tick labels
    'legend.fontsize': 16,     # Legend font size
    'figure.titlesize': 26,    # Figure title size
})

# Create separate directories for different types of results
base_results_dir = 'results/correlation_analysis'
pearson_dir = os.path.join(base_results_dir, 'pearson')
cca_dir = os.path.join(base_results_dir, 'cca')
multiple_corr_dir = os.path.join(base_results_dir, 'multiple_correlation')
averaging_dir = os.path.join(base_results_dir, 'averaging_analysis')

# Create directories if they don't exist
for directory in [base_results_dir, pearson_dir, cca_dir, multiple_corr_dir, averaging_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Directory containing merged CSV files
merged_dir = 'results/merged'

# Get list of all CSV files in the merged directory, excluding expert CSV
merged_files = [f for f in os.listdir(merged_dir) if f.endswith('.csv') and "expert" not in f.lower()]

# List to store all data for correlation analysis
all_data = []

# Keep track of row counts
total_rows_before_filtering = 0
total_rows_after_filtering = 0

# Process each file
for file in merged_files:
    file_path = os.path.join(merged_dir, file)
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        total_rows_before_filtering += len(df)
        
        # Filter out rows with 'AVERAGE' or 'STD_DEV'
        df = df[~df['paper_number'].astype(str).str.contains('AVERAGE|STD_DEV', case=False)]
        total_rows_after_filtering += len(df)
        
        # Check if all required columns exist
        required_columns = ['depth_score', 'actionable_insights', 'adherence_score', 'coverage', 
                           'semantic_similarity', 'factual_correctness']
        
        # Only process files that have all the required columns
        if all(column in df.columns for column in required_columns):
            # Add model name (extracted from filename) for reference
            df['model'] = file.replace('.csv', '')
            # Append to all_data
            all_data.append(df)
            print(f"Processed {file} - {len(df)} rows")
        else:
            missing = [col for col in required_columns if col not in df.columns]
            print(f"Skipping {file}: Missing columns {missing}")
    
    except Exception as e:
        print(f"Error processing {file}: {e}")

# Number of files in all_data
print(f"\nNumber of files processed: {len(all_data)}")
print(f"Total rows before filtering AVERAGE/STD_DEV: {total_rows_before_filtering}")
print(f"Total rows after filtering AVERAGE/STD_DEV: {total_rows_after_filtering}")

if all_data:
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Total rows in combined dataset: {len(combined_data)}")
    
    # Required columns for correlation
    metrics = ['depth_score', 'actionable_insights', 'adherence_score', 'coverage', 
              'semantic_similarity', 'factual_correctness']
    
    # Verify we're working with individual data points, not averages
    print("\nData point verification (first 5 rows):")
    print(combined_data[['paper_number', 'conference'] + metrics].head())
    
    # Check for any remaining "AVERAGE" or "STD_DEV" values
    avg_rows = combined_data[combined_data['paper_number'].astype(str).str.contains('AVERAGE|STD_DEV', case=False)]
    if len(avg_rows) > 0:
        print(f"\nWARNING: {len(avg_rows)} rows with AVERAGE/STD_DEV still present - removing them")
        combined_data = combined_data[~combined_data['paper_number'].astype(str).str.contains('AVERAGE|STD_DEV', case=False)]
    else:
        print("\nConfirmed: No AVERAGE or STD_DEV rows present in the dataset")
    
    print(f"\nPerforming correlation analysis on {len(combined_data)} individual data points...")
    
    # ==================== 1. PEARSON CORRELATION ====================
    print("\n========== PEARSON CORRELATION ANALYSIS ==========")
    # Calculate correlation matrix
    correlation_matrix = combined_data[metrics].corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
    
    # Save correlation matrix to CSV
    correlation_matrix.to_csv(os.path.join(pearson_dir, 'correlation_matrix.csv'))
    
    # Calculate p-values for correlations
    p_values = pd.DataFrame(np.zeros((len(metrics), len(metrics))), 
                           index=metrics, columns=metrics)
    
    # Count the number of pairs
    num_pairs = 0
    
    for i, metric1 in enumerate(metrics):
        for j, metric2 in enumerate(metrics):
            if i != j:  # Skip self correlation
                corr, p_value = stats.pearsonr(combined_data[metric1], combined_data[metric2])
                p_values.loc[metric1, metric2] = p_value
                num_pairs += 1
    
    print(f"Analyzed {num_pairs} pairs of metrics (all pairwise combinations)")
    
    # Save p-values to CSV
    p_values.to_csv(os.path.join(pearson_dir, 'correlation_pvalues.csv'))
    print("\nP-values:")
    print(p_values)
    
    # Create a heatmap for the correlation matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
               annot_kws={"size": 24}, fmt=".2f")
    plt.title('Correlation Matrix of Metrics', fontsize=32)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(pearson_dir, 'correlation_heatmap.png'), dpi=300)
    
    # Generate scatter plots for each pair of metrics
    plt.figure(figsize=(20, 16))
    with sns.plotting_context(rc={"font.size": 18, "axes.titlesize": 24, "axes.labelsize": 22}):
        scatter_grid = sns.pairplot(combined_data[metrics], diag_kind='kde', height=3.5, aspect=1.2)
        for ax in scatter_grid.axes.flatten():
            ax.tick_params(labelsize=18)
        plt.suptitle('Scatter Plots of Metrics', y=1.02, fontsize=36)
        plt.tight_layout()
    plt.savefig(os.path.join(pearson_dir, 'metric_scatter_plots.png'), dpi=300)
    
    # ==================== 2. CANONICAL CORRELATION ANALYSIS ====================
    print("\n========== CANONICAL CORRELATION ANALYSIS ==========")
    
    # Create a DataFrame to store CCA results
    cca_results = []
    
    # Count total combinations
    total_cca_combinations = 0
    for k in range(1, 4):
        total_cca_combinations += len(list(combinations(range(len(metrics)), k)))
    print(f"Analyzing {total_cca_combinations} CCA combinations (all ways to split 6 metrics into 2 sets)")
    
    # Consider different ways to split the metrics (from 1 vs 5 to 3 vs 3)
    for k in range(1, 4):
        # Get all combinations of k metrics for the first set
        for set1_indices in combinations(range(len(metrics)), k):
            set1_metrics = [metrics[i] for i in set1_indices]
            set2_metrics = [m for m in metrics if m not in set1_metrics]
            
            if len(set1_metrics) > 0 and len(set2_metrics) > 0:
                # Prepare the data
                X = combined_data[set1_metrics].values
                Y = combined_data[set2_metrics].values
                
                # Fit CCA
                n_components = min(len(set1_metrics), len(set2_metrics))
                cca = CCA(n_components=n_components)
                
                try:
                    cca.fit(X, Y)
                    # Get the canonical correlations (we need to compute them from the transformed data)
                    X_c, Y_c = cca.transform(X, Y)
                    
                    # Calculate correlations for each component
                    can_cors = []
                    for i in range(n_components):
                        can_cor = np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
                        can_cors.append(can_cor)
                    
                    # Store the results
                    cca_results.append({
                        'set1': ', '.join(set1_metrics),
                        'set2': ', '.join(set2_metrics),
                        'n_components': n_components,
                        'canonical_correlations': can_cors
                    })
                    
                    print(f"\nCCA between {set1_metrics} and {set2_metrics}:")
                    print(f"Canonical correlations: {can_cors}")
                    
                except Exception as e:
                    print(f"Error in CCA for {set1_metrics} vs {set2_metrics}: {e}")
    
    # Save CCA results to CSV
    if cca_results:
        cca_df = pd.DataFrame(cca_results)
        # Convert canonical correlations list to separate columns
        max_components = max(cca_df['n_components'])
        for i in range(max_components):
            cca_df[f'corr_{i+1}'] = cca_df['canonical_correlations'].apply(
                lambda x: x[i] if i < len(x) else np.nan
            )
        # Drop the list column
        cca_df = cca_df.drop('canonical_correlations', axis=1)
        cca_df.to_csv(os.path.join(cca_dir, 'cca_results.csv'), index=False)
    
    # =============== NEW CODE: CCA VISUALIZATIONS ===============
    print("\nGenerating CCA visualizations...")
    
    # 1. Create a heatmap of first canonical correlations for pairs of metrics
    single_vs_rest = cca_df[cca_df['n_components'] == 1].copy()  # Use copy() to avoid SettingWithCopyWarning
    if not single_vs_rest.empty:
        # Extract single metric names
        single_vs_rest['metric'] = single_vs_rest['set1'].apply(
            lambda x: x.replace('[', '').replace(']', '').replace("'", "").strip().split(', ')[0]
        )
        
        # Create a DataFrame for the plot
        metric_values = []
        for _, row in single_vs_rest.iterrows():
            metric_values.append({
                'metric': row['metric'],
                'correlation': row['corr_1']
            })
        
        if metric_values:
            plot_df = pd.DataFrame(metric_values)
            plot_df = plot_df.sort_values(by='correlation', ascending=False)
            
            # Create a bar plot instead of a heatmap
            plt.figure(figsize=(10, 8))
            bars = plt.bar(plot_df['metric'], plot_df['correlation'], color='skyblue')
            plt.ylabel('First Canonical Correlation', fontsize=20)
            plt.title('First Canonical Correlation: Single Metric vs. All Others', fontsize=24)
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            
            # Add value labels
            for i, bar in enumerate(bars):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f"{plot_df['correlation'].iloc[i]:.3f}", ha='center', fontsize=16)
            
            plt.tight_layout()
            plt.savefig(os.path.join(cca_dir, 'single_metric_cca_barchart.png'), dpi=300)

    # 2. Visualize top canonical correlations by number of components
    # Group by the size of the first set
    plt.figure(figsize=(12, 8))
    
    # Get the top canonical correlation for each set size
    set_sizes = []
    for idx, row in cca_df.iterrows():
        try:
            set_size = len(row['set1'].split(', '))
            set_sizes.append(set_size)
        except:
            set_sizes.append(0)  # Default fallback
    
    cca_df_with_sizes = cca_df.copy()
    cca_df_with_sizes['set_size'] = set_sizes
    
    # Group by set size and get max correlation
    # Make sure the column name is consistent
    grouped = cca_df_with_sizes.groupby('set_size')['corr_1'].max().reset_index()
    grouped.columns = ['set_size', 'max_correlation']  # Renamed from 'max_corr' to 'max_correlation'
    
    # Plot
    plt.bar(grouped['set_size'], grouped['max_correlation'], color='skyblue')
    plt.xlabel('Number of Metrics in First Set', fontsize=20)
    plt.ylabel('Maximum First Canonical Correlation', fontsize=20)
    plt.title('Maximum Canonical Correlation by Set Size', fontsize=24)
    plt.xticks(grouped['set_size'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(grouped['max_correlation']):
        plt.text(grouped['set_size'].iloc[i], v + 0.01, f"{v:.3f}", ha='center', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(cca_dir, 'max_cca_by_set_size.png'), dpi=300)
    
    # 3. Visualize the top 5 strongest canonical correlations
    top_cca = cca_df.sort_values(by='corr_1', ascending=False).head(5)
    
    plt.figure(figsize=(16, 10))  # Increase figure width to accommodate longer labels
    bars = plt.bar(range(len(top_cca)), top_cca['corr_1'], color='lightblue')
    plt.xlabel('CCA Configuration', fontsize=20)
    plt.ylabel('First Canonical Correlation', fontsize=20)
    plt.title('Top 5 Strongest Canonical Correlations', fontsize=24)
    
    # Create labels with full set1 vs set2 information without truncation
    labels = []
    for _, row in top_cca.iterrows():
        labels.append(f"{row['set1']} vs\n{row['set2']}")
    
    plt.xticks(range(len(top_cca)), labels, rotation=45, ha='right', fontsize=14)
    plt.subplots_adjust(bottom=0.4)  # Increase bottom margin for long labels
    
    # Add value labels
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f"{top_cca['corr_1'].iloc[i]:.3f}", ha='center', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(cca_dir, 'top5_cca_configurations.png'), dpi=300)
    
    # ==================== 3. MULTIPLE CORRELATION ANALYSIS ====================
    print("\n========== MULTIPLE CORRELATION ANALYSIS ==========")
    
    # Create a DataFrame to store all Multiple Correlation results
    all_mc_results = []
    
    # For each metric as the target
    for target in metrics:
        mc_results = []
        
        # Get all other metrics
        other_metrics = [m for m in metrics if m != target]
        
        # For all possible combinations of predictors (from 1 to all 5)
        num_combinations = 0
        for k in range(1, len(other_metrics) + 1):
            for predictor_combination in combinations(other_metrics, k):
                num_combinations += 1
                predictors = list(predictor_combination)
                
                # Prepare the data
                X = combined_data[predictors]
                y = combined_data[target]
                
                # Create a formula for statsmodels (for the F-test)
                formula = f"{target} ~ {' + '.join(predictors)}"
                model = ols(formula, data=combined_data).fit()
                
                # Get R² and F-test p-value
                r_squared = model.rsquared
                f_pvalue = model.f_pvalue
                
                # Store results
                mc_results.append({
                    'target': target,
                    'predictors': ', '.join(predictors),
                    'num_predictors': len(predictors),
                    'r_squared': r_squared,
                    'r': np.sqrt(r_squared),  # Multiple correlation coefficient
                    'f_pvalue': f_pvalue,
                    'is_significant': f_pvalue < 0.05
                })
        
        # Find the best model (highest R²) for this target
        best_model = max(mc_results, key=lambda x: x['r_squared'])
        print(f"\nTarget: {target}")
        print(f"Analyzed {num_combinations} combinations of predictors")
        print(f"Best model: {best_model['predictors']}")
        print(f"R² = {best_model['r_squared']:.4f}")
        print(f"Multiple R = {best_model['r']:.4f}")
        print(f"F-test p-value = {best_model['f_pvalue']:.4e}")
        print(f"Is significant (p<0.05)? {'Yes' if best_model['is_significant'] else 'No'}")
        
        # Append all results for this target to the main results list
        all_mc_results.extend(mc_results)
    
    # Save Multiple Correlation results to CSV
    if all_mc_results:
        mc_df = pd.DataFrame(all_mc_results)
        mc_df.to_csv(os.path.join(multiple_corr_dir, 'multiple_correlation.csv'), index=False)
        
        # Create a separate file with just the best models for each target
        best_models = []
        for target in metrics:
            target_models = [m for m in all_mc_results if m['target'] == target]
            best_model = max(target_models, key=lambda x: x['r_squared'])
            best_models.append(best_model)
        
        pd.DataFrame(best_models).to_csv(os.path.join(multiple_corr_dir, 'best_models.csv'), index=False)
    
    # =============== NEW CODE: MULTIPLE REGRESSION VISUALIZATIONS ===============
    print("\nGenerating Multiple Regression visualizations...")
    
    # 1. Bar chart of R² values for best models for each target metric
    best_models_df = pd.DataFrame(best_models)
    
    # Sort by R-squared
    best_models_df = best_models_df.sort_values(by='r_squared', ascending=False)
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(best_models_df['target'], best_models_df['r_squared'], color='lightgreen')
    
    # Add a horizontal line at R²=0.05 for reference
    plt.axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='R²=0.05 (5% explained variance)')
    
    plt.xlabel('Target Metric', fontsize=20)
    plt.ylabel('R²', fontsize=20)
    plt.title('Explained Variance (R²) for Best Models by Target Metric', fontsize=24)
    plt.xticks(rotation=45, ha='right', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f"{bar.get_height():.3f}", ha='center', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(multiple_corr_dir, 'best_models_r_squared.png'), dpi=300)
    
    # 2. Create a visualization showing R² by number of predictors for each target
    # Create a subplot for each target
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, target in enumerate(metrics):
        # Get models for this target
        target_models = [m for m in all_mc_results if m['target'] == target]
        if not target_models:
            continue
            
        # Convert to DataFrame for easier manipulation
        target_df = pd.DataFrame(target_models)
        
        # Group by number of predictors and get the max R²
        grouped_data = target_df.groupby('num_predictors')['r_squared'].max().reset_index()
        
        # Plot
        axes[i].bar(grouped_data['num_predictors'], grouped_data['r_squared'], color='skyblue', alpha=0.7)
        axes[i].set_title(f'Target: {target}', fontsize=20)
        axes[i].set_xlabel('Number of Predictors', fontsize=18)
        axes[i].set_ylabel('Max R²', fontsize=18)
        axes[i].tick_params(axis='both', which='major', labelsize=14)
        axes[i].grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add value labels
        for j, v in enumerate(grouped_data['r_squared']):
            axes[i].text(grouped_data['num_predictors'].iloc[j], v + 0.01, f"{v:.3f}", ha='center', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(multiple_corr_dir, 'r_squared_by_num_predictors.png'), dpi=300)
    
    # 3. Create a visualization showing which predictors are used in the best model for each target
    # Initialize a matrix with zeros
    importance_matrix = pd.DataFrame(0, index=metrics, columns=metrics)
    
    # For each target metric and its best model
    for best_model in best_models:
        target = best_model['target']
        # Split the predictors string and process each predictor
        predictors_str = best_model['predictors'] 
        try:
            predictors = [p.strip() for p in predictors_str.split(',')]
            # Mark each predictor as used (1) in the importance matrix
            for predictor in predictors:
                if predictor in metrics:  # Ensure the predictor is valid
                    importance_matrix.loc[target, predictor] = 1
        except:
            print(f"Error processing predictor string: {predictors_str}")
    
    # Create a figure for the heatmap
    plt.figure(figsize=(12, 10))
    
    # Plot the heatmap
    sns.heatmap(importance_matrix, cmap='Blues', annot=True, fmt='d', cbar=False, 
               annot_kws={"size": 16})
    plt.title('Predictors Used in Best Models for Each Target', fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(multiple_corr_dir, 'predictor_importance_heatmap.png'), dpi=300)
    
    # ==================== 4. AVERAGING ANALYSIS ====================
    print("\n========== AVERAGING ANALYSIS ==========")
    
    # Calculate the unified score (average of all metrics) for each row
    combined_data['unified_score'] = combined_data[metrics].mean(axis=1)
    
    # Dictionary to store results
    averaging_results = []
    
    # For each reviewer/model in the dataset
    for model in combined_data['model'].unique():
        model_data = combined_data[combined_data['model'] == model]
        
        # Calculate the baseline average of all metrics for this model
        baseline_avg = model_data['unified_score'].mean()
        
        print(f"\nModel: {model}")
        print(f"Baseline average score: {baseline_avg:.4f}")
        
        # For each metric, calculate the average score without that metric
        for metric in metrics:
            # Calculate the average score for each row without this metric
            # First, calculate the sum of all metrics
            row_sum = model_data[metrics].sum(axis=1)
            # Then subtract the metric to be removed
            adjusted_sum = row_sum - model_data[metric]
            # Calculate the new average (dividing by number of metrics - 1)
            adjusted_avg = adjusted_sum / (len(metrics) - 1)
            
            # Calculate the average of these adjusted scores across all papers for this model
            model_adjusted_avg = adjusted_avg.mean()
            
            # Calculate the absolute and relative change
            abs_change = model_adjusted_avg - baseline_avg
            rel_change = abs_change / baseline_avg * 100
            
            # Also calculate the contribution of this metric to the unified score
            # (How much of the original unified score was due to this metric)
            contribution = (model_data[metric] / len(metrics)).mean()
            contribution_pct = contribution / baseline_avg * 100
            
            print(f"  Without {metric}: {model_adjusted_avg:.4f} (Change: {abs_change:.4f}, {rel_change:.2f}%)")
            
            # Store the results
            averaging_results.append({
                'model': model,
                'metric_removed': metric,
                'baseline_avg': baseline_avg,
                'adjusted_avg': model_adjusted_avg,
                'absolute_change': abs_change,
                'relative_change_pct': rel_change,
                'metric_contribution': contribution,
                'metric_contribution_pct': contribution_pct
            })
    
    # Convert to DataFrame and save
    avg_df = pd.DataFrame(averaging_results)
    avg_df.to_csv(os.path.join(averaging_dir, 'metric_importance.csv'), index=False)
    
    # Calculate summary statistics across all models
    metric_summary = avg_df.groupby('metric_removed').agg({
        'absolute_change': ['mean', 'std'],
        'relative_change_pct': ['mean', 'std'],
        'metric_contribution': ['mean', 'std'],
        'metric_contribution_pct': ['mean', 'std']
    })
    
    # Flatten the column hierarchy for better readability
    metric_summary.columns = ['_'.join(col).strip() for col in metric_summary.columns.values]
    metric_summary = metric_summary.reset_index()
    
    # Sort by absolute mean change (largest impact first)
    metric_summary = metric_summary.sort_values(by='absolute_change_mean', key=abs, ascending=False)
    
    # Save summary
    metric_summary.to_csv(os.path.join(averaging_dir, 'metric_importance_summary.csv'), index=False)
    
    print("\nMetric importance summary (sorted by absolute impact):")
    print(metric_summary[['metric_removed', 'absolute_change_mean', 'relative_change_pct_mean']])
    
    # Create a bar chart of metric importance
    plt.figure(figsize=(16, 10))
    
    # Sort metrics by absolute change for the plot
    plot_data = metric_summary.sort_values(by='absolute_change_mean', key=abs, ascending=False)
    
    # Plot the absolute changes
    bars = plt.bar(plot_data['metric_removed'], plot_data['absolute_change_mean'])
    
    # Color the bars based on positive/negative change
    for i, bar in enumerate(bars):
        if plot_data['absolute_change_mean'].iloc[i] < 0:
            bar.set_color('g')  # Green for metrics that decrease the score when removed
        else:
            bar.set_color('r')  # Red for metrics that increase the score when removed
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Impact of Removing Each Metric on Unified Score', fontsize=32)
    plt.xlabel('Metric Removed', fontsize=28)
    plt.ylabel('Change in Unified Score', fontsize=28)
    plt.xticks(rotation=45, ha='right', fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on the bars
    for i, v in enumerate(plot_data['absolute_change_mean']):
        plt.text(i, v + (0.001 if v >= 0 else -0.001), 
                f"{v:.4f}", ha='center', va='bottom' if v >= 0 else 'top', fontsize=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(averaging_dir, 'metric_importance_chart.png'), dpi=300)
    
    # Create a pie chart showing each metric's contribution to the unified score
    plt.figure(figsize=(16, 14))
    plt.clf()  # Clear any previous plots
    
    # Calculate average contribution percentages across all models
    contributions = avg_df.groupby('metric_removed')['metric_contribution_pct'].mean()
    
    # Custom colors with higher contrast
    colors = plt.cm.tab10(np.arange(len(contributions)))
    
    # Plot pie chart with larger text - removed shadow, added explode for better visibility
    explode = [0.02] * len(contributions)  # Slight separation between wedges
    
    plt.pie(contributions, labels=contributions.index, autopct='%1.1f%%', 
            startangle=90, shadow=False, textprops={'fontsize': 24, 'weight': 'bold'}, 
            colors=colors, wedgeprops={'linewidth': 0.5, 'edgecolor': 'black'},
            explode=explode)
    
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Average Contribution of Each Metric to Unified Score', fontsize=32, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(averaging_dir, 'metric_contribution_pie.png'), dpi=300)
    
    print("\nAnalysis complete! Results saved to separate directories:")
    print(f"  - Pearson correlations: {pearson_dir}")
    print(f"  - Canonical Correlation Analysis: {cca_dir}")
    print(f"  - Multiple Correlation Analysis: {multiple_corr_dir}")
    print(f"  - Averaging Analysis: {averaging_dir}")
else:
    print("No data was found with all required metrics.") 