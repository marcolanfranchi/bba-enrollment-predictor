import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Data Preparation & Feature Engineering
def prepare_data(df):
    # Filter to 300–499 level courses
    df["CatalogNbr_numeric"] = (
        df["CatalogNbr"].astype(str)
        .str.replace("W", "", regex=False)
        .str.strip()
    )
    df["CatalogNbr_numeric"] = pd.to_numeric(df["CatalogNbr_numeric"], errors="coerce")
    df = df[(df["CatalogNbr_numeric"] >= 300) & (df["CatalogNbr_numeric"] < 500)]
    df = df.drop(columns=["CatalogNbr_numeric"])
    
    # Convert CourseTerm to season features
    df["CourseTerm_str"] = df["CourseTerm"].astype(str)
    df["is_spring"] = df["CourseTerm_str"].str.endswith("1").astype(int)
    df["is_summer"] = df["CourseTerm_str"].str.endswith("4").astype(int)
    df["is_fall"]   = df["CourseTerm_str"].str.endswith("7").astype(int)
    df = df.drop(columns=["CourseTerm_str"])
    
    # Clean CatalogNbr
    df["CatalogNbr"] = (
        df["CatalogNbr"].astype(str)
        .str.replace("W", "", regex=False)
        .str.strip()
    )
    df["CatalogNbr"] = pd.to_numeric(df["CatalogNbr"], errors="coerce").fillna(-1).astype(int)
    
    # FEATURE ENGINEERING
    # Historical mean & std per course (for normalization)
    hist_stats = df.groupby("CatalogNbr")["enrollment_count"].agg(["mean", "std"]).rename(
        columns={"mean": "hist_mean", "std": "hist_std"}
    )
    df = df.merge(hist_stats, left_on="CatalogNbr", right_index=True, how="left")
    
    # Normalize lag/rolling/trend features per course
    lag_cols = [c for c in df.columns if c.startswith("enrollment_lag") or "rolling_avg" in c or "trend" in c]
    for col in lag_cols:
        df[f"{col}_norm"] = (df[col] - df["hist_mean"]) / (df["hist_std"] + 1e-6)
    
    # Trend slope: last lag minus two lags ago
    if "enrollment_lag_1" in df.columns and "enrollment_lag_2" in df.columns:
        df["trend_slope"] = df["enrollment_lag_1"] - df["enrollment_lag_2"]
        df["trend_slope_norm"] = df["trend_slope"] / (df["hist_std"] + 1e-6)
    else:
        df["trend_slope_norm"] = 0
    
    # Drop raw lag/trend columns (keep normalized)
    df = df.drop(columns=lag_cols + ["trend_slope"], errors="ignore")
    
    # Fill remaining NA
    df = df.fillna(0)
    
    # Convert campus boolean columns to int
    campus_cols = ["offered_burnaby", "offered_surrey", "offered_van"]
    for col in campus_cols:
        if col in df.columns:
            df[col] = df[col].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0).astype(int)
    
    return df


def plot_actual_vs_predicted(results_df, output_path='plots/actual_vs_predicted.png'):
    """Scatter plot of actual vs predicted enrollments"""
    plt.figure(figsize=(10, 8))
    plt.scatter(results_df['Actual'], results_df['Predicted'], alpha=0.5, s=20)
    
    # Add perfect prediction line
    min_val = min(results_df['Actual'].min(), results_df['Predicted'].min())
    max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Enrollment', fontsize=12)
    plt.ylabel('Predicted Enrollment', fontsize=12)
    plt.title('Actual vs Predicted Enrollment (All Terms)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_residuals(results_df, output_path='plots/residuals.png'):
    """Residual plot to check for patterns"""
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['Predicted'], results_df['Error'], alpha=0.5, s=20)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    
    plt.xlabel('Predicted Enrollment', fontsize=12)
    plt.ylabel('Residual (Actual - Predicted)', fontsize=12)
    plt.title('Residual Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_error_distribution(results_df, output_path='plots/error_distribution.png'):
    """Histogram of prediction errors"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Absolute errors
    axes[0].hist(results_df['Abs_Error'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(results_df['Abs_Error'].mean(), color='r', linestyle='--', 
                    lw=2, label=f'Mean: {results_df["Abs_Error"].mean():.1f}')
    axes[0].set_xlabel('Absolute Error', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Absolute Errors', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Percentage errors
    axes[1].hist(results_df['Pct_Error'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1].axvline(results_df['Pct_Error'].mean(), color='r', linestyle='--', 
                    lw=2, label=f'Mean: {results_df["Pct_Error"].mean():.1f}%')
    axes[1].set_xlabel('Percentage Error (%)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Percentage Errors', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_metrics_over_time(metrics_df, output_path='plots/metrics_over_time.png'):
    """Line plot showing how metrics change over test terms"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # MAE over time
    axes[0, 0].plot(metrics_df['Term'], metrics_df['MAE'], marker='o', linewidth=2, label='Model MAE')
    axes[0, 0].plot(metrics_df['Term'], metrics_df['Baseline_MAE'], marker='s', linewidth=2, label='Baseline MAE')
    axes[0, 0].set_xlabel('Term', fontsize=11)
    axes[0, 0].set_ylabel('MAE', fontsize=11)
    axes[0, 0].set_title('MAE Over Time', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # RMSE over time
    axes[0, 1].plot(metrics_df['Term'], metrics_df['RMSE'], marker='o', linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Term', fontsize=11)
    axes[0, 1].set_ylabel('RMSE', fontsize=11)
    axes[0, 1].set_title('RMSE Over Time', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # R² over time
    axes[1, 0].plot(metrics_df['Term'], metrics_df['R2'], marker='o', linewidth=2, color='green')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', lw=1)
    axes[1, 0].set_xlabel('Term', fontsize=11)
    axes[1, 0].set_ylabel('R²', fontsize=11)
    axes[1, 0].set_title('R² Over Time', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # MAPE over time
    axes[1, 1].plot(metrics_df['Term'], metrics_df['MAPE'], marker='o', linewidth=2, color='purple')
    axes[1, 1].set_xlabel('Term', fontsize=11)
    axes[1, 1].set_ylabel('MAPE (%)', fontsize=11)
    axes[1, 1].set_title('MAPE Over Time', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_top_errors(results_df, n=20, output_path='plots/top_errors.png'):
    """Bar plot of courses with highest prediction errors"""
    # Get top N errors
    top_errors = results_df.nlargest(n, 'Abs_Error')[['CatalogNbr', 'Actual', 'Predicted', 'Abs_Error']].copy()
    top_errors['Course'] = 'BUS ' + top_errors['CatalogNbr'].astype(str)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(top_errors))
    width = 0.35
    
    ax.bar(x - width/2, top_errors['Actual'], width, label='Actual', alpha=0.8)
    ax.bar(x + width/2, top_errors['Predicted'], width, label='Predicted', alpha=0.8)
    
    ax.set_xlabel('Course', fontsize=12)
    ax.set_ylabel('Enrollment', fontsize=12)
    ax.set_title(f'Top {n} Courses by Prediction Error', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top_errors['Course'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_best_predictions(results_df, n=20, output_path='plots/best_predictions.png'):
    """Bar plot of courses with smallest prediction errors (best predictions)"""
    # Get top N best predictions (smallest absolute errors)
    best_preds = results_df.nsmallest(n, 'Abs_Error')[['CatalogNbr', 'Actual', 'Predicted', 'Abs_Error']].copy()
    best_preds['Course'] = 'BUS ' + best_preds['CatalogNbr'].astype(str)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(best_preds))
    width = 0.35
    
    ax.bar(x - width/2, best_preds['Actual'], width, label='Actual', alpha=0.8, color='green')
    ax.bar(x + width/2, best_preds['Predicted'], width, label='Predicted', alpha=0.8, color='lightgreen')
    
    ax.set_xlabel('Course', fontsize=12)
    ax.set_ylabel('Enrollment', fontsize=12)
    ax.set_title(f'Top {n} Best Predictions (Lowest Error)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(best_preds['Course'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_all_predictions(results_df, output_path='plots/all_predictions_comprehensive.png'):
    """
    Comprehensive plot showing ALL predictions for every course-term combination.
    Organized by term, showing all courses within each term.
    """
    # Sort by term, then by course number for logical ordering
    results_sorted = results_df.sort_values(['CV_Term', 'CatalogNbr']).reset_index(drop=True)
    
    # Create labels for x-axis
    results_sorted['Label'] = 'BUS' + results_sorted['CatalogNbr'].astype(str) + '\n(T' + results_sorted['CV_Term'].astype(str) + ')'
    
    # Calculate figure height based on number of predictions
    n_predictions = len(results_sorted)
    fig_height = max(12, n_predictions * 0.15)  # At least 12 inches, scale with data
    
    fig, ax = plt.subplots(figsize=(16, fig_height))
    
    x = np.arange(n_predictions)
    width = 0.4
    
    # Plot bars
    bars1 = ax.barh(x - width/2, results_sorted['Actual'], width, label='Actual', alpha=0.8, color='steelblue')
    bars2 = ax.barh(x + width/2, results_sorted['Predicted'], width, label='Predicted', alpha=0.8, color='coral')
    
    # Add term separators
    terms = results_sorted['CV_Term'].unique()
    term_positions = []
    for term in terms:
        term_idx = results_sorted[results_sorted['CV_Term'] == term].index
        if len(term_idx) > 0:
            mid_point = term_idx[len(term_idx)//2]
            term_positions.append(mid_point)
            # Draw separator line
            if term != terms[0]:
                ax.axhline(y=term_idx[0] - 0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add term labels on the right side
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(term_positions)
    ax2.set_yticklabels([f'Term {t}' for t in terms], fontsize=10, fontweight='bold')
    
    ax.set_yticks(x)
    ax.set_yticklabels(results_sorted['Label'], fontsize=7)
    ax.set_xlabel('Enrollment Count', fontsize=12)
    ax.set_ylabel('Course (Term)', fontsize=12)
    ax.set_title('All Predictions: Every Course, Every Term', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path} ({n_predictions} predictions)")
    print(f"  → Figure size: 16 x {fig_height:.1f} inches")

# -----------------------------
# Train + Evaluate One CV Fold
# -----------------------------
def train_and_evaluate(train_df, test_df, term, course_max_dict):
    y_train_raw = train_df["enrollment_count"].values
    y_train = np.log1p(y_train_raw)
    X_train = train_df.drop(columns=["enrollment_count"])
    
    y_test_raw = test_df["enrollment_count"].values
    y_test = np.log1p(y_test_raw)
    X_test = test_df.drop(columns=["enrollment_count"])
    
    meta_test = test_df[["CourseTerm", "CatalogNbr"]].copy()
    
    X_train = X_train.drop(columns=["CatalogNbr", "CourseTerm"], errors="ignore")
    X_test  = X_test.drop(columns=["CatalogNbr", "CourseTerm"], errors="ignore")
    
    sample_weight = y_train_raw / np.mean(y_train_raw)
    
    model = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=4,
        random_state=42
    )
    
    model.fit(X_train, y_train, sample_weight=sample_weight)
    
    y_pred = np.expm1(model.predict(X_test))
    
    # Clip predictions per course
    y_pred = np.array([np.clip(p, 0, course_max_dict.get(c, 300))
                       for p, c in zip(y_pred, meta_test["CatalogNbr"])])
    
    # Standard metrics
    mae = mean_absolute_error(y_test_raw, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred))
    r2 = r2_score(y_test_raw, y_pred)
    
    # Percentage error metrics
    pct_errors = np.abs((y_test_raw - y_pred) / (y_test_raw + 1e-6)) * 100
    mape = np.mean(pct_errors)
    
    # Baseline: predict most recent previous enrollment
    last_enrollment = train_df.groupby("CatalogNbr")["enrollment_count"].last().to_dict()
    baseline_pred = test_df["CatalogNbr"].map(last_enrollment).fillna(train_df["enrollment_count"].median())
    baseline_mae = mean_absolute_error(y_test_raw, baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_test_raw, baseline_pred))
    baseline_r2 = r2_score(y_test_raw, baseline_pred)
    
    results_df = meta_test.copy()
    results_df["Actual"] = y_test_raw
    results_df["Predicted"] = y_pred
    results_df["Baseline_Pred"] = baseline_pred
    results_df["Error"] = results_df["Actual"] - results_df["Predicted"]
    results_df["Abs_Error"] = np.abs(results_df["Error"])
    results_df["Pct_Error"] = np.abs(results_df["Error"]) / (results_df["Actual"] + 1e-6) * 100
    results_df["CV_Term"] = term
    
    print("\n" + "="*50)
    print(f"PREDICTIONS FOR TERM {term}")
    print("="*50)
    print(results_df.head(10).to_string(index=False))
    print(f"\nMetrics - Model: MAE {mae:.2f}, RMSE {rmse:.2f}, R² {r2:.4f}, MAPE {mape:.2f}%")
    print(f"Baseline: MAE {baseline_mae:.2f}, RMSE {baseline_rmse:.2f}, R² {baseline_r2:.4f}")
    
    return results_df, mae, rmse, r2, mape, baseline_mae

# Main Cross-Term CV Loop
def main():
    df = pd.read_csv("data/enrolment_counts.csv")
    df = prepare_data(df)
    df = df.sort_values("CourseTerm")
    
    course_max_dict = df.groupby("CatalogNbr")["enrollment_count"].max().to_dict()
    
    unique_terms = sorted(df["CourseTerm"].unique())
    all_results = []
    metrics = []
    
    for term in unique_terms:
        test_df = df[df["CourseTerm"] == term]
        train_df = df[df["CourseTerm"] < term]
        
        if len(test_df) < 5 or len(train_df) < 50:
            print(f"Skipping term {term} (not enough data)")
            continue
        
        results_df, mae, rmse, r2, mape, baseline_mae = train_and_evaluate(
            train_df, test_df, term, course_max_dict
        )
        
        all_results.append(results_df)
        metrics.append({
            "Term": term, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape,
            "Baseline_MAE": baseline_mae
        })
    
    # Combine all results
    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df.to_csv("data/demand_predictions_cv.csv", index=False)
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv("data/cv_metrics_by_term.csv", index=False)
    
    # Print summary
    print("\n" + "="*70)
    print("CROSS-TERM CV SUMMARY")
    print("="*70)
    print(metrics_df.to_string(index=False))
    print("\n" + "-"*70)
    print(f"Average MAE:          {metrics_df['MAE'].mean():.2f}")
    print(f"Average RMSE:         {metrics_df['RMSE'].mean():.2f}")
    print(f"Average R²:           {metrics_df['R2'].mean():.4f}")
    print(f"Average MAPE:         {metrics_df['MAPE'].mean():.2f}%")
    print(f"Average Baseline MAE: {metrics_df['Baseline_MAE'].mean():.2f}")
    print("="*70)
    
    # Generate all plots
    print("\nGenerating evaluation plots...")
    plot_actual_vs_predicted(all_results_df)
    plot_residuals(all_results_df)
    plot_error_distribution(all_results_df)
    plot_metrics_over_time(metrics_df)
    plot_top_errors(all_results_df, n=20)
    plot_best_predictions(all_results_df, n=20)
    plot_all_predictions(all_results_df)
    
    print("\nAll plots saved to 'plots/' directory")
    print("Pipeline complete!")

if __name__ == "__main__":
    main()