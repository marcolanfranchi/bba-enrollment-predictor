import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # type: ignore
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

def plot_feature_selection_results(results_df, output_path='plots/feature_selection.png'):
    """Plot feature selection results showing performance vs number of features"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # MAE vs number of features
    axes[0, 0].plot(results_df['n_features'], results_df['avg_mae'], marker='o', linewidth=2, markersize=8)
    axes[0, 0].axhline(y=results_df['avg_mae'].min(), color='r', linestyle='--', alpha=0.5, label='Best MAE')
    axes[0, 0].set_xlabel('Number of Features', fontsize=11)
    axes[0, 0].set_ylabel('Average MAE', fontsize=11)
    axes[0, 0].set_title('MAE vs Number of Features', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # R² vs number of features
    axes[0, 1].plot(results_df['n_features'], results_df['avg_r2'], marker='o', linewidth=2, markersize=8, color='green')
    axes[0, 1].axhline(y=results_df['avg_r2'].max(), color='r', linestyle='--', alpha=0.5, label='Best R²')
    axes[0, 1].set_xlabel('Number of Features', fontsize=11)
    axes[0, 1].set_ylabel('Average R²', fontsize=11)
    axes[0, 1].set_title('R² vs Number of Features', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # MAPE vs number of features
    axes[1, 0].plot(results_df['n_features'], results_df['avg_mape'], marker='o', linewidth=2, markersize=8, color='orange')
    axes[1, 0].axhline(y=results_df['avg_mape'].min(), color='r', linestyle='--', alpha=0.5, label='Best MAPE')
    axes[1, 0].set_xlabel('Number of Features', fontsize=11)
    axes[1, 0].set_ylabel('Average MAPE (%)', fontsize=11)
    axes[1, 0].set_title('MAPE vs Number of Features', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # RMSE vs number of features
    axes[1, 1].plot(results_df['n_features'], results_df['avg_rmse'], marker='o', linewidth=2, markersize=8, color='purple')
    axes[1, 1].axhline(y=results_df['avg_rmse'].min(), color='r', linestyle='--', alpha=0.5, label='Best RMSE')
    axes[1, 1].set_xlabel('Number of Features', fontsize=11)
    axes[1, 1].set_ylabel('Average RMSE', fontsize=11)
    axes[1, 1].set_title('RMSE vs Number of Features', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_top_features(feature_importance, n=20, output_path='plots/top_features.png'):
    """Bar plot of top N most important features"""
    top_n = feature_importance.head(n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(top_n)), top_n['importance'], color='steelblue')
    ax.set_yticks(range(len(top_n)))
    ax.set_yticklabels(top_n['feature'], fontsize=10)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Top {n} Most Important Features', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Invert y-axis to show highest importance at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

# Train + Evaluate One CV Fold
def train_and_evaluate(train_df, test_df, term, course_max_dict, feature_subset=None):
    """
    Train and evaluate model for one CV fold.
    
    Args:
        feature_subset: List of feature names to use. If None, uses all features.
    """
    y_train_raw = train_df["enrollment_count"].values
    y_train = np.log1p(y_train_raw)
    X_train = train_df.drop(columns=["enrollment_count"])
    
    y_test_raw = test_df["enrollment_count"].values
    y_test = np.log1p(y_test_raw)
    X_test = test_df.drop(columns=["enrollment_count"])
    
    meta_test = test_df[["CourseTerm", "CatalogNbr"]].copy()
    
    X_train = X_train.drop(columns=["CatalogNbr", "CourseTerm"], errors="ignore")
    X_test  = X_test.drop(columns=["CatalogNbr", "CourseTerm"], errors="ignore")
    
    # Apply feature selection if specified
    if feature_subset is not None:
        available_features = [f for f in feature_subset if f in X_train.columns]
        X_train = X_train[available_features]
        X_test = X_test[available_features]
    
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
    
    # Get feature importance if using all features
    feature_importance = None
    if feature_subset is None:
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    
    return results_df, mae, rmse, r2, mape, baseline_mae, feature_importance

def get_feature_importance_from_full_model(df):
    """
    Train a full model on all data to get feature importance rankings.
    Uses the most recent term as validation to avoid overfitting.
    """
    df = df.sort_values("CourseTerm")
    unique_terms = sorted(df["CourseTerm"].unique())
    
    if len(unique_terms) < 2:
        return None
    
    # Use all but last term for training
    train_df = df[df["CourseTerm"] < unique_terms[-1]]
    test_df = df[df["CourseTerm"] == unique_terms[-1]]
    
    if len(train_df) < 50 or len(test_df) < 5:
        return None
    
    course_max_dict = df.groupby("CatalogNbr")["enrollment_count"].max().to_dict()
    _, _, _, _, _, _, feature_importance = train_and_evaluate(
        train_df, test_df, unique_terms[-1], course_max_dict, feature_subset=None
    )
    
    if feature_importance is not None:
        print("\nTop 20 Most Important Features:")
        print("-" * 70)
        for idx, row in feature_importance.head(20).iterrows():
            print(f"  {idx+1:2d}. {row['feature']:40s} (importance: {row['importance']:.4f})")
        print("-" * 70)
    
    return feature_importance

def evaluate_feature_subsets(df, feature_importance, n_features_list=[10, 15, 20, 25, 30, 50, 100]):
    """
    Evaluate models with different numbers of top features.
    
    Args:
        df: Prepared dataframe
        feature_importance: DataFrame with feature importance rankings
        n_features_list: List of feature counts to test
    """
    if feature_importance is None or len(feature_importance) == 0:
        print("Warning: Could not get feature importance. Skipping feature selection.")
        return None
    
    # Get top features
    top_features = feature_importance['feature'].tolist()
    max_features = len(top_features)
    
    # Adjust n_features_list to not exceed available features
    n_features_list = [n for n in n_features_list if n <= max_features]
    n_features_list.append(max_features)  # Always include "all features"
    n_features_list = sorted(list(set(n_features_list)))
    
    print("\n" + "="*70)
    print("FEATURE SELECTION ANALYSIS")
    print("="*70)
    print(f"Testing models with {n_features_list} features")
    
    course_max_dict = df.groupby("CatalogNbr")["enrollment_count"].max().to_dict()
    unique_terms = sorted(df["CourseTerm"].unique())
    
    feature_selection_results = []
    
    for n_features in n_features_list:
        if n_features == max_features:
            feature_subset = None  # Use all features
            feature_names = "all_features"
        else:
            feature_subset = top_features[:n_features]
            feature_names = f"top_{n_features}"
        
        print(f"\nEvaluating with {feature_names}...")
        
        all_results = []
        metrics_list = []
        
        for term in unique_terms:
            test_df = df[df["CourseTerm"] == term]
            train_df = df[df["CourseTerm"] < term]
            
            if len(test_df) < 5 or len(train_df) < 50:
                continue
            
            results_df, mae, rmse, r2, mape, baseline_mae, _ = train_and_evaluate(
                train_df, test_df, term, course_max_dict, feature_subset=feature_subset
            )
            
            all_results.append(results_df)
            metrics_list.append({
                "Term": term, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape
            })
        
        if len(metrics_list) == 0:
            continue
        
        metrics_df = pd.DataFrame(metrics_list)
        avg_mae = metrics_df['MAE'].mean()
        avg_rmse = metrics_df['RMSE'].mean()
        avg_r2 = metrics_df['R2'].mean()
        avg_mape = metrics_df['MAPE'].mean()
        
        feature_selection_results.append({
            'n_features': n_features,
            'feature_set': feature_names,
            'avg_mae': avg_mae,
            'avg_rmse': avg_rmse,
            'avg_r2': avg_r2,
            'avg_mape': avg_mape
        })
        
        print(f"  Average MAE: {avg_mae:.2f}, R²: {avg_r2:.4f}, MAPE: {avg_mape:.2f}%")
    
    # Find best feature count
    if feature_selection_results:
        fs_results_df = pd.DataFrame(feature_selection_results)
        best_idx = fs_results_df['avg_mae'].idxmin()
        best_config = fs_results_df.loc[best_idx]
        
        print("\n" + "-"*70)
        print("BEST FEATURE CONFIGURATION:")
        print(f"  Features: {best_config['feature_set']} ({best_config['n_features']} features)")
        print(f"  MAE: {best_config['avg_mae']:.2f}")
        print(f"  R²: {best_config['avg_r2']:.4f}")
        print(f"  MAPE: {best_config['avg_mape']:.2f}%")
        print("\nFeatures selected for final model:")
        print("-" * 70)
        selected_features = top_features[:best_config['n_features']]
        for idx, feat in enumerate(selected_features, 1):
            feat_importance = feature_importance[feature_importance['feature'] == feat]['importance'].values[0]
            print(f"  {idx:2d}. {feat:40s} (importance: {feat_importance:.4f})")
        print("="*70)
        
        # Save results
        fs_results_df.to_csv("data/feature_selection_results.csv", index=False)
        feature_importance.to_csv("data/feature_importance.csv", index=False)
        
        print("\nSaved feature selection results to: data/feature_selection_results.csv")
        print("Saved feature importance rankings to: data/feature_importance.csv")
        
        # Generate plots
        plot_feature_selection_results(fs_results_df)
        plot_top_features(feature_importance, n=20)
        
        return best_config['n_features'], top_features[:best_config['n_features']]
    
    return None, None

# Main Cross-Term CV Loop
def main():
    df = pd.read_csv("data/enrolment_counts.csv")
    df = prepare_data(df)
    df = df.sort_values("CourseTerm")
    
    # Step 1: Get feature importance from full model
    print("\n" + "="*70)
    print("STEP 1: GETTING FEATURE IMPORTANCE")
    print("="*70)
    feature_importance = get_feature_importance_from_full_model(df)
    
    # Step 2: Evaluate different feature subsets
    best_n_features, best_features = evaluate_feature_subsets(df, feature_importance)
    
    # Step 3: Train final model with best feature set (or all features if selection skipped)
    print("\n" + "="*70)
    print("STEP 2: FINAL MODEL TRAINING")
    print("="*70)
    if best_features is not None and feature_importance is not None:
        print(f"Using best feature set: {len(best_features)} features")
        print("\nFinal model features:")
        print("-" * 70)
        for idx, feat in enumerate(best_features, 1):
            feat_imp_row = feature_importance[feature_importance['feature'] == feat]
            if len(feat_imp_row) > 0:
                feat_importance_val = feat_imp_row['importance'].values[0]
                print(f"  {idx:2d}. {feat:40s} (importance: {feat_importance_val:.4f})")
            else:
                print(f"  {idx:2d}. {feat:40s}")
        print("="*70)
        feature_subset = best_features
    else:
        print("Using all features")
        if feature_importance is not None:
            print(f"\nTotal features available: {len(feature_importance)}")
            print("\nTop 10 features by importance:")
            print("-" * 70)
            for idx, row in feature_importance.head(10).iterrows():
                print(f"  {idx+1:2d}. {row['feature']:40s} (importance: {row['importance']:.4f})")
            print("="*70)
        feature_subset = None
    
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
        
        results_df, mae, rmse, r2, mape, baseline_mae, _ = train_and_evaluate(
            train_df, test_df, term, course_max_dict, feature_subset=feature_subset
        )
        
        all_results.append(results_df)
        metrics.append({
            "Term": term, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape,
            "Baseline_MAE": baseline_mae
        })
    
    # Combine all results
    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df.to_csv("data/demand_predictions_cv.csv", index=False)
    
    for term in unique_terms:
        test_df = df[df["CourseTerm"] == term]
        train_df = df[df["CourseTerm"] < term]
        
        if len(test_df) < 5 or len(train_df) < 50:
            print(f"Skipping term {term} (not enough data)")
            continue
        
        results_df, mae, rmse, r2, mape, baseline_mae, _ = train_and_evaluate(
            train_df, test_df, term, course_max_dict, feature_subset=feature_subset
        )
        
        all_results.append(results_df)
        metrics.append({
            "Term": term, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape,
            "Baseline_MAE": baseline_mae
        })
        
        # Print predictions for this term
        print("\n" + "="*70)
        print(f"TERM {term} PREDICTIONS")
        print("="*70)
        print(results_df[['CatalogNbr', 'Actual', 'Predicted', 'Error', 'Abs_Error', 'Pct_Error']].to_string(index=False))
        print("\n" + "-"*70)
        print(f"Term {term} Metrics:")
        print(f"  MAE:           {mae:.2f}")
        print(f"  RMSE:          {rmse:.2f}")
        print(f"  R²:            {r2:.4f}")
        print(f"  MAPE:          {mape:.2f}%")
        print(f"  Baseline MAE:  {baseline_mae:.2f}")
        print("="*70)
    
    # Save final feature list if feature selection was used
    if feature_subset is not None and feature_importance is not None:
        importance_values = []
        for f in feature_subset:
            feat_imp_row = feature_importance[feature_importance['feature'] == f]
            if len(feat_imp_row) > 0:
                importance_values.append(feat_imp_row['importance'].values[0])
            else:
                importance_values.append(0.0)
        
        final_features_df = pd.DataFrame({
            'feature': feature_subset,
            'rank': range(1, len(feature_subset) + 1),
            'importance': importance_values
        })
        final_features_df.to_csv("data/final_model_features.csv", index=False)
        print("\nSaved final model features to: data/final_model_features.csv")
        print(f"  Total features in final model: {len(feature_subset)}")
    elif feature_importance is not None:
        # Save all features if using all features
        feature_importance.to_csv("data/final_model_features.csv", index=False)
        print("\nSaved all features to: data/final_model_features.csv")
        print(f"  Total features in final model: {len(feature_importance)}")
    
    # Create final dashboard-ready CSV with predictions
    # Load original aggregated data to get all features
    df_original = pd.read_csv("data/enrolment_counts.csv")
    
    # Ensure CatalogNbr and CourseTerm have matching types for merge
    # Convert CatalogNbr to string in both dataframes to match original format
    all_results_df['CatalogNbr'] = all_results_df['CatalogNbr'].astype(str)
    df_original['CatalogNbr'] = df_original['CatalogNbr'].astype(str)
    all_results_df['CourseTerm'] = all_results_df['CourseTerm'].astype(str)
    df_original['CourseTerm'] = df_original['CourseTerm'].astype(str)
    
    # Merge predictions with original data
    dashboard_df = df_original.merge(
        all_results_df[['CourseTerm', 'CatalogNbr', 'Predicted', 'Actual', 'Error', 'Abs_Error', 'Pct_Error', 'CV_Term']],
        on=['CourseTerm', 'CatalogNbr'],
        how='inner'
    )
    
    # Rename enrollment_count to Actual_Enrollment for clarity, keep Predicted
    dashboard_df = dashboard_df.rename(columns={'enrollment_count': 'Actual_Enrollment'})
    
    # Select key columns for dashboarding (keep all original features + predictions)
    dashboard_df.to_csv("data/enrollment_predictions_dashboard.csv", index=False)
    print("\nSaved dashboard-ready CSV: data/enrollment_predictions_dashboard.csv")
    print(f"  Shape: {dashboard_df.shape}")
    print(f"  Columns: {len(dashboard_df.columns)}")
    
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