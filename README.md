# BBA Enrollment Predictor

Warm-up project for BUS 439 - Analytics Project. 

Our pipeline includes data cleaning, pre-processing, and model training. Given a course name, its term, and a student, the final model is capable of predicting whether or not a student will enroll in a future upper-division BBA course.

Our model is useful for estimating future course demand, ultimately supporting academic planning decisions such as scheduling, capacity planning, and resource allocation.

## Getting Started

### Pre-requisites

Make sure you have Python 3.13.7 installed.

If you don’t have Python installed yet:

1. Go to the official Python website: [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Download Python 3.13.7 for your operating system.
3. During installation (especially on Windows), make sure to check: “Add Python to PATH”

You can verify your installation by running:

```bash
python3 --version
```

or
```bash
python --version
```

You should see something like:

```bash
Python 3.13.7
```

### 1. Clone the repository

```bash
git clone https://github.com/marcolanfranchi/bba-enrollment-predictor.git
cd bba-enrollment-predictor
```

### 2. Create and activate a virtual environment

On MacOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```
On Windows:

```bash
python3 -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Cleaning

The first step in the pipeline is to clean and preprocess the raw enrolment data.

From the project root directory, run:

```bash
python3 model/1-clean.py
```

This script:
- Cleans the raw enrolment and historical data
- Removes unnecessary time/day columns
- Creates boolean features for course components (LAB, TUT, SEM)
- Adds campus availability features (Burnaby, Surrey, Vancouver)
- Merges historical data into the main dataset
- Outputs a cleaned dataset ready for more feature engineering
- After running the script, the processed file will be saved to: `data/enrolment_clean.csv`

This cleaned dataset is used as the input for the next steps in the modeling pipeline.

### 2. Feature Engineering

The second step performs feature engineering and aggregates the data to course-term level for demand forecasting.

From the project root directory, run:

```bash
python3 model/2-preprocess.py
```

This script:
- Joins student information from the declared student dataset
- Engineers student-level features (course progression, concentration progress, credits completed, etc.)
- Aggregates student-level data to course-term level for demand forecasting
- Adds historical demand features (lags, rolling averages, trends, seasonality)
- Adds term-level growth features (program expansion/contraction metrics)
- Adds course prerequisite and difficulty features
- Adds enrollment volatility and stability features
- Adds course popularity and market share features
- Adds capacity constraint features
- Outputs a preprocessed dataset ready for model training

After running the script, the processed file will be saved to: `data/enrolment_counts.csv`

This preprocessed dataset contains all engineered features and is used as input for model training.

### 3. Model Training

The final step trains the enrollment prediction model using cross-validation and performs feature selection.

From the project root directory, run:

```bash
python3 model/3-train.py
```

This script:
- Loads the preprocessed course-term level data
- Filters to 300-499 level courses (upper-division BBA courses)
- Performs feature importance analysis to identify the most impactful features
- Evaluates models with different feature subsets (10, 15, 20, 25, 30, 50, 100, all features)
- Selects the optimal feature set based on cross-validated performance
- Trains the final model using leave-one-term-out cross-validation
- Generates evaluation plots and metrics
- Saves predictions and model outputs**Outputs:** to the `data/` directory
- `plots/` - Directory containing evaluation visualizations
<!-- - `data/demand_predictions_cv.csv` - Cross-validation predictions for all terms
- `data/cv_metrics_by_term.csv` - Performance metrics (MAE, RMSE, R², MAPE) for each term
- `data/enrollment_predictions_dashboard.csv` - Dashboard-ready CSV with predictions and all features
- `data/feature_selection_results.csv` - Performance comparison across different feature counts
- `data/feature_importance.csv` - Feature importance rankings
- `data/final_model_features.csv` - List of features used in the final model -->
  <!-- - `actual_vs_predicted.png` - Scatter plot of predictions vs actuals
  - `residuals.png` - Residual analysis plot
  - `error_distribution.png` - Distribution of prediction errors
  - `metrics_over_time.png` - Model performance across terms
  - `top_errors.png` - Courses with highest prediction errors
  - `best_predictions.png` - Courses with best predictions
  - `all_predictions_comprehensive.png` - All predictions organized by term
  - `feature_selection.png` - Performance vs number of features
  - `top_features.png` - Top 20 most important features -->

**Model Performance:**
The model uses XGBoost regression and typically achieves:
- R² score: ~60%
- Mean Absolute Error (MAE): ~20 students per course
- Mean Absolute Percentage Error (MAPE): ~15-20%

These metrics are reasonable given the limited historical data (~7 terms) and real-world enrollment variability.

## Notes

- The pipeline processes data sequentially: cleaning → feature engineering → model training
- Each step depends on outputs from the previous step
- Feature selection automatically identifies the optimal number of features (typically 15-25 features)
- The final model focuses on upper-division BBA courses (300-499 level)
- All predictions use only information available before the target term (no data leakage)
