# Libraries
import pandas as pd

def fix_course_names(df):
    # remove random spaces
    df["CatalogNbr"] = df["CatalogNbr"].str.strip()
    df["CatalogNbr"] = df["CatalogNbr"].str.rstrip()
    return df

def add_num_bus_courses(df):
    """
    Adds a feature counting the number of completed BUS courses
    a student has taken up to (but not including) the given term
    (since # of credits per course are unavailable).
    """
    df = df.copy()
    df["CourseTerm"] = pd.to_numeric(df["CourseTerm"], errors="coerce")
    
    # Ensure proper ordering
    df = df.sort_values(by=["STD_INDEX", "CourseTerm"])
    
    # Indicator for BUS courses
    df["is_bus_course"] = df["Subject"] == "BUS"
    
    # Count BUS courses per term per student
    term_counts = (
        df.groupby(["STD_INDEX", "CourseTerm"])["is_bus_course"]
          .sum()
          .groupby(level=0)
          .cumsum()
          .shift(1)
          .reset_index(name="std_bus_courses_completed")
    )
    
    # Merge back to row-level data
    df = df.merge(
        term_counts,
        on=["STD_INDEX", "CourseTerm"],
        how="left"
    )
    
    # Replace NaNs (first term) with 0
    df["std_bus_courses_completed"] = df["std_bus_courses_completed"].fillna(0)
    
    return df

def add_course_level_counts(df):
    """
    Adds features counting courses taken at each level (100, 200, 300, 400)
    up to (but not including) the current term.
    Shows student progression pattern.
    """
    df = df.copy()
    df["CourseTerm"] = pd.to_numeric(df["CourseTerm"], errors="coerce")
    df = df.sort_values(by=["STD_INDEX", "CourseTerm"])
    
    # Extract course level from CatalogNbr
    df['temp_catalog'] = df['CatalogNbr'].astype(str).str.replace('W', '', regex=False)
    df['temp_catalog'] = pd.to_numeric(df['temp_catalog'], errors='coerce').fillna(0).astype(int)
    df['temp_level'] = df['temp_catalog'] // 100 * 100
    
    # Create indicators for each level
    for level in [100, 200, 300, 400]:
        df[f'is_{level}_level'] = (df['temp_level'] == level).astype(int)
    
    # Count cumulative courses at each level per student
    for level in [100, 200, 300, 400]:
        level_counts = (
            df.groupby(["STD_INDEX", "CourseTerm"])[f'is_{level}_level']
              .sum()
              .groupby(level=0)
              .cumsum()
              .shift(1)
              .reset_index(name=f"num_{level}_taken")
        )
        
        df = df.merge(
            level_counts,
            on=["STD_INDEX", "CourseTerm"],
            how="left"
        )
        
        df[f"num_{level}_taken"] = df[f"num_{level}_taken"].fillna(0)
        df = df.drop(columns=[f'is_{level}_level'])
    
    # Clean up temp columns
    df = df.drop(columns=['temp_catalog', 'temp_level'])
    
    return df

def add_time_since_admission(df):
    """
    Adds a feature showing how many terms have passed since admission.
    term_since_admit = CourseTerm - AdmitTerm (in coded format)
    """
    df = df.copy()
    df["CourseTerm"] = pd.to_numeric(df["CourseTerm"], errors="coerce")
    df["AdmitTerm"] = pd.to_numeric(df["AdmitTerm"], errors="coerce")
    
    # Calculate difference (approximation in terms)
    df["terms_since_admit"] = df["CourseTerm"] - df["AdmitTerm"]
    
    # Convert to rough term count (each year has 3 terms: 1, 4, 7)
    # Simplification: divide by 3 for a rough estimate
    df["terms_since_admit"] = (df["terms_since_admit"] / 3).fillna(0).astype(int)
    
    return df

def add_concentration_progress(df):
    """
    Counts how many concentration-specific courses a student has completed
    for each concentration they've declared.
    Uses course number ranges since all courses are BUS.
    """
    df = df.copy()
    df["CourseTerm"] = pd.to_numeric(df["CourseTerm"], errors="coerce")
    df = df.sort_values(by=["STD_INDEX", "CourseTerm"])
    
    # Extract numeric catalog number for range checking
    df['temp_catalog'] = df['CatalogNbr'].astype(str).str.replace('W', '', regex=False)
    df['temp_catalog'] = pd.to_numeric(df['temp_catalog'], errors='coerce').fillna(0).astype(int)
    
    # Map course number ranges to concentrations
    # Based on SFU BBA Program Requirements (Spring 2026 Calendar)
    concentration_ranges = {
        'CORECON': [(300, 300), (303, 303), (312, 312), (343, 343), (360, 360), 
                    (373, 374), (381, 381), (393, 393), (478, 478), (496, 496)],
        'COOPCON': [(225, 225), (325, 327), (425, 425)],
        'ACCTCON': [(320, 322), (420, 420), (424, 424), (426, 426), (428, 428)],
        'ENTINNCON': [(314, 314), (338, 339), (361, 361), (394, 395), (406, 406), 
                      (443, 443), (450, 450), (453, 453), (477, 477)],
        'FINCON': [(313, 315), (410, 412), (414, 414), (417, 419)],
        'HRMCON': [(374, 374), (381, 389)],
        'INBUCON': [(346, 346), (418, 418), (430, 432), (434, 435), (447, 447)],
        'MISCON': [(361, 362), (462, 466), (468, 468)],
        'OPERMGTCON': [(336, 336), (437, 437), (440, 440), (445, 445), (473, 475)],
        'MKTGCON': [(343, 343), (345, 347), (441, 449), (455, 455)],
        'STANCON': [(307, 307), (371, 371), (471, 471), (478, 479)],
    }
    
    # For each concentration, count relevant courses completed
    for conc_abbr, ranges in concentration_ranges.items():
        # Create indicator: is this course in any of the ranges for this concentration?
        df[f'is_{conc_abbr}_course'] = 0
        for min_course, max_course in ranges:
            df.loc[
                (df['temp_catalog'] >= min_course) & (df['temp_catalog'] <= max_course),
                f'is_{conc_abbr}_course'
            ] = 1
        
        # Count cumulative courses in this concentration
        conc_counts = (
            df.groupby(["STD_INDEX", "CourseTerm"])[f'is_{conc_abbr}_course']
              .sum()
              .groupby(level=0)
              .cumsum()
              .shift(1)
              .reset_index(name=f"{conc_abbr}_courses_completed")
        )
        
        df = df.merge(conc_counts, on=["STD_INDEX", "CourseTerm"], how="left")
        df[f"{conc_abbr}_courses_completed"] = df[f"{conc_abbr}_courses_completed"].fillna(0)
        df = df.drop(columns=[f'is_{conc_abbr}_course'])
    
    # Clean up temp column
    df = df.drop(columns=['temp_catalog'])
    
    return df

def splitting_concentrations(df):
    """
    Splits the Subplan column into one column per concentration and includes 
    1 if the student declared that concentration and 0 if they didn't 
    """
    
    df = df.copy()
    # Step 1: split the strings into lists
    df['Subplan'] = df['Subplan'].fillna('')
    df['Subplan_list'] = df['Subplan'].str.split(',')
    
    # Step 2: create one-hot encoding
    df_ohe = df['Subplan_list'].str.join('|').str.get_dummies()
    df_ohe = df_ohe.astype(bool)
    
    # Step 3: combine with original dataframe
    df.reset_index(drop=True, inplace=True)
    df_ohe.reset_index(drop=True, inplace=True)
    df = pd.concat([df, df_ohe], axis=1)
    
    # drop helper columns
    df = df.drop(columns=['Subplan', 'Subplan_list'])
    
    return df

def decoding_year_term(df):
    """
    Decodes the CourseTerm column into two columns: enrol_year includes the year the student enrolled in the course 
    and enrol_term is the term the student enrolled in the term. Does the same for the AdmitTerm column, turing it into 
    two columns: admit_year and admit_term
    """
    df = df.copy()
    
    # extract term (1 for Spring, 4 for Summer, 7 for Fall)
    df['enrol_term'] = df['CourseTerm'] % 10
    
    # Extract year (middle two digits)
    # 1. Remove last digit: floor divide by 10
    # 2. Take last two digits as year
    df['enrol_year'] = (df['CourseTerm'] // 10) % 100
    df['enrol_year'] = 2000 + df['enrol_year']
    
    # repeat for AdmitTerm
    df['admit_term'] = df['AdmitTerm'] % 10
    df['admit_year'] = (df['AdmitTerm'] // 10) % 100
    df['admit_year'] = 2000 + df['admit_year']
    
    return df

def course_level(df):
    """
    Takes a dataframe with a numeric course column (CatalogNbr)
    and adds course_level one-hot columns (100, 200, 300, 400) and is_writing binary column 
    (1 if course contains 'W', 0 otherwise)
    
    Also splits CatalogNbr into level and course_number_within_level
    """
    df = df.copy()
    
    # Step 1: Detect writing courses
    df['is_writing'] = df['CatalogNbr'].astype(str).str.upper().str.contains('W').astype(bool)
    
    # Step 2: Remove 'W' to clean numeric part
    df['Catalog_clean'] = df['CatalogNbr'].astype(str).str.replace('W', '', regex=False)
    
    # Step 3: Convert to numeric
    df['Catalog_clean'] = pd.to_numeric(df['Catalog_clean'], errors='coerce')
    
    # Step 4: Fill NaN with 0 (or some placeholder) instead of dropping rows
    df['Catalog_clean'] = df['Catalog_clean'].fillna(0).astype(int)
    
    # Step 5: Extract course level (hundreds digit)
    df['course_level'] = df['Catalog_clean'] // 100 * 100
    
    # Extract course number within level (e.g., 360 -> 60)
    df['course_number_within_level'] = df['Catalog_clean'] % 100
    
    # Step 6: One-hot encode course_level
    df = pd.get_dummies(df, columns=['course_level'], prefix='level')
    
    # Step 7: Drop temporary cleaned column
    df = df.drop(columns=['Catalog_clean'])
    
    return df

def day_of_week_features(df):
    """
    NEW: Adds boolean features for each day of the week the course is offered.
    Simple feature based on term pattern (Spring=1, Summer=4, Fall=7).
    """
    df = df.copy()
    
    # Extract term indicator
    df['enrol_term_indicator'] = df['CourseTerm'] % 10
    
    # Drop temp column
    df = df.drop(columns=['enrol_term_indicator'])
    
    return df

from course.credits import courses
def num_bus_credits(df):
    def get_credits(row):
        # why is there 2XX in the catalognbr?
        if 'X' in row['CatalogNbr']:
            return 0
        return courses[(row['CatalogNbr'],row['CourseTerm'])]
    df['credits'] = df[['CourseTerm','CatalogNbr']].apply(lambda x: get_credits(x), axis=1)
    std = df.groupby(["STD_INDEX","CourseTerm"])
    def sum_credits(row):
        std = df.loc[row["STD_INDEX"]==df["STD_INDEX"]]
        return sum(std["credits"][row["CourseTerm"] > df["CourseTerm"]])
    df['num_bus_credits'] = df[['STD_INDEX','CourseTerm']].apply(
        lambda x: sum_credits(x), axis=1)
    return df

def engineered_features(df):
    """Builds new features onto the cleaned enrolment data."""
    
    df = fix_course_names(df)
    # Example feature: number of completed BUS courses
    df = add_num_bus_courses(df)
    df = add_course_level_counts(df)
    df = add_time_since_admission(df)
    df = add_concentration_progress(df)
    df = splitting_concentrations(df)
    df = decoding_year_term(df)
    df = course_level(df)
    df = num_bus_credits(df)
    
    return df

def aggregate_to_course_term_level(df):
    """
    Aggregates student-level data to course-term level for demand forecasting.
    
    Instead of predicting individual enrollments, we predict total enrollment per course per term.
    This is more useful for course planning and removes the need for negative sampling.
    """
    print("\nAggregating data to course-term level...")
    
    df = df.copy()
    df["CourseTerm"] = pd.to_numeric(df["CourseTerm"], errors="coerce")
    
    # Group by course and term
    course_term_groups = df.groupby(['CatalogNbr', 'CourseTerm'])
    
    # Aggregate enrollment count (target variable)
    enrollment_counts = course_term_groups.size().reset_index(name='enrollment_count')
    
    # Course characteristics (same for all instances of a course)
    course_features = course_term_groups.agg({
        'is_writing': 'first',
        'has_lab': 'first',
        'has_tut': 'first',
        'has_sem': 'first',
        'offered_burnaby': 'first',
        'offered_surrey': 'first',
        'offered_van': 'first',
        'course_number_within_level': 'first',
        'credits': 'first',
    }).reset_index()
    
    # Get level indicators (one-hot encoded)
    level_cols = [c for c in df.columns if c.startswith('level_')]
    if level_cols:
        level_features = course_term_groups[level_cols].first().reset_index()
        course_features = course_features.merge(level_features, on=['CatalogNbr', 'CourseTerm'])
    
    # Term features
    term_features = course_term_groups.agg({
        'enrol_term': 'first',
        'enrol_year': 'first',
    }).reset_index()
    
    # Student cohort features (aggregated across all students in that term)
    cohort_features = course_term_groups.agg({
        'std_bus_courses_completed': 'mean',
        'terms_since_admit': 'mean',
        'num_bus_credits': 'mean',
        'num_300_taken': 'mean',
        'num_400_taken': 'mean',
    }).reset_index()
    
    cohort_features.columns = ['CatalogNbr', 'CourseTerm', 
                                'avg_student_bus_completed', 'avg_terms_since_admit', 'avg_num_bus_credits',
                                'avg_300_level_taken', 'avg_400_level_taken']
    
    # Concentration distribution (what % of students have each concentration)
    conc_cols = [c for c in df.columns if c.endswith('CON') and not c.endswith('_courses_completed')]
    if conc_cols:
        conc_features = course_term_groups[conc_cols].mean().reset_index()
        conc_features.columns = ['CatalogNbr', 'CourseTerm'] + [f'pct_{c}' for c in conc_cols]
    else:
        conc_features = enrollment_counts[['CatalogNbr', 'CourseTerm']].copy()
    
    # Merge all aggregated features
    aggregated = enrollment_counts
    for features_df in [course_features, term_features, cohort_features, conc_features]:
        aggregated = aggregated.merge(features_df, on=['CatalogNbr', 'CourseTerm'], how='left')
    
    print(f"Aggregated to {len(aggregated)} course-term combinations")
    
    return aggregated

def add_historical_demand_features(df):
    """
    Adds historical demand features for forecasting enrollment.
    Includes: Multiple lag features, rolling averages (3 & 5 terms), 
    same-term seasonal averages, and course-level historical baseline.
    """
    print("\nAdding historical demand features...")
    
    df = df.copy()
    df = df.sort_values(['CatalogNbr', 'CourseTerm'])
    
    # Basic Lag Features
    df['enrollment_lag_1'] = df.groupby('CatalogNbr')['enrollment_count'].shift(1)
    df['enrollment_lag_2'] = df.groupby('CatalogNbr')['enrollment_count'].shift(2)
    df['enrollment_lag_3'] = df.groupby('CatalogNbr')['enrollment_count'].shift(3)
    
    # Rolling Averages
    df['enrollment_rolling_avg_3'] = (
        df.groupby('CatalogNbr')['enrollment_count']
          .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    )
    
    # Trend Features
    df['enrollment_trend_1'] = df['enrollment_lag_1'] - df['enrollment_lag_2']
    df['enrollment_trend_2'] = df['enrollment_lag_2'] - df['enrollment_lag_3']
    
    # Same-Term Seasonality    
    df['enrollment_same_term_avg'] = (
        df.groupby(['CatalogNbr', 'enrol_term'])['enrollment_count']
          .transform(lambda x: x.shift(1).expanding().mean())
    )
    
    # Course Historical Baseline
    df['course_historical_avg'] = (
        df.groupby('CatalogNbr')['enrollment_count']
          .transform(lambda x: x.shift(1).expanding().mean())
    )
    
    # Fill Missing Values
    hist_cols = [
        'enrollment_lag_1', 'enrollment_lag_2', 'enrollment_lag_3',
        'enrollment_rolling_avg_3',
        'enrollment_trend_1', 'enrollment_trend_2',
        'enrollment_same_term_avg', 'course_historical_avg'
    ]
    
    df[hist_cols] = df[hist_cols].fillna(0)
    
    print(f"Added {len(hist_cols)} historical demand features")
    
    return df

# Main Pipeline Step for Preprocessing the Cleaned Enrolment Data
def main():
    
    # Step 1: Load cleaned enrolment data
    clean_data = pd.read_csv('data/enrolment_clean.csv')
    
    # Step 2: Join student info from delcared csv
    student_declared = pd.read_csv('data/STUDENT_Declared20260105.csv')
    clean_data = clean_data.merge(student_declared, on='STD_INDEX', how='left')
    clean_data = clean_data.drop(columns=['Program'])
    
    # Step 3: Apply feature engineering function (keeps group members' code)
    clean_data_detailed = engineered_features(clean_data)
    clean_data_detailed.to_csv('data/enrolment_engineered.csv', index=False)
    
    # Step 4: Aggregate to course-term level for demand forecasting
    aggregated_data = aggregate_to_course_term_level(clean_data_detailed)
    
    # Step 5: Add historical demand features
    aggregated_data = add_historical_demand_features(aggregated_data)
    
    # Step 6: Save the preprocessed data
    aggregated_data.to_csv('data/enrolment_counts.csv', index=False)
    print("\nSaved course demand dataset to: data/enrolment_counts.csv")
    print(f"Shape: {aggregated_data.shape}")
    print("\nTarget variable (enrollment_count) statistics:")
    print(aggregated_data['enrollment_count'].describe())

# Run the preprocessing pipeline step
if __name__ == "__main__":
    main()
