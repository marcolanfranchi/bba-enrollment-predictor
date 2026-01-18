# Libraries
import pandas as pd
from course.credits import courses

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

def num_bus_credits(df):
    def get_credits(row):
        # Handle special course codes (e.g., 2XX) that don't have credit mappings
        if 'X' in str(row['CatalogNbr']):
            return 0
        try:
            return courses[(row['CatalogNbr'], row['CourseTerm'])]
        except KeyError:
            return 0
    
    df['credits'] = df[['CourseTerm','CatalogNbr']].apply(lambda x: get_credits(x), axis=1)
    
    def sum_credits(row):
        # Filter to student's records and sum credits from previous terms
        student_data = df[df["STD_INDEX"] == row["STD_INDEX"]]
        return sum(student_data["credits"][student_data["CourseTerm"] < row["CourseTerm"]])
    
    df['num_bus_credits'] = df[['STD_INDEX','CourseTerm']].apply(
        lambda x: sum_credits(x), axis=1)
    return df

def engineered_features(df):
    """Builds new features onto the cleaned enrolment data."""
    
    df = fix_course_names(df)
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
    Only uses information available before the target term.
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
    
    # Student cohort features from previous terms.
    # These represent the characteristics of students who could take the course
    # based on prior term data
    cohort_features = course_term_groups.agg({
        'std_bus_courses_completed': 'mean',
        'terms_since_admit': 'mean',
        'num_bus_credits': 'mean',
        'num_300_taken': 'mean',
        'num_400_taken': 'mean',
    }).reset_index()
    
    cohort_features.columns = ['CatalogNbr', 'CourseTerm', 
                                'prev_avg_student_bus_completed', 'prev_avg_terms_since_admit', 
                                'prev_avg_num_bus_credits', 'prev_avg_300_level_taken', 
                                'prev_avg_400_level_taken']
    
    # Concentration distribution from previous term
    conc_cols = [c for c in df.columns if c.endswith('CON') and not c.endswith('_courses_completed')]
    if conc_cols:
        conc_features = course_term_groups[conc_cols].mean().reset_index()
        conc_features.columns = ['CatalogNbr', 'CourseTerm'] + [f'prev_pct_{c}' for c in conc_cols]
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

def add_term_growth_features(df):
    """
    Adds term-level enrollment growth features to capture program expansion/contraction.
    """
    print("\nAdding term-level enrollment growth features...")
    
    df = df.copy()
    df = df.sort_values(['CourseTerm'])
    
    # Calculate total program enrollment per term (lagged to previous term)
    term_totals = df.groupby('CourseTerm')['enrollment_count'].sum().reset_index()
    term_totals.columns = ['CourseTerm', 'term_total_enrollment']
    
    # Shift to get PREVIOUS term's total enrollment
    term_totals['prev_term_total_enrollment'] = term_totals['term_total_enrollment'].shift(1)
    
    # Calculate growth metrics
    term_totals['term_enrollment_growth'] = (
        term_totals['term_total_enrollment'] - term_totals['prev_term_total_enrollment']
    )
    
    term_totals['term_enrollment_growth_pct'] = (
        term_totals['term_enrollment_growth'] / term_totals['prev_term_total_enrollment']
    ).fillna(0) * 100
    
    # Rolling average of term enrollments (3-term window)
    term_totals['term_enrollment_rolling_avg_3'] = (
        term_totals['term_total_enrollment'].shift(1).rolling(window=3, min_periods=1).mean()
    )
    
    # YoY growth (comparing to same term last year)
    term_totals['yoy_enrollment_growth'] = (
        term_totals['term_total_enrollment'] - term_totals['term_total_enrollment'].shift(3)
    )
    
    # Count of unique courses offered per term (diversity of offerings)
    courses_per_term = df.groupby('CourseTerm')['CatalogNbr'].nunique().reset_index()
    courses_per_term.columns = ['CourseTerm', 'num_courses_offered']
    courses_per_term['prev_num_courses_offered'] = courses_per_term['num_courses_offered'].shift(1)
    
    # Merge term-level features back to main dataframe
    # Drop current term's total (would be leakage), keep only lagged features
    term_features = term_totals.drop(columns=['term_total_enrollment'])
    term_features = term_features.merge(
        courses_per_term[['CourseTerm', 'prev_num_courses_offered']], 
        on='CourseTerm', 
        how='left'
    )
    
    df = df.merge(term_features, on='CourseTerm', how='left')
    
    # Fill NaNs (first terms have no historical data)
    growth_cols = [
        'prev_term_total_enrollment', 'term_enrollment_growth', 'term_enrollment_growth_pct',
        'term_enrollment_rolling_avg_3', 'yoy_enrollment_growth', 'prev_num_courses_offered'
    ]
    
    df[growth_cols] = df[growth_cols].fillna(0)
    
    print(f"Added {len(growth_cols)} term-level growth features")
    
    return df

def add_course_prerequisite_features(df):
    """
    Adds features based on prerequisite structure and course difficulty.
    """
    print("\nAdding course prerequisite and difficulty features...")
    
    df = df.copy()
    df = df.sort_values(['CatalogNbr', 'CourseTerm'])
    
    # Extract numeric course number
    df['temp_catalog'] = df['CatalogNbr'].astype(str).str.replace('W', '', regex=False)
    df['temp_catalog'] = pd.to_numeric(df['temp_catalog'], errors='coerce').fillna(0).astype(int)
    
    # Course number within level (e.g., 360 -> 60)
    # Lower numbers often = earlier in sequence
    df['course_sequence_position'] = df['temp_catalog'] % 100
    
    # Track enrollment in prerequisite-heavy course levels
    # 300-level courses often require 200-level prereqs
    for level in [200, 300, 400]:
        level_enrollments = df[df['temp_catalog'] >= level].groupby('CourseTerm').size().reset_index()
        level_enrollments.columns = ['CourseTerm', f'prev_level_{level}_plus_enrollment']
        level_enrollments[f'prev_level_{level}_plus_enrollment'] = (
            level_enrollments[f'prev_level_{level}_plus_enrollment'].shift(1)
        )
        df = df.merge(level_enrollments, on='CourseTerm', how='left')
        df[f'prev_level_{level}_plus_enrollment'] = df[f'prev_level_{level}_plus_enrollment'].fillna(0)
    
    # Concentration-specific demand (lagged)
    # If FINCON had high enrollment last term, FIN courses likely see more demand
    df['temp_conc'] = None
    conc_map = {
        range(313, 320): 'FINCON',
        range(320, 330): 'ACCTCON',
        range(343, 350): 'MKTGCON',
        range(360, 370): 'MISCON',
        range(381, 390): 'HRMCON',
    }
    
    for course_range, conc in conc_map.items():
        mask = df['temp_catalog'].isin(course_range)
        df.loc[mask, 'temp_conc'] = conc
    
    # For each concentration, track lagged enrollment
    if 'temp_conc' in df.columns:
        conc_enrollment = df.groupby(['temp_conc', 'CourseTerm']).size().reset_index()
        conc_enrollment.columns = ['temp_conc', 'CourseTerm', 'conc_enrollment']
        conc_enrollment['prev_conc_enrollment'] = (
            conc_enrollment.groupby('temp_conc')['conc_enrollment'].shift(1)
        )
        df = df.merge(
            conc_enrollment[['temp_conc', 'CourseTerm', 'prev_conc_enrollment']], 
            on=['temp_conc', 'CourseTerm'], 
            how='left'
        )
        df['prev_conc_enrollment'] = df['prev_conc_enrollment'].fillna(0)
        df = df.drop(columns=['temp_conc'])
    
    df = df.drop(columns=['temp_catalog'])
    
    print("Added course prerequisite and difficulty features")
    
    return df

def add_enrollment_volatility_features(df):
    """
    Adds features capturing enrollment volatility and stability.
    High volatility = harder to predict, may need capacity buffer
    Low volatility = stable demand, easier planning
    """
    print("\nAdding enrollment volatility features...")
    
    df = df.copy()
    df = df.sort_values(['CatalogNbr', 'CourseTerm'])
    
    # Calculate rolling standard deviation (volatility)
    df['enrollment_rolling_std_3'] = (
        df.groupby('CatalogNbr')['enrollment_count']
          .transform(lambda x: x.shift(1).rolling(window=3, min_periods=2).std())
    )
    
    # Coefficient of variation (normalized volatility)
    df['enrollment_cv_3'] = (
        df['enrollment_rolling_std_3'] / 
        (df['enrollment_rolling_avg_3'] + 1)  # +1 to avoid division by zero
    )
    
    # Calculate enrollment momentum (is it accelerating or decelerating?)
    df['enrollment_momentum'] = (
        df['enrollment_trend_1'] - df['enrollment_trend_2']
    )
    
    # Seasonal stability: does this course have consistent enrollment in its term?
    df['enrollment_same_term_std'] = (
        df.groupby(['CatalogNbr', 'enrol_term'])['enrollment_count']
          .transform(lambda x: x.shift(1).expanding().std())
    )
    
    # Fill NaNs
    volatility_cols = [
        'enrollment_rolling_std_3', 'enrollment_cv_3', 
        'enrollment_momentum', 'enrollment_same_term_std'
    ]
    df[volatility_cols] = df[volatility_cols].fillna(0)
    
    print(f"Added {len(volatility_cols)} enrollment volatility features")
    
    return df

def add_course_popularity_features(df):
    """
    Adds features measuring course popularity and market share.
    
    Captures relative demand compared to other courses at same level.
    """
    print("\nAdding course popularity features...")
    
    df = df.copy()
    df = df.sort_values(['CatalogNbr', 'CourseTerm'])
    
    # Extract course level
    df['temp_catalog'] = df['CatalogNbr'].astype(str).str.replace('W', '', regex=False)
    df['temp_catalog'] = pd.to_numeric(df['temp_catalog'], errors='coerce').fillna(0).astype(int)
    df['temp_level'] = df['temp_catalog'] // 100 * 100
    
    # Calculate market share within course level (lagged)
    level_totals = df.groupby(['temp_level', 'CourseTerm'])['enrollment_count'].sum().reset_index()
    level_totals.columns = ['temp_level', 'CourseTerm', 'level_total_enrollment']
    
    df = df.merge(level_totals, on=['temp_level', 'CourseTerm'], how='left')
    
    # Shift level totals to previous term
    df['prev_level_total_enrollment'] = (
        df.groupby('temp_level')['level_total_enrollment'].shift(1)
    )
    
    # Market share (what % of level enrollment does this course capture?)
    df['prev_course_market_share'] = (
        df['enrollment_lag_1'] / (df['prev_level_total_enrollment'] + 1) * 100
    )
    
    # Rank within level (is this a top-3 course in its level?)
    df['prev_enrollment_rank_in_level'] = (
        df.groupby(['temp_level', 'CourseTerm'])['enrollment_lag_1']
          .rank(method='dense', ascending=False)
    )
    
    # Is this course growing faster than its level average?
    df['prev_level_avg_enrollment'] = (
        df.groupby('temp_level')['prev_level_total_enrollment']
          .transform('mean')
    )
    
    df['prev_course_vs_level_growth'] = (
        df['enrollment_trend_1'] - 
        (df['prev_level_total_enrollment'] - df.groupby('temp_level')['prev_level_total_enrollment'].shift(1))
    )
    
    # Clean up temp columns
    df = df.drop(columns=['temp_catalog', 'temp_level', 'level_total_enrollment'])
    
    # Fill NaNs
    popularity_cols = [
        'prev_level_total_enrollment', 'prev_course_market_share', 
        'prev_enrollment_rank_in_level', 'prev_level_avg_enrollment', 
        'prev_course_vs_level_growth'
    ]
    df[popularity_cols] = df[popularity_cols].fillna(0)
    
    print(f"Added {len(popularity_cols)} course popularity features")
    
    return df

def add_capacity_constraint_features(df):
    """
    Adds features related to course capacity and section offerings.
    
    If a course was "full" last term, it may have unmet demand.
    """
    print("\nAdding capacity constraint features...")
    
    df = df.copy()
    df = df.sort_values(['CatalogNbr', 'CourseTerm'])
    
    # Identify courses with unusual enrollment spikes
    df['had_enrollment_spike'] = (
        (df['enrollment_trend_1'] > df['enrollment_rolling_std_3'] * 2) & 
        (df['enrollment_rolling_std_3'] > 0)
    ).astype(int)
    
    # Identify courses with unusual drops
    df['had_enrollment_drop'] = (
        (df['enrollment_trend_1'] < -df['enrollment_rolling_std_3'] * 2) & 
        (df['enrollment_rolling_std_3'] > 0)
    ).astype(int)
    
    # Course offering frequency
    # How many times has this course been offered in the last 6 terms?
    df['course_offering_frequency'] = (
        df.groupby('CatalogNbr')['enrollment_count']
          .transform(lambda x: x.shift(1).rolling(window=6, min_periods=1).count())
    )
    
    # Average gap between offerings (if course isn't offered every term)
    # This is complex to calculate exactly, so we'll use a proxy
    df['is_every_term_course'] = (df['course_offering_frequency'] >= 5).astype(int)
    
    print("Added capacity constraint features")
    
    return df

# Main Pipeline Step for Preprocessing the Cleaned Enrolment Data
def main():
    
    # Step 1: Load cleaned enrolment data
    clean_data = pd.read_csv('data/enrolment_clean.csv')
    
    # Step 2: Join student info from declared csv
    student_declared = pd.read_csv('data/STUDENT_Declared20260105.csv')
    clean_data = clean_data.merge(student_declared, on='STD_INDEX', how='left')
    clean_data = clean_data.drop(columns=['Program'])
    
    # Step 3: Apply feature engineering function
    clean_data_detailed = engineered_features(clean_data)
    clean_data_detailed.to_csv('data/enrolment_engineered.csv', index=False)
    
    # Step 4: Aggregate to course-term level for demand forecasting
    aggregated_data = aggregate_to_course_term_level(clean_data_detailed)
    
    # Step 5: Add historical demand features
    aggregated_data = add_historical_demand_features(aggregated_data)
    
    # Step 6: Add term-level enrollment growth features
    aggregated_data = add_term_growth_features(aggregated_data)
    
    # Step 7: Add course prerequisite features
    aggregated_data = add_course_prerequisite_features(aggregated_data)
    
    # Step 8: Add enrollment volatility features
    aggregated_data = add_enrollment_volatility_features(aggregated_data)
    
    # Step 9: Add course popularity features
    aggregated_data = add_course_popularity_features(aggregated_data)
    
    # Step 10: Add capacity constraint features
    aggregated_data = add_capacity_constraint_features(aggregated_data)
    
    # Step 11: Save the preprocessed data
    aggregated_data.to_csv('data/enrolment_counts.csv', index=False)
    print("\nSaved course demand dataset to: data/enrolment_counts.csv")
    print(f"Shape: {aggregated_data.shape}")
    print("\nTarget variable (enrollment_count) statistics:")
    print(aggregated_data['enrollment_count'].describe())

# Run the preprocessing pipeline step
if __name__ == "__main__":
    main()
