# Libraries
import pandas as pd


def has_component(series, component):
    """Return True if a component (LAB/TUT/SEM) appears in the group."""
    return series.str.contains(component).any()


def offered_campus(series, campus):
    """Return True if a campus (BURNABY/SURREY/VAN) appears in the group."""
    return series.str.contains(campus, case=False).any()


# Main Pipeline
def main():

    # Load enrolment data
    enrol_data = pd.read_csv('data/ENROL_Declared20260105.csv')
    
    # Rename `EnolledSemester` column for consistency
    enrol_data = enrol_data.rename(columns={'EnolledSemester': 'CourseTerm'})

    # Load historical data
    enrol_data_hist = pd.read_excel(
        'data/ENROL_HIST_DECLARED20260113.xlsx',
        engine='openpyxl'
    )

    # Remove spaces from both datasets column names
    enrol_data.columns = enrol_data.columns.str.replace(" ", "")
    enrol_data_hist.columns = enrol_data_hist.columns.str.replace(" ", "")

    # Step 1: Remove time/day columns, then remove duplicates
    cols_to_remove = [
        "HourStart", "MinuteStart", "HourEnd", "MinuteEnd",
        "MON", "Tues", "WED", "Thurs", "FRI"
    ]

    enrol_data = enrol_data.drop(columns=cols_to_remove).drop_duplicates()

    # Step 2: Define grouping keys (one row per student-course-term)
    keys = ['STD_INDEX', 'CourseTerm', 'Subject', 'CatalogNbr']

    # Step 3: Convert SsrComponent (tut, lab, sem) to Boolean features
    component_features = enrol_data.groupby(keys).agg(
        has_lab=('SsrComponent', lambda x: has_component(x, 'LAB')),
        has_tut=('SsrComponent', lambda x: has_component(x, 'TUT')),
        has_sem=('SsrComponent', lambda x: has_component(x, 'SEM')),
    ).reset_index()

    # Step 4: Convert Location to Boolean campus features
    location_features = enrol_data.groupby(keys).agg(
        offered_burnaby=('Location', lambda x: offered_campus(x, 'BURNABY')),
        offered_surrey=('Location', lambda x: offered_campus(x, 'SURREY')),
        offered_van=('Location', lambda x: offered_campus(x, 'VAN')),
    ).reset_index()

    # Step 5: Merge all features
    final_features = component_features.merge(
        location_features,
        on=keys,
        how='inner'
    )

    # Step 6: Manipulate historical data to match other datasets format for appending
    # Step 6.1: Convert CourseName to Subject + CatalogNbr
    enrol_data_hist['CourseName'] = enrol_data_hist['CourseName'].str.strip()
    enrol_data_hist[['Subject', 'CatalogNbr']] = enrol_data_hist['CourseName'].str.extract(r'([A-Z]+)([0-9A-Z]+)')
    enrol_data_hist = enrol_data_hist.drop(columns=['CourseName'])

    # Step 6.2: Align columns
    hist_missing_cols = set(final_features.columns) - set(enrol_data_hist.columns)
    for col in hist_missing_cols:
        enrol_data_hist[col] = None

    final_missing_cols = set(enrol_data_hist.columns) - set(final_features.columns)
    for col in final_missing_cols:
        final_features[col] = None

    # Step 7: Append historical data
    final_features = pd.concat([final_features, enrol_data_hist], ignore_index=True)

    # Step 8: Compute defaults for imputing missing values
    # Step 8.1 Component defaults (course components, lab/tut/sem, assumed constant across terms)
    component_defaults = final_features.groupby(['CatalogNbr']).agg(
        has_lab_default=('has_lab', 'max'),
        has_tut_default=('has_tut', 'max'),
        has_sem_default=('has_sem', 'max'),
        course_title_default=('CourseTitle', 'first')
    ).reset_index()

    final_features = final_features.merge(
        component_defaults,
        on=['CatalogNbr'],
        how='left'
    )

    # Step 8.2 Campus offering defaults (term-specific)
    campus_defaults = final_features.groupby(['CourseTerm', 'CatalogNbr']).agg(
        offered_burnaby_default=('offered_burnaby', 'max'),
        offered_surrey_default=('offered_surrey', 'max'),
        offered_van_default=('offered_van', 'max')
    ).reset_index()

    final_features = final_features.merge(
        campus_defaults,
        on=['CourseTerm', 'CatalogNbr'],
        how='left'
    )

    # Step 9: Fill only where defaults exist (leave nulls otherwise)
    component_cols = ['has_lab', 'has_tut', 'has_sem']
    campus_cols = ['offered_burnaby', 'offered_surrey', 'offered_van']

    for col in component_cols:
        final_features[col] = final_features[col].fillna(final_features[f"{col}_default"])

    for col in campus_cols:
        final_features[col] = final_features[col].fillna(final_features[f"{col}_default"])

    final_features['CourseTitle'] = final_features['CourseTitle'].fillna(
        final_features['course_title_default']
    )

    # Drop all *_default columns
    final_features = final_features.drop(
        columns=[c for c in final_features.columns if c.endswith('_default')]
    )

    # Step 10: Save + Sanity Check
    final_features.to_csv('data/enrolment_clean.csv', index=False)

    print("Saved cleaned dataset to: data/enrolment_clean.csv")
    print("\nPreview of final dataset:")
    print(final_features.head())

    print("\nMissing value counts:")
    print(final_features.isna().sum())


if __name__ == "__main__":
    main()
