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
    a student has taken up to (and including) the given term
    (since # of credits per course are unavailable).
    """

    # Sort data by student and term to ensure proper cumulative counting
    df = df.sort_values(by=["STD_INDEX", "CourseTerm"])
    
    # Create indicator for BUS courses (all data is BUS, but just to be safe)
    df["is_bus_course"] = df["Subject"] == "BUS"
    
    # Cumulative count of BUS courses per student
    df["std_bus_courses_completed"] = (
        df.groupby("STD_INDEX")["is_bus_course"]
          .cumsum()
    )

    # Clean up helper column
    df = df.drop(columns=["is_bus_course"])
    
    return df

def add_prev_term_courses(df):
    """
    Adds a feature counting how many courses a student took in the previous term.
    """
    df = df.copy()
    
    # Sort by student and term to ensure proper ordering
    df = df.sort_values(by=['STD_INDEX', 'CourseTerm']).reset_index(drop=True)
    
    # Count number of courses per student per term
    term_counts = df.groupby(['STD_INDEX', 'CourseTerm']).size().rename('courses_this_term')
    
    # Merge counts back to the DataFrame
    df = df.merge(term_counts, on=['STD_INDEX', 'CourseTerm'], how='left')
    
    # Shift counts per student to get previous term
    df['courses_prev_term'] = df.groupby('STD_INDEX')['courses_this_term'].shift(1)
    
    # Fill NaN with 0 for students’ first term
    df['courses_prev_term'] = df['courses_prev_term'].fillna(0).astype(int)
    
    # Drop the helper column
    df = df.drop(columns=['courses_this_term'])
    
    return df


def splitting_concentrations(df):
    """
    Splits the Subplan column into one column per concentration and includes 
    1 if the student declared that concentration and 0 if they didn't 
    """

    # Step 1: split the strings into lists
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

    # drop helper columns (can drop these if want, since we have the other columns now that have the same info)
    #df = df.drop(columns=['CourseTerm', 'AdmitTerm'])

    return df

def course_level(df):
    """
    Takes a dataframe with a numeric course column (CatalogNbr)
    and adds course_level one-hot columns (100, 200, 300, 400) and is_writing binary column 
    (1 if course contains 'W', 0 otherwise)
    """

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
    
    # Step 6: One-hot encode course_level
    df = pd.get_dummies(df, columns=['course_level'], prefix='level')
    
    # Step 7: Drop temporary cleaned column
    df = df.drop(columns=['Catalog_clean'])
    
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
    
    # TODO: Future engineered features can be added here:
    # e.g., course_level (100, 200, 300, 400), is_writing (True/False)

    df = splitting_concentrations(df)

    df = decoding_year_term(df)

    df = course_level(df)

    df = add_prev_term_courses(df)

    df = num_bus_credits(df)

    
    return df


# Main Pipeline Step for Preprocessing the Cleaned Enrolment Data
def main():
    
    # Step 1: Load cleaned enrolment data
    clean_data = pd.read_csv('data/enrolment_clean.csv')

    # Step 2: Join student info from delcared csv
    student_declared = pd.read_csv('data/STUDENT_Declared20260105.csv')
    clean_data = clean_data.merge(student_declared, on='STD_INDEX', how='left')
    clean_data = clean_data.drop(columns=['Program'])

    # Step 3: Apply feature engineering function
    clean_data_detailed = engineered_features(clean_data)

    # Step 4: Save the preprocessed data
    clean_data_detailed.to_csv('data/enrolment_final.csv', index=False)

    print("Saved preprocessed dataset to: data/enrolment_final.csv")


# Run the preprocessing pipeline step
if __name__ == "__main__":
    main()
