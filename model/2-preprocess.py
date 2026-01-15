# Libraries
import pandas as pd


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


def engineered_features(df):
    """Builds new features onto the cleaned enrolment data."""
    
    # Example feature: number of completed BUS courses
    df = add_num_bus_courses(df)
    
    # TODO: Future engineered features can be added here:
    # e.g., course_level (100, 200, 300, 400), is_writing (True/False)
    
    return df


# Main Pipeline Step for Preprocessing the Cleaned Enrolment Data
def main():
    
    # Step 1: Load cleaned enrolment data
    clean_data = pd.read_csv('data/enrolment_clean.csv')

    # Step 2: Join student info from delcared csv
    # TODO: 

    # Step 3: Apply feature engineering function
    clean_data_detailed = engineered_features(clean_data)

    # Step 4: Save the preprocessed data
    clean_data_detailed.to_csv('data/enrolment_final.csv', index=False)

    print("Saved preprocessed dataset to: data/enrolment_final.csv")


# Run the preprocessing pipeline step
if __name__ == "__main__":
    main()
