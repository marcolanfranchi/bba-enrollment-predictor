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



### 3. Model Training

