import numpy as np
from datallm import DataLLM

# Initialize DataLLM client with API key
API_KEY = 'your_api_key'
BASE_URL = 'https://data.mostly.ai'
datallm = DataLLM(api_key=API_KEY, base_url=BASE_URL)

# Define the mappings and probabilities for cases and controls
probabilities = {
    "cases": {
        "age": (58, 8),  # Mean age = 58, Std Dev = 8 (Colorectal cancer typically presents at older ages)
        "BMI": (30, 4),  # Mean BMI = 30, Std Dev = 4 (Higher BMI is linked to increased colorectal cancer risk)
        "red_meat": (70, 15),  # Mean red meat intake = 70g/day, Std Dev = 15g (Higher consumption linked to cancer)
        "processed_meat": (30, 10),  # Mean processed meat intake = 30g/day, Std Dev = 10g (Linked to colorectal cancer)
        "alcohol_use": (25, 10),  # Mean alcohol use = 25g/day, Std Dev = 10g (Alcohol use linked to colorectal cancer risk)
        "smoking_duration": (25, 7),  # Mean smoking duration = 25 years, Std Dev = 7 years (Smoking duration impacts risk)
        "pack_years": (20, 8),  # Mean pack-years = 20, Std Dev = 8 (Pack-years is a measure of smoking exposure)
        "moderate_activity": (60, 20),  # Mean moderate physical activity = 60 min/week, Std Dev = 20 min
        "vigorous_activity": (30, 15),  # Mean vigorous physical activity = 30 min/week, Std Dev = 15 min
        "strength_activity": (1, 0.5),  # Mean strength-based activity = 1 time/week, Std Dev = 0.5
        "dietary_fiber": (60, 10),  # Mean dietary fiber = 60g/day, Std Dev = 10g (Higher fiber intake may reduce risk)
        "sex": [0.48, 0.52],  # Distribution of sexes in cases: 48% male, 52% female
        "diabetes": [0.84, 0.16],  # 84% of cases do not have diabetes, 16% do (Diabetes increases cancer risk)
        "famhx1": [0.81, 0.19]  # Family history: 81% do not have a family history, 19% do
    },
    "controls": {
        "age": (54, 8),  # Mean age = 54, Std Dev = 8 (Control group tends to be slightly younger than cases)
        "BMI": (25, 3),  # Mean BMI = 25, Std Dev = 3 (Controls tend to have lower BMI than cancer cases)
        "red_meat": (30, 10),  # Mean red meat intake = 30g/day, Std Dev = 10g (Lower than cases)
        "processed_meat": (25, 7),  # Mean processed meat intake = 25g/day, Std Dev = 7g (Lower than cases)
        "alcohol_use": (12.5, 6),  # Mean alcohol use = 12.5g/day, Std Dev = 6g (Lower than cases)
        "smoking_duration": (8, 5),  # Mean smoking duration = 8 years, Std Dev = 5 years (Lower smoking duration for controls)
        "pack_years": (8, 5),  # Mean pack-years = 8, Std Dev = 5 (Lower smoking exposure in controls)
        "moderate_activity": (150, 40),  # Mean moderate physical activity = 150 min/week, Std Dev = 40 min (Higher physical activity in controls)
        "vigorous_activity": (100, 40),  # Mean vigorous physical activity = 100 min/week, Std Dev = 40 min (Higher in controls)
        "strength_activity": (3, 1),  # Mean strength-based activity = 3 times/week, Std Dev = 1 (Higher than cases)
        "dietary_fiber": (30, 5),  # Mean dietary fiber = 30g/day, Std Dev = 5g (Lower fiber intake than cases)
        "sex": [0.52, 0.48],  # Distribution of sexes in controls: 52% male, 48% female
        "diabetes": [0.88, 0.12],  # 88% of controls do not have diabetes, 12% do (Lower diabetes prevalence in controls)
        "famhx1": [0.81, 0.19]  # Family history: 81% do not have a family history, 19% do (Same family history distribution)
    }
}

# Generate synthetic data using MostlyAI
df = datallm.mock(
    n=100,
    data_description="Synthetic dataset for colorectal cancer study",
    columns={
        "SUBJECT_ID": {"prompt": "Unique identifier for each subject", "dtype": "integer", "range": [1, 100]},
        "caseclrt_Crs_Frst": {"categories": ["0", "1"], "dtype": "category", "probabilities": [0.50, 0.50]},  # Case/control status
        "age": {"prompt": "Age at diagnosis", "dtype": "float", "distribution": {"type": "normal", "mean": 55, "std_dev": 10}},
        "sex": {"categories": ["0", "1"], "dtype": "category", "probabilities": [0.60, 0.40]},  # Sex distribution, placeholder
        "famhx1": {"categories": ["0", "1"], "dtype": "category", "probabilities": [0.7, 0.3]},  # Family history
        "BMI": {"prompt": "BMI kg/m2", "dtype": "float", "distribution": {"type": "normal", "mean": 30, "std_dev": 5}},
        "diabetes": {"categories": ["0", "1"], "dtype": "category", "probabilities": [0.50, 0.50]},  # Placeholder, adjust later
        "red_meat": {"prompt": "Red meat consumed in grams per day", "dtype": "float", "distribution": {"type": "normal", "mean": 50, "std_dev": 20}},
        "processed_meat": {"prompt": "Processed meat consumed in grams per day", "dtype": "float", "distribution": {"type": "normal", "mean": 30, "std_dev": 10}},
        "alcohol_use": {"prompt": "Alcohol consumption in grams per day", "dtype": "float", "distribution": {"type": "normal", "mean": 20, "std_dev": 15}},
        "smoking_duration": {"prompt": "Years of smoking", "dtype": "float", "distribution": {"type": "normal", "mean": 20, "std_dev": 10}},
        "pack_years": {"prompt": "Pack-years of smoking", "dtype": "float", "distribution": {"type": "normal", "mean": 20, "std_dev": 10}},
        "moderate_activity": {"prompt": "Hours of moderate physical activity per week", "dtype": "float", "distribution": {"type": "normal", "mean": 5, "std_dev": 2}},
        "vigorous_activity": {"prompt": "Hours of vigorous physical activity per week", "dtype": "float", "distribution": {"type": "normal", "mean": 5, "std_dev": 2}},
        "strength_activity": {"prompt": "Times of strength-based physical activity per week", "dtype": "float", "distribution": {"type": "normal", "mean": 5, "std_dev": 2}},
        "dietary_fiber": {"prompt": "Dietary fiber intake in grams per day", "dtype": "float", "range": [0, 50]}
    }
)

# Function to apply case/control specific attributes
def apply_probabilities(row, col_name):
    group = 'cases' if row["caseclrt_Crs_Frst"] == "1" else 'controls'

    # Adjust probabilities for sex and diabetes based on case/control status
    if col_name == 'sex':
        return np.random.choice(["0", "1"], p=probabilities[group]['sex'])

    if col_name == 'diabetes':
        return np.random.choice(["0", "1"], p=probabilities[group]['diabetes'])

    # Adjust smoking duration based on age, ensuring non-negative values
    if col_name == 'smoking_duration':
        age = row['age']
        max_smoking_duration = min(age - 15, 40)  # Cap smoking duration by age
        smoking_duration = np.random.normal(loc=(age - 15) / 2, scale=5)
        return max(0, min(smoking_duration, max_smoking_duration))  # Ensure no negative values

    # Adjust BMI based on diabetes status, ensuring non-negative values
    if col_name == 'BMI':
        diabetes = row['diabetes']
        if diabetes == '1':  # Increase BMI if diabetic
            return max(0, np.random.normal(loc=probabilities[group]['BMI'][0] + 3, scale=2))
        else:
            return max(0, np.random.normal(loc=probabilities[group]['BMI'][0], scale=2))

    return row[col_name]  # Return original value if no adjustment needed

# Apply probabilities and correlations to the synthetic dataset
for col in df.columns:
    if col in probabilities['cases']:  # Apply only to relevant columns
        df[col] = df.apply(lambda row: apply_probabilities(row, col), axis=1)

# Save DataFrame to CSV file
output_file_path = r"D:\Colon-Cancer-Predicter\data\data.csv"
df.to_csv(output_file_path, index=False)

print(f"Data saved to {output_file_path}")