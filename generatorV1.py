import numpy as np
from datallm import DataLLM

# Initialize DataLLM client with API key
API_KEY = 'your_api_key'
BASE_URL = 'https://data.mostly.ai'
datallm = DataLLM(api_key=API_KEY, base_url=BASE_URL)

# Define the mappings and probabilities
probabilities = {
    "cases": {
        "age": (60, 7),  # Mean = 60, Standard Deviation = 7 (for age distribution)
        "BMI": (30, 4),  # Mean = 30, Standard Deviation = 4 (for BMI)
        "red_meat": (52.5, 15),  # Mean = 52.5, Standard Deviation = 15 (grams of red meat consumed per day)
        "processed_meat": (30, 10),  # Mean = 30, Standard Deviation = 10 (grams of processed meat consumed per day)
        "alcohol_use": (25, 10),  # Mean = 25, Standard Deviation = 10 (grams of alcohol consumed per day)
        "smoking_duration": (20, 10),  # Mean = 20, Standard Deviation = 10 (years of smoking)
        "cigarettes_per_day": (10, 5),  # Mean = 10, Standard Deviation = 5 (cigarettes smoked per day)
        "pack_years": (20, 10),  # Mean = 20, Standard Deviation = 10 (pack-years of smoking)
        "physical_activity": (5, 2),  # Mean = 5, Standard Deviation = 2 (hours of physical activity per week)
        "dietary_fiber": (60, 10),  # Mean = 60, Standard Deviation = 10 (grams of dietary fiber intake per day)
        "sex": [0.48, 0.52],  # Sex distribution for cases: [Male, Female]
        "diabetes": [0.15, 0.85],  # Diabetes distribution for cases: [No, Yes]
        "famhx1": [0.4, 0.6],  # Family history: [No, Yes]
    },
    "controls": {
        "age": (55, 10),  # Mean = 55, Standard Deviation = 10 (for age distribution)
        "BMI": (25, 3),  # Mean = 25, Standard Deviation = 3 (for BMI)
        "red_meat": (45, 10),  # Mean = 45, Standard Deviation = 10 (grams of red meat consumed per day)
        "processed_meat": (25, 7),  # Mean = 25, Standard Deviation = 7 (grams of processed meat consumed per day)
        "alcohol_use": (12.5, 6),  # Mean = 12.5, Standard Deviation = 6 (grams of alcohol consumed per day)
        "smoking_duration": (15, 7),  # Mean = 15, Standard Deviation = 7 (years of smoking)
        "cigarettes_per_day": (10, 5),  # Mean = 10, Standard Deviation = 5 (cigarettes smoked per day)
        "pack_years": (15, 7),  # Mean = 15, Standard Deviation = 7 (pack-years of smoking)
        "physical_activity": (5, 2),  # Mean = 5, Standard Deviation = 2 (hours of physical activity per week)
        "dietary_fiber": (30, 5),  # Mean = 30, Standard Deviation = 5 (grams of dietary fiber intake per day)
        "sex": [0.52, 0.48],  # Sex distribution for controls: [Male, Female]
        "diabetes": [0.88, 0.12],  # Diabetes distribution for controls: [No, Yes]
        "famhx1": [0.7, 0.3],  # Family history: [No, Yes]
    }
}

# Generate synthetic data using MostlyAI
df = datallm.mock(
    n=100,
    data_description="Synthetic dataset for colorectal cancer study",
    columns={
        "SUBJECT_ID": {"prompt": "Unique identifier for each subject", "dtype": "integer", "range": [1, 100]},
        "caseclrt_Crs_Frst": {"categories": ["0", "1"], "dtype": "category", "probabilities": [0.50, 0.50]},  # Case/control status
        "age": {"prompt": "Age at diagnosis", "dtype": "float", "distribution": {"type": "normal", "mean": 55, "std_dev": 10}},  # Mean = 55, Std Dev = 10
        "sex": {"categories": ["0", "1"], "dtype": "category", "probabilities": [0.60, 0.40]},  # Placeholder, will adjust later
        "famhx1": {"categories": ["0", "1"], "dtype": "category", "probabilities": [0.7, 0.3]},  # Family history
        "BMI": {"prompt": "BMI kg/m2", "dtype": "float", "distribution": {"type": "normal", "mean": 30, "std_dev": 5}},  # Mean = 30, Std Dev = 5
        "diabetes": {"categories": ["0", "1"], "dtype": "category", "probabilities": [0.50, 0.50]},  # Placeholder, will adjust later
        "red_meat": {"prompt": "Red meat consumed in grams per day", "dtype": "float", "distribution": {"type": "normal", "mean": 50, "std_dev": 20}},  # Mean = 50, Std Dev = 20
        "processed_meat": {"prompt": "Processed meat consumed in grams per day", "dtype": "float", "distribution": {"type": "normal", "mean": 30, "std_dev": 10}},  # Mean = 30, Std Dev = 10
        "alcohol_use": {"prompt": "Alcohol consumption in grams per day", "dtype": "float", "distribution": {"type": "normal", "mean": 20, "std_dev": 15}},  # Mean = 20, Std Dev = 15
        "smoking_duration": {"prompt": "Years of smoking", "dtype": "float", "distribution": {"type": "normal", "mean": 20, "std_dev": 10}},  # Mean = 20, Std Dev = 10
        "cigarettes_per_day": {"prompt": "Cigarettes smoked per day", "dtype": "float", "distribution": {"type": "normal", "mean": 10, "std_dev": 5}},  # Mean = 10, Std Dev = 5
        "pack_years": {"prompt": "Pack-years of smoking", "dtype": "float", "distribution": {"type": "normal", "mean": 20, "std_dev": 10}},  # Mean = 20, Std Dev = 10
        "physical_activity": {"prompt": "Hours of physical activity per week", "dtype": "float", "distribution": {"type": "normal", "mean": 5, "std_dev": 2}},  # Mean = 5, Std Dev = 2
        "dietary_fiber": {"prompt": "Dietary fiber intake in grams per day", "dtype": "float", "range": [0, 50]}  # Dietary fiber range
    }
)

# Function to apply case/control specific attributes, with correlations
def apply_probabilities(row, col_name):
    group = 'cases' if row["caseclrt_Crs_Frst"] == "1" else 'controls'

    # Adjust probabilities for sex and diabetes based on case/control status
    if col_name == 'sex':
        return np.random.choice(["0", "1"], p=probabilities[group]['sex'])
    if col_name == 'diabetes':
        return np.random.choice(["0", "1"], p=probabilities[group]['diabetes'])

# Adjusting correlations for other variables
    if col_name == 'smoking_duration':
        age = row['age']
        max_smoking_duration = min(age - 15, 40)  # Max smoking duration limited by age - 15
        smoking_duration = np.random.normal(loc=(age - 15) / 2, scale=5)
        return min(smoking_duration, max_smoking_duration)  # Ensure it does not exceed max

    if col_name == 'BMI':
        diabetes = row['diabetes']
        if diabetes == '1':  # If the person has diabetes, increase BMI
            return np.random.normal(loc=probabilities[group]['BMI'][0] + 3, scale=2)
        else:
            return np.random.normal(loc=probabilities[group]['BMI'][0], scale=2)

    if col_name == 'pack_years':
        smoking_duration = row['smoking_duration']
        cigarettes_per_day = row['cigarettes_per_day']
        return (smoking_duration * cigarettes_per_day) / 20  # Calculate pack-years

    return np.nan

# Apply the probabilities to the synthetic dataset
for col in df.columns:
    df[col] = df.apply(lambda row: apply_probabilities(row, col) if col in probabilities['cases'] else row[col], axis=1)

# Save the DataFrame to a CSV file
output_file_path = r"D:\scifair2025\data\data.csv"
df.to_csv(output_file_path, index=False)

print(f"Data saved to {output_file_path}")