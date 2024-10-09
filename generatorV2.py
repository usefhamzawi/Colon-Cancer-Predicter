import numpy as np
import pandas as pd
from datallm import DataLLM

# Initialize DataLLM client with API key
API_KEY = 'zpka_70513e3546da4445acf8834922ed1f0b_7f9500b4'
BASE_URL = 'https://data.mostly.ai'
datallm = DataLLM(api_key=API_KEY, base_url=BASE_URL)

# Define the mappings and probabilities for cases and controls
conditions = {
    "0": {
        "age": (54, 8),  # Normal distribution for age
        "BMI": (25, 3),  # Normal distribution for BMI
        "red_meat": (51, 10),  # Log-normal: Mean red meat intake = 51g/day, Std Dev = 10 
        "processed_meat": (38.9, 10),  # Log-normal: Mean processed meat intake = 38.9g/day, Std Dev = 10 
        "alcohol_use": (15, 4.5),  # Log-normal: Mean alcohol use = 15g/day, Std Dev = 4.5 
        "smoking_duration": (12.9, 4.1),  # Log-normal: Mean smoking duration = 12.9 years, Std Dev = 4.1 
        "pack_years": (8, 5),  # Log-normal: Mean pack-years = 8, Std Dev = 5 
        "moderate_activity": (150, 40),  # Log-normal: Mean moderate physical activity = 150 min/week, Std Dev = 40 
        "vigorous_activity": (100, 40),  # Log-normal: Mean vigorous physical activity = 100 min/week, Std Dev = 40 
        "strength_activity": (3, 1),  # Log-normal: Mean strength-based activity = 3 times/week, Std Dev = 1
        "dietary_fiber": (30, 5),  # Log-normal: Mean dietary fiber = 30g/day, Std Dev = 5
        "sex": [0.52, 0.48],  # Distribution of sexes in controls: 52% female, 48% male
        "diabetes": [0.884, 0.116],  # 88.4% of controls do not have diabetes, 11.6% do
        "famhx1": [0.906, 0.094]  # Family history: 90.6% do not have a family history, 9.4% do
    },
    "1": {
        "age": (58, 8),  # Normal distribution for age
        "BMI": (30, 4),  # Normal distribution for BMI
        "red_meat": (59.67, 10),  # Log-normal: Mean = 59.67g/day, Std Dev = 10
        "processed_meat": (58, 10),  # Log-normal: Mean = 58g/day, Std Dev = 10 
        "alcohol_use": (22.8, 11.4),  # Log-normal: Mean alcohol use = 22.8g/day, Std Dev = 11.4 
        "smoking_duration": (16.705, 5.31),  # Log-normal: Mean smoking duration = 16.705 years, Std Dev = 5.31
        "pack_years": (20, 8),  # Log-normal: Mean pack-years = 20, Std Dev = 8 
        "moderate_activity": (60, 20),  # Log-normal: Mean moderate physical activity = 60 min/week, Std Dev = 20 
        "vigorous_activity": (30, 15),  # Log-normal: Mean vigorous physical activity = 30 min/week, Std Dev = 15 
        "strength_activity": (1, 0.5),  # Log-normal: Mean strength-based activity = 1 time/week, Std Dev = 0.5 
        "dietary_fiber": (25.5, 5),  # Log-normal: Mean dietary fiber = 25.5g/day, Std Dev = 5
        "sex": [0.48, 0.52],  # Distribution of sexes in cases: 48% female, 52% male
        "diabetes": [0.8378, 0.1622],  # 84% of cases do not have diabetes, 16% do
        "famhx1": [0.7913, 0.2087]  # Family history: 79.13% do not have a family history, 20.87% do
    }
}

# Softmax function to calculate probabilities
def softmax(x):
    e_x = np.exp(x - np.max(x))  # Numerical stability
    return e_x / e_x.sum()  # Normalize

# Colorectal cancer incidence by race
crc_incidence_rates = {
    "White": 37.0 / 100000,
    "Black": 41.9 / 100000,
    "Asian/PI": 31.7 / 100000,
    "American Indian": 39.3 / 100000,
    "Hispanic": 33.5 / 100000
}

# Calculate softmax probabilities for cases and controls
incidence_values = np.array(list(crc_incidence_rates.values()))
race_probs_cases = softmax(incidence_values)
race_probs_controls = softmax(1 - incidence_values)

# categories
sex_categories = [0, 1] # 0 for female 1 for male
diabetes_categories = [0, 1] # 0 for without diabetes 1 for with diabetes
famhx1_categories = [0, 1] # 0 for without family history 1 for with family history
races = [0, 1, 2, 3, 4] # 0 White, 1 Black, 2 Asian/PI, 3 American Indian, 4 Hispanic

df = datallm.mock(
    n=100,
    data_description="Synthetic dataset for colorectal cancer study",
    columns={
        "id": 
        {
            "prompt": "unique id for each case/control",
            "dtype": "integer",
        },
        "case_control":
        { 
            "dtype": "category", 
            "categories": ["0", "1"], # 0 for control and 1 for case
            "probabilities": [0.50, 0.50]  
        }
    }
)

def normal_to_lognormal(mu_X, sigma_X):
    """
    Convert normal distribution parameters to lognormal distribution parameters.

    Parameters:
    mu_X (float): Mean of the normal distribution.
    sigma_X (float): Standard deviation of the normal distribution.

    Returns:
    tuple: Mean (mu_Y) and standard deviation (sigma_Y) of the corresponding lognormal distribution.
    """
    # Calculate mean and standard deviation for the log-normal distribution
    mu_Y = np.log(mu_X**2 / np.sqrt(sigma_X**2 + mu_X**2))
    sigma_Y = np.sqrt(np.log(sigma_X**2 / mu_X**2 + 1))
    
    return mu_Y, sigma_Y

def generate_probability_vars(row, column_name, category):
    probabilities = conditions[row["case_control"]][column_name]
    return pd.Series(category).sample(weights=probabilities).values[0]

def generate_num_vars(row, column_name, is_lognormal):
    mean, stdv = conditions[row["case_control"]][column_name]
    if is_lognormal:
        mean, stdv = normal_to_lognormal(mean, stdv)
        return np.random.lognormal(mean=mean, sigma=stdv)
    else:
        return np.random.normal(loc=mean, scale=stdv)
    
# Function to adjust meat intake based on BMI
def adjust_meat_intake(BMI, red_meat, processed_meat):
    red_meat_adjusted, processed_meat_adjusted = red_meat, processed_meat
    if BMI < 18.5:  # Underweight
        red_meat_adjusted *= 0.9  # Decrease by 10%
        processed_meat_adjusted *= 0.9
    elif 18.5 <= BMI < 24.9:  # Normal weight
        red_meat_adjusted *= 1.0
        processed_meat_adjusted *= 1.0
    elif 25 <= BMI < 29.9:  # Overweight
        red_meat_adjusted *= 1.1  # Increase by 10%
        processed_meat_adjusted *= 1.1
    else:  # Obese
        red_meat_adjusted *= 1.2  # Increase by 20%
        processed_meat_adjusted *= 1.2
    
    # Ensure values are within realistic bounds
    red_meat_adjusted = min(max(red_meat_adjusted, 10), 100)  # Limits between 10g and 100g
    processed_meat_adjusted = min(max(processed_meat_adjusted, 5), 90)  # Limits between 5g and 90g
    
    return red_meat_adjusted, processed_meat_adjusted

# Function to adjust smoking duration based on pack years
def adjust_smoking_duration(pack_years, smoking_duration):
    # Set boundaries for smoking duration
    smoking_duration_adjusted = smoking_duration
    
    # Ensure consistency between smoking duration and pack years
    if smoking_duration == 0:
        pack_years = 0
    elif pack_years == 0:
        smoking_duration = 0
    
    if pack_years < 5:
        smoking_duration_adjusted *= 0.75  # Decrease for low pack years
    elif 5 <= pack_years < 15:
        smoking_duration_adjusted *= 1.0  # No adjustment
    else:
        smoking_duration_adjusted *= 1.25  # Increase for high pack years
    
    # Ensure realistic bounds
    smoking_duration_adjusted = min(max(smoking_duration_adjusted, 0), 40) 
    
    return smoking_duration_adjusted

# Apply the functions to generate the additional variables
for column_name, is_lognormal in [
    ("age", False),
    ("BMI", False),
    ("red_meat", True),
    ("processed_meat", True),
    ("alcohol_use", True),
    ("smoking_duration", True),
    ("pack_years", True),
    ("moderate_activity", True),
    ("vigorous_activity", True),
    ("strength_activity", True),
    ("dietary_fiber", True)
]:
    df[column_name] = df.apply(generate_num_vars, axis=1, column_name=column_name, is_lognormal=is_lognormal)

# After generating the initial values for red_meat, processed_meat, and smoking_duration
df['red_meat'], df['processed_meat'] = zip(*df.apply(
    lambda row: adjust_meat_intake(row['BMI'], row['red_meat'], row['processed_meat']), axis=1))

df['smoking_duration'] = df.apply(
    lambda row: adjust_smoking_duration(row['pack_years'], row['smoking_duration']), axis=1)

for column_name, category in [
    ("sex", sex_categories),
    ("diabetes", diabetes_categories),
    ("famhx1", famhx1_categories)
]:
    df[column_name] = df.apply(generate_probability_vars, axis=1, column_name=column_name, category=category)

# Define a function to generate race based on case_control status
def generate_race(row):
    if row['case_control'] == "0":  # Control
        probabilities = race_probs_controls
    else:  # Case
        probabilities = race_probs_cases
    
    return np.random.choice(races, p=probabilities)

# Apply the function to generate the race column
df['race'] = df.apply(generate_race, axis=1)

# Format numeric outputs to two decimal places
numeric_columns = ['age', 'BMI', 'red_meat', 'processed_meat', 'alcohol_use', 'smoking_duration', 'pack_years', 'moderate_activity', 'vigorous_activity', 'strength_activity', 'dietary_fiber']
df[numeric_columns] = df[numeric_columns].round(2)

# Save DataFrame to CSV file
output_file_path = r"D:\Colon-Cancer-Predicter\data\data.csv"
df.to_csv(output_file_path, index=False)

print(f"Data saved to {output_file_path}")