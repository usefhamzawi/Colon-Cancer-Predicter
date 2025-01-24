import numpy as np
import pandas as pd
import random

# Define the mappings and probabilities for cases and controls
conditions = {
    "0": {  # Controls
        "age": (38.9, 13.4),
        "BMI": (28.35, 5.33),
        "red_meat": (65.6, 10),
        "processed_meat": (38.9, 10),
        "alcohol_use": (8, 4.5),
        "drinking_status": [0.62, 0.38],  # Non-drinker, Drinker
        "smoking_duration": (19.58, 10.98),
        "pack_years": (18.09, 14.58),
        "smoking_status": [0.47, 0.43, 0.10],  # Never, Former, Current
        "physical_activity": (7.8, 4.3),
        "sedentary_activity": (49, 12),
        "dietary_fiber": (16, 6),
        "sex": [0.5047, 0.4953],  # Female, Male
        "diabetes": [0.853, 0.147],  # No, Yes
        "family_history": [0.9345, 0.0655],  # No, Yes
    },
    "1": {  # Cases
        "age": (51.6, 6.7),
        "BMI": (34.10505, 6.41199),
        "red_meat": (74.21, 11.3),
        "processed_meat": (42.401, 10.9),
        "alcohol_use": (10.8, 6.075),
        "drinking_status": [0.16, 0.84],  # Non-drinker, Drinker
        "smoking_duration": (22.3, 12.5),
        "pack_years": (20.6, 16.6),
        "smoking_status": [0.43, 0.45, 0.12],  # Never, Former, Current
        "physical_activity": (6.5988, 3.6378),
        "sedentary_activity": (63.7, 15.6),
        "dietary_fiber": (13.6, 5.1),
        "sex": [0.4794, 0.5206],  # Female, Male
        "diabetes": [0.8303, 0.1697],  # No, Yes
        "family_history": [0.8657, 0.1343],  # No, Yes
    }
}

# Race probabilities for controls
race_probs_controls = [
    0.584,  # White
    0.137,  # Black
    0.067,  # Asian/PI
    0.017,  # American Indian
    0.195   # Hispanic
]


# Race incidence rates for cases
race_probs_cases = [
    0.2017, # White
    0.2285, # Black
    0.1728, # Asian/PI
    0.2143, # American Indian
    0.1827  # Hispanic
]

# Categories for case-control and other binary categories
sex_categories = [0, 1]  # 0 for female, 1 for male
diabetes_categories = [0, 1]  # 0 for without diabetes, 1 for with diabetes
family_history_categories = [0, 1]  # 0 for without family history, 1 for with family history
smoking_categories = [0, 1, 2]  # 0 for Never-smokers, 1 for Former-smokers, 2 for Current-smokers
drinking_categories = [0, 1]  # 0 for non-drinkers, 1 for drinkers
races = [0, 1, 2, 3, 4]  # 0 White, 1 Black, 2 Asian/PI, 3 American Indian, 4 Hispanic

# Create initial mock dataset
n = 100000
noise_level = 0.05

df = pd.DataFrame({
    "id": range(n),
    "case_control": np.random.choice(["0", "1"], size=n, p=[0.50, 0.50])
})


def normal_to_lognormal(mu_X, sigma_X):
    # Calculate mean and standard deviation for the log-normal distribution
    mu_Y = np.log(mu_X**2 / np.sqrt(sigma_X**2 + mu_X**2))
    sigma_Y = np.sqrt(np.log(sigma_X**2 / mu_X**2 + 1))
    
    return mu_Y, sigma_Y

# Function to generate race based on case/control status
def generate_race(row):
    race_probs = race_probs_cases if row['case_control'] == "1" else race_probs_controls
    return np.random.choice(races, p=race_probs)


# Standardize a column
def standardize(column):
    mean = np.mean(column)
    std = np.std(column)
    return (column - mean) / std


# Add Gaussian noise to a column
def add_gaussian_noise(column, noise_level=0.02):
    noise = np.random.normal(loc=0, scale=noise_level, size=len(column))
    return column + noise


# Apply the race generation function
df['race'] = df.apply(generate_race, axis=1)


# Function to generate probability based variables based on the case_control
def generate_probability_vars(row, column_name, category):
    probabilities = conditions[row["case_control"]][column_name]
    return np.random.choice(category, p=probabilities)


# Function to generate number based variables based on the case_control
def generate_num_vars(row, column_name, is_lognormal):
    mean, stdv = conditions[row["case_control"]][column_name]
    if is_lognormal:
        mean, stdv = normal_to_lognormal(mean, stdv)
        return np.random.lognormal(mean=mean, sigma=stdv)
    else:
        return np.random.normal(loc=mean, scale=stdv)
    

def adjust_BMI(BMI, red_meat, processed_meat, diabetes, physical_activity, sedentary_activity, smoking_status, dietary_fiber):
    # Adding a cap to BMI for better consistency
    BMI_adjustment = 0
    
    # Adjusting based on red meat and processed meat
    BMI_adjustment += (red_meat * random.uniform(0.005, 0.015)) + (processed_meat * random.uniform(0.015, 0.025))
    
    # Adjusting based on physical activity
    BMI_adjustment -= physical_activity * random.uniform(0.025, 0.035)
    
    # Adjusting based on sedentary activity
    BMI_adjustment += sedentary_activity * random.uniform(0.015, 0.025)
    
    # Diabetes impact
    if diabetes == 1:
        BMI_adjustment += random.uniform(0.4, 0.6)
    
    # Smoking impact
    if smoking_status == 1:
        BMI_adjustment -= random.uniform(0.15, 0.25)
    
    # Dietary fiber impact
    BMI_adjustment -= dietary_fiber * random.uniform(0.03, 0.05)
    
    # Cap BMI adjustments to realistic limits
    BMI_adjustment = max(min(BMI + BMI_adjustment, 50), 10)
    
    return BMI_adjustment


def adjust_alcohol_consumption(alcohol_use, drinking_status, sex, age):
    if drinking_status == 0:  # Non-drinker
        return 0
    
    # Females drink less on average than males 
    if sex == 0:
        alcohol_use *= random.uniform(0.3, 0.4)

    # Older individuals (above 60) drink less 
    if age > 60:
        alcohol_use *= random.uniform(0.45, 0.55)

    return alcohol_use

def adjust_smoking_duration_pack_years(smoking_duration, pack_years, smoking_status, age, sex, BMI, drinking_status, diabetes):
    adjusted_smoking_duration = smoking_duration
    adjusted_pack_years = pack_years

    if smoking_status == 0:  # Never smoked
        return 0, 0
    elif smoking_status == 1:  # Former smoker
        adjusted_smoking_duration *= random.uniform(0.75, 0.85)  # Reduced duration as smoking ceased
        adjusted_pack_years *= random.uniform(0.65, 0.75)  # Accumulated pack years but no ongoing smoking
    elif smoking_status == 2:  # Current smoker
        if age > 40:
            adjusted_pack_years *= random.uniform(1.15, 1.25)  # Longer smoking history
        if BMI > 30:
            adjusted_pack_years *= random.uniform(1.05, 1.15)  # Obesity correlated with heavier smoking
        if diabetes == 1:
            adjusted_pack_years *= random.uniform(1.25, 1.35)  # Health factors correlate with heavier smoking
        if drinking_status == 1:
            adjusted_pack_years *= random.uniform(1.05, 1.15)  # Drinking increases likelihood of heavier smoking

    return adjusted_smoking_duration, adjusted_pack_years


# Generate numeric and categorical variables
for column_name, is_lognormal in [
    ("age", False),
    ("BMI", False),
    ("red_meat", True),
    ("processed_meat", True),
    ("alcohol_use", True),
    ("smoking_duration", True),
    ("pack_years", True),
    ("sedentary_activity", True),
    ("physical_activity", True),
    ("dietary_fiber", True),
]:
    df[column_name] = df.apply(lambda row: generate_num_vars(row, column_name, is_lognormal), axis=1)


for column_name, category in [
    ("sex", sex_categories),
    ("diabetes", diabetes_categories),
    ("family_history", family_history_categories),
    ("smoking_status", smoking_categories),
    ("drinking_status", drinking_categories),
]:
    df[column_name] = df.apply(lambda row: generate_probability_vars(row, column_name, category), axis=1)



# After generating the initial values for BMI, alchohol_use, and smoking_duration/pack_years
df['BMI'] = df.apply(
    lambda row: adjust_BMI(
        row['BMI'], 
        row['red_meat'], 
        row['processed_meat'], 
        row['diabetes'], 
        row['physical_activity'], 
        row['sedentary_activity'], 
        row['smoking_status'], 
        row['dietary_fiber']
        ), 
        axis=1
)


# Apply the alcohol adjustment function to the DataFrame
df['alcohol_use'] = df.apply(
    lambda row: adjust_alcohol_consumption(
        row['alcohol_use'],
        row['drinking_status'],
        row['sex'],
        row['age']     
        ),
        axis=1
)


# Apply the smoking adjustment function to the DataFrame
df['smoking_duration'], df['pack_years'] = zip(*df.apply(
    lambda row: adjust_smoking_duration_pack_years(
        row['smoking_duration'], 
        row['pack_years'], 
        row['smoking_status'], 
        row['age'], 
        row['sex'], 
        row['BMI'], 
        row['drinking_status'], 
        row['diabetes']
    ), 
    axis=1
))


# Standardize and add noise to columns
columns_to_standardize_and_add_noise = [
    'age', 'BMI', 'red_meat', 'processed_meat', 'alcohol_use', 'smoking_duration', 
    'pack_years', 'physical_activity', 'sedentary_activity', 'dietary_fiber'
]

normal_df = df.copy()

for col in columns_to_standardize_and_add_noise:
    df[col] = standardize(df[col])
    df[col] = add_gaussian_noise(df[col], noise_level=noise_level)

# Function to save DataFrame to CSV
def save_to_csv(df, file_path):
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

# Define output file paths
normal_file_path = r"D:\\Colon-Cancer-Predicter\\data\\normal_data.csv"
preprocessed_file_path = r"D:\\Colon-Cancer-Predicter\\data\\preprocessed_data.csv"

# Save DataFrames to CSV
save_to_csv(normal_df, normal_file_path)
save_to_csv(df, preprocessed_file_path)