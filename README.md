# Colorectal Cancer Predictor

## Overview

This project generates synthetic datasets for a colorectal cancer study and uses a machine learning model to predict colorectal cancer risk. The synthetic dataset simulates real-world health and lifestyle factors, including age, BMI, red meat consumption, processed meat consumption, alcohol use, smoking history, physical activity, sedentary activity, dietary fiber intake, and family history. The data generation process includes normal and log-normal distributions, with added noise for variability, making it suitable for training and testing the predictive model. The model achieves high accuracy in distinguishing between case and control groups, as detailed in the Models and Pictures folder.

## Features

- **Data Generation**: Create a synthetic dataset with key health metrics related to colorectal cancer.
- **Customizable Attributes**: Adjust mean values and distributions for various health factors, including age, BMI, dietary habits, and smoking history.
- **Case/Control Differentiation**: Apply different probability distributions and correlations for case and control groups.
- **Export to CSV**: Save the generated synthetic dataset to a CSV file for further analysis.
- **Machine Learning Model**: Predict colorectal cancer risk using the generated dataset.

## Requirements

- Python 3.x
- NumPy
- Pandas
- Random
- Tensorflow
- Scikit-learn
- Matplotlib
- SHAP

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/usefhamzawi/Colon-Cancer-Predicter.git
   cd Colon-Cancer-Predicter
   ```

2. Install the required packages:

   ```bash
   pip install requirements.txt
   ```

## Usage

1. Open the main script (e.g., `generatorV3.py`).

2. Adjust the `probabilities` dictionary to fit your study requirements, modifying means and standard deviations as needed. The values in the dictionary have been informed by the following studies (see **Acknowledgments**).

3. Run the Synthetic Data Generator script:

   ```bash
   cd src
   python generatorV3.py
   ```

4. The synthetic dataset will be saved as `normal_data.csv` in the specified output directory. Another preprocessed dataset will be saved as `preprocessed_data.csv`

5. Run the Model script

   ```bash
   cd src
   python model.py
   ```

## Code Explanation

## Synthetic Data Generator

1. **Initialization**:

   - Imports necessary libraries (`numpy`, `pandas`, `random`) and sets up data structures for synthetic data generation.

2. **Probability Definitions**:

   - Defines the mean and standard deviation for various health-related factors, such as age, BMI, alcohol use, smoking, and family history, for both control and case groups. Categorical distributions for attributes like sex, smoking status, and diabetes are also defined based on research studies.

3. **Synthetic Data Generation**:

   - Uses random sampling and the normal/log-normal distribution to generate a synthetic dataset, ensuring diversity and realistic variation in the generated values.

4. **Data Adjustment**:

   - Applies case-control-specific logic to adjust variables like BMI, alcohol use, and smoking behavior based on known health correlations. Adjustments account for factors such as age, sex, and health conditions like diabetes and smoking status.

5. **Standarization and Adding Guassian Noise**

   - After the Data Adjustments the script standarizes the data then adds Guassian Noise. In the data folder the `preprocessed_data.csv` has a level of Guassian Noise at a level of 0.05, the `preprocessed_data_added_noise.csv` has a level of Guassian Noise of 0.08, lastly the `preprocessed_data_extra_noise` has a level of Guassian Noise of 0.1. Guassian Noise is introduced in Machine Learning datasets to improve the models overfitting and overconfidence by augmenting the data to have more variation. The noise is generated using `np.random.normal(loc=0, scale=noise_level, size=len(column))`, this creates randin values from a normal distribution with a mean of 0 and a standard defined by the `noise_level`.

## Model

1. **Initialization**:

   - The `__init__` method sets up the model's input shape and prepares a `StandardScaler` for feature normalization.

2. **Model Building**:

   - The `build_model` method defines a neural network architecture with configurable hidden layers, dropout rates, and L2 regularization. The model uses ReLU activation for hidden layers and a sigmoid output for binary classification.

3. **Data Preparation**:

   - The `prepare_data` method splits the dataset into training and test sets, scaling the features using `StandardScaler`.

4. **Training**:

   - The `train` method trains the model on the training data for a specified number of epochs, using a batch size and validation split.

5. **Evaluation**:

   - The `evaluate` method evaluates the model on the test data, calculating performance metrics such as AUC-ROC, F1 score, and confusion matrix.

6. **Calibration**:

   - The `calibrate_probabilities` method calibrates the model's predicted probabilities using cross-validation and a specified method (e.g., sigmoid).

7. **Calibration Evaluation**:

   - The `evaluate_calibration` method plots a reliability diagram and computes calibration metrics like Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

8. **Model Saving**:

   - The `save_model` method saves the trained model, its architecture, and weights to specified paths.

9. **Detailed Evaluation**:

   - The `detailed_evaluation` method provides an in-depth evaluation, saving the risk scores (predicted probabilities) to a CSV file.

10. **SHAP Analysis**:

    - The `plot_shap_summary` method generates SHAP (Shapley value) plots to visualize feature importance and model explainability. It produces a beeswarm plot for the top 10 most impactful features and the remaining features.

11. **Training History Plotting**:
    - The `plot_training_history` method generates loss and accuracy curves from the training history.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

The following research articles were referenced to inform the probabilities and distributions used in the synthetic data generation:

1. **Population Estimates**  
   [https://www.census.gov/newsroom/press-releases/2023/population-estimates-characteristics.html](https://www.census.gov/newsroom/press-releases/2023/population-estimates-characteristics.html)  
   [https://pmc.ncbi.nlm.nih.gov/articles/PMC7042874/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7042874/)

2. **BMI Study (Lei et al. 2020)**  
   [https://pmc.ncbi.nlm.nih.gov/articles/PMC9905196/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9905196/)  
   [https://pmc.ncbi.nlm.nih.gov/articles/PMC3651246/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3651246/)

3. **Red and Processed Meat Consumption Study**  
   [https://pmc.ncbi.nlm.nih.gov/articles/PMC4698595/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4698595/)  
   [https://pmc.ncbi.nlm.nih.gov/articles/PMC9991741/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9991741/)

4. **Alcohol Consumption**  
   [https://www.cancer.gov/about-cancer/causes-prevention/risk/alcohol/alcohol-fact-sheet#:~:text=Colorectal%20cancer:%20Moderate%20to%20heavy,cancers%20(4%2C%2015)](<https://www.cancer.gov/about-cancer/causes-prevention/risk/alcohol/alcohol-fact-sheet#:~:text=Colorectal%20cancer:%20Moderate%20to%20heavy,cancers%20(4%2C%2015).>)  
   [https://news.gallup.com/poll/467507/percentage-americans-drink-alcohol.aspx#:~:text=How%20Much%20Do%20Drinkers%20Consume,drink%20in%20the%20past%20week](https://news.gallup.com/poll/467507/percentage-americans-drink-alcohol.aspx#:~:text=How%20Much%20Do%20Drinkers%20Consume,drink%20in%20the%20past%20week.)

5. **Drinking Status**  
   [https://www.pewresearch.org/short-reads/2024/01/03/10-facts-about-americans-and-alcohol-as-dry-january-begins/#:~:text=Overall%2C%2062%25%20of%20U.S.%20adults,a%20July%202023%20Gallup%20survey](https://www.pewresearch.org/short-reads/2024/01/03/10-facts-about-americans-and-alcohol-as-dry-january-begins/#:~:text=Overall%2C%2062%25%20of%20U.S.%20adults,a%20July%202023%20Gallup%20survey.)  
   [https://www.sciencedirect.com/science/article/pii/S0002916523041783#:~:text=Descriptive%20characteristics%20of%20the%203121,drinkers%2C%20the%20majority%20were%20men](https://www.sciencedirect.com/science/article/pii/S0002916523041783#:~:text=Descriptive%20characteristics%20of%20the%203121,drinkers%2C%20the%20majority%20were%20men.)

6. **Smoking Duration and Pack Years Study**  
   [https://pmc.ncbi.nlm.nih.gov/articles/PMC7368133/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7368133/)

7. **Smoking Status Graph**  
   [https://pmc.ncbi.nlm.nih.gov/articles/PMC3493822/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3493822/)

8. **Physical Activity**  
   [https://pmc.ncbi.nlm.nih.gov/articles/PMC7048166/#:~:text=Participant%20characteristics%20by%20leisure%2Dtime%20physical%20activity%20are,7.6%20and%208.0%20MET%20hours/week%20(Data%20Supplement)](<https://pmc.ncbi.nlm.nih.gov/articles/PMC7048166/#:~:text=Participant%20characteristics%20by%20leisure%2Dtime%20physical%20activity%20are,7.6%20and%208.0%20MET%20hours/week%20(Data%20Supplement).>)

9. **Sedentary Activity**  
   [https://www.ahajournals.org/doi/10.1161/cir.0000000000000440#:~:text=On%20the%20basis%20of%20objective%20measurement%20from,time%20remained%20stable%20from%202003%E2%80%932004%20to%202005%E2%80%932006](https://www.ahajournals.org/doi/10.1161/cir.0000000000000440#:~:text=On%20the%20basis%20of%20objective%20measurement%20from,time%20remained%20stable%20from%202003%E2%80%932004%20to%202005%E2%80%932006.)

10. **Dietary Fiber Study**  
    [https://pmc.ncbi.nlm.nih.gov/articles/PMC4588743/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4588743/)

11. **Sex Distribution**  
    [https://www.statista.com/statistics/737923/us-population-by-gender/#:~:text=Projection%20estimates%20calculated%20using%20the,US%20Census%20data%20for%202021](https://www.statista.com/statistics/737923/us-population-by-gender/#:~:text=Projection%20estimates%20calculated%20using%20the,US%20Census%20data%20for%202021.)  
    [https://www.cancer.org/cancer/types/colon-rectal-cancer/about/key-statistics.html](https://www.cancer.org/cancer/types/colon-rectal-cancer/about/key-statistics.html)

12. **Diabetes Prevalence**  
    [https://www.cdc.gov/diabetes/php/data-research/index.html](https://www.cdc.gov/diabetes/php/data-research/index.html)

13. **Family History Prevalence**  
    [https://pmc.ncbi.nlm.nih.gov/articles/PMC4955831/#:~:text=Article%20flow%20diagram-,Prevalence,specific%20prevalence%20of%20family%20history](https://pmc.ncbi.nlm.nih.gov/articles/PMC4955831/#:~:text=Article%20flow%20diagram-,Prevalence,specific%20prevalence%20of%20family%20history.)

14. **Race Distribution**  
    [https://www.census.gov/quickfacts/fact/table/US/RHI125223](https://www.census.gov/quickfacts/fact/table/US/RHI125223)  
    [https://pmc.ncbi.nlm.nih.gov/articles/PMC9069392/table/T1/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9069392/table/T1/)

Special thanks to these studies for providing the necessary data and statistics used in building the synthetic datasets for colorectal cancer research.

---
