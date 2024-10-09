# Synthetic Data Generation for Colorectal Cancer Study

## Overview

This project generates synthetic datasets for a colorectal cancer study using the `DataLLM` library from MostlyAI. The synthetic data simulates real-world health and lifestyle factors, allowing researchers to analyze patterns and relationships without compromising patient privacy.

## Features

- **Data Generation**: Create a synthetic dataset with key health metrics related to colorectal cancer.
- **Customizable Attributes**: Adjust mean values and distributions for various health factors, including age, BMI, dietary habits, and smoking history.
- **Case/Control Differentiation**: Apply different probability distributions and correlations for case and control groups.
- **Export to CSV**: Save the generated synthetic dataset to a CSV file for further analysis.

## Requirements

- Python 3.x
- NumPy
- DataLLM library (install using `pip install datallm`)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Colon-Cancer-Predicter.git
   cd Colon-Cancer-Predicter
   ```

2. Install the required packages:

   ```bash
   pip install numpy datallm
   ```

3. Replace `'your-api-key'` in the code with your actual API key from MostlyAI.

## Usage

1. Open the main script (e.g., `generatorV2.py`).
2. Adjust the `probabilities` dictionary to fit your study requirements, modifying means and standard deviations as needed.
3. Run the script:

   ```bash
   python generatorV2.py
   ```

4. The synthetic dataset will be saved as `data.csv` in the specified output directory.

## Code Explanation

- **Initialization**: Sets up the DataLLM client with the provided API key and base URL.
- **Probability Definitions**: Specifies the mean and standard deviation for various health-related factors, along with categorical distributions for attributes like sex and diabetes status.
- **Synthetic Data Generation**: Uses the `mock` method from the DataLLM client to create a dataset of specified size.
- **Data Adjustment**: A function applies case/control-specific logic to adjust certain attributes based on correlations and realistic health patterns.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MostlyAI](https://mostly.ai/) for providing the DataLLM library used in this project.
- Contributions and feedback from the research community.

---

Feel free to modify any sections to better fit your project specifics!
