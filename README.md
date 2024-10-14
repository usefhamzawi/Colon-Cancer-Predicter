# Colorectal Cancer Predictor

Note: This project contains a synthetic data generator that simulates realistic health and lifestyle data for colorectal cancer research. The model for predicting colorectal cancer is still under development. Only the data generation component is complete at this stage.

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
- Pandas
- DataLLM library (install using `pip install datallm`)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Colon-Cancer-Predicter.git
   cd Colon-Cancer-Predicter
   ```

2. Install the required packages:

   ```bash
   pip install numpy pandas datallm
   ```

3. Replace `'your-api-key'` in the code with your actual API key from MostlyAI.

## Usage

1. Open the main script (e.g., `generatorV2.py`).
2. Adjust the `probabilities` dictionary to fit your study requirements, modifying means and standard deviations as needed. The values in the dictionary have been informed by the following studies (see **Acknowledgments**).
3. Run the script:

   ```bash
   cd src
   python generatorV2.py
   ```

4. The synthetic dataset will be saved as `data.csv` in the specified output directory.

## Code Explanation

- **Initialization**: Sets up the DataLLM client with the provided API key and base URL.
- **Probability Definitions**: Specifies the mean and standard deviation for various health-related factors, along with categorical distributions for attributes like sex and diabetes status. These are supported by multiple studies referenced in the code.
- **Synthetic Data Generation**: Uses the `mock` method from the DataLLM client to create a dataset of specified size.
- **Data Adjustment**: A function applies case/control-specific logic to adjust certain attributes based on correlations and realistic health patterns derived from research.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

The following research articles were referenced to inform the probabilities and distributions used in the synthetic data generation:

1. [Colorectal Genetics PDQ](https://www.cancer.gov/types/colorectal/hp/colorectal-genetics-pdq#:~:text=About%2075%25%20of%20patients%20with,overall.%5B3%2C4%5D)
2. [American Cancer Society Colorectal Risk Factors](https://www.cancer.org/cancer/types/colon-rectal-cancer/causes-risks-prevention/risk-factors.html#:~:text=probably%20lowers%20risk.-,Smoking,best%20not%20to%20drink%20alcohol.)
3. [American Cancer Society Key Statistics](https://www.cancer.org/cancer/types/colon-rectal-cancer/about/key-statistics.html#:~:text=Overall%2C%20the%20lifetime%20risk%20of,risk%20factors%20for%20colorectal%20cancer.)
4. [Colon Cancer Statistics](https://acsjournals.onlinelibrary.wiley.com/doi/10.3322/caac.21772)
5. [NCBI - Colorectal Cancer Article 1](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9069392/)
6. [NCBI - Colorectal Cancer Article 2](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4698595/)
7. [NCBI - Colorectal Cancer Article 3](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8926870/#:~:text=In%202018%2C%20mean%20global%20intake,were%20generally%20similar%20by%20sex.)
8. [Processed Meat Consumption Study](<https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9991741/#:~:text=Similarly%2C%20the%20American%20Heart%20Association,for%20processed%20meat(16).>)
9. [Smoking Behavior and Prognosis](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8561259/#:~:text=Among%20stage%20IV%20patients%2C%201151,later%20than%2010%20years%20before.)
10. [Analysis of Smoking and Colorectal Cancer](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3493822/#:~:text=The%20basic%20characteristics%20of%20each,by%20colon%20and%20rectal%20cancers.&text=Forest%20plot%20for%20smoking%20status,RE%20model:%20random%20effect%20model.)
11. [NCBI - Smoking Duration and Pack Years](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7368133/)
12. [NCBI - Smoking Study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2697260/)
13. [Red Meat Consumption and Cancer Risk](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5836070/)
14. [Alcohol Consumption and Risk of Colon Cancer](https://pubmed.ncbi.nlm.nih.gov/15623458/)
15. [Gender Differences in the Epidemiology of Alcohol Use](https://arcr.niaaa.nih.gov/volume/40/2/gender-differences-epidemiology-alcohol-use-and-related-harms-united-states#:~:text=In%20the%20United%20States%2C%20more,%2C%206.7%20liters%20for%20females).)
16. [Secular Trends in Alcohol Consumption](<https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2562028/#:~:text=Among%20drinkers%2C%20there%20was%20a,women%20(p%3C%200.001)%2C>)
17. [Global Trends in Smoking](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10488173/)
18. [Pack Years Calculation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6522766/)
19. [Physical Activity Study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6700697/)
20. [Smoking and Colorectal Cancer](<https://www.cghjournal.org/article/S1542-3565(19)31384-9/fulltext#:~:text=Of%2028%2C711%20responders%20to%20the,%E2%80%9354%20y%2C%20respectively.>)
21. [Processed Meat Consumption Research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4588743/)
22. [Colorectal Cancer Epidemiology](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10646729/)
23. [Colorectal Cancer and Diet](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8920658/)

Special thanks to these studies for providing the necessary data and statistics used in building the synthetic datasets for colorectal cancer research.

---
