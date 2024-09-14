
**Parkinson's Disease Classification**

This project involves classifying Parkinson's Disease status using various machine learning algorithms. The models are implemented to predict the status of patients based on features extracted from their medical records. The following algorithms are utilized:

- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Modified K-Nearest Neighbors (Modified KNN)**
- **Random Forest (RF)**
- **Gaussian Mixture Model (GMM)**

The project includes visualizations such as confusion matrices and accuracy plots for each algorithm to assess their performance.

## Features

- **Data Preprocessing**: Normalizes the dataset using Min-Max scaling.
- **Model Training and Evaluation**: Trains various classifiers and evaluates their performance.
- **Visualization**: Generates confusion matrices and accuracy plots for model evaluation.

---

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:

- `pandas`
- `numpy`
- `sklearn`
- `seaborn`
- `matplotlib`

You can install the necessary packages using pip:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

### Dataset

The dataset used in this project is `Parkinsons.csv`. Ensure that this file is located in the same directory as the script or provide the correct path in the code.

### Running the Code

To run the classification models and visualize the results, execute the following Python script:

```bash
python classification_models.py
```

This script will:

1. **Load and Preprocess Data**: Read the `Parkinsons.csv` dataset and normalize the features.
2. **Train and Evaluate Models**: Use SVM, KNN, Modified KNN, Random Forest, and Gaussian Mixture Model classifiers to train and evaluate the models.
3. **Generate Visualizations**: Display confusion matrices and accuracy plots for each classifier.

---

## Code Overview

### 1. **SVM Algorithm (`SVM_Algo`)**

Trains a Support Vector Machine with a linear kernel, evaluates its performance, and displays a confusion matrix.

### 2. **KNN Algorithm (`KNN_Algo`)**

Trains a K-Nearest Neighbors classifier with different values of k (1, 5, 9, 13), evaluates performance, and visualizes confusion matrices and accuracy.

### 3. **Modified KNN Algorithm (`Mod_KNN_Algo`)**

Implements a modified KNN classifier using Mahalanobis distance and distance-weighted voting, evaluates performance, and visualizes results.

### 4. **Random Forest Algorithm (`RF_Algo`)**

Trains a Random Forest classifier with 40 trees and a maximum depth of 15, evaluates performance, and visualizes the confusion matrix.

### 5. **Gaussian Mixture Model (GMM) (`GMM1` and `GMM2`)**

Trains and evaluates Gaussian Mixture Models with varying numbers of components, calculates accuracy, and visualizes the confusion matrix.

---

## Results

The final section of the script summarizes the accuracy of each classifier and provides a comparative bar chart.

Example Output

Accuracy for SVM: 88.136%
Accuracy for Random Forest : 94.915%

Contributing

If you would like to contribute to this project, please fork the repository and create a pull request with your proposed changes.

---

License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to adjust any sections to better match your project's specifics or personal preferences!
