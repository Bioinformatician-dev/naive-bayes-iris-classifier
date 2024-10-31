# Naive Bayes Classifier on Iris Dataset

This project demonstrates the application of various Naive Bayes classifiers on the Iris dataset to classify iris flowers based on their petal and sepal characteristics. It uses Scikit-Learn's built-in dataset and models to evaluate performance across different Naive Bayes variants.


## Project Structure
```bash
naive-bayes-iris-classifier/
├── classifier.ipynb                   # Main script to run the models
├── requirements.txt          # List of dependencies
└── README.md                 # Project documentation
```


## Overview

The project covers:
- Loading and splitting the Iris dataset
- Training and evaluating different Naive Bayes models
  - **GaussianNB**
  - **MultinomialNB**
  - **BernoulliNB**
  - **ComplementNB**
- Comparing model performances using accuracy, confusion matrices, and classification reports.

## Dependencies

* Python 3.7+
* notebook
* scikit-learn
* pandas
* numpy
* matplotlib
* seaborn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Bioinformatician-dev/naive-bayes-iris-classifier.git
   cd naive-bayes-iris-classifier
   ```

2.Install the required packages
```bash
   pip install -r requirements.txt
```
## Usage
1.Run the main script to train and evaluate each model: Open Jupyter Notebook and navigate to the classifier.ipynb file:
```bash
    classifier.ipynb
```
2.The results, including confusion matrices and classification reports, will be printed in the console.



3.Install these dependencies by running:

```bash
   pip install notebook scikit-learn pandas numpy matplotlib seaborn
```

## Results
Each Naive Bayes model is evaluated using the following metrics:

Accuracy Score
Confusion Matrix
Classification Report
A confusion matrix visualization will be shown for each classifier.
