#Import the necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, ComplementNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np




# **Load Dataset**




#LoadtheIrisdataset
iris = load_iris()

X = iris.data
y = iris.target

# Combine X and y arrays
combined_data = np.concatenate((X, y.reshape(-1, 1)), axis=1)

# Create a DataFrame
df_combined = pd.DataFrame(combined_data, columns=iris.feature_names + ['target'])

# Replace target values with corresponding species names
df_combined['target'].replace({0: 'setosa', 1: 'versicolor', 2: 'virginica'}, inplace=True)

# Plot the distributions based on y
for feature in df_combined.columns[:-1]:
    sns.displot(data=df_combined, x=feature, hue='target', kde=True)
    plt.title(f"Distribution of {feature} by target")
    plt.show()

# **Split Dataset into train-test sets**

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **Classification by Naive Bayes Algorithm**

# List of Naive Bayes classifiers
classifiers = [
    GaussianNB(),
    MultinomialNB(),
    BernoulliNB(),
    ComplementNB()
]

# Store evaluation results
results = []

# Train and evaluate each classifier
for classifier in classifiers:
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, target_names=iris.target_names)

    # Store results in the list
    results.append({
        "Classifier": classifier.__class__.__name__,
        "Accuracy": accuracy,
        "Precision (macro avg)": report['macro avg']['precision'],
        "Recall (macro avg)": report['macro avg']['recall'],
        "F1 Score (macro avg)": report['macro avg']['f1-score']
    })

    # Display the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {classifier.__class__.__name__}')
    plt.show()

# Create a DataFrame for the results
results_df = pd.DataFrame(results)
print("Evaluation Metrics for Naive Bayes Classifiers on Iris Dataset")
print(results_df)

# Plot the accuracies
labels = [result["Classifier"] for result in results]
accuracies = [result["Accuracy"] for result in results]
plt.bar(labels, accuracies)
plt.xlabel('Naive Bayes Classifier')
plt.ylabel('Accuracy')
plt.title('Accuracy of Naive Bayes Classifiers on Iris Dataset')

# Add the value on top of each bar
for i, v in enumerate(accuracies):
    plt.text(i, v, str(round(v, 4)), ha='center', va='bottom')
plt.show()
