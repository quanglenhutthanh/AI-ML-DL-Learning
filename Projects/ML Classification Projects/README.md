# Introduction

This project demonstrate demonstrates the process of using various machine learning algorithms on a given dataset.

The project use scikit-learn, one of the most popular machine learning libraries in Python, to implement and evaluate the model.

# Prerequisites
## scikit-learn
scikit-learn is a popular open source library for machine learning in Python. It provides simple and efficient tools for data mining and data analysis, it alse offer variety of machine learning algorithms for both supervised and unsupervised learning. In this project, scikit-learn is used for:
- Date preprocessing task like scaling and splitting the data.
- Implemeting machine learning algorithms such as Support Vector Machine (SVM), Random Forest.
- Evaluation model performance based on metrics such as accuracy, precision, recall, F1 score.

## pandas
pandas is an open-source library for data manipulation and data analysis in Python. It provides powerful, flexible and easy-to-use data structures, such as DataFrame. With pandas, use can easily load data from various format (such as CSV, excel, json,...), clean and transform datasets, perform data analysis and integrate with others libraries for machine learning and visualization.

## ydata-profiling
ydata-profiling is a Python library that generates a data analysis report for a given dataset. It helps in quickly understanding the data, identifying missing values, detecting correlations, and generating visualizations, all with just a few lines of code.

# Dataset

## Diabetes Dataset Overview
This dataset contains medical records of patients, aimed at predicting whether or not they have diabetes based on certain health measurements. Below is a brief explanation of each feature:

- Pregnancies: Number of times the patient has been pregnant.

- Glucose: Blood sugar levels after a glucose tolerance test.

- BloodPressure: Diastolic blood pressure (the lower number in a blood pressure reading).

- SkinThickness: Thickness of skin on the triceps, used to estimate body fat.

- Insulin: Insulin levels in the blood two hours after a test.

- BMI: Body Mass Index, calculated from height and weight, indicating body fat levels.

- DiabetesPedigreeFunction: A score that indicates the patient’s likelihood of diabetes based on their family history.

- Age: The patient’s age.

- Outcome: The result of whether the patient has diabetes (1) or not (0).

## Stroke dataset overview

# Implamentation
## Data Preparation
The dataset is loaded using pandas.

read_csv
```python
data = pd.read_csv("diabetes.csv")
print(data.head())
```

read_excel
```python
data = pd.read_excel("Classification/data/stroke_classification.xlsx",dtype=str)
```

Use ydata-profiling to gain insights in to the dataset.

```python
# profiling
profile = ProfileReport(data,title="Dataset Report", explorative=True)
profile.to_file("dataset_report.html")

```
## Data Preporcessing

In this project, we will use some of essential functions provided by scikit-learn:

**SimpleImputer** is used to handle missing data by filling in with the specified strategy, such as mean, median, or most frequent value.

**LabelEncoder** converts caterical variebles into numerical values.

using **train_test_split** function to device dataset into 2 subsets, one for training model and one for testing it.

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
```
**StandardScaler** 
Data scaler
```python
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
```
## Model Selection
### Support Vector Machine
Support Vector Machine is a supervised machine learning algorithm use for classification and regression tasks. The primary goal of SVM is find the optimal hyperplate that seperate data points in different classes.

### Decision Tree
The Decision Tree algorithm is a popular supervised learning method used for both classification and regression tasks. It models decisions in a tree-like structure, where each internal node represents a "decision" or "test" on an attribute, each branch represents the outcome of the decision, and each leaf node represents a class label (in classification) or a value (in regression).

![image](/Machine%20Learning/Classification/img/decision_tree.png)

### Random Forest
Random Forest is an ensemble learning method primarily used for classification and regression tasks. It builds multiple decision trees and merges them together to get a more accurate and stable prediction. The idea is to combine the predictions of multiple decision trees to reduce the likelihood of overfitting, improve accuracy, and enhance robustness.

![image](/Machine%20Learning/Classification/img/random_forest.jpg)

```python
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

 models ={
        'SVM' : SVC(),
        'RandomForest': RandomForestClassifier(),
        'DecisionTree': DecisionTreeClassifier()
    }
```

Train and test the model
- The `fit()` function will train the model using training data `x_train` and the correspond lable `y_train'
- After training the model make prediction on the test data `x_test` using `model.predict`


```python
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print(y_predict)
```
## Model Evaluation

Evaluate performance of the model by calculate metrics such as: accuracy, precision, recall and f1 score

```python
'accuracy' : accuracy_score(y_test, y_predict),
'precision' : precision_score(y_test, y_predict,average="weighted"),
'recall' : recall_score(y_test, y_predict, average="weighted"),
'f1_score' : f1_score(y_test, y_predict, average="weighted")
```
Show confusion matrix
```python
conf_matrix = confusion_matrix(y_test, y_predict)
```

View classification report
```python
class_report = classification_report(y_test, y_predict)
```

## Model Optimization
Use gridSearchCV to perform hyperparameter tuning for algorithms

example:
```python
params = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf", "poly"],
    "gamma": ["scale", "auto"]
}
clf = GridSearchCV(
    estimator=SVC(random_state=100),
    param_grid=params,
    scoring="f1",
    cv=6,
    verbose=1,
    n_jobs=6
)
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
print(clf.best_score_)
print(clf.best_params_)
```
