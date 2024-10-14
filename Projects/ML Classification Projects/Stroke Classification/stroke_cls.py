import pandas as pd
from run_models import run_classification_models
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

data = pd.read_excel("Classification/data/stroke_classification.xlsx",dtype=str)
le = LabelEncoder()

# Convert 'gender' to numerical values
le = LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])
print(data.head())

# Fill NaN values with the mean
imputer = SimpleImputer(strategy="mean")
data['bmi'] = imputer.fit_transform(data[['bmi']])

run_classification_models(data, "stroke")




