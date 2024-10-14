import pandas as pd
from run_models import run_classification_models

root_path = '/Users/quanglnt/Documents/AI_ML/Github Learning/AI_ML_Learning/Machine Learning/Classification/data/diabetes.csv'
data = pd.read_csv(root_path)

run_classification_models(data,"Outcome")


