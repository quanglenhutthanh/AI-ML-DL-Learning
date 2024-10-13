import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

def generate_profile_report(data):
    profile = ProfileReport(data,title="Dataset Report", explorative=True)
    profile.to_file("dataset_report.html")

def run_classification_models(data, target_column):
    
    x = data.drop(target_column, axis = 1)
    y = data[target_column]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    models ={
        'SVM' : SVC(),
        'RandomForest': RandomForestClassifier(),
        'DecisionTree': DecisionTreeClassifier()
    }

    metrics = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1_score': f1_score
    }

    results = {}
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        results[model_name] = {
            'accuracy' : accuracy_score(y_test, y_predict),
            'precision' : precision_score(y_test, y_predict,average="weighted"),
            'recall' : recall_score(y_test, y_predict, average="weighted"),
            'f1_score' : f1_score(y_test, y_predict, average="weighted"),
        }
    
    # Print results
    print(f"{'Algorithm':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 60)
    for model_name, metric_scores in results.items():
        print(f"{model_name:<20} {metric_scores['accuracy']:<10.4f} {metric_scores['precision']:<10.4f} {metric_scores['recall']:<10.4f} {metric_scores['f1_score']:<10.4f}")

    conf_matrix = confusion_matrix(y_test, y_predict)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Classification Report (includes precision, recall, f1-score)
    class_report = classification_report(y_test, y_predict)
    print("Classification Report:")
    print(class_report)










