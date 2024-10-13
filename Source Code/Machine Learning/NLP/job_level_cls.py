import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from
import re

def filter_location(location):
    result = re.findall(r",\s[A-Z]{2}", location)
    if len(result):
        return result[0][2:]
    else:
        return location

data = pd.read_excel("./ML_Practice/NLP/final_project.ods", engine="odf", dtype=str)
data["location"] = data["location"].apply(filter_location)
#print(data.head())
#print(data["career_level"].value_counts())

target = "career_level"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(len(x_train['function'].unique()))

# vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
# result = vectorizer.fit_transform(x_train["title"])
# print(vectorizer.vocabulary_)
# print(result.shape)

# encoder = OneHotEncoder()
# result = encoder.fit_transform(x_train[["location"]])
# print(result.shape)


preprocessor = ColumnTransformer(transformers=[
    ('title_feature', TfidfVectorizer(stop_words="english", ngram_range=(1,1)), 'title'),
    ('nom_feature', OneHotEncoder(),['location','function']),
    ('des_feature', TfidfVectorizer(stop_words="english", ngram_range=(1,2)), 'description'),
    ('industry_feature', TfidfVectorizer(stop_words="english", ngram_range=(1,1)), 'industry')
])


model = Pipeline(steps=[
    (),
])