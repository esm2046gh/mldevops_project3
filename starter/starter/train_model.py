# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.data import Dump
from ml.model import train_model
from ml.model import model_performance
from ml.model import model_performance_on_slices
from ml.model import inference
from ml.model import Metrics


# Add code to load in the data.
data_local_path = '../data/census_clean.csv'
data = pd.read_csv(data_local_path)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
output_feature = "salary"

models = {}

model = LogisticRegression(random_state=42)
model_param_grid = {
    'C': [0.1, 1, 10, 100],
    'max_iter': [200, 1000]
}
models["lrc"] = [model, model_param_grid]

model = RandomForestClassifier(random_state=42)
model_param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
}
models["rfc"] = [model, model_param_grid]

model = SVC()
model_param_grid = {
'C': [0.1, 1, 10, 100, 1000],
'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
'kernel': ['rbf']}
models["svc"] = [model, model_param_grid]


model = DecisionTreeClassifier(random_state=42)
model_param_grid = {
    'criterion':['gini','entropy'],
    'max_depth':[4,10,20,50,100]
}
models["dtc"] = [model, model_param_grid]

# Optional enhancement, use K-fold cross validation instead of a train-test split.
#train, test = train_test_split(data, test_size=0.2)
def train_test_split_stratified(data, test_size=0.2, output_label=output_feature):
    train0 = None
    test0 = None
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
    for train_index, test_index in splitter.split(data, data[output_label]):
        train0, test0 = data.iloc[train_index], data.iloc[test_index]

    return train0, test0

if __name__ == "__main__":
    #train, test = train_test_split(data, test_size=0.2)
    train, test = train_test_split_stratified(data, test_size=0.2, output_label=output_feature)
    X_train, y_train, encoder, lb, scaler = process_data(train, categorical_features=cat_features,
                                                         label=output_feature, training=True)

    model_key = "dtc" #"lrc", "rfc", "svc", "dtc"
    model = train_model(X_train, y_train, models[model_key][0], models[model_key][1], cv=3)
    print(f"Overall Performance({model_key}): Train")
    model_performance(model, train, cat_features, output_feature, encoder, lb, scaler)
    print(f"Overall Performance({model_key}): Test")
    model_performance(model, test, cat_features, output_feature, encoder, lb, scaler)
    print(f"Slice Performance({model_key}): Test")
    model_performance_on_slices(model, test, cat_features, output_feature, encoder, lb, scaler)

    dump = Dump('joblib')
    dump.save(model, '../model/model.pkl')
    dump.save(train, '../model/train.pkl')
    dump.save(test, '../model/test.pkl')
    dump.save(cat_features, '../model/cat_features.pkl')
    dump.save(output_feature, '../model/output_feature.pkl')
    dump.save(encoder, '../model/encoder.pkl')
    dump.save(lb, '../model/lb.pkl')
    dump.save(scaler, '../model/scaler.pkl')



# Model inference

