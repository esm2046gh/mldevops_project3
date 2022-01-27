import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from dataclasses import dataclass
from data import process_data

@dataclass
class Metrics:
    '''
    - A class containing the model metrics

    Attributes:
    ----------
    - fbeta (number): fbeta-score
    - precision (number): precision-score
    - recall (number): recall-score
    - f1 (number): f1-score
    - accuracy (number): accuracy_score
    '''

    fbeta = np.nan
    precision = np.nan
    recall = np.nan
    f1 = np.nan
    accuracy = np.nan

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, model, model_param_grid, cv=5):
    """
    Trains a machine learning model and returns it.
    It performs hyperparameter tuning using GridSearchCV

    Inputs
    ------
    - X_train (np.array): Input Training data.
    - y_train (np.array): Target training data (output labels).
    - model : sklearn model for classification
    - model_param_grid (dict): model params grid to tune
    - cv (int): Cross validation parameter. default = 5
    Returns
    -------
    - model (GridSearchCV.best_estimator_): Trained machine learning model.
    """
    print(f"train_model: {type(model).__name__}, cv: {cv}")
    print(f"params_grid: {model_param_grid}")

    cv_grid_searcher = GridSearchCV(estimator=model, param_grid=model_param_grid, cv=cv, verbose=3)
    cv_grid_searcher.fit(X_train, y_train)
    print(f"best_params_ : {cv_grid_searcher.best_params_}")
    print("returning best_estimator_ ...")
    return cv_grid_searcher.best_estimator_

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using  fbeta, precision, recall, f1 and accuracy.

    Inputs
    ------
    - y (np.array): Known labels, binarized.
    - preds(np.array): Predicted labels, binarized.

    Returns
    -------
    - A Metrics object (see class Metrics)
    metrics.precision (float). See sklearn.metrics
    metrics.recall (float).
    metrics.fbeta (float).
    metrics.f1 (float).
    """
    metrics = Metrics()
    metrics.fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    metrics.precision = precision_score(y, preds, zero_division=1)
    metrics.recall = recall_score(y, preds, zero_division=1)
    metrics.f1 = f1_score(y, preds, zero_division=1)
    metrics.accuracy = accuracy_score(y, preds)

    return metrics


def inference(model, X):
    """
    Run model inferences and return the predictions.

    Inputs
    ------
    - model (sklearn model for classification): Trained machine learning model.
    - X (np.array): Data used for prediction.
    Returns
    -------
    - preds (np.array): Predictions from the model.
    """
    preds = model.predict(X)

    return preds


def model_performance(model, data, cat_features, output_feature, encoder, lb, scaler):
    """
    Calculates the models performance

    Inputs
    ------
   - model (sklearn model for classification): Trained machine learning model
   - data (dataframe): Data for assessment
   - cat_features (list): list of categorical features
   - output_feature (str): The output label
   - encoder (OneHotEncoder-object): Encoder for categorical features
   - lb: label binarizer. See process_data in data.py
   - scaler (MinMaxscaler-object): scaler for continous data

    Returns
    -------
    print(...)
    """
    X_data, y_data, encoder, lb, scaler = process_data(
        data, categorical_features=cat_features, label=output_feature, training=False, encoder=encoder, lb=lb, scaler=scaler)

    preds = inference(model, X_data)

    res = compute_model_metrics(y_data, preds)

    print(f"Overall Performance. fbeta: {res.fbeta:.4f}| precision: {res.precision:.4f}| recall: {res.recall:.4f}| accuracy: {res.accuracy:.4f}")


def model_performance_on_slices(model, data, cat_features, output_feature, encoder, lb, scaler):
    """
    Calculates the models performance on the slices of categorical columns.

    Inputs
    ------
   - model (sklearn model for classification): Trained machine learning model
   - data (dataframe): Data for assessment
   - cat_features (list): list of categorical features
   - output_feature (str): The output label
   - encoder (OneHotEncoder-object): Encoder for categorical features
   - lb: label binarizer. See process_data in data.py
   - scaler (MinMaxscaler-object): scaler for continous data

    Returns
    -------
    print(...)
    """
    for cat_feat in cat_features:
        for feat_class in data[cat_feat].unique():
            data_temp = data[data[cat_feat] == feat_class]

            X_data, y_data, encoder, lb, scaler = process_data(
                data_temp, categorical_features=cat_features, label=output_feature, training=False, encoder=encoder, lb=lb, scaler=scaler)

            preds = inference(model, X_data)

            res = compute_model_metrics(y_data, preds)

            print(f"Slice Performance. cat_feat: {cat_feat} | feat_class: {feat_class}| fbeta: {res.fbeta:.4f}| precision: {res.precision:.4f}| recall: {res.recall:.4f}| accuracy: {res.accuracy:.4f}")




