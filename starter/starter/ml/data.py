import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

class Dump:
    '''
    A class for saving and loading object using either 'pickle' or 'joblib'

    Attributes:
    ----------
        - dumper(str): 'joblib' or 'pickle'. default = 'joblib'

    Methods:
    -------
        - save(obj, filename): Saves the object 'obj' on path 'filename'
        - obj = load(filename): Returns  and object from path 'filename'
    '''

    def __init__(self, dumper='joblib'):
        try:
            assert dumper in ['joblib', 'pickle']

            self.dumper = __import__(dumper)
            self.dumper_str = dumper
        except AssertionError as err:
            #logging.error("Wrong object dumper. %s.", dumper)
            raise err

    #@ee_log("Dump.save")
    def save(self, obj, file_path):
        """
        save(obj, filename): Saves the object 'obj' on path 'filename'
        """
        print(f"FILE PATH: {file_path}")
        func = getattr(self.dumper, 'dump')
        with open(file_path, 'wb') as a_file:
            if self.dumper_str == 'pickle':
                func(obj, a_file, self.dumper.HIGHEST_PROTOCOL)
            else:
                func(obj, a_file)

    #@ee_log("Dump.load")
    def load(self, file_path):
        """
        obj = load(filename): Returns  and object from path 'filename'
        """
        print(f"FILE PATH: {file_path}")
        func = getattr(self.dumper, 'load')
        with open(file_path, 'rb') as a_file:
            obj = func(a_file)
            a_file.close()
        return obj

def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None, scaler=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    scaler: sklearn.preprocessing.MinMaxScaler
        MinMaxScaler for continous features, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer passed in.
    scaler : sklearn.preprocessing.MinMaxScaler
        Trained MinMaxScaler if training is True, otherwise returns the scaler passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()

        scaler = MinMaxScaler()
        X_continuous = scaler.fit_transform(X_continuous)
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

        X_continuous = scaler.transform(X_continuous)

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb, scaler
