from typing import Tuple, Union, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

columns = (['duration'
,'protocol_type'
,'service'
,'flag'
,'src_bytes'
,'dst_bytes'
,'land'
,'wrong_fragment'
,'urgent'
,'hot'
,'num_failed_logins'
,'logged_in'
,'num_compromised'
,'root_shell'
,'su_attempted'
,'num_root'
,'num_file_creations'
,'num_shells'
,'num_access_files'
,'num_outbound_cmds'
,'is_host_login'
,'is_guest_login'
,'count'
,'srv_count'
,'serror_rate'
,'srv_serror_rate'
,'rerror_rate'
,'srv_rerror_rate'
,'same_srv_rate'
,'diff_srv_rate'
,'srv_diff_host_rate'
,'dst_host_count'
,'dst_host_srv_count'
,'dst_host_same_srv_rate'
,'dst_host_diff_srv_rate'
,'dst_host_same_src_port_rate'
,'dst_host_srv_diff_host_rate'
,'dst_host_serror_rate'
,'dst_host_srv_serror_rate'
,'dst_host_rerror_rate'
,'dst_host_srv_rerror_rate'
,'attack'
,'level'])

def get_model_params(model: LogisticRegression) -> LogRegParams:
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params

def set_model_params(
        model: LogisticRegression, params: LogRegParams
    ) -> LogisticRegression:
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def set_initial_params(model: LogisticRegression):
    n_classes = 2
    n_features = 10
    model.classes_ = np.array([i for i in range(2)])
    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes))
    return

def preprocessing(file_path_train, file_path_test):
    df_train = pd.read_csv(file_path_train, header=None, names=columns)
    df_test = pd.read_csv(file_path_test, header=None, names=columns)
    
    # No duplicate data in the dataset
    # No null data

    # Classify attack types to attack and non-attack
    df_train['bi_attack'] = df_train.attack.map(lambda val: 'normal' if val == 'normal' else 'abnormal')
    df_train.drop('attack', axis=1, inplace=True)
    df_test['bi_attack'] = df_test.attack.map(lambda val: 'normal' if val == 'normal' else 'abnormal')
    df_test.drop('attack', axis=1, inplace=True)

    # Encode objects
    le = LabelEncoder()
    cols = ['protocol_type', 'service', 'flag', 'bi_attack']
    for col in cols:
        df_train[col] = le.fit_transform(df_train[col])
        df_test[col] = le.fit_transform(df_test[col])

    x_train = df_train.drop('bi_attack', axis=1)
    y_train = df_train.bi_attack

    x_test = df_test.drop('bi_attack', axis=1)
    y_test = df_test.bi_attack


    # only use the top 10
    col=['service', 'flag', 'src_bytes', 'dst_bytes', 'logged_in',
       'same_srv_rate', 'diff_srv_rate', 'dst_host_srv_count',
       'dst_host_same_srv_rate', 'dst_host_diff_srv_rate']
    x_train = x_train[col]
    x_test = x_test[col]

    # scale
    scaler = MinMaxScaler()
    x_train= scaler.fit_transform(x_train)
    x_test= scaler.fit_transform(x_test)

    return (x_train, y_train), (x_test, y_test)

def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )