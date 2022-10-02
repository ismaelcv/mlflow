import random
from typing import Tuple, Union

import numpy as np
import pandas as pd


def enforce_ts_completeness(X: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    This function makes sure that the original data X contains all the observations along the time frame
    It assumes a ts column on datetime format with constant time increments
    freq could be 5m, H, D etc

    """

    original_length = len(X)

    X = pd.DataFrame({"ts": pd.date_range(X.ts.min(), X.ts.max(), freq=freq)}).merge(X)

    if original_length != len(X):
        print(f"The dataset is missing {len(X) - original_length} observations")

    return X


def split_in_CV_sets(X_y: pd.DataFrame, n_splits: int, val_size_perc: float = 0.1) -> dict:
    """
    This function splits a dataset into n CV sets and includes a validation set at the end of the
    timeseries.
    It returns a dictionary including all sets
    """
    X_y_dict = {}

    df_size = len(X_y)

    val_size = int(df_size * val_size_perc)
    X_y_dict["val"] = {"X_y": X_y.iloc[df_size - val_size :]}

    obs_per_cv_split = int((df_size - val_size) / n_splits)

    for i in range(n_splits):
        X_y_dict[f"cv{i}"] = {"X_y": X_y.iloc[i * obs_per_cv_split : (i + 1) * obs_per_cv_split]}

    return X_y_dict


def create_X_and_y_per_cv_split(X_y_dict: dict) -> dict:
    """
    Split in cv

    """
    for item, X_y in X_y_dict.items():
        X_y = X_y["X_y"].sort_values("ts").reset_index(drop=True).drop(columns=["ts"])

        if item == "val":
            X_val, y_val = create_supervised_array(
                X_y,
                number_of_samples=400,
                target_variable="pollution",
                days_of_training=30,
                days_of_prediction=1,
                observations_per_hour=1,
            )

            X_y_dict[item] = {
                "X_val": X_val,
                "y_val": y_val,
                **X_y_dict[item],
            }

        else:
            X_train, y_train, X_test, y_test = create_supervised_split(
                X_y,
                number_of_training_samples=1000,
                number_of_testing_samples=200,
                target_variable="pollution",
                days_of_training=30,
                days_of_prediction=1,
                observations_per_hour=1,
            )

            X_y_dict[item] = {
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
                **X_y_dict[item],
            }

    return X_y_dict


def create_supervised_split(
    X_y: pd.DataFrame,
    number_of_training_samples: int,
    number_of_testing_samples: int,
    target_variable: str,
    days_of_training: int,
    days_of_prediction: int,
    observations_per_hour: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    split for lstm
    """
    train_to_test_ratio = number_of_training_samples / (number_of_training_samples + number_of_testing_samples)

    X_y_train = X_y.iloc[: int(len(X_y) * train_to_test_ratio)]
    X_y_test = X_y.iloc[int(len(X_y) * train_to_test_ratio) :]

    X_train, y_train = create_supervised_array(
        X_y_train,
        number_of_samples=number_of_training_samples,
        target_variable=target_variable,
        days_of_training=days_of_training,
        days_of_prediction=days_of_prediction,
        observations_per_hour=observations_per_hour,
    )

    X_test, y_test = create_supervised_array(
        X_y_test,
        number_of_samples=number_of_testing_samples,
        target_variable=target_variable,
        days_of_training=days_of_training,
        days_of_prediction=days_of_prediction,
        observations_per_hour=observations_per_hour,
    )

    return X_train, y_train, X_test, y_test


def create_supervised_array(
    X_y: pd.DataFrame,
    number_of_samples: int,
    target_variable: str,
    days_of_training: int,
    days_of_prediction: int,
    observations_per_hour: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    something i will fix later
    """

    no_of_obs_per_sample = days_of_training * 24 * observations_per_hour
    no_of_obs_to_predict = days_of_prediction * 24 * observations_per_hour

    X = np.ndarray((number_of_samples, no_of_obs_per_sample, X_y.shape[1]))
    y = np.ndarray((number_of_samples, no_of_obs_to_predict))

    i = 0
    while i < number_of_samples:
        split = random.randint(0, len(X_y))

        sample_X = X_y[split - no_of_obs_per_sample : split]

        if len(sample_X) != no_of_obs_per_sample:
            continue

        sample_y = X_y[split : split + no_of_obs_to_predict][target_variable]

        if len(sample_y) != no_of_obs_to_predict:
            continue

        X[i] = sample_X.to_numpy()
        y[i] = sample_y.to_numpy()

        i += 1

    return X, y


def update_list_in_dict(dictionary: dict, keys: list, value: Union[str, int, float, np.ndarray]) -> None:
    """
    Updates a list inside a nested dictionary
    """

    def retrieve_dict_values(dictionary: dict, keys: list) -> list:
        """
        Retrieve a list of values from a nested dictionary.
        """
        d = dictionary.copy()
        for key in keys:
            d = d[key]
        return d  # type: ignore

    def update_nested_list_in_dict(dictionary: dict, keys: list, value: Union[str, int, float, np.ndarray]) -> None:
        """
        Update a nested list in a dictionary.
        """
        for key in keys[:-1]:
            dictionary = dictionary.setdefault(key, {})
        dictionary[keys[-1]] = value

    updated_list = retrieve_dict_values(dictionary, keys) + [value]
    update_nested_list_in_dict(dictionary, keys, updated_list)
