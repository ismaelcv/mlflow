from typing import Optional

from sklearn.compose import ColumnTransformer
from sklearn.experimental import (  # pylint: disable=unused-import
    enable_iterative_imputer,
)
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import IterativeImputer  # isort: split

from mlflow.models.frameworks.frameworks_definition import FRAMEWORKS


def get_model(
    numerical_variables: list,
    categorical_variables: Optional[list],
    framework: str,
    user_inputed_hyperparameters: Optional[dict] = None,
) -> Pipeline:

    """
    Returns a sklearn pipeline for the GradientBoostingRegressor model.
    It applies:
        - IterativeImputer to the numerical variables
        - OneHotEncoder to the categorical variables

    """

    if user_inputed_hyperparameters is None:
        user_inputed_hyperparameters = {}

    # Combine both the default and user inputted hyperparameters
    hyper_params = {**FRAMEWORKS[framework]["default_params"], **user_inputed_hyperparameters}

    categorical_pipeline = make_pipeline(
        ColumnTransformer([("onehot_encode", OneHotEncoder(handle_unknown="ignore"), categorical_variables)])
    )

    numerical_pipeline = make_pipeline(ColumnTransformer([("imputer", IterativeImputer(), numerical_variables)]))

    return make_pipeline(
        FeatureUnion([("categorical_pipeline", categorical_pipeline), ("numerical_pipeline", numerical_pipeline)]),
        FRAMEWORKS[framework]["model"](**hyper_params),
    )
