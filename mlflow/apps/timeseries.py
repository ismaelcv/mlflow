import os
import pathlib
from datetime import date, datetime, timedelta
from typing import Tuple

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from PIL import Image
from plotly.subplots import make_subplots

from mlflow.apps.BaseDashApp import BaseDashApp


DATASETS_PATH = pathlib.Path(__file__).parents[2] / "data" / "datasets"


def define_app_controls() -> list:
    """
    This function defines the controls of the dash app.
    The controls will be placed on the left side of the app.
    """

    return []


def define_training_content() -> list:
    """
    This functions defines the content of the dash app.
    Including the graph, sliders and the table.
    """

    return [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Br(),
                        html.Br(),
                        dbc.Label("Select Dataset:"),
                        dcc.Dropdown(id="dataset_dropdown"),
                        html.Br(),
                        dbc.Label("Select timestamp column:"),
                        dcc.Dropdown(id="timestamp_dropdown"),
                        html.Br(),
                        dbc.Label("Select categorical variables:"),
                        dcc.Dropdown(id="categorical_dropdown", multi=True),
                        
                    ]
                ),
                dbc.Col(),
                dbc.Col(),
            ]
        )
    ]


dashApp = BaseDashApp()
dashApp = dashApp.add_controls(define_app_controls()).add_training_content(define_training_content()).render_layout()


@dashApp.app.callback(
    Output("dataset_dropdown", "options"),
    Output("dataset_dropdown", "value"),
    Output("timestamp_dropdown", "options"),
    Output("timestamp_dropdown", "value"),
    Output("categorical_dropdown", "options"),
    Output("categorical_dropdown", "value"),
    [
        Input("dataset_dropdown", "value"),
        Input("timestamp_dropdown", "value"),
        Input("categorical_dropdown", "value"),
    ],
)
def update_training_panel(
    dataset_filename: str,
    timestamp_column: str,
    categorical_variables: list,
) -> Tuple[list, str, list, str, list, str]:
    """
    Update the training panel
    """

    panel_outcome = (
        _load_dataset(dataset_filename)
        + _select_timestamp_column(timestamp_column)
        + _select_categorical_variables(categorical_variables)
    )

    return tuple(panel_outcome)


def _select_categorical_variables(categorical_variables: str):
    """
    This function looks for parquet in the specified folder
    """

    col_list = [item[0] for item in dashApp.X_y.dtypes.items() if item[1] not in ["datetime64[ns]", "float", "int"]]

    if len(col_list) == 0:
        return [], []

    if categorical_variables is None:
        categorical_variables = []

    return col_list, categorical_variables


def _select_timestamp_column(timestamp_column: str):
    """
    This function looks for parquet in the specified folder
    """

    col_list = [item[0] for item in dashApp.X_y.dtypes.items() if item[1] == "datetime64[ns]"]

    if len(col_list) == 0:
        return [], ""

    if timestamp_column is None:
        timestamp_column = ""

    return col_list, timestamp_column


def _load_dataset(dataset_filename: str):
    """
    This function looks for parquet in the specified folder
    """

    dataset_list = [file for file in os.listdir(DATASETS_PATH) if file.endswith(".parquet")]

    if dataset_list is []:
        return [], ""

    if dataset_filename == dashApp.dataset_filename:
        return dataset_list, dataset_filename

    if dataset_filename is None:
        dataset_filename = dataset_list[0]

    dashApp.X_y = pd.read_parquet(DATASETS_PATH / dataset_filename)
    dashApp.dataset_filename = dataset_filename

    return dataset_list, dataset_filename


def run_app() -> None:
    """
    Helper function to call the app.run_server() function with poetry
    """
    dashApp.app.run_server(debug=True, port="8086", use_reloader=False)


if __name__ == "__main__":
    dashApp.app.run_server(debug=True, port="8093")
