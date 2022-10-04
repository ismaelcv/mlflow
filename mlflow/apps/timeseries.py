import os
import pathlib
from datetime import timedelta
from typing import Tuple

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

from mlflow.apps.BaseDashApp import BaseDashApp
from mlflow.models.model_utils import get_train_and_test_split_dt, split_in_CV_sets

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
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Timestamp column:"),
                                        dcc.Dropdown(id="timestamp_dropdown"),
                                    ]
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Target variable:"),
                                        dcc.Dropdown(id="target_dropdown"),
                                    ]
                                ),
                            ]
                        ),
                        dbc.Label("Categorical variables:"),
                        dcc.Dropdown(id="categorical_dropdown", multi=True),
                        dbc.Label("Numerical variables:"),
                        dcc.Dropdown(id="numerical_dropdown", multi=True),
                    ]
                ),
                dbc.Col(
                    [
                        dbc.Row([dcc.Graph(id="cv_graph", style={"height": "40vh", "width": "50vh"})]),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("No. of crossval sets:"),
                                        dcc.Slider(
                                            min=3,
                                            max=10,
                                            value=5,
                                            marks={3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10"},
                                            id="cv_slider",
                                        ),
                                    ]
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Validation set size:"),
                                        dcc.Dropdown(
                                            options=["40%", "30%", "20%", "10%", "5%"],
                                            value="20%",
                                            id="val_perc_dropdown",
                                        ),
                                    ]
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Train/Test split:"),
                                        dcc.Dropdown(
                                            options=["60/40%", "70/30%", "80/20%", "90/10%", "95/5%"],
                                            value="70/30%",
                                            id="train_test_dropdown",
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ]
                ),
                dbc.Col(),
            ]
        )
    ]


dashApp = BaseDashApp()
dashApp = dashApp.add_controls(define_app_controls()).add_training_content(define_training_content()).render_layout()


@dashApp.app.callback(
    Output("cv_graph", "figure"),
    Output("val_perc_dropdown", "value"),
    Output("train_test_dropdown", "value"),
    [
        Input("val_perc_dropdown", "value"),
        Input("train_test_dropdown", "value"),
        Input("cv_slider", "value"),
    ],
)
def update_cv_plot(
    val_perc_str: str,
    train_test_perc_str: str,
    no_of_cv_sets: float,
) -> Tuple[make_subplots, str, str]:
    """
    Update the training panel
    """

    if train_test_perc_str is None:
        train_test_perc_str = "70/30%"

    if val_perc_str is None:
        val_perc_str = "20%"

    val_perc = int(val_perc_str.replace("%", "")) / 100
    test_perc = 1 - int(train_test_perc_str.split("/")[0]) / 100

    if dashApp.timestamp_column != "":
        cv_fig = _plot_cv_sets(int(no_of_cv_sets), val_perc, test_perc)
    else:
        cv_fig = make_subplots()

    return cv_fig, val_perc_str, train_test_perc_str


@dashApp.app.callback(
    Output("dataset_dropdown", "options"),
    Output("dataset_dropdown", "value"),
    Output("timestamp_dropdown", "options"),
    Output("timestamp_dropdown", "value"),
    Output("categorical_dropdown", "options"),
    Output("categorical_dropdown", "value"),
    Output("numerical_dropdown", "options"),
    Output("numerical_dropdown", "value"),
    Output("target_dropdown", "options"),
    Output("target_dropdown", "value"),
    Output("cv_slider", "value"),
    [
        Input("dataset_dropdown", "value"),
        Input("timestamp_dropdown", "value"),
        Input("categorical_dropdown", "value"),
        Input("numerical_dropdown", "value"),
        Input("target_dropdown", "value"),
    ],
)
def update_training_panel(
    dataset_filename: str,
    timestamp_column: str,
    categorical_variables: list,
    numerical_variables: list,
    target_variable: str,
) -> Tuple[list, str, list, str, list, list, list, list, list, str, float]:
    """
    Update the training panel
    """

    panel_outcome = (
        _load_dataset(dataset_filename)
        + _select_timestamp_column(timestamp_column)
        + _select_categorical_variables(categorical_variables)
        + _select_numerical_variables(numerical_variables)
        + _select_target_variable(target_variable)
        + [5.0]
    )

    return tuple(panel_outcome)  # type: ignore


def _plot_cv_sets(n_cv_sets: int, val_perc: float, test_perc: float) -> list:
    cv_fig = make_subplots()

    if len(dashApp.X_y) == 0:
        return [cv_fig]

    X_y_dict = split_in_CV_sets(dashApp.X_y, n_cv_sets, val_perc)

    X_y_dict = get_train_and_test_split_dt(X_y_dict, dashApp.timestamp_column, test_perc)

    traces_to_plot = []

    y_left_margin = (
        (dashApp.X_y[dashApp.timestamp_column].max() - dashApp.X_y[dashApp.timestamp_column].min()).total_seconds()
        * 0.15
    ) / 3600
    text_pos = (
        (dashApp.X_y[dashApp.timestamp_column].max() - dashApp.X_y[dashApp.timestamp_column].min()).total_seconds()
        * 0.1
    ) / 3600

    i = 1
    for key, cv_set in X_y_dict.items():
        X_y = cv_set["X_y"]
        split_dt = cv_set["train_test_split_dt"]

        traces_to_plot += [
            {
                "x": [
                    X_y[X_y[dashApp.timestamp_column] < split_dt][dashApp.timestamp_column].min(),
                    X_y[X_y[dashApp.timestamp_column] < split_dt][dashApp.timestamp_column].max(),
                ],
                "y": [i + 1, i + 1],
                "line_color": "#1890ff",
                "mode": "lines",
                "line_width": 2,
                "showlegend": False,
            },
            {
                "x": [
                    X_y[X_y[dashApp.timestamp_column] >= split_dt][dashApp.timestamp_column].min(),
                    X_y[X_y[dashApp.timestamp_column] >= split_dt][dashApp.timestamp_column].max(),
                ],
                "y": [i + 1, i + 1],
                "line_color": "red",
                "mode": "lines",
                "line_width": 2,
                "showlegend": False,
            },
            {
                "x": [dashApp.X_y[dashApp.timestamp_column].min() - timedelta(hours=text_pos)],
                "y": [i + 1],
                "mode": "text",
                "text": key,
                "textfont": {"size": 13},
                "showlegend": False,
            },
        ]

        i += 1

    for trace in traces_to_plot:
        cv_fig.add_trace(go.Scatter(**trace))

    return cv_fig.update_layout(
        {
            "legend": None,
            "paper_bgcolor": "rgba(255,255,255,1)",
            "plot_bgcolor": "rgba(255,255,255,1)",
            "xaxis": dict(
                showgrid=True, linewidth=1, linecolor="black", gridcolor="rgba(100,100,100,.07)", gridwidth=1
            ),
            "yaxis": {"visible": False, "showticklabels": False},
            "xaxis_range": [
                dashApp.X_y[dashApp.timestamp_column].min() - timedelta(hours=y_left_margin),
                dashApp.X_y[dashApp.timestamp_column].max(),
            ],
            "yaxis_range": [1, i + 1],
        }
    )


def _select_target_variable(target_variable: str) -> list:
    """
    This function alllows the selection of the target variable
    """

    col_list = [item[0] for item in dashApp.X_y.dtypes.items() if item[1] in ["float", "int"]]

    if len(col_list) == 0:
        return [[], ""]

    if (target_variable is None) | dashApp.reset_dataset_values:
        target_variable = ""

    dashApp.target_variable = target_variable
    dashApp.reset_dataset_values = False

    return [col_list, target_variable]


def _select_numerical_variables(numerical_variables: list) -> list:
    """
    This function looks for parquet in the specified folder
    """

    col_list = [item[0] for item in dashApp.X_y.dtypes.items() if item[1] in ["float", "int"]]

    if len(col_list) == 0:
        return [[], []]

    if (numerical_variables is None) | dashApp.reset_dataset_values:
        numerical_variables = []

    dashApp.numerical_variables = numerical_variables

    return [col_list, numerical_variables]


def _select_categorical_variables(categorical_variables: list) -> list:
    """
    This function looks for parquet in the specified folder
    """

    col_list = [item[0] for item in dashApp.X_y.dtypes.items() if item[1] not in ["datetime64[ns]", "float", "int"]]

    if len(col_list) == 0:
        return [[], []]

    if (categorical_variables is None) | dashApp.reset_dataset_values:
        categorical_variables = []

    dashApp.categorical_variables = categorical_variables

    return [col_list, categorical_variables]


def _select_timestamp_column(timestamp_column: str) -> list:
    """
    This function looks for parquet in the specified folder
    """

    col_list = [item[0] for item in dashApp.X_y.dtypes.items() if item[1] == "datetime64[ns]"]

    if len(col_list) == 0:
        return [[], ""]

    if (timestamp_column is None) | dashApp.reset_dataset_values:
        timestamp_column = ""

    dashApp.timestamp_column = timestamp_column

    return [col_list, timestamp_column]


def _load_dataset(dataset_filename: str) -> list:
    """
    This function looks for parquet in the specified folder
    """

    dataset_list = [file for file in os.listdir(DATASETS_PATH) if file.endswith(".parquet")]

    if dataset_list == []:
        return [[], ""]

    if dataset_filename == dashApp.dataset_filename:
        return [dataset_list, dataset_filename]

    dashApp.reset_dataset_values = True

    if dataset_filename is None:
        dataset_filename = dataset_list[0]

    dashApp.reset_values(dataset_filename)

    return [dataset_list, dataset_filename]


def run_app() -> None:
    """
    Helper function to call the app.run_server() function with poetry
    """
    dashApp.app.run_server(debug=True, port="8086", use_reloader=False)


if __name__ == "__main__":
    dashApp.app.run_server(debug=True, port="8093")
