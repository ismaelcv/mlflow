from __future__ import annotations

import os
import pathlib

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html

DATASETS_PATH = pathlib.Path(__file__).parents[2] / "data" / "datasets"


class BaseDashApp:  # pylint: disable = too-many-instance-attributes
    """
    Class to elegantly costruct a dash app
    """

    def __init__(self) -> None:
        self.timestamp_column = ""  # type:str
        self.results_content = []  # type: list
        self.training_content = []  # type: list
        self.controls = []  # type: list
        self.dataset_filename = ""  # type: str
        self.X_y = pd.DataFrame()
        self.app = None  # type: dash.Dash
        self.categorical_variables = []  # type: list
        self.numerical_variables = []  # type: list
        self.target_variable = ""  # type:str
        self.reset_dataset_values = False

    def add_controls(self, controls: list) -> BaseDashApp:
        """
        Using this function you can add control functionality to the app
        Thigs like labels, spaces, inputs, checkboxes etc
        This elements need to be defined as a list of lists
        """
        self.controls = controls

        return self

    def reset_values(self, dataset_filename: str) -> None:
        """
        Auxiliary function to reset the values when a new dataset is selected
        """

        self.timestamp_column = ""
        self.target_variable = ""
        self.dataset_filename = dataset_filename
        self.X_y = pd.read_parquet(DATASETS_PATH / dataset_filename)
        self.dataset_filename = dataset_filename
        self.categorical_variables = []
        self.numerical_variables = []

    def add_training_content(self, content: list) -> BaseDashApp:
        """
        Adds content to the training panel
        """
        self.training_content += content

        return self

    def add_results_content(self, content: list) -> BaseDashApp:
        """
        adds content to the results panel
        """
        self.results_content += content

        return self

    def render_layout(self) -> BaseDashApp:
        """
        Once the control and the content has been defined, this function renders
        the elements and returns a dash.Dash app
        """

        # Define assets to be able to import images from the /assets folder

        assets_path = [dir for dir in os.walk(os.getcwd()) if dir[0].endswith("assets")][0][0]
        app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], assets_folder=assets_path)

        controls = html.Div(self.controls)
        sidebar = html.Div(
            [
                html.Br(),
                html.Div(
                    html.Img(src=app.get_asset_url("source_white.png"), width="250", height="80"),
                    style={"textAlign": "center"},
                ),
                html.Br(),
                controls,
            ],
            style={"background-color": "#f8f9fa"},
        )

        tabs = html.Div(
            [
                dcc.Tabs(
                    id="tabs",
                    value="tab-1",
                    children=[
                        dcc.Tab(value="tab-1", children=self.training_content, label="Training Grounds"),
                        dcc.Tab(value="tab-2", children=[], label="Plot"),
                    ],
                ),
                html.Div(id="tabs-content"),
            ]
        )

        app.layout = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(sidebar, width=3),
                        dbc.Col(tabs, width=9),
                    ]
                )
            ]
        )

        self.app = app

        return self
