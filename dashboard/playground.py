# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import text_processing
import numpy as np

app = Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
# Load data
df_train = text_processing.load_and_process_data("./train.csv")
df_test = text_processing.load_and_process_data("./test.csv")

features = df_test.select_dtypes(include=np.number).columns[1:]
fig_train = px.histogram(df_train, x="char_count", marginal="box",
                             color_discrete_sequence=['turquoise'])


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(children={

    html.H1(
        children='Explaining USE Predictions with LIME',
        style={
            "textAlign": "center",
            "color": "#2BBBB"  # colors["text"]
        }
    ),

    html.H2("Overview"),
    dcc.Dropdown(
        features,  # Don't include ID
        "char_count",
        id="feature-selector"
    ),
    # ----------------------- Training Data
    html.Div([
        dbc.Row([
            dbc.Col(
                dbc.Container(
                    dbc.Card(
                        html.Div(
                            dbc.Table.from_dataframe(
                                df_train[["id", "text", "target"]]),
                            style={"maxHeight": "450px", "overflow": "scroll"},
                        ),
                        body=True,
                    ),
                    className="p-5",
                ), width={"size": 6, "order": "first"}
            ),
            dbc.Col(
                html.Div([
                    dcc.Graph(id="train-hist")
                ]), width={"size": 6, "order": "last"},
            )
        ]),
    ],
    ),

    # ----------------------- Testing Data
    html.Div([
        dbc.Row([
            dbc.Col(
                    html.Div([
                        dbc.Table.from_dataframe(df_test[["id", "text"]]),
                        # style={"maxHeight": "450px", "overflow": "scroll"},
                    ]), width={"size": 6, "order": "first"}
                    ),
            dbc.Col(
                html.Div([
                    dcc.Graph(id="test-hist")
                ]), width={"size": 6, "order": "last"},
            )
        ]),
    ]),

}, style={"padding": "5%"})



@app.callback(
    Output(component_id='train-hist', component_property='figure'),
    Input(component_id='feature-selector', component_property='value')
)
def update_figure(feature):
    fig_train = px.histogram(df_train, x=feature, marginal="box",
                             color_discrete_sequence=['turquoise'])
    fig_train.update_layout(height=600)
    return fig_train


@app.callback(
    Output(component_id='test-hist', component_property='figure'),
    Input(component_id='feature-selector', component_property='value')
)
def update_figure(feature):
    fig_test = px.histogram(df_test, x=feature, marginal="box",
                            color_discrete_sequence=['indianred'])
    fig_test.update_layout(height=600)
    return fig_test

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
